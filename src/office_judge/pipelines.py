"""The three office pipelines on top of judge-core.

Every pipeline sends one primary-judge request per message: all sentences and
all questions go into a single call (Jev's speculative fan-out pattern).
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from judge_core import AuditLog, Cascade, Decision, Item, load_policies
from judge_core.backends.human import HumanQueue
from judge_core.types import Judge

from office_judge.detectors import LABELS, build_rule_judge
from office_judge.text import Vault, clean, split_sentences, window
from office_judge.writer import Writer

HERE = Path(__file__).parent
UNASSIGNED = "未指定"


def load_company(path: str | Path | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or HERE / "company.yaml").read_text(encoding="utf-8"))


def _trace(d: Decision) -> dict[str, Any]:
    return {
        "status": d.status,
        "decided_by": d.decided_by,
        "path": [
            {k: v for k, v in s.model_dump().items() if v not in (None, "", 0.0) or k == "layer"}
            for s in d.path
        ],
    }


# ---------- results ----------

class SentenceFlag(BaseModel):
    index: int
    text: str  # masked text
    status: str
    probability: float | None
    trace: dict[str, Any]


class DlpResult(BaseModel):
    action: str  # pass | mask | block
    reasons: list[str] = Field(default_factory=list)
    masked_text: str
    mapping: dict[str, str] = Field(default_factory=dict)  # placeholder -> original, for unmask
    rule_hits: list[dict[str, str]] = Field(default_factory=list)
    flagged: list[SentenceFlag] = Field(default_factory=list)
    sentences_checked: int = 0


class TicketResult(BaseModel):
    is_ticket: bool = True  # False: not a service request, routing skipped
    team: str | None  # None means waiting in the human queue (or not a ticket)
    team_confidence: float | None
    urgency: float | None
    urgency_label: str | None
    trace: dict[str, Any]


class ActionItem(BaseModel):
    index: int
    sentence: str
    owner: str | None
    owner_confidence: float | None
    probability: float | None
    task: str | None = None
    due_date: str | None = None
    due_text: str | None = None


class ActionResult(BaseModel):
    items: list[ActionItem]
    sentences_checked: int
    postprocessed: bool


# ---------- pipeline ----------

class OfficeJudge:
    def __init__(
        self,
        primary: Judge,
        llm: Judge | None = None,
        writer: Writer | None = None,
        human: HumanQueue | None = None,
        audit: AuditLog | None = None,
        company: dict[str, Any] | None = None,
        policy_dir: str | Path | None = None,
    ):
        self.company = company or load_company()
        self.policies = load_policies(Path(policy_dir or HERE / "policies"))
        self.rule = build_rule_judge(self.company.get("codenames", []), self.company.get("customers", []))
        self.cascade = Cascade(primary=primary, llm=llm, human=human, audit=audit)
        self.writer = writer

    # --- DLP (gating) ---
    def dlp(self, text: str, explain: bool = True) -> DlpResult:
        text = clean(text)
        findings = self.rule.scan(text)
        vault = Vault()
        masked = vault.mask_findings(text, findings)  # known formats never leave the process
        sentences = split_sentences(masked)
        pol = self.policies["dlp_sensitive"]
        items = {
            f"dlp#{i}": Item(pol, pol.question(f"{pol.instructions}\n\n句子：「{s.text}」"), {"sentence": s.text})
            for i, s in enumerate(sentences)
        }
        decisions = self.cascade.run(masked, items, pipeline="dlp") if items else {}

        reasons: list[str] = []
        secrets = sorted({LABELS[f.detector] for f in findings if f.category == "secret"})
        if secrets:
            reasons.append(f"含有{('、'.join(secrets))}。金鑰或密碼一旦貼出就應視為外洩，請移除並盡快更換。")
        pii = sorted({LABELS[f.detector] for f in findings if f.category == "pii"})
        if pii:
            reasons.append(f"含有個人資料（{'、'.join(pii)}），已自動遮罩。")
        if any(f.category == "codename" for f in findings):
            reasons.append("提到內部專案代號，已自動遮罩。")
        if any(f.category == "customer" for f in findings):
            reasons.append("提到客戶名稱，已自動遮罩。")

        flagged, to_mask, unresolved = [], [], []
        for i, s in enumerate(sentences):
            d = decisions[f"dlp#{i}"]
            if d.answer is not True:
                continue
            prob = next((st.probability for st in reversed(d.path) if st.probability is not None), None)
            flagged.append(SentenceFlag(index=i, text=s.text, status=d.status, probability=prob, trace=_trace(d)))
            (to_mask if d.status == "decided" else unresolved).append(s)

        if to_mask:
            reason = None
            if explain and self.writer:
                try:
                    reason = self.writer.explain(" ".join(s.text for s in to_mask), "語意判斷為未公開或機密資訊")
                except Exception:  # writer is best-effort; the decision already stands
                    reason = None
            reasons.append((reason or f"有 {len(to_mask)} 句可能透露未公開資訊") + "（已整句遮罩）")
            masked = vault.replace_spans(masked, [(s.start, s.end, "機密內容") for s in to_mask])
        if unresolved:
            reasons.append(f"有 {len(unresolved)} 句無法自動確認是否機密，已送人工審核，審核完成前先不送出。")

        if secrets or unresolved:
            action = "block"
        elif vault.mapping:
            action = "mask"
        else:
            action = "pass"
        return DlpResult(
            action=action,
            reasons=reasons,
            masked_text=masked,
            mapping=vault.mapping,
            rule_hits=[{"type": LABELS.get(f.detector, f.detector), "category": f.category} for f in findings],
            flagged=flagged,
            sentences_checked=len(sentences),
        )

    # --- ticket routing (routing + ranking) ---
    def route_ticket(self, text: str) -> TicketResult:
        text = clean(text)
        vault = Vault()
        masked = vault.mask_findings(text, [f for f in self.rule.scan(text) if f.category in ("pii", "secret")])
        team_pol, urg_pol = self.policies["ticket_team"], self.policies["ticket_urgency"]
        req_pol = self.policies["ticket_is_request"]
        # The human fallback is applied below, only once we know this is a ticket,
        # so chat or meeting notes never land in the routing queue.
        team_no_human = team_pol.model_copy(update={"fallback": [x for x in team_pol.fallback if x != "human"]})
        team_q = team_pol.question(criteria=self.company["teams"])
        items = {
            "is_request": Item(req_pol, req_pol.question()),
            "team": Item(team_no_human, team_q),
            "urgency": Item(urg_pol, urg_pol.question()),
        }
        d = self.cascade.run(masked, items, pipeline="ticket")
        trace = {k: _trace(d[k]) for k in ("is_request", "team", "urgency")}
        if d["is_request"].answer is False:
            return TicketResult(is_ticket=False, team=None, team_confidence=None,
                                urgency=None, urgency_label=None, trace=trace)
        if d["team"].status != "decided" and "human" in team_pol.fallback and self.cascade.human:
            self.cascade.human.enqueue(masked, "team", team_q)
            d["team"].status, d["team"].answer = "pending_human", None
            trace["team"] = _trace(d["team"])
        urgency = d["urgency"].answer
        levels = urg_pol.criteria or []
        label = levels[min(len(levels) - 1, max(0, round(urgency)))].split("：")[0] if urgency is not None else None
        return TicketResult(
            team=d["team"].answer,
            team_confidence=d["team"].confidence,
            urgency=urgency,
            urgency_label=label,
            trace=trace,
        )

    # --- action items (extraction + routing) ---
    def extract_actions(
        self, text: str, attendees: list[str] | None = None, today: dt.date | None = None
    ) -> ActionResult:
        text = clean(text)
        attendees = attendees or self.company.get("default_attendees", [])
        vault = Vault()
        masked = vault.mask_findings(text, [f for f in self.rule.scan(text) if f.category in ("pii", "secret")])
        sentences = split_sentences(masked)
        act_pol, own_pol = self.policies["is_action_item"], self.policies["action_owner"]
        owners = {name: f"與會者 {name}" for name in attendees}
        owners[UNASSIGNED] = "句中沒有指出負責人，或負責人不在與會者名單中"
        items: dict[str, Item] = {}
        for i, s in enumerate(sentences):
            before, after = window(sentences, i)
            ctx = f"\n\n前文：「{before}」\n本句：「{s.text}」\n後文：「{after}」"
            items[f"act#{i}"] = Item(act_pol, act_pol.question(act_pol.instructions + ctx))
            # speculative: asked for every sentence in the same request, used only for action items
            items[f"own#{i}"] = Item(own_pol, own_pol.question(own_pol.instructions + ctx, criteria=owners))
        d = self.cascade.run({"attendees": attendees, "text": masked}, items, pipeline="actions") if items else {}

        found: list[ActionItem] = []
        for i, s in enumerate(sentences):
            act = d[f"act#{i}"]
            if act.answer is not True:
                continue
            own = d[f"own#{i}"]
            owner = own.answer if own.answer not in (None, UNASSIGNED) else None
            prob = next((st.probability for st in act.path if st.probability is not None), None)
            found.append(ActionItem(index=i, sentence=s.text, owner=owner, owner_confidence=own.confidence, probability=prob))

        postprocessed = False
        if found and self.writer:
            try:
                resolved = self.writer.resolve_actions(
                    [{"i": a.index, "sentence": a.sentence, "owner": a.owner} for a in found],
                    (today or dt.date.today()).isoformat(),
                )
                by_i = {r["i"]: r for r in resolved}
                for a in found:
                    r = by_i.get(a.index, {})
                    a.task, a.due_date, a.due_text = r.get("task"), r.get("due_date"), r.get("due_text")
                postprocessed = True
            except Exception:
                postprocessed = False
        for a in found:
            a.sentence = vault.unmask(a.sentence)
            a.task = vault.unmask(a.task) if a.task else None
        return ActionResult(items=found, sentences_checked=len(sentences), postprocessed=postprocessed)

    def analyze(self, text: str, attendees: list[str] | None = None) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_dlp = ex.submit(self.dlp, text)
            f_ticket = ex.submit(self.route_ticket, text)
            f_act = ex.submit(self.extract_actions, text, attendees)
            return {
                "dlp": f_dlp.result().model_dump(),
                "ticket": f_ticket.result().model_dump(),
                "actions": f_act.result().model_dump(),
            }
