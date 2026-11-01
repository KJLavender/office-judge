"""Cascade: Rule -> primary (Jev) -> fallback layers, batched per layer.

Each layer receives *all* still-open questions for one state in a single
call, matching Jev's many-questions-per-request design.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from judge_core.audit import AuditLog
from judge_core.backends.human import HumanQueue
from judge_core.backends.rule import RuleJudge
from judge_core.policy import Policy
from judge_core.types import Judge, JudgeError, JudgeResult, Question, State


class Step(BaseModel):
    layer: str
    outcome: str  # decided | gray | error | queued | skipped
    answer: Any = None
    probability: float | None = None
    confidence: float | None = None
    model: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


class Decision(BaseModel):
    qid: str
    policy_id: str
    role: str
    answer: Any = None
    # decided: a layer was confident; pending_human: waiting in queue (gating stays blocked);
    # failsafe / failopen: every layer failed, fail_mode applied;
    # gray_default: still gray after the last layer, role default applied;
    # unprocessed: extraction skipped because every layer failed
    status: str
    decided_by: str = ""
    confidence: float | None = None
    path: list[Step] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.path)


@dataclass
class Item:
    policy: Policy
    question: Question
    context: dict[str, Any] = field(default_factory=dict)  # stored with human-queue items


@dataclass
class Cascade:
    primary: Judge
    llm: Judge | None = None
    rule: RuleJudge | None = None
    human: HumanQueue | None = None
    audit: AuditLog | None = None

    def _chain(self, pol: Policy) -> list[str]:
        chain = (["rule"] if self.rule else []) + ["primary"]
        for layer in pol.fallback:
            if layer == "llm" and self.llm:
                chain.append("llm")
            elif layer == "human" and self.human:
                chain.append("human")
        return chain

    def run(self, state: State, items: dict[str, Item], pipeline: str = "default") -> dict[str, Decision]:
        t0 = time.perf_counter()
        chains = {qid: self._chain(it.policy) for qid, it in items.items()}
        decisions = {
            qid: Decision(qid=qid, policy_id=it.policy.id, role=it.policy.role, status="open")
            for qid, it in items.items()
        }
        leaning: dict[str, tuple[Any, float, str]] = {}  # last gray answer per question
        model_answered: dict[str, bool] = {qid: False for qid in items}
        step = 0
        while True:
            open_q = [q for q, d in decisions.items() if d.status == "open" and step < len(chains[q])]
            if not open_q:
                break
            groups: dict[str, list[str]] = {}
            for qid in open_q:
                groups.setdefault(chains[qid][step], []).append(qid)
            for layer, qids in groups.items():
                self._run_layer(layer, state, items, qids, decisions, leaning, model_answered)
            step += 1

        for qid, d in decisions.items():
            if d.status == "open":
                self._finalize(d, items[qid].policy, leaning.get(qid), model_answered[qid])

        if self.audit:
            self.audit.write(
                pipeline, state, [d.model_dump() for d in decisions.values()],
                (time.perf_counter() - t0) * 1000, sum(d.cost_usd for d in decisions.values()),
            )
        return decisions

    def _run_layer(self, layer, state, items, qids, decisions, leaning, model_answered) -> None:
        if layer == "human":
            for qid in qids:
                self.human.enqueue(state, qid, items[qid].question, items[qid].context)  # type: ignore[union-attr]
                d = decisions[qid]
                d.path.append(Step(layer="human", outcome="queued"))
                d.status, d.decided_by = "pending_human", "human"
                # gating stays blocked until a person says otherwise
                d.answer = True if d.role == "gating" else None
            return
        backend = {"rule": self.rule, "primary": self.primary, "llm": self.llm}[layer]
        questions = {qid: items[qid].question for qid in qids}
        try:
            results: dict[str, JudgeResult] = backend.judge_many(state, questions)
        except JudgeError as e:
            for qid in qids:
                decisions[qid].path.append(Step(layer=layer, outcome="error", error=str(e)))
            return
        for qid in qids:
            d = decisions[qid]
            r = results.get(qid)
            if r is None:
                d.path.append(Step(layer=layer, outcome="skipped"))
                continue
            model_answered[qid] = True
            decided, answer = items[qid].policy.verdict(r)
            d.path.append(Step(
                layer=layer, outcome="decided" if decided else "gray", answer=answer,
                probability=r.probability, confidence=r.confidence, model=r.model,
                latency_ms=r.latency_ms, cost_usd=r.cost_usd,
            ))
            if decided:
                d.status, d.answer, d.decided_by, d.confidence = "decided", answer, layer, r.confidence
            else:
                leaning[qid] = (answer, r.confidence, layer)

    @staticmethod
    def _finalize(d: Decision, pol: Policy, lean: tuple | None, answered: bool) -> None:
        if not answered or lean is None:
            if pol.role == "gating":
                closed = pol.effective_fail_mode == "closed"
                d.status, d.answer = ("failsafe", True) if closed else ("failopen", False)
            elif pol.role == "extraction":
                d.status, d.answer = "unprocessed", None
            else:
                d.status, d.answer = "failopen", None
            d.decided_by = "fail_mode"
            return
        answer, conf, layer = lean
        d.status, d.decided_by, d.confidence = "gray_default", layer, conf
        if pol.role == "gating":
            d.answer = True  # never pass a gray gating decision
        elif pol.role == "routing":
            d.answer = None
        else:
            d.answer = answer
