from judge_core import AuditLog, Cascade, Item, load_policies
from judge_core.backends import FakeJudge, HumanQueue, RuleJudge
from judge_core.backends.rule import lexicon_detector

POLICIES = load_policies("""
id: leak
role: gating
kind: noul
instructions: leak?
thresholds: {yes_above: 0.8, no_below: 0.1}
fallback: [llm, human]
---
id: team
role: routing
kind: choice
instructions: team?
criteria: {IT: computers, HR: people}
thresholds: {min_confidence: 0.6}
fallback: [human]
---
id: todo
role: extraction
kind: noul
instructions: todo?
---
id: urgency
role: ranking
kind: score
instructions: urgent?
criteria: [low, high]
""")


def items(*ids):
    return {i: Item(POLICIES[i.split("#")[0]], POLICIES[i.split("#")[0]].question()) for i in ids}


def by_policy(values):
    """Responder answering from a {policy_id: value} map (None -> skip)."""
    def respond(state, qid, q):
        return values.get(qid.split("#")[0])
    return respond


def test_confident_primary_decides_in_one_call():
    jev = FakeJudge(by_policy({"leak": 0.95, "team": ("IT", 0.9), "todo": 0.7, "urgency": (1.0, 0.8)}))
    llm = FakeJudge(by_policy({}))
    d = Cascade(primary=jev, llm=llm).run("s", items("leak", "team", "todo", "urgency"))
    assert len(jev.calls) == 1 and len(jev.calls[0]) == 4
    assert llm.calls == []
    assert d["leak"].answer is True and d["leak"].decided_by == "primary"
    assert d["team"].answer == "IT"
    assert d["todo"].answer is True
    assert d["urgency"].answer == 1.0


def test_gray_escalates_to_llm_batched():
    jev = FakeJudge(by_policy({"leak": 0.5}))
    llm = FakeJudge(by_policy({"leak": 0.02}))
    d = Cascade(primary=jev, llm=llm).run("s", items("leak#0", "leak#1", "leak#2"))
    assert len(llm.calls) == 1 and len(llm.calls[0]) == 3
    assert all(x.answer is False and x.decided_by == "llm" for x in d.values())
    assert [s.outcome for s in d["leak#0"].path] == ["gray", "decided"]


def test_gating_still_gray_goes_to_human_and_stays_blocked(tmp_path):
    hq = HumanQueue(str(tmp_path / "q.db"))
    jev = FakeJudge(by_policy({"leak": 0.5}))
    llm = FakeJudge(by_policy({"leak": 0.4}))
    d = Cascade(primary=jev, llm=llm, human=hq).run("s", items("leak"))
    assert d["leak"].status == "pending_human" and d["leak"].answer is True
    assert len(hq.pending()) == 1


def test_gating_gray_without_human_never_passes():
    d = Cascade(primary=FakeJudge(by_policy({"leak": 0.5}))).run("s", items("leak"))
    assert d["leak"].status == "gray_default" and d["leak"].answer is True


def test_gating_fails_closed_when_every_layer_is_down():
    d = Cascade(primary=FakeJudge(by_policy({}), fail=True),
                llm=FakeJudge(by_policy({}), fail=True)).run("s", items("leak"))
    assert d["leak"].status == "failsafe" and d["leak"].answer is True
    assert [s.outcome for s in d["leak"].path] == ["error", "error"]


def test_primary_outage_falls_back_to_llm():
    d = Cascade(primary=FakeJudge(by_policy({}), fail=True),
                llm=FakeJudge(by_policy({"leak": 0.01}))).run("s", items("leak"))
    assert d["leak"].answer is False and d["leak"].decided_by == "llm"


def test_routing_gray_goes_to_human_queue(tmp_path):
    hq = HumanQueue(str(tmp_path / "q.db"))
    d = Cascade(primary=FakeJudge(by_policy({"team": ("HR", 0.3)})), human=hq).run("s", items("team"))
    assert d["team"].status == "pending_human" and d["team"].answer is None


def test_extraction_outage_marks_unprocessed():
    d = Cascade(primary=FakeJudge(by_policy({}), fail=True)).run("s", items("todo"))
    assert d["todo"].status == "unprocessed"


def test_rule_layer_short_circuits():
    rule = RuleJudge([lexicon_detector("cn", "codename", ["Polaris"])], rules={"leak": ["codename"]})
    jev = FakeJudge(by_policy({"leak": 0.01}))
    d = Cascade(primary=jev, rule=rule).run("Polaris launch", items("leak"))
    assert d["leak"].decided_by == "rule" and d["leak"].answer is True
    assert jev.calls == []


def test_audit_log_records_every_layer(tmp_path):
    audit = AuditLog(str(tmp_path / "a.db"))
    jev = FakeJudge(by_policy({"leak": 0.5}))
    llm = FakeJudge(by_policy({"leak": 0.95}))
    Cascade(primary=jev, llm=llm, audit=audit).run("secret text", items("leak"), pipeline="dlp")
    [rec] = audit.records("dlp")
    assert len(rec["input_hash"]) == 64 and "secret text" not in str(rec)
    [dec] = rec["decisions"]
    assert [s["layer"] for s in dec["path"]] == ["primary", "llm"]
    assert dec["decided_by"] == "llm"


def test_load_policies_from_directory():
    from pathlib import Path

    import office_judge
    pols = load_policies(Path(office_judge.__file__).parent / "policies")
    assert {"dlp_sensitive", "ticket_team", "ticket_urgency", "is_action_item", "action_owner"} <= set(pols)
    assert pols["dlp_sensitive"].effective_fail_mode == "closed"


def test_model_specific_thresholds():
    from judge_core.types import noul_result
    pol = load_policies("""
id: leak
role: gating
kind: noul
instructions: leak?
thresholds: {yes_above: 0.5, no_below: 0.05}
model_thresholds:
  kev-4b: {yes_above: 0.5, no_below: 0.15}
""")["leak"]
    assert pol.verdict(noul_result(0.1, model="local:jev-1.14.0")) == (False, False)  # gray under default
    assert pol.verdict(noul_result(0.1, model="kev-4b")) == (True, False)  # decided "no" for kev
