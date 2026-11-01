import pytest

from judge_core.eval import calibration, compare, rescue_rate, sweep_thresholds
from judge_core.backends import FakeJudge
from judge_core.types import Question


def test_sweep_widening_gray_zone_trades_coverage_for_errors():
    probs = [0.95, 0.9, 0.6, 0.45, 0.1, 0.05]
    labels = [True, True, False, True, False, False]
    pts = sweep_thresholds(probs, labels, steps=10)
    assert pts[0].auto_rate == 1.0 and pts[0].error_rate == pytest.approx(2 / 6)
    assert pts[-1].auto_rate == 0.0
    mid = next(p for p in pts if p.yes_above >= 0.85)
    assert mid.error_rate == 0.0


def test_calibration_perfect_and_bad():
    assert calibration([1.0, 1.0, 0.0, 0.0], [True, True, False, False]).ece == pytest.approx(0.0)
    assert calibration([1.0, 1.0], [False, False]).ece == pytest.approx(1.0)


def test_compare_reports_accuracy_and_latency():
    q = {"x": Question(kind="noul", instructions="?")}
    ds = [("a", q, {"x": True}), ("b", q, {"x": False})]
    always_yes = FakeJudge(lambda s, i, qq: 0.9, name="yes")
    rows = compare([always_yes], ds)
    assert rows[0]["backend"] == "yes" and rows[0]["accuracy"] == 0.5 and rows[0]["answered"] == 2


def test_rescue_rate():
    def dec(qid, prim, llm):
        return {"qid": qid, "path": [
            {"layer": "primary", "outcome": "gray", "answer": prim},
            {"layer": "llm", "outcome": "decided", "answer": llm},
        ]}
    decisions = [dec("a", True, False), dec("b", True, False), dec("c", False, False)]
    r = rescue_rate(decisions, {"a": False, "b": True, "c": False})
    assert r == {"n": 3, "rescued": pytest.approx(1 / 3), "broke": pytest.approx(1 / 3), "agreed": pytest.approx(1 / 3)}
