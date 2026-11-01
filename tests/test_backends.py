"""Every backend passes the same contract: judge_many(state, questions) -> {qid: JudgeResult}."""

import json

import httpx
import pytest

from judge_core.backends import HumanQueue, JevJudge, LLMJudge, RuleJudge
from judge_core.backends import llm as llm_mod
from judge_core.backends.rule import lexicon_detector
from judge_core.types import JudgeError, JudgeResult, Question

QUESTIONS = {
    "leak": Question(kind="noul", instructions="Does it leak a secret?"),
    "team": Question(kind="choice", instructions="Which team?", criteria={"IT": "computers", "HR": "people"}),
    "urgency": Question(kind="score", instructions="How urgent?", criteria=["low", "mid", "high"]),
}
STATE = "My VPN password is hunter2 and Polaris ships next week."


def jev_transport(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request, body))
        return httpx.Response(200, json={
            "model": "jev-1.14.0",
            "answers": {
                "leak": {"type": "noul", "noul": 0.97},
                "team": {"type": "choice", "choice": "IT", "confidence": 0.8, "probabilities": {"IT": 0.9, "HR": 0.1}},
                "urgency": {"type": "score", "score": 1.6, "confidence": 0.5,
                            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "legend": {}},
            },
            "usage": {"input_tokens": 1000, "output_tokens": 0},
        })
    return httpx.MockTransport(handler)


def make_jev():
    seen = []
    return JevJudge(api_key="k", client=httpx.Client(transport=jev_transport(seen))), seen


def make_llm(monkeypatch):
    def fake_claude_json(prompt, system, schema, model="sonnet", timeout=180):
        assert schema["required"] == ["q0", "q1", "q2"]  # safe keys; real ids may contain '#'
        return {
            "q0": {"probability_yes": 0.9},
            "q1": {"answer": "IT", "confidence": 0.7},
            "q2": {"score": 2, "confidence": 0.6},
        }, 0.003
    monkeypatch.setattr(llm_mod, "claude_json", fake_claude_json)
    return LLMJudge()


@pytest.fixture(params=["jev", "llm"])
def model_backend(request, monkeypatch):
    if request.param == "jev":
        return make_jev()[0]
    return make_llm(monkeypatch)


def test_model_backend_contract(model_backend):
    out = model_backend.judge_many(STATE, QUESTIONS)
    assert set(out) == set(QUESTIONS)
    assert all(isinstance(r, JudgeResult) for r in out.values())
    assert out["leak"].answer is True and 0.5 < out["leak"].probability <= 1
    assert out["team"].answer == "IT"
    assert isinstance(out["urgency"].answer, float)
    assert all(0 <= r.confidence <= 1 for r in out.values())
    assert model_backend.judge_many(STATE, {}) == {}


def test_rule_backend_contract():
    rule = RuleJudge([lexicon_detector("codename", "codename", ["Polaris"])], rules={"leak": ["codename"]})
    out = rule.judge_many(STATE, QUESTIONS)
    assert set(out) == {"leak"}  # only answers what it has a rule for
    assert out["leak"].answer is True and out["leak"].probability == 1.0
    assert rule.judge_many("nothing here", QUESTIONS) == {}


def test_human_backend_contract(tmp_path):
    hq = HumanQueue(str(tmp_path / "q.db"))
    assert hq.judge_many(STATE, QUESTIONS) == {}  # never decides synchronously
    pending = hq.pending()
    assert [p["question_id"] for p in pending] == list(QUESTIONS)
    hq.resolve(pending[0]["id"], True)
    assert hq.labels() == [{"question_id": "leak", "state": STATE, "answer": True}]
    assert len(hq.pending()) == 2


def test_jev_wire_format():
    jev, seen = make_jev()
    out = jev.judge_many(STATE, QUESTIONS)
    req, body = seen[0]
    assert len(seen) == 1  # all questions in one request
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["authorization"] == "Bearer k"
    assert body["state"] == STATE and body["model"] == "jev-1.14.0"
    assert body["questions"]["team"] == {"type": "choice", "instructions": "Which team?",
                                         "criteria": {"IT": "computers", "HR": "people"}}
    assert body["questions"]["leak"] == {"type": "noul", "instructions": "Does it leak a secret?"}
    assert out["leak"].model == "jev-1.14.0"
    assert out["leak"].cost_usd == pytest.approx(1000 * 0.042e-6 / 3)


def test_jev_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"model": "m", "answers": {"leak": {"type": "noul", "noul": 0.1}},
                                         "usage": {"input_tokens": 5, "output_tokens": 0}})
    jev = JevJudge(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = jev.judge_many("x", {"leak": QUESTIONS["leak"]})
    assert calls["n"] == 3 and out["leak"].answer is False


def test_jev_errors_raise_judge_error():
    jev = JevJudge(api_key="k", client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(401, json={"detail": "bad key"}))))
    with pytest.raises(JudgeError):
        jev.judge_many("x", {"leak": QUESTIONS["leak"]})


def test_jev_base_url_override_for_local_engine():
    seen = []
    jev = JevJudge(base_url="http://127.0.0.1:8089/", client=httpx.Client(transport=jev_transport(seen)))
    jev.judge_many(STATE, QUESTIONS)
    assert str(seen[0][0].url) == "http://127.0.0.1:8089/v1/systemone"
    assert "authorization" not in seen[0][0].headers


def test_question_validation():
    with pytest.raises(ValueError):
        Question(kind="choice", instructions="x")
    with pytest.raises(ValueError):
        Question(kind="score", instructions="x", criteria=["only one"])
    with pytest.raises(ValueError):
        Question(kind="noul", instructions="x", criteria={"maybe": "y"})
