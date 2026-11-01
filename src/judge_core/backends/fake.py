"""Scripted backend for tests and offline demos."""

from __future__ import annotations

from typing import Callable

from judge_core.types import JudgeError, JudgeResult, Question, State, noul_result

Responder = Callable[[State, str, Question], float | str | tuple[str, float] | None]


class FakeJudge:
    """responder(state, qid, question) returns:
    noul -> P(yes); choice -> (option, confidence); score -> (score, confidence) as (str|float, float);
    None -> leave unanswered. Set fail=True to simulate an outage."""

    def __init__(self, responder: Responder, name: str = "fake", fail: bool = False):
        self.responder = responder
        self.name = name
        self.fail = fail
        self.calls: list[dict[str, Question]] = []

    def judge_many(self, state: State, questions: dict[str, Question]) -> dict[str, JudgeResult]:
        self.calls.append(questions)
        if self.fail:
            raise JudgeError(f"{self.name} is down")
        out = {}
        for qid, q in questions.items():
            r = self.responder(state, qid, q)
            if r is None:
                continue
            if q.kind == "noul":
                out[qid] = noul_result(float(r), model=self.name)  # type: ignore[arg-type]
            else:
                ans, conf = r  # type: ignore[misc]
                out[qid] = JudgeResult(answer=ans if q.kind == "choice" else float(ans), confidence=conf, model=self.name)
        return out
