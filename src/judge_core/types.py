"""Core data types shared by every judge backend."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

Kind = Literal["noul", "choice", "score"]
State = str | dict[str, Any] | list[Any]


class Question(BaseModel):
    """One typed question, mirroring Jev's question objects.

    criteria:
      noul   -> optional {"true": ..., "false": ...}
      choice -> {option: description}
      score  -> [level descriptions, low to high]
    """

    kind: Kind
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _check_criteria(self) -> "Question":
        if self.kind == "choice" and not (isinstance(self.criteria, dict) and self.criteria):
            raise ValueError("choice question needs criteria as {option: description}")
        if self.kind == "score" and not (isinstance(self.criteria, list) and len(self.criteria) >= 2):
            raise ValueError("score question needs at least two levels")
        if self.kind == "noul" and self.criteria is not None:
            if not isinstance(self.criteria, dict) or not set(self.criteria) <= {"true", "false"}:
                raise ValueError("noul criteria may only have 'true' and 'false' keys")
        return self

    @property
    def options(self) -> list[str]:
        if self.kind == "choice":
            return list(self.criteria)  # type: ignore[arg-type]
        if self.kind == "score":
            return [str(i) for i in range(len(self.criteria))]  # type: ignore[arg-type]
        return ["true", "false"]


class JudgeResult(BaseModel):
    answer: bool | str | float
    probability: float | None = None  # noul: P(yes)
    probabilities: dict[str, float] | None = None
    confidence: float  # choice/score: backend value; noul: |2p - 1|
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    model: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


def noul_result(p: float, **kw: Any) -> JudgeResult:
    return JudgeResult(answer=p >= 0.5, probability=p, confidence=abs(2 * p - 1), **kw)


class JudgeError(RuntimeError):
    """A backend failed (network, rate limit, bad response)."""


@runtime_checkable
class Judge(Protocol):
    name: str

    def judge_many(self, state: State, questions: dict[str, Question]) -> dict[str, JudgeResult]:
        """Answer every question against the same state, ideally in one call."""
        ...
