"""Jev (TypeSafe System One) backend.

Speaks the POST /v1/systemone wire format, so pointing base_url at a
jev-rs server (`jev serve`) gives a fully local backend with no code change.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from judge_core.types import JudgeError, JudgeResult, Question, State, noul_result

PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000
RETRY_STATUS = {429, 529, 502, 503, 504}


def to_wire(q: Question) -> dict[str, Any]:
    body: dict[str, Any] = {"type": q.kind, "instructions": q.instructions}
    if q.criteria is not None:
        body["criteria"] = q.criteria
    return body


def from_wire(q: Question, a: dict[str, Any], **meta: Any) -> JudgeResult:
    if a.get("type") != q.kind:
        raise JudgeError(f"expected {q.kind} answer, got {a.get('type')}")
    if q.kind == "noul":
        return noul_result(float(a["noul"]), **meta)
    if q.kind == "choice":
        return JudgeResult(
            answer=a["choice"],
            probabilities=a["probabilities"],
            confidence=float(a["confidence"]),
            **meta,
        )
    return JudgeResult(
        answer=float(a["score"]),
        probabilities=a["probabilities"],
        confidence=float(a["confidence"]),
        **meta,
    )


class JevJudge:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 4,
        long_context: bool = False,
        client: httpx.Client | None = None,
        price_per_input_token: float | None = None,  # default: Jev price on TypeSafe, 0 elsewhere
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-1.14.0")
        self.max_retries = max_retries
        self.long_context = long_context
        hosted = self.base_url == "https://api.typesafe.ai"
        if price_per_input_token is None:
            price_per_input_token = PRICE_PER_INPUT_TOKEN if hosted else 0.0
        self.price = price_per_input_token
        self.name = f"jev:{self.model}" if hosted else f"local:{self.model}"
        self._client = client or httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.long_context:
            h["TypeSafe-Danger-Zone-Long-Context"] = "true"
        return h

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/v1/systemone"
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post(url, json=body, headers=self._headers())
            except httpx.HTTPError as e:
                if attempt == self.max_retries:
                    raise JudgeError(f"jev connection failed: {e}") from e
                time.sleep(0.5 * 2**attempt)
                continue
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                retry_after = r.headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else 0.5 * 2**attempt)
                continue
            if r.status_code != 200:
                raise JudgeError(f"jev HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        raise JudgeError("jev retries exhausted")

    def judge_many(self, state: State, questions: dict[str, Question]) -> dict[str, JudgeResult]:
        if not questions:
            return {}
        body = {
            "model": self.model,
            "state": state,
            "questions": {qid: to_wire(q) for qid, q in questions.items()},
        }
        t0 = time.perf_counter()
        data = self._post(body)
        latency = (time.perf_counter() - t0) * 1000
        answers = data.get("answers", {})
        tokens = data.get("usage", {}).get("input_tokens", 0)
        cost_each = tokens * self.price / len(questions)
        model = data.get("model", self.model)
        out = {}
        for qid, q in questions.items():
            if qid not in answers:
                raise JudgeError(f"jev response missing answer for {qid}")
            out[qid] = from_wire(
                q, answers[qid], latency_ms=latency, cost_usd=cost_each, model=model, raw=answers[qid]
            )
        return out
