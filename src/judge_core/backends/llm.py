"""LLM backend via the local Claude Code CLI (subscription auth, no API key).

Used as the second cascade layer and as the quality ceiling in comparisons.
`claude_json` is also used by office_judge.writer to phrase block reasons,
since Jev cannot explain itself.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any

from judge_core.types import JudgeError, JudgeResult, Question, State, noul_result

DEFAULT_MODEL = os.environ.get("JUDGE_LLM_MODEL", "sonnet")


def claude_json(
    prompt: str,
    system: str,
    schema: dict[str, Any],
    model: str = DEFAULT_MODEL,
    timeout: float = 180,
) -> tuple[dict[str, Any], float]:
    """Run `claude -p` with a JSON schema; return (structured_output, cost_usd)."""
    exe = shutil.which("claude")
    if not exe:
        raise JudgeError("claude CLI not found on PATH")
    cmd = [
        exe, "-p",
        "--output-format", "json",
        "--model", model,
        "--effort", "medium",
        "--system-prompt", system,
        "--json-schema", json.dumps(schema, ensure_ascii=False),
        "--tools", "",
        "--setting-sources", "",
        "--no-session-persistence",
    ]
    try:
        proc = subprocess.run(
            cmd, input=prompt, text=True, encoding="utf-8", capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise JudgeError("claude CLI timed out") from e
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise JudgeError(f"claude CLI returned non-JSON: {proc.stdout[:200]} {proc.stderr[:200]}") from e
    if data.get("is_error") or "structured_output" not in data:
        raise JudgeError(f"claude CLI error: {str(data.get('result'))[:300]}")
    return data["structured_output"], float(data.get("total_cost_usd") or 0.0)


def _answer_schema(q: Question) -> dict[str, Any]:
    if q.kind == "noul":
        return {
            "type": "object",
            "properties": {"probability_yes": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["probability_yes"],
        }
    if q.kind == "choice":
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": q.options},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["answer", "confidence"],
        }
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": len(q.criteria) - 1},  # type: ignore[arg-type]
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["score", "confidence"],
    }


def _describe(qid: str, q: Question) -> str:
    lines = [f"### {qid} ({q.kind})", q.instructions]
    if q.kind == "choice":
        lines += [f"- {k}: {v}" for k, v in q.criteria.items()]  # type: ignore[union-attr]
    elif q.kind == "score":
        lines += [f"- level {i}: {v}" for i, v in enumerate(q.criteria)]  # type: ignore[arg-type]
    elif q.criteria:
        lines += [f"- yes means: {q.criteria.get('true', '')}", f"- no means: {q.criteria.get('false', '')}"]  # type: ignore[union-attr]
    return "\n".join(lines)


SYSTEM = (
    "You are a careful classifier. Answer every question strictly from the given state. "
    "Report honest probabilities: use values near 0.5 when the state is ambiguous."
)


class LLMJudge:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.name = f"llm:{model}"

    def judge_many(self, state: State, questions: dict[str, Question]) -> dict[str, JudgeResult]:
        if not questions:
            return {}
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        # schema property names must match ^[a-zA-Z0-9_.-]{1,64}$, so use positional keys
        keys = {qid: f"q{i}" for i, qid in enumerate(questions)}
        prompt = "## State\n" + state_text + "\n\n## Questions\n" + "\n\n".join(
            _describe(keys[qid], q) for qid, q in questions.items()
        )
        schema = {
            "type": "object",
            "properties": {keys[qid]: _answer_schema(q) for qid, q in questions.items()},
            "required": list(keys.values()),
        }
        t0 = time.perf_counter()
        out, cost = claude_json(prompt, SYSTEM, schema, model=self.model)
        latency = (time.perf_counter() - t0) * 1000
        meta = {"latency_ms": latency, "cost_usd": cost / len(questions), "model": self.name}
        results = {}
        for qid, q in questions.items():
            a = out.get(keys[qid])
            if a is None:
                raise JudgeError(f"llm response missing {qid}")
            if q.kind == "noul":
                results[qid] = noul_result(float(a["probability_yes"]), raw=a, **meta)
            elif q.kind == "choice":
                results[qid] = JudgeResult(answer=a["answer"], confidence=float(a["confidence"]), raw=a, **meta)
            else:
                results[qid] = JudgeResult(answer=float(a["score"]), confidence=float(a["confidence"]), raw=a, **meta)
        return results
