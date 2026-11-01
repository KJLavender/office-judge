"""Deterministic rule backend: regex + checksum detectors and lexicons.

Unlike model backends it returns *where* it matched, so callers can mask.
It only answers questions it has a rule for; everything else falls through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from judge_core.types import JudgeResult, Question, State, noul_result


@dataclass(frozen=True)
class Finding:
    start: int
    end: int
    text: str
    detector: str
    category: str  # e.g. pii, secret, codename


@dataclass
class Detector:
    name: str
    category: str
    pattern: re.Pattern[str]
    validate: Callable[[str], bool] | None = None
    group: int = 0

    def find(self, text: str) -> list[Finding]:
        out = []
        for m in self.pattern.finditer(text):
            s = m.group(self.group)
            if self.validate is None or self.validate(s):
                out.append(Finding(m.start(self.group), m.end(self.group), s, self.name, self.category))
        return out


def lexicon_detector(name: str, category: str, terms: list[str]) -> Detector:
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE) if terms else re.compile(r"(?!x)x")
    return Detector(name, category, pattern)


def merge_overlaps(findings: list[Finding]) -> list[Finding]:
    """Keep the longest finding where spans overlap; sort by position."""
    out: list[Finding] = []
    for f in sorted(findings, key=lambda f: (f.start, -(f.end - f.start))):
        if out and f.start < out[-1].end:
            continue
        out.append(f)
    return out


@dataclass
class RuleJudge:
    detectors: list[Detector]
    # policy/question id -> categories that decide "yes" for it
    rules: dict[str, list[str]] = field(default_factory=dict)
    name: str = "rule"

    def scan(self, text: str) -> list[Finding]:
        return merge_overlaps([f for d in self.detectors for f in d.find(text)])

    def judge_many(self, state: State, questions: dict[str, Question]) -> dict[str, JudgeResult]:
        text = state if isinstance(state, str) else str(state)
        findings = self.scan(text)
        out = {}
        for qid, q in questions.items():
            wanted = self.rules.get(qid)
            if q.kind != "noul" or not wanted:
                continue
            hits = [f for f in findings if f.category in wanted]
            if hits:
                out[qid] = noul_result(1.0, model=self.name, raw={"hits": [f.detector for f in hits]})
        return out
