"""Shared preprocessing for human-written text: cleaning, sentence split, reversible masking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from judge_core.backends.rule import Finding

from office_judge.detectors import LABELS


def clean(text: str) -> str:
    """Drop quoted reply lines, normalise whitespace, keep line breaks."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if not ln.lstrip().startswith(">")]
    out = "\n".join(re.sub(r"[ \t　]+", " ", ln) for ln in lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    text: str


_CJK_SPLIT = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]*")
_EN_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def split_sentences(text: str) -> list[Sentence]:
    out = []
    for m in _CJK_SPLIT.finditer(text):
        seg_start = m.start()
        seg = m.group()
        pos = 0
        for part in _EN_SPLIT.split(seg) + [None]:
            if part is None:
                break
            idx = seg.index(part, pos)
            pos = idx + len(part)
            stripped = part.strip()
            if not stripped or re.fullmatch(r"[\W_]+", stripped):
                continue
            s = seg_start + idx + (len(part) - len(part.lstrip()))
            out.append(Sentence(s, s + len(stripped), stripped))
    return out


def window(sentences: list[Sentence], i: int, k: int = 1) -> tuple[str, str]:
    before = " ".join(s.text for s in sentences[max(0, i - k):i])
    after = " ".join(s.text for s in sentences[i + 1:i + 1 + k])
    return before, after


@dataclass
class Vault:
    """Reversible masking: same value -> same placeholder; unmask restores originals."""

    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original
    _by_value: dict[tuple[str, str], str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def placeholder(self, label: str, original: str) -> str:
        key = (label, original)
        if key not in self._by_value:
            self._counts[label] = self._counts.get(label, 0) + 1
            ph = f"[{label}_{self._counts[label]}]"
            self._by_value[key] = ph
            self.mapping[ph] = original
        return self._by_value[key]

    def replace_spans(self, text: str, spans: list[tuple[int, int, str]]) -> str:
        """spans: (start, end, label), non-overlapping."""
        ordered = sorted(spans, key=lambda s: s[0])
        names = [self.placeholder(label, text[start:end]) for start, end, label in ordered]  # number left to right
        out = text
        for (start, end, _), ph in reversed(list(zip(ordered, names))):
            out = out[:start] + ph + out[end:]
        return out

    def mask_findings(self, text: str, findings: list[Finding]) -> str:
        return self.replace_spans(text, [(f.start, f.end, LABELS.get(f.detector, f.detector)) for f in findings])

    def unmask(self, text: str) -> str:
        # longest first so [X_12] is not hit by [X_1]; repeat because a masked
        # sentence can itself contain earlier placeholders
        order = sorted(self.mapping, key=len, reverse=True)
        for _ in range(3):
            before = text
            for ph in order:
                text = text.replace(ph, self.mapping[ph])
            if text == before:
                break
        return text
