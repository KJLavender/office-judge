"""Policies: one YAML document per judgment, validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from judge_core.types import JudgeResult, Kind, Question

Role = Literal["routing", "gating", "ranking", "extraction"]
Layer = Literal["llm", "human"]


class Thresholds(BaseModel):
    yes_above: float = 0.5  # noul: P(yes) >= this -> yes
    no_below: float = 0.5  # noul: P(yes) <= this -> no; between is the gray zone
    min_confidence: float = 0.0  # choice: below this is the gray zone


class Policy(BaseModel):
    id: str
    role: Role
    kind: Kind
    instructions: str
    criteria: dict[str, str] | list[str] | None = None
    thresholds: Thresholds = Field(default_factory=Thresholds)
    # per-model overrides, matched as a substring of the answering model's name.
    # Thresholds are tuned per model: probabilities from different models are not comparable.
    model_thresholds: dict[str, Thresholds] = Field(default_factory=dict)
    fallback: list[Layer] = Field(default_factory=list)
    fail_mode: Literal["open", "closed"] | None = None

    @field_validator("criteria", mode="before")
    @classmethod
    def _yaml_bool_keys(cls, v: Any) -> Any:
        # YAML reads `true:` / `false:` keys as booleans
        if isinstance(v, dict):
            return {(str(k).lower() if isinstance(k, bool) else str(k)): val for k, val in v.items()}
        return v

    def question(self, instructions: str | None = None, criteria: dict[str, str] | list[str] | None = None) -> Question:
        """Build a concrete question; pipelines override text per sentence or per meeting."""
        return Question(
            kind=self.kind,
            instructions=instructions or self.instructions,
            criteria=criteria if criteria is not None else self.criteria,
        )

    @property
    def effective_fail_mode(self) -> str:
        return self.fail_mode or ("closed" if self.role == "gating" else "open")

    def thresholds_for(self, model: str) -> Thresholds:
        for key, t in self.model_thresholds.items():
            if key in (model or ""):
                return t
        return self.thresholds

    def verdict(self, r: JudgeResult) -> tuple[bool, object]:
        """(decided?, answer). Undecided means the result is in the gray zone."""
        t = self.thresholds_for(r.model)
        if self.kind == "noul":
            p = r.probability if r.probability is not None else float(bool(r.answer))
            if p >= t.yes_above:
                return True, True
            if p <= t.no_below:
                return True, False
            return False, p >= 0.5
        if self.kind == "choice":
            return r.confidence >= t.min_confidence, r.answer
        return True, r.answer  # score / ranking: never gated


def load_policies(source: str | Path) -> dict[str, Policy]:
    """Load from a YAML file, a directory of YAML files, or a YAML string (multi-doc allowed)."""
    is_path = isinstance(source, Path) or ("\n" not in source and Path(source).exists())
    p = Path(source)
    if is_path and p.is_dir():
        texts = [f.read_text(encoding="utf-8") for f in sorted(p.glob("*.y*ml"))]
    elif is_path:
        texts = [p.read_text(encoding="utf-8")]
    else:
        texts = [str(source)]
    out: dict[str, Policy] = {}
    for text in texts:
        for doc in yaml.safe_load_all(text):
            for item in doc if isinstance(doc, list) else [doc]:
                if item:
                    pol = Policy.model_validate(item)
                    if pol.id in out:
                        raise ValueError(f"duplicate policy id {pol.id}")
                    out[pol.id] = pol
    return out
