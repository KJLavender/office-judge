"""Evaluation tools: threshold sweeps, calibration, backend comparison, LLM rescue rate."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from judge_core.types import Judge, JudgeError, Question, State


@dataclass
class SweepPoint:
    yes_above: float
    no_below: float
    auto_rate: float  # share decided without escalation
    error_rate: float  # errors among auto-decided items


def sweep_thresholds(
    probs: Sequence[float], labels: Sequence[bool], steps: int = 20
) -> list[SweepPoint]:
    """Symmetric gray zone [0.5-w, 0.5+w] for w in 0..0.5."""
    out = []
    n = len(probs)
    for i in range(steps + 1):
        w = 0.5 * i / steps
        hi, lo = 0.5 + w, 0.5 - w
        auto = [(p >= hi, y) for p, y in zip(probs, labels) if p >= hi or p <= lo]
        errors = sum(pred != y for pred, y in auto)
        out.append(SweepPoint(hi, lo, len(auto) / n if n else 0.0, errors / len(auto) if auto else 0.0))
    return out


@dataclass
class Calibration:
    bins: list[dict[str, float]]  # lo, hi, count, mean_prob, frac_positive
    ece: float


def calibration(probs: Sequence[float], labels: Sequence[bool], n_bins: int = 10) -> Calibration:
    bins = []
    n = len(probs)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        if not idx:
            bins.append({"lo": lo, "hi": hi, "count": 0, "mean_prob": 0.0, "frac_positive": 0.0})
            continue
        mp = sum(probs[i] for i in idx) / len(idx)
        fp = sum(bool(labels[i]) for i in idx) / len(idx)
        ece += len(idx) / n * abs(mp - fp)
        bins.append({"lo": lo, "hi": hi, "count": len(idx), "mean_prob": mp, "frac_positive": fp})
    return Calibration(bins, ece)


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def compare(
    backends: Sequence[Judge],
    dataset: Sequence[tuple[State, dict[str, Question], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """dataset: (state, questions, gold answers by qid). noul gold is bool; choice gold is str."""
    rows = []
    for be in backends:
        correct = total = errors = 0
        lat: list[float] = []
        cost = 0.0
        for state, questions, gold in dataset:
            t0 = time.perf_counter()
            try:
                res = be.judge_many(state, questions)
            except JudgeError:
                errors += 1
                continue
            lat.append((time.perf_counter() - t0) * 1000)
            for qid, g in gold.items():
                if qid in res:
                    total += 1
                    correct += res[qid].answer == g
                    cost += res[qid].cost_usd
        rows.append({
            "backend": be.name,
            "accuracy": correct / total if total else 0.0,
            "answered": total,
            "errors": errors,
            "p50_ms": _pct(lat, 0.5),
            "p95_ms": _pct(lat, 0.95),
            "usd_per_1k": cost / total * 1000 if total else 0.0,
        })
    return rows


def rescue_rate(decision_dicts: Sequence[dict[str, Any]], gold: dict[str, bool]) -> dict[str, float]:
    """Among questions where the primary layer was gray and the LLM answered:
    rescued = LLM disagreed with the primary's leaning and was right;
    broke   = LLM disagreed and was wrong; agreed = LLM kept the leaning."""
    rescued = broke = agreed = 0
    for d in decision_dicts:
        g = gold.get(d["qid"])
        if g is None:
            continue
        steps = {s["layer"]: s for s in d["path"]}
        prim, llm = steps.get("primary"), steps.get("llm")
        if not prim or prim["outcome"] != "gray" or not llm or llm["answer"] is None:
            continue
        if llm["answer"] == prim["answer"]:
            agreed += 1
        elif llm["answer"] == g:
            rescued += 1
        else:
            broke += 1
    n = rescued + broke + agreed
    return {
        "n": n,
        "rescued": rescued / n if n else 0.0,
        "broke": broke / n if n else 0.0,
        "agreed": agreed / n if n else 0.0,
    }


def plot_sweep_and_calibration(points: list[SweepPoint], cal: Calibration, path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot([p.auto_rate for p in points], [p.error_rate for p in points], marker="o", color="#3b6ea5")
    a.set_xlabel("auto-decided share")
    a.set_ylabel("error rate among auto-decided")
    a.set_title("Threshold sweep")
    a.grid(alpha=0.3)
    used = [x for x in cal.bins if x["count"]]
    b.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    b.plot([x["mean_prob"] for x in used], [x["frac_positive"] for x in used], marker="o", color="#3b6ea5")
    b.set_xlabel("predicted P(yes)")
    b.set_ylabel("observed share yes")
    b.set_title(f"Calibration (ECE {cal.ece:.3f})")
    b.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
