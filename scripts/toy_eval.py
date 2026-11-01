"""Run the DLP sentence judgment over a labelled JSONL file and produce the
threshold sweep, calibration curve, backend comparison and LLM rescue rate.

  uv run python scripts/toy_eval.py                       # demo heuristic, offline
  uv run python scripts/toy_eval.py --backends jev llm    # needs TYPESAFE_API_KEY / claude CLI
  uv run python scripts/toy_eval.py --data data/out/dlp.jsonl --cascade
  # local decision models behind llama-server (/v1/systemone)
  uv run python scripts/toy_eval.py --base-url http://127.0.0.1:8080 --backends local:julia-1 local:laya
  uv run python scripts/toy_eval.py --base-url ... --backends local:julia-1 --cascade --second local:laya

Backend specs: demo | llm | jev | jev:<model id> | local:<model name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from judge_core import Cascade, Item, load_policies
from judge_core.backends import JevJudge, LLMJudge
from judge_core.eval import calibration, compare, plot_sweep_and_calibration, rescue_rate, sweep_thresholds
from office_judge.demo_backend import demo_judge

ROOT = Path(__file__).resolve().parent.parent


def make_backend(spec: str, base_url: str | None):
    kind, _, model = spec.partition(":")
    if kind == "demo":
        return demo_judge()
    if kind == "llm":
        return LLMJudge(model) if model else LLMJudge()
    if kind == "jev":
        return JevJudge(model=model or None)
    if kind == "local":
        if not base_url:
            sys.exit("local:<model> needs --base-url (llama-server address)")
        judge = JevJudge(base_url=base_url, model=model, api_key="", timeout=60, price_per_input_token=0.0)
        judge.name = f"local:{model}"
        return judge
    sys.exit(f"unknown backend {spec}")


class _Recorder:
    """Wraps a backend so compare() and the plots share one call per row."""

    def __init__(self, inner, probs: list[float]):
        self.inner, self.probs, self.name = inner, probs, inner.name

    def judge_many(self, state, questions):
        out = self.inner.judge_many(state, questions)
        self.probs.append(out["dlp"].probability)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "toy_dlp.jsonl"))
    ap.add_argument("--backends", nargs="+", default=["demo"])
    ap.add_argument("--base-url", default=None, help="llama-server address for local:<model>")
    ap.add_argument("--cascade", action="store_true", help="also run first backend -> second layer, report rescue rate")
    ap.add_argument("--second", default="llm", help="second layer for --cascade (default llm)")
    ap.add_argument("--out", default=str(ROOT / "data" / "out"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    pol = load_policies(ROOT / "src" / "office_judge" / "policies")["dlp_sensitive"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def q(text):
        return pol.question(f"{pol.instructions}\n\n句子：「{text}」")

    backends = [make_backend(b, args.base_url) for b in args.backends]
    labels = [r["label"] for r in rows]
    print(f"{len(rows)} rows from {args.data}\n")
    print(f"{'backend':28} {'acc':>6} {'n':>4} {'err':>4} {'p50ms':>8} {'p95ms':>8} {'$/1k':>8}")
    all_probs = {}
    for be in backends:
        # one question per row; the row text is the state. compare() also records each answer.
        probs: list[float] = []
        recorder = _Recorder(be, probs)
        dataset = [(r["text"], {"dlp": q(r["text"])}, {"dlp": r["label"]}) for r in rows]
        [row] = compare([recorder], dataset)
        all_probs[be.name] = probs
        print(f"{row['backend']:28} {row['accuracy']:6.3f} {row['answered']:4d} {row['errors']:4d} "
              f"{row['p50_ms']:8.0f} {row['p95_ms']:8.0f} {row['usd_per_1k']:8.4f}")

    for be in backends:
        probs = all_probs[be.name]
        if len(probs) != len(rows):
            print(f"\n[{be.name}] skipped plots: {len(rows) - len(probs)} rows failed")
            continue
        points = sweep_thresholds(probs, labels)
        cal = calibration(probs, labels)
        safe = be.name.replace(":", "_")
        path = out_dir / f"sweep_calibration_{safe}.png"
        plot_sweep_and_calibration(points, cal, str(path), title=f"dlp_sensitive / {be.name}")
        missed = sum(p < pol.thresholds.no_below and y for p, y in zip(probs, labels))
        print(f"\n[{be.name}] ECE={cal.ece:.3f}  plot: {path}")
        print(f"  leaks auto-passed at no_below={pol.thresholds.no_below}: {missed}")
        for p in points[::5]:
            print(f"  gray zone [{p.no_below:.2f}, {p.yes_above:.2f}]  auto {p.auto_rate:5.1%}  error {p.error_rate:5.1%}")

    if args.cascade:
        second = make_backend(args.second, args.base_url)
        cascade = Cascade(primary=backends[0], llm=second)
        decisions, gold = [], {}
        for i, r in enumerate(rows):
            d = cascade.run(r["text"], {f"row{i}": Item(pol, q(r["text"]))})[f"row{i}"]
            decisions.append(d.model_dump())
            gold[f"row{i}"] = r["label"]
        rr = rescue_rate(decisions, gold)
        print(f"\n{second.name} rescue rate over {rr['n']} gray items: rescued {rr['rescued']:.1%}, "
              f"broke {rr['broke']:.1%}, agreed {rr['agreed']:.1%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
