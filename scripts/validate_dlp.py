"""Held-out validation: tune the threshold and gray band on the dev set, freeze them,
then apply to a separate validation set.

  uv run python scripts/validate_dlp.py --dev data/out/dlp_100.jsonl --val data/out/dlp_val.jsonl \
      --kev http://127.0.0.1:8195

Dev probabilities come from bench_dlp's cache (kev-4b / zh+criteria, claude-sonnet / zh+criteria).
Also writes a CSV for human labelling, so the Claude-generated labels can be checked.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from judge_core.backends import JevJudge

sys.path.insert(0, str(Path(__file__).parent))
from bench_dlp import VARIANTS, auc, run_llm  # noqa: E402

from office_judge.detectors import build_rule_judge  # noqa: E402
from office_judge.pipelines import load_company  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BANDS = [(lo, hi) for lo in (0.02, 0.05, 0.1, 0.15) for hi in (0.2, 0.25, 0.3, 0.4, 0.5) if lo < hi]


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def score(pred, labels):
    pos = sum(labels)
    neg = len(labels) - pos
    return {
        "acc": sum(a == y for a, y in zip(pred, labels)) / len(labels),
        "miss": sum((not a) and y for a, y in zip(pred, labels)) / pos if pos else 0.0,
        "false_block": sum(a and not y for a, y in zip(pred, labels)) / neg if neg else 0.0,
    }


def cascade(hits, kev, llm, lo, hi):
    pred, to_llm = [], 0
    for h, p, q in zip(hits, kev, llm):
        if h:
            pred.append(True)
        elif p >= hi:
            pred.append(True)
        elif p <= lo:
            pred.append(False)
        else:
            to_llm += 1
            pred.append(q >= 0.5)
    return pred, to_llm / len(kev)


def best_threshold(probs, labels):
    # ties broken toward the lower threshold: fewer leaks
    return max(sorted(set(probs)), key=lambda t: (sum((p >= t) == y for p, y in zip(probs, labels)), -t))


def best_band(hits, kev, llm, labels, max_llm=0.25):
    """Fewest misses, then highest accuracy, among bands sending at most max_llm to the LLM."""
    cands = []
    for lo, hi in BANDS:
        pred, share = cascade(hits, kev, llm, lo, hi)
        if share <= max_llm:
            s = score(pred, labels)
            cands.append((s["miss"], -s["acc"], share, lo, hi))
    _, _, _, lo, hi = min(cands)
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=str(ROOT / "data" / "out" / "dlp_100.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "data" / "out" / "dlp_val.jsonl"))
    ap.add_argument("--kev", default="http://127.0.0.1:8195")
    ap.add_argument("--human-sheet", default=str(ROOT / "data" / "out" / "human_labels.csv"))
    args = ap.parse_args()

    company = load_company()
    rule = build_rule_judge(company["codenames"], company["customers"])
    dev, val = load(args.dev), load(args.val)
    dev_cache = Path(args.dev).parent / (Path(args.dev).stem + "_cache")
    dev_kev = json.loads((dev_cache / "kev-4b__zh_criteria.json").read_text(encoding="utf-8"))["probs"]
    dev_llm = json.loads((dev_cache / "claude-sonnet__zh_criteria.json").read_text(encoding="utf-8"))["probs"]
    dev_y = [r["label"] for r in dev]
    dev_hits = [bool(rule.scan(r["text"])) for r in dev]

    # --- tune on dev, then freeze ---
    t = best_threshold(dev_kev, dev_y)
    lo, hi = best_band(dev_hits, dev_kev, dev_llm, dev_y)
    print(f"tuned on dev ({len(dev)} rows): threshold {t:.3f}, gray band [{lo}, {hi})")

    # --- run on val ---
    val_cache = Path(args.val).parent / (Path(args.val).stem + "_cache")
    val_cache.mkdir(parents=True, exist_ok=True)
    f = val_cache / "kev-4b__zh_criteria.json"
    if f.exists():
        val_kev = json.loads(f.read_text(encoding="utf-8"))["probs"]
        lat = json.loads(f.read_text(encoding="utf-8"))["lat"]
    else:
        judge = JevJudge(base_url=args.kev, model="kev-4b", api_key="", timeout=120, price_per_input_token=0.0)
        val_kev, lat = [], []
        for r in val:
            t0 = time.perf_counter()
            val_kev.append(judge.judge_many(r["text"], {"q": VARIANTS["zh+criteria"](r["text"])})["q"].probability)
            lat.append((time.perf_counter() - t0) * 1000)
        f.write_text(json.dumps({"probs": val_kev, "lat": lat}), encoding="utf-8")
    val_llm = run_llm(val, val_cache)["zh+criteria"]["probs"]
    val_y = [r["label"] for r in val]
    val_hits = [bool(rule.scan(r["text"])) for r in val]

    def row(name, pred, share=None):
        s = score(pred, val_y)
        extra = f"  to LLM {share:4.0%}" if share is not None else ""
        print(f"  {name:38} acc {s['acc']:.2f}  miss {s['miss']:.2f}  false_block {s['false_block']:.2f}{extra}")

    print(f"\nvalidation ({len(val)} rows, {sum(val_y)} positive); Kev AUC {auc(val_kev, val_y):.2f}, "
          f"p50 {sorted(lat)[len(lat) // 2]:.0f} ms")
    row("rules only", val_hits)
    row("Kev @0.5", [p >= 0.5 for p in val_kev])
    row(f"Kev @{t:.3f} (frozen from dev)", [p >= t for p in val_kev])
    row(f"rules + Kev @{t:.3f}", [h or p >= t for h, p in zip(val_hits, val_kev)])
    row("Claude @0.5", [q >= 0.5 for q in val_llm])
    pred, share = cascade(val_hits, val_kev, val_llm, lo, hi)
    row(f"rules -> Kev [{lo},{hi}) -> Claude (frozen)", pred, share)
    t_val = best_threshold(val_kev, val_y)
    row(f"(oracle) Kev @{t_val:.3f} tuned on val", [p >= t_val for p in val_kev])

    # --- human labelling sheet: 20 dev + 20 val, models' answers hidden ---
    rng = random.Random(7)
    picks = [("dev", i, dev[i]) for i in rng.sample(range(len(dev)), 20)] + \
            [("val", i, val[i]) for i in rng.sample(range(len(val)), 20)]
    rng.shuffle(picks)
    with open(args.human_sheet, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "row", "text", "your_label (1=機密/0=可以送出)", "note"])
        for s, i, r in picks:
            w.writerow([s, i, r["text"], "", ""])
    print(f"\nhuman labelling sheet: {args.human_sheet}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
