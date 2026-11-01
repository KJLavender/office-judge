"""End-to-end DLP check: every row goes through OfficeJudge.dlp() exactly as the app would run it.

  TYPESAFE_BASE_URL=http://127.0.0.1:8195 TYPESAFE_DEFAULT_MODEL=kev-4b \
      uv run python scripts/validate_pipeline.py --data data/out/dlp_val.jsonl

A row counts as flagged when the action is mask or block. Block reasons are not generated
(writer off) to keep the run fast; decisions are unaffected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "out" / "dlp_val.jsonl"))
    args = ap.parse_args()
    os.environ.setdefault("OFFICE_JUDGE_WRITER", "off")
    from office_judge.app import build_from_env

    oj = build_from_env(ROOT / "data" / "out" / "pipeline_eval")
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"primary {oj.cascade.primary.name}, fallback {oj.cascade.llm.name if oj.cascade.llm else None}, {len(rows)} rows")

    flagged, lat, sentences, to_llm, to_human = [], [], 0, 0, 0
    errors = []
    for r in rows:
        t0 = time.perf_counter()
        res = oj.dlp(r["text"], explain=False)
        lat.append(time.perf_counter() - t0)
        flagged.append(res.action != "pass")
        last = oj.cascade.audit.records("dlp")[-1]["decisions"] if res.sentences_checked else []
        sentences += len(last)
        to_llm += sum(any(s["layer"] == "llm" for s in d["path"]) for d in last)
        to_human += sum(d["status"] == "pending_human" for d in last)
        if (res.action != "pass") != r["label"]:
            errors.append((r["label"], res.action, r["text"][:70]))

    y = [r["label"] for r in rows]
    pos, neg = sum(y), len(y) - sum(y)
    miss = sum((not f) and t for f, t in zip(flagged, y))
    fb = sum(f and not t for f, t in zip(flagged, y))
    lat.sort()
    print(f"acc {1 - (miss + fb) / len(y):.2f}  miss {miss}/{pos}  false flag {fb}/{neg}")
    print(f"sentences {sentences}: to LLM {to_llm} ({to_llm / max(sentences, 1):.0%}), to human {to_human}")
    print(f"latency per message p50 {lat[len(lat) // 2]:.2f}s  p95 {lat[int(len(lat) * 0.95)]:.2f}s")
    print("\nerrors (label, action, text):")
    for e in errors:
        print(f"  {'LEAK' if e[0] else 'ok  '} {e[1]:5} {e[2]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
