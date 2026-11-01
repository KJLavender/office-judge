"""Benchmark DLP sentence judgments across local decision models, prompt variants and an LLM ceiling.

  uv run python scripts/bench_dlp.py --data data/out/dlp_100.jsonl \
      --local julia-1=http://127.0.0.1:8191 laya=http://127.0.0.1:8192 --llm

Per model x prompt variant: accuracy@0.5, AUC, ECE, leak miss rate, false-block rate, p50 latency.
"System" = rules (regex/checksum/lexicon) OR the model, i.e. what the DLP pipeline actually does.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from judge_core.backends import JevJudge, LLMJudge
from judge_core.eval import calibration
from judge_core.types import Question
from office_judge.detectors import build_rule_judge
from office_judge.pipelines import load_company

ROOT = Path(__file__).resolve().parent.parent

ZH_I = "這句話是否透露客戶資料、個人資料、帳號密碼或金鑰、或未公開的內部商業資訊？"
EN_I = "Does the state disclose customer data, personal data, credentials or keys, or non-public internal business information?"
ZH_C = {"true": "例如未公布的營收或財測、併購或募資、未發表的產品規格與時程、客戶名單、合約或報價金額、內部系統架構與漏洞、個資、金鑰",
        "false": "公開資訊、一般工作溝通與行政事務、問候、通用技術問題"}
EN_C = {"true": "e.g. unannounced revenue or forecasts, M&A or fundraising, unreleased product specs or timelines, customer lists, "
                "contract or quote amounts, internal system weaknesses, personal data, keys or passwords",
        "false": "public information, routine work coordination, admin requests, greetings, generic technical questions"}
VARIANTS = {
    "zh+criteria": lambda t: Question(kind="noul", instructions=f"{ZH_I}\n\n句子：「{t}」", criteria=ZH_C),
    "zh": lambda t: Question(kind="noul", instructions=ZH_I),
    "en+criteria": lambda t: Question(kind="noul", instructions=EN_I, criteria=EN_C),
    "en": lambda t: Question(kind="noul", instructions=EN_I),
}


def auc(probs, labels):
    pos = [p for p, y in zip(probs, labels) if y]
    neg = [p for p, y in zip(probs, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def metrics(probs, labels, rule_hits=None, lat=None):
    pred = [p >= 0.5 for p in probs]
    if rule_hits is not None:
        pred = [a or b for a, b in zip(pred, rule_hits)]
    pos = sum(labels)
    neg = len(labels) - pos
    return {
        "acc": sum(a == y for a, y in zip(pred, labels)) / len(labels),
        "auc": auc(probs, labels),
        "ece": calibration(probs, labels).ece,
        "miss": sum((not a) and y for a, y in zip(pred, labels)) / pos if pos else 0.0,
        "false_block": sum(a and not y for a, y in zip(pred, labels)) / neg if neg else 0.0,
        "p50_ms": sorted(lat)[len(lat) // 2] if lat else None,
    }


def run_local(name, url, rows, variants, cache: Path):
    """Results are cached per (model, variant) so an interrupted run resumes."""
    judge = JevJudge(base_url=url, model=name, api_key="", timeout=120, price_per_input_token=0.0)
    out = {}
    for vn in variants:
        f = cache / f"{name}__{vn.replace('+', '_')}.json"
        if f.exists():
            out[vn] = json.loads(f.read_text(encoding="utf-8"))
            print(f"  {name:18} {vn:12} cached", flush=True)
            continue
        probs, lat = [], []
        for r in rows:
            t0 = time.perf_counter()
            probs.append(judge.judge_many(r["text"], {"q": VARIANTS[vn](r["text"])})["q"].probability)
            lat.append((time.perf_counter() - t0) * 1000)
        out[vn] = {"probs": probs, "lat": lat}
        f.write_text(json.dumps(out[vn]), encoding="utf-8")
        print(f"  {name:18} {vn:12} done", flush=True)
    return out


def run_llm(rows, cache: Path, chunk=20):
    """Ask the LLM about many rows per call: state is the list, one question per row."""
    f = cache / "claude-sonnet__zh_criteria.json"
    if f.exists():
        return {"zh+criteria": json.loads(f.read_text(encoding="utf-8"))}
    judge = LLMJudge()
    probs, cost = [], 0.0
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        state = {"rows": [r["text"] for r in part]}
        qs = {f"r{j}": Question(kind="noul", instructions=f"{ZH_I}（只判斷 rows[{j}]）", criteria=ZH_C) for j in range(len(part))}
        res = judge.judge_many(state, qs)
        probs += [res[f"r{j}"].probability for j in range(len(part))]
        cost += sum(x.cost_usd for x in res.values())
        print(f"  llm {i + len(part)}/{len(rows)}", flush=True)
    result = {"probs": probs, "lat": None, "cost_usd": cost}
    f.write_text(json.dumps(result), encoding="utf-8")
    return {"zh+criteria": result}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "out" / "dlp_100.jsonl"))
    ap.add_argument("--local", nargs="*", default=[], help="name=url pairs")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "out" / "bench_dlp.json"))
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    labels = [r["label"] for r in rows]
    company = load_company()
    rule = build_rule_judge(company["codenames"], company["customers"])
    rule_hits = [bool(rule.scan(r["text"])) for r in rows]
    is_zh = [bool(re.search(r"[一-鿿]", r["text"])) for r in rows]
    hard = [bool(r.get("hard")) for r in rows]
    print(f"{len(rows)} rows, {sum(labels)} positive, {sum(is_zh)} zh, {sum(hard)} hard; rules alone flag {sum(rule_hits)}")

    cache = Path(args.out).parent / (Path(args.data).stem + "_cache")
    cache.mkdir(parents=True, exist_ok=True)
    raw: dict[str, dict] = {}
    for spec in args.local:
        name, url = spec.split("=", 1)
        raw[name] = run_local(name, url, rows, args.variants, cache)
    if args.llm:
        raw["claude-sonnet"] = run_llm(rows, cache)

    rules_only = metrics([float(h) for h in rule_hits], labels)
    print(f"\nrules only: acc {rules_only['acc']:.2f}  miss {rules_only['miss']:.2f}  false_block {rules_only['false_block']:.2f}")
    header = f"{'model':18} {'prompt':12} {'acc':>5} {'AUC':>5} {'ECE':>5} {'miss':>5} {'fblk':>5} | {'sys acc':>7} {'sys miss':>8} {'sys fblk':>8} | {'zh acc':>6} {'hard acc':>8} {'p50ms':>6}"
    print("\n" + header + "\n" + "-" * len(header))
    report = {"n": len(rows), "rules_only": rules_only, "results": {}}
    for name, by_variant in raw.items():
        for vn, d in by_variant.items():
            p = d["probs"]
            m = metrics(p, labels, lat=d["lat"])
            s = metrics(p, labels, rule_hits)
            zh = metrics([x for x, z in zip(p, is_zh) if z], [y for y, z in zip(labels, is_zh) if z])
            hd = metrics([x for x, h in zip(p, hard) if h], [y for y, h in zip(labels, hard) if h])
            report["results"][f"{name}/{vn}"] = {"model": m, "system": s, "zh": zh, "hard": hd, "probs": p,
                                                 "cost_usd": d.get("cost_usd")}
            lat = f"{m['p50_ms']:6.0f}" if m["p50_ms"] is not None else "     -"
            print(f"{name:18} {vn:12} {m['acc']:5.2f} {m['auc']:5.2f} {m['ece']:5.2f} {m['miss']:5.2f} {m['false_block']:5.2f} | "
                  f"{s['acc']:7.2f} {s['miss']:8.2f} {s['false_block']:8.2f} | {zh['acc']:6.2f} {hd['acc']:8.2f} {lat}")
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
