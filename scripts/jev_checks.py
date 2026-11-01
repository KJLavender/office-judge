"""Day-one checks against the real Jev API (needs TYPESAFE_API_KEY).

  1. Traditional Chinese accuracy, jev-1.13.0 vs jev-1.14.0
  2. Batched vs one-question-per-call: same answers? latency / tokens?
  3. Choice option order sensitivity: shuffle options, does the answer move?

  uv run python scripts/jev_checks.py [--data data/toy_dlp.jsonl]
  uv run python scripts/jev_checks.py --base-url http://127.0.0.1:8191 --models julia-1   # llama-server
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from judge_core.backends import JevJudge
from judge_core.eval import calibration
from judge_core.types import Question
from office_judge.pipelines import load_company

ROOT = Path(__file__).resolve().parent.parent
BASE_URL: str | None = None
TICKETS = [
    "VPN 從早上就連不上，要怎麼處理？",
    "我想請下週五的特休，要找誰簽核？",
    "上個月出差的計程車收據要怎麼報帳？",
    "3 樓會議室的冷氣壞了，好熱。",
    "新人的筆電什麼時候會到？",
    "這個月薪資單少了加班費。",
    "門禁卡掉了要去哪裡補辦？",
    "廠商說發票抬頭寫錯了要重開。",
    "Outlook 一直跳出要輸入密碼。",
    "停車位可以申請嗎？",
]


def dlp_question(text: str) -> Question:
    return Question(kind="noul", instructions=f"這句話是否透露客戶資料、個人資料、金鑰、或未公開的內部商業資訊？\n\n句子：「{text}」")


def judge(model):
    price = {"price_per_input_token": 0.0} if BASE_URL else {}
    return JevJudge(model=model, base_url=BASE_URL, timeout=60, **price)


def check_models(rows, models):
    print("== 1. 繁中準確率與校準 ==")
    for model in models:
        jev = judge(model)
        probs = []
        for r in rows:
            probs.append(jev.judge_many(r["text"], {"q": dlp_question(r["text"])})["q"].probability)
        labels = [r["label"] for r in rows]
        acc = sum((p >= 0.5) == y for p, y in zip(probs, labels)) / len(rows)
        print(f"  {model}: accuracy {acc:.3f}  ECE {calibration(probs, labels).ece:.3f}  (n={len(rows)})")


def check_batching(rows, model):
    print(f"== 2. 合併送出 vs 逐題（{model}） ==")
    jev = judge(model)
    state = {"sentences": [r["text"] for r in rows]}
    questions = {f"s{i}": dlp_question(r["text"]) for i, r in enumerate(rows)}
    t0 = time.perf_counter()
    batched = jev.judge_many(state, questions)
    t_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    single = {qid: jev.judge_many(state, {qid: q})[qid] for qid, q in questions.items()}
    t_single = time.perf_counter() - t0
    diffs = [abs(batched[k].probability - single[k].probability) for k in questions]
    flips = sum(batched[k].answer != single[k].answer for k in questions)
    cost_b = sum(r.cost_usd for r in batched.values())
    cost_s = sum(r.cost_usd for r in single.values())
    print(f"  max |Δp| {max(diffs):.3f}  answer flips {flips}/{len(questions)}")
    print(f"  wall time batched {t_batch:.2f}s vs single {t_single:.2f}s;  cost ${cost_b:.6f} vs ${cost_s:.6f}")


def check_choice_order(model):
    print(f"== 3. Choice 選項順序（{model}） ==")
    jev = judge(model)
    teams = load_company()["teams"]
    agree = 0
    for t in TICKETS:
        items = list(teams.items())
        a = jev.judge_many(t, {"q": Question(kind="choice", instructions="這張工單應該由哪個團隊處理？", criteria=dict(items))})["q"]
        random.shuffle(items)
        b = jev.judge_many(t, {"q": Question(kind="choice", instructions="這張工單應該由哪個團隊處理？", criteria=dict(items))})["q"]
        agree += a.answer == b.answer
        flag = "" if a.answer == b.answer else "  <-- changed"
        print(f"  {t[:20]:22} {a.answer:10} {a.confidence:.2f} | {b.answer:10} {b.confidence:.2f}{flag}")
    print(f"  agreement {agree}/{len(TICKETS)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "toy_dlp.jsonl"))
    ap.add_argument("--base-url", default=None, help="llama-server / jev-rs address instead of TypeSafe")
    ap.add_argument("--models", nargs="+", default=["jev-1.13.0", "jev-1.14.0"])
    args = ap.parse_args()
    global BASE_URL
    BASE_URL = args.base_url
    if not (BASE_URL or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_BASE_URL")):
        sys.exit("set TYPESAFE_API_KEY, or pass --base-url for a local llama-server")
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    check_models(rows, args.models)
    check_batching(rows, args.models[-1])
    check_choice_order(args.models[-1])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
