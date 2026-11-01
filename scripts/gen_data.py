"""Generate labelled synthetic data for the fictional company with the local Claude CLI.

Labels are produced together with the text; spot-check 10% by hand afterwards.
Personal data comes from fixed fake patterns, never from real people.

  uv run python scripts/gen_data.py dlp --n 300
  uv run python scripts/gen_data.py ticket --n 300
  uv run python scripts/gen_data.py action --n 100      # n meetings, many sentences each
  uv run python scripts/gen_data.py dlp --n 100 --model haiku --hint "..." --out data/out/dlp_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from judge_core.backends.llm import claude_json
from office_judge.pipelines import load_company

ROOT = Path(__file__).resolve().parent.parent
BATCH = 20

HARD = (
    "其中約 30% 要是困難樣本：換句話說的機密描述（不出現「機密」「營收」等關鍵字）、中英夾雜、"
    "語氣模糊、看似敏感其實是公開資訊、看似無害其實透露內部資訊。至少一半用繁體中文，其餘英文或中英夾雜。"
)

SPECS = {
    "dlp": {
        "system": "你在為資料外洩防護模型產生測試資料。只產生虛構內容。",
        "task": lambda c, n: (
            f"公司：{c['name']}；內部專案代號：{'、'.join(c['codenames'])}；客戶：{'、'.join(c['customers'])}。\n"
            f"產生 {n} 句員工可能貼到外部 AI 工具的句子。label=true 代表透露客戶資料、個資、金鑰或未公開商業資訊。"
            f"true 與 false 約各半。category 填 business / customer / pii / secret / none。{HARD}"
        ),
        "item": {"text": {"type": "string"}, "label": {"type": "boolean"}, "category": {"type": "string"},
                 "hard": {"type": "boolean"}},
    },
    "ticket": {
        "system": "你在為工單分流模型產生測試資料。只產生虛構內容。",
        "task": lambda c, n: (
            "團隊與負責範圍：" + "；".join(f"{k}: {v}" for k, v in c["teams"].items()) + "\n"
            f"產生 {n} 張員工寫給內部服務台的工單（主旨＋內文，口語）。team 為正確團隊，四隊大致平均；"
            f"urgency 為 0（不急）到 3（影響多人或資安）。{HARD}"
        ),
        "item": {"text": {"type": "string"}, "team": {"type": "string", "enum": ["IT", "HR", "Finance", "Facilities"]},
                 "urgency": {"type": "integer", "minimum": 0, "maximum": 3}, "hard": {"type": "boolean"}},
    },
    "action": {
        "system": "你在為會議待辦萃取模型產生測試資料。只產生虛構內容。",
        "task": lambda c, n: (
            f"與會者：{'、'.join(c['default_attendees'])}。產生 {n} 份短會議記錄，每份 6–12 句，格式「名字：發言」。"
            "sentences 列出每一句，is_action 標記是否指派或承諾了待辦，owner 為負責人（不明確填 null），"
            "due_text 為期限原文（沒有填 null）。每份約 2–4 個待辦，要有語氣模糊的待辦（「也許可以看一下」）"
            "和看起來像待辦其實不是的句子（已完成的事、假設語氣）。"
        ),
        "item": {
            "sentences": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"}, "is_action": {"type": "boolean"},
                "owner": {"type": ["string", "null"]}, "due_text": {"type": ["string", "null"]},
            }, "required": ["text", "is_action", "owner", "due_text"]}},
        },
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(SPECS))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None, help="generator model; use a different one from the judge to limit self-agreement")
    ap.add_argument("--hint", default="", help="extra instructions, e.g. a different scenario mix for a held-out set")
    args = ap.parse_args()
    spec = SPECS[args.kind]
    company = load_company()
    out = Path(args.out or ROOT / "data" / "out" / f"{args.kind}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    batch = 5 if args.kind == "action" else BATCH
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": spec["item"], "required": list(spec["item"])}}},
        "required": ["items"],
    }
    done, cost = 0, 0.0
    with out.open("a", encoding="utf-8") as f:
        while done < args.n:
            n = min(batch, args.n - done)
            task = spec["task"](company, n) + ("\n\n" + args.hint if args.hint else "")
            kw = {"model": args.model} if args.model else {}
            data, c = claude_json(task, spec["system"], schema, timeout=600, **kw)
            cost += c
            for item in data["items"][:n]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            done += min(n, len(data["items"]))
            print(f"{done}/{args.n}  (${cost:.3f})", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
