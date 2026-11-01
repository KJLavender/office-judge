"""Text-generating helpers (System 2): block reasons and action-item deadlines.

Kept apart from the judge cascade: Jev decides, the writer only phrases or
converts. Every call is optional; pipelines degrade gracefully without it.
"""

from __future__ import annotations

from typing import Any, Protocol

from judge_core.backends.llm import DEFAULT_MODEL, claude_json


class Writer(Protocol):
    def explain(self, text: str, finding: str) -> str: ...

    def resolve_actions(self, items: list[dict[str, Any]], today: str) -> list[dict[str, Any]]:
        """items: [{i, sentence, owner}] -> [{i, task, due_date (YYYY-MM-DD|None), due_text}]"""
        ...


class ClaudeWriter:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def explain(self, text: str, finding: str) -> str:
        out, _ = claude_json(
            f"## 內容\n{text}\n\n## 判定\n{finding}",
            "你是公司的資料外洩防護助理。用一句繁體中文（40 字內）告訴員工這段內容為什麼不能直接送出，"
            "具體指出是哪一類資訊，語氣友善，不要重複敏感內容本身。",
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
            model=self.model,
        )
        return out["reason"]

    def resolve_actions(self, items: list[dict[str, Any]], today: str) -> list[dict[str, Any]]:
        if not items:
            return []
        lines = "\n".join(f"{it['i']}. （負責人：{it['owner'] or '未指定'}）{it['sentence']}" for it in items)
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "i": {"type": "integer"},
                            "task": {"type": "string"},
                            "due_date": {"type": ["string", "null"]},
                            "due_text": {"type": ["string", "null"]},
                        },
                        "required": ["i", "task", "due_date", "due_text"],
                    },
                }
            },
            "required": ["items"],
        }
        out, _ = claude_json(
            f"今天是 {today}（ISO 日期）。\n\n{lines}",
            "把每一句待辦改寫成簡短的繁體中文任務（動詞開頭，20 字內），保留 [類別_編號] placeholder 原樣。"
            "若句中有期限，due_text 填原文（例如「下週三前」），due_date 換算成 YYYY-MM-DD；"
            "週的起點是週一。沒有期限兩者都填 null。i 照抄編號。",
            schema,
            model=self.model,
        )
        return out["items"]
