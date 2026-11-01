"""Human review queue backed by SQLite. Answers come back as labels."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

from judge_core.types import JudgeResult, Question, State


class HumanQueue:
    name = "human"

    def __init__(self, path: str = "human_queue.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created REAL, question_id TEXT, state TEXT, question TEXT,
                context TEXT, status TEXT DEFAULT 'pending', answer TEXT, answered REAL)"""
        )
        self.db.commit()

    def enqueue(self, state: State, qid: str, q: Question, context: dict[str, Any] | None = None) -> int:
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO queue (created, question_id, state, question, context) VALUES (?,?,?,?,?)",
                (time.time(), qid, json.dumps(state, ensure_ascii=False), q.model_dump_json(),
                 json.dumps(context or {}, ensure_ascii=False)),
            )
            self.db.commit()
        return int(cur.lastrowid)

    def judge_many(self, state: State, questions: dict[str, Question]) -> dict[str, JudgeResult]:
        """Enqueue everything; humans answer asynchronously, so nothing is decided now."""
        for qid, q in questions.items():
            self.enqueue(state, qid, q)
        return {}

    def pending(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id, question_id, state, question, context FROM queue WHERE status='pending' ORDER BY id"
        ).fetchall()
        return [
            {"id": r[0], "question_id": r[1], "state": json.loads(r[2]),
             "question": json.loads(r[3]), "context": json.loads(r[4])}
            for r in rows
        ]

    def resolve(self, item_id: int, answer: Any) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE queue SET status='done', answer=?, answered=? WHERE id=?",
                (json.dumps(answer, ensure_ascii=False), time.time(), item_id),
            )
            self.db.commit()

    def labels(self) -> list[dict[str, Any]]:
        """Resolved items, ready to recycle into threshold tuning / classifier training."""
        rows = self.db.execute(
            "SELECT question_id, state, answer FROM queue WHERE status='done' ORDER BY id"
        ).fetchall()
        return [{"question_id": r[0], "state": json.loads(r[1]), "answer": json.loads(r[2])} for r in rows]
