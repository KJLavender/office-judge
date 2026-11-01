"""Audit log: every cascade run, every layer, in SQLite."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any


def input_hash(state: Any) -> str:
    raw = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str = "audit.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, pipeline TEXT,
                input_hash TEXT, decisions TEXT, latency_ms REAL, cost_usd REAL)"""
        )
        self.db.commit()

    def write(self, pipeline: str, state: Any, decisions: list[dict[str, Any]],
              latency_ms: float, cost_usd: float) -> int:
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO audit (ts, pipeline, input_hash, decisions, latency_ms, cost_usd) VALUES (?,?,?,?,?,?)",
                (time.time(), pipeline, input_hash(state), json.dumps(decisions, ensure_ascii=False),
                 latency_ms, cost_usd),
            )
            self.db.commit()
        return int(cur.lastrowid)

    def records(self, pipeline: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT id, ts, pipeline, input_hash, decisions, latency_ms, cost_usd FROM audit"
        rows = self.db.execute(q + (" WHERE pipeline=?" if pipeline else "") + " ORDER BY id",
                               (pipeline,) if pipeline else ()).fetchall()
        return [
            {"id": r[0], "ts": r[1], "pipeline": r[2], "input_hash": r[3],
             "decisions": json.loads(r[4]), "latency_ms": r[5], "cost_usd": r[6]}
            for r in rows
        ]
