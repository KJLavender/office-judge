"""Web API + demo page.  Run: uv run uvicorn office_judge.app:app --reload

Backends are picked from the environment:
  OFFICE_JUDGE_PRIMARY   jev | llm | demo   (default: jev if TYPESAFE_API_KEY/TYPESAFE_BASE_URL is set, else demo)
  OFFICE_JUDGE_FALLBACK  on | off           LLM as second layer (default: on when the claude CLI exists)
  OFFICE_JUDGE_WRITER    on | off           LLM writes block reasons and resolves deadlines (same default)
  TYPESAFE_BASE_URL      point at a jev-rs server to keep everything local
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from judge_core import AuditLog
from judge_core.backends import HumanQueue, JevJudge, LLMJudge

from office_judge.demo_backend import demo_judge
from office_judge.pipelines import OfficeJudge
from office_judge.text import Vault
from office_judge.writer import ClaudeWriter

HERE = Path(__file__).parent


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.lower() in ("1", "on", "true", "yes")


def build_from_env(data_dir: str | Path = "data/out") -> OfficeJudge:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    has_jev = bool(os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_BASE_URL"))
    kind = os.environ.get("OFFICE_JUDGE_PRIMARY", "jev" if has_jev else "demo")
    has_claude = shutil.which("claude") is not None
    primary = {"jev": JevJudge, "llm": LLMJudge, "demo": demo_judge}[kind]()
    llm = LLMJudge() if _flag("OFFICE_JUDGE_FALLBACK", has_claude) and kind != "llm" else None
    writer = ClaudeWriter() if _flag("OFFICE_JUDGE_WRITER", has_claude) else None
    return OfficeJudge(
        primary=primary,
        llm=llm,
        writer=writer,
        human=HumanQueue(str(Path(data_dir) / "human_queue.db")),
        audit=AuditLog(str(Path(data_dir) / "audit.db")),
    )


app = FastAPI(title="office-judge")
oj = build_from_env()


class TextIn(BaseModel):
    text: str
    attendees: list[str] | None = None


class UnmaskIn(BaseModel):
    text: str
    mapping: dict[str, str]


class AnswerIn(BaseModel):
    answer: Any


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/config")
def config() -> dict[str, Any]:
    c = oj.cascade
    return {
        "primary": c.primary.name,
        "llm": c.llm.name if c.llm else None,
        "writer": type(oj.writer).__name__ if oj.writer else None,
        "attendees": oj.company.get("default_attendees", []),
        "company": oj.company.get("name"),
    }


@app.post("/api/analyze")
def analyze(body: TextIn) -> dict[str, Any]:
    return oj.analyze(body.text, body.attendees)


@app.post("/api/dlp")
def dlp(body: TextIn) -> dict[str, Any]:
    return oj.dlp(body.text).model_dump()


@app.post("/api/ticket")
def ticket(body: TextIn) -> dict[str, Any]:
    return oj.route_ticket(body.text).model_dump()


@app.post("/api/actions")
def actions(body: TextIn) -> dict[str, Any]:
    return oj.extract_actions(body.text, body.attendees).model_dump()


@app.post("/api/unmask")
def unmask(body: UnmaskIn) -> dict[str, str]:
    """Restore placeholders in, e.g., an external AI's reply to masked text."""
    return {"text": Vault(mapping=dict(body.mapping)).unmask(body.text)}


@app.get("/api/queue")
def queue() -> list[dict[str, Any]]:
    return oj.cascade.human.pending() if oj.cascade.human else []


@app.post("/api/queue/{item_id}")
def answer(item_id: int, body: AnswerIn) -> dict[str, str]:
    if oj.cascade.human:
        oj.cascade.human.resolve(item_id, body.answer)
    return {"status": "ok"}
