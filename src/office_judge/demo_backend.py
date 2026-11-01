"""Keyword heuristics that mimic a judge, so the demo and tests run with no API key.

Not a baseline worth reporting: it exists to exercise the cascade end to end.
Questions are recognised by the policy text the pipelines put in them.
"""

from __future__ import annotations

import re

from judge_core.backends.fake import FakeJudge
from judge_core.types import Question, State

SENSITIVE = ["營收", "財測", "併購", "募資", "未公開", "尚未公布", "合約金額", "報價", "客戶名單", "漏洞", "毛利", "裁員"]
MAYBE_SENSITIVE = ["預計", "財報", "內部", "機密", "架構"]
ACTION = ["請", "麻煩", "我來", "我會", "負責", "處理一下", "交給", "之前完成", "前完成", "前寄", "前交", "跟進", "TODO"]
MAYBE_ACTION = ["也許", "可以看一下", "有空的話", "考慮"]
TEAM_WORDS = {
    "IT": ["VPN", "密碼", "電腦", "網路", "印表機", "軟體", "帳號", "Wi-Fi", "outlook", "Teams"],
    "HR": ["請假", "薪資", "勞保", "健保", "到職", "離職", "特休", "教育訓練"],
    "Finance": ["報帳", "發票", "付款", "採購", "預算", "請款"],
    "Facilities": ["門禁", "會議室", "冷氣", "空調", "椅子", "桌子", "文具", "停車"],
}
URGENT = ["整個", "所有人", "大家都", "全部", "資安", "中毒", "勒索"]
BLOCKING = ["沒辦法", "無法", "不能", "連不上", "壞了", "馬上", "急"]


def _quoted(instructions: str, label: str) -> str:
    m = re.search(label + r"：「(.*?)」", instructions, re.S)
    return m.group(1) if m else ""


def _respond(state: State, qid: str, q: Question):
    ins = q.instructions
    if "透露客戶資料" in ins:
        s = _quoted(ins, "句子")
        if any(w in s for w in SENSITIVE):
            return 0.92
        if any(w in s for w in MAYBE_SENSITIVE):
            return 0.3  # gray zone -> escalates
        return 0.02
    if "是否指派了某人" in ins:
        s = _quoted(ins, "本句")
        if any(w in s for w in MAYBE_ACTION):
            return 0.55
        return 0.88 if any(w in s for w in ACTION) else 0.08
    if "由誰負責" in ins:
        s = _quoted(ins, "本句")
        names = [o for o in q.options if o != "未指定"]
        body = re.sub(r"^[^：:]{1,10}[：:]", "", s)  # drop the "speaker:" prefix
        assigned = [n for n in names if re.search(r"(請|交給|麻煩)" + re.escape(n), body)]
        hit = assigned or [n for n in names if n in body]
        if hit:
            return hit[0], 0.9
        speaker = re.match(r"^([^：:]{1,10})[：:]", s)
        if re.search(r"我來|我會", body) and speaker and speaker.group(1) in names:
            return speaker.group(1), 0.85
        return "未指定", 0.7
    text = state if isinstance(state, str) else str(state)
    if "內部服務台" in ins:  # is this a ticket at all?
        hit = any(w.lower() in text.lower() for ws in TEAM_WORDS.values() for w in ws) or any(w in text for w in BLOCKING)
        return 0.9 if hit else 0.15
    if q.kind == "choice":  # team routing
        scores = {t: sum(w.lower() in text.lower() for w in ws) for t, ws in TEAM_WORDS.items() if t in q.options}
        best = max(scores, key=scores.get) if scores else q.options[0]
        if not scores or scores[best] == 0:
            return best, 0.2
        return best, 0.9 if scores[best] >= 2 else 0.65
    if q.kind == "score":
        if any(w in text for w in URGENT):
            return 3.0, 0.8
        if any(w in text for w in BLOCKING):
            return 2.0, 0.7
        return 1.0, 0.6
    return None


def demo_judge() -> FakeJudge:
    return FakeJudge(_respond, name="demo-heuristic")
