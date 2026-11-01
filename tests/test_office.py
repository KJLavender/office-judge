import datetime as dt

import pytest

from judge_core.backends import FakeJudge, HumanQueue
from office_judge.demo_backend import demo_judge
from office_judge.detectors import build_rule_judge, valid_tw_id, valid_ubn
from office_judge.pipelines import OfficeJudge
from office_judge.text import Vault, split_sentences

RULE = build_rule_judge(["北極星"], ["宏遠銀行"])


def kinds(text):
    return [f.detector for f in RULE.scan(text)]


@pytest.mark.parametrize("text,expected", [
    ("身分證 A123456789", ["tw_id"]),
    ("身分證 A123456788", []),  # bad checksum
    ("居留證 A800000014", ["tw_arc"]),
    ("統編 22099131", ["tw_ubn"]),
    ("訂單編號 22099131", []),  # 8 digits without context
    ("手機 0912-345-678", ["tw_mobile"]),
    ("卡號 4111 1111 1111 1111", ["credit_card"]),
    ("卡號 4111 1111 1111 1112", []),
    ("key AKIAABCDEFGHIJKLMNOP", ["aws_key"]),
    ("token ghp_" + "a" * 36, ["github_token"]),
    ("api_key=sk-proj-abcdefghijklmnopqrstuvwx", ["api_key"]),
    ("密碼是 Hunter2024!", ["password"]),
    ("北極星要延期", ["codename"]),
    ("客戶是宏遠銀行", ["customer"]),
])
def test_detectors(text, expected):
    assert kinds(text) == expected


def test_checksums():
    assert valid_tw_id("A123456789") and not valid_tw_id("A123456780")
    assert valid_ubn("04595257") and not valid_ubn("12345678")


def test_split_sentences_mixed_language():
    s = split_sentences("第一句。Second one. Third!\n第四句？")
    assert [x.text for x in s] == ["第一句。", "Second one.", "Third!", "第四句？"]
    text = "甲。乙！"
    assert all(text[x.start:x.end] == x.text for x in split_sentences(text))


def test_vault_roundtrip_and_stable_placeholders():
    text = "A123456789 和 A123456789 以及 B123456780"
    v = Vault()
    findings = RULE.scan(text)
    masked = v.mask_findings(text, findings)
    assert "A123456789" not in masked
    assert masked.count("[身分證號_1]") == 2
    assert v.unmask(masked) == text
    # external AI reply that reuses the placeholders
    assert v.unmask("已處理 [身分證號_1]") == "已處理 A123456789"


class FakeWriter:
    def explain(self, text, finding):
        return "這句提到未公開的營收數字"

    def resolve_actions(self, items, today):
        return [{"i": it["i"], "task": "寄出報價單", "due_date": "2026-11-04", "due_text": "下週三前"} for it in items]


@pytest.fixture
def oj(tmp_path):
    return OfficeJudge(primary=demo_judge(), writer=FakeWriter(), human=HumanQueue(str(tmp_path / "q.db")))


def test_dlp_pass(oj):
    r = oj.dlp("明天下午三點開會，記得帶筆電。")
    assert r.action == "pass" and r.masked_text == "明天下午三點開會，記得帶筆電。"


def test_dlp_masks_pii_and_semantic(oj):
    r = oj.dlp("聯絡人手機 0912-345-678。下一季營收預估 3 億。謝謝。")
    assert r.action == "mask"
    assert "0912" not in r.masked_text and "3 億" not in r.masked_text
    assert "[機密內容_1]" in r.masked_text and "謝謝" in r.masked_text
    assert any("營收" in x for x in r.reasons)  # writer reason used
    assert Vault(mapping=r.mapping).unmask(r.masked_text).startswith("聯絡人手機 0912-345-678")


def test_dlp_blocks_secrets(oj):
    r = oj.dlp("測試環境 api_key=sk-proj-abcdefghijklmnopqrstuvwx 先給你")
    assert r.action == "block" and any("更換" in x for x in r.reasons)


def test_dlp_gray_sentence_blocks_pending_review(oj):
    r = oj.dlp("這是內部的系統說明。")  # demo judge: gray, no LLM -> human queue
    assert r.action == "block"
    assert r.flagged[0].status == "pending_human"
    assert len(oj.cascade.human.pending()) == 1


def test_dlp_never_sends_known_pii_to_judge(tmp_path):
    seen = []
    spy = FakeJudge(lambda s, qid, q: (seen.append((s, q.instructions)), 0.01)[1])
    OfficeJudge(primary=spy).dlp("身分證 A123456789，北極星下週上線。")
    assert seen and all("A123456789" not in s and "北極星" not in s and "北極星" not in ins for s, ins in seen)


def test_dlp_fails_closed(tmp_path):
    down = FakeJudge(lambda *a: None, fail=True)
    r = OfficeJudge(primary=down).dlp("一般內容。")
    assert r.action == "block" and r.flagged[0].status == "failsafe"


def test_ticket_routing(oj):
    r = oj.route_ticket("整個業務部 VPN 連不上，outlook 也收不到信")
    assert r.team == "IT" and r.urgency == 3.0 and r.urgency_label == "非常緊急"


def test_ticket_unclear_goes_to_human(oj):
    r = oj.route_ticket("你好，我的東西壞了沒辦法用")
    assert r.team is None and r.trace["team"]["status"] == "pending_human"


def test_actions(oj):
    text = "Kevin：客訴處理完了。\n怡君：報價單下週三前要寄出，這個請志明處理。\nAmy：我來整理教材。\n大家辛苦了。"
    r = oj.extract_actions(text, ["Kevin", "怡君", "志明", "Amy"], today=dt.date(2026, 10, 29))
    sentences = [a.sentence for a in r.items]
    assert len(r.items) == 2 and r.postprocessed
    assert "志明" in sentences[0] and r.items[0].owner == "志明"
    assert r.items[0].due_date == "2026-11-04"
    # one primary call for all sentences and all questions
    assert len(oj.cascade.primary.calls) == 1


def test_analyze_runs_all_three(oj):
    out = oj.analyze("請建宏幫忙申請 VPN 帳號，身分證 A123456789。")
    assert set(out) == {"dlp", "ticket", "actions"}
    assert out["dlp"]["action"] == "mask"
    assert out["ticket"]["team"] == "IT"
    assert out["actions"]["items"]


def test_non_ticket_skips_routing_and_queue(oj):
    r = oj.route_ticket("今天天氣很好，大家中午一起吃飯吧。")
    assert r.is_ticket is False and r.team is None
    assert oj.cascade.human.pending() == []
