"""Deterministic detectors: Taiwan PII (with checksums), cards, secrets, company lexicons.

Presidio ships no Taiwan recognizers, so these are written from the official
check-digit rules. Every pattern here is cheap, local and exact.
"""

from __future__ import annotations

import re

from judge_core.backends.rule import Detector, RuleJudge, lexicon_detector

# National ID / ARC first-letter codes (A=10 ... Z=33, with the historic I/O/W/X/Y/Z gaps)
LETTER_CODES = dict(zip("ABCDEFGHJKLMNPQRSTUVXYWZIO", range(10, 36)))


def _tw_id_checksum(code: int, digits: str) -> bool:
    nums = [code // 10, code % 10] + [int(c) for c in digits]
    weights = [1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1]
    return sum(n * w for n, w in zip(nums, weights)) % 10 == 0


def valid_tw_id(s: str) -> bool:
    """National ID (A123456789) and new-format ARC (A800000014): letter + [1289] + 8 digits."""
    s = s.upper()
    if not re.fullmatch(r"[A-Z][1289]\d{8}", s):
        return False
    return _tw_id_checksum(LETTER_CODES[s[0]], s[1:])


def valid_old_arc(s: str) -> bool:
    """Old-format ARC (AA00000009): second letter contributes its code's last digit."""
    s = s.upper()
    if not re.fullmatch(r"[A-Z][A-D]\d{8}", s):
        return False
    return _tw_id_checksum(LETTER_CODES[s[0]], str(LETTER_CODES[s[1]] % 10) + s[2:])


def valid_ubn(s: str) -> bool:
    """Unified Business Number (統一編號). Since 2023 the rule is sum % 5 == 0."""
    if not re.fullmatch(r"\d{8}", s):
        return False
    weights = [1, 2, 1, 2, 1, 2, 4, 1]
    total = sum(sum(divmod(int(c) * w, 10)) for c, w in zip(s, weights))
    return total % 5 == 0 or (s[6] == "7" and (total + 1) % 5 == 0)


def luhn(s: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


# Human-readable labels, also used as mask placeholders
LABELS = {
    "tw_id": "身分證號",
    "tw_arc": "居留證號",
    "tw_ubn": "統一編號",
    "tw_mobile": "手機號碼",
    "tw_nhi": "健保卡號",
    "credit_card": "信用卡號",
    "email": "Email",
    "aws_key": "AWS 金鑰",
    "github_token": "GitHub Token",
    "slack_token": "Slack Token",
    "api_key": "API 金鑰",
    "google_key": "Google 金鑰",
    "private_key": "私鑰",
    "jwt": "JWT",
    "password": "密碼",
    "codename": "專案代號",
    "customer": "客戶名稱",
}

_B = r"(?<![A-Za-z0-9])"  # left boundary that also works next to CJK
_E = r"(?![A-Za-z0-9])"

PII_DETECTORS = [
    Detector("tw_id", "pii", re.compile(_B + r"[A-Za-z][12]\d{8}" + _E), valid_tw_id),
    Detector("tw_arc", "pii", re.compile(_B + r"[A-Za-z][89]\d{8}" + _E), valid_tw_id),
    Detector("tw_arc", "pii", re.compile(_B + r"[A-Za-z][A-Da-d]\d{8}" + _E), valid_old_arc),
    # 8 digits are everywhere; only count them next to a context word
    Detector("tw_ubn", "pii", re.compile(r"(?:統編|統一編號|VAT|Tax ID)\s*[:：#]?\s*(\d{8})" + _E, re.I), valid_ubn, group=1),
    Detector("tw_nhi", "pii", re.compile(r"健保卡(?:號|卡號)?\s*[:：]?\s*(\d{12})" + _E), group=1),
    Detector("tw_mobile", "pii", re.compile(r"(?<!\d)(?:\+886[-\s]?9|09)\d{2}[-\s]?\d{3}[-\s]?\d{3}(?!\d)")),
    Detector("credit_card", "pii", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), luhn),
    Detector("email", "pii", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]

# Patterns follow the shapes used by gitleaks / detect-secrets
SECRET_DETECTORS = [
    Detector("aws_key", "secret", re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}" + _E)),
    Detector("github_token", "secret", re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,})")),
    Detector("slack_token", "secret", re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}")),
    Detector("api_key", "secret", re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    Detector("google_key", "secret", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    Detector("private_key", "secret", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z]+ )?PRIVATE KEY-----|$)")),
    Detector("jwt", "secret", re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    Detector(
        "password", "secret",
        re.compile(r"(?:password|passwd|pwd|密碼|secret|token|api[_-]?key)\s*(?:是|為|[:=：])\s*[\"']?([^\s\"'，。,;；]{6,})", re.I),
        group=1,
    ),
]


def build_rule_judge(codenames: list[str], customers: list[str]) -> RuleJudge:
    detectors = PII_DETECTORS + SECRET_DETECTORS + [
        lexicon_detector("codename", "codename", codenames),
        lexicon_detector("customer", "customer", customers),
    ]
    return RuleJudge(detectors=detectors)
