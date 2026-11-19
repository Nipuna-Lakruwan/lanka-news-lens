"""Unicode-aware text cleaning for Sinhala, Tamil and English news.

Python's ``\\w`` treats Sinhala/Tamil combining vowel signs (category Mc/Mn) as
non-word characters on some code paths, which splits words in the middle. We
never tokenise with ``\\w``; instead we keep every letter *and* mark, and keep
ZWJ/ZWNJ (U+200D/U+200C) inside words because Sinhala conjuncts such as
``ශ්‍රී`` depend on them.
"""
from __future__ import annotations

import re
import unicodedata

_DROP = {"﻿": "", " ": " ", "​": ""}
_DROP_RE = re.compile("|".join(map(re.escape, _DROP)))
_WS = re.compile(r"[ \t\r\f\v]+")


def clean(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _DROP_RE.sub(lambda m: _DROP[m.group(0)], text)
    text = _WS.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_word_char(ch: str) -> bool:
    if ch in "‌‍":
        return True
    cat = unicodedata.category(ch)
    return cat[0] in "LMN"  # letters, combining marks, digits


def tokens(text: str) -> list[str]:
    """Split into words, keeping combining marks and joiners inside words."""
    out, cur = [], []
    for ch in text.lower():
        if _is_word_char(ch):
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return [t.strip("‌‍") for t in out if t.strip("‌‍")]
