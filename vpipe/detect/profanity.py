"""Profanity detection by root/stem (with an allow-list of false positives)."""
from __future__ import annotations

import re
from typing import Optional

from ..config import ProfanityCfg, ProfanityLists
from ..models import TYPE_PROFANITY, CutSegment, Word
from ..textnorm import normalize


class ProfanityMatcher:
    """Reusable matcher — also used by subtitle text masking."""

    def __init__(self, lists: ProfanityLists):
        self.allow = {normalize(w) for w in lists.allow}
        # Inputs are normalized (ё->е) before matching, so fold ё in the roots
        # too — otherwise a root that literally requires ё (e.g. "ёб") is dead.
        # Only ё is folded; the rest of each root keeps its regex syntax.
        roots = [r.replace("ё", "е") for r in lists.roots if r]
        self.rx: Optional[re.Pattern] = (
            re.compile("|".join(roots)) if roots else None)

    def is_profane(self, raw_word: str) -> bool:
        n = normalize(raw_word)
        if not n or self.rx is None:
            return False
        if n in self.allow:
            return False
        return self.rx.search(n) is not None


def detect(words: list[Word], cfg: ProfanityCfg,
           lists: ProfanityLists) -> list[CutSegment]:
    matcher = ProfanityMatcher(lists)
    out: list[CutSegment] = []
    for w in words:
        if matcher.is_profane(w.word):
            out.append(CutSegment(
                id="", start=round(w.start, 3), end=round(w.end, 3),
                type=TYPE_PROFANITY, action=cfg.action, enabled=True,
                text=w.word.strip(), word=w.word.strip()))
    return out
