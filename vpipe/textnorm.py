"""Shared text normalization for Russian word matching."""
from __future__ import annotations

import re

_NONLETTER = re.compile(r"[^а-я]+")


def normalize(word: str) -> str:
    """Lowercase, fold ё→е, and strip everything but Cyrillic letters.

    Used so that "Ну,", "ну" and "НУ" all match the same filler/profanity rule.
    """
    w = word.lower().replace("ё", "е")
    return _NONLETTER.sub("", w)
