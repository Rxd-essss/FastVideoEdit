"""Filler-word detection (mumbles, single words, multi-word phrases)."""
from __future__ import annotations

import re

from ..config import FillerLists, FillersCfg
from ..models import ACTION_REMOVE, TYPE_FILLER, CutSegment, Word
from ..textnorm import normalize


# A mumble must normalise to at least this many characters before it is removed,
# so a single bare letter the ASR emits ("а", "э", "м") — which is usually the
# legitimate conjunction/interjection, not a stretched hesitation — is left in.
# Stretched forms ("аа", "эээ", "ммм", "нуу") are >= 2 chars and still caught.
_MIN_MUMBLE_LEN = 2


def _compile_mumble_matcher(lists: FillerLists) -> re.Pattern:
    """Anchored regex over the stretched-sound (mumble) patterns only."""
    pats = [m for m in lists.mumbles if m]   # already regexes (e.g. "э+")
    if not pats:
        return re.compile(r"(?!x)x")         # matches nothing
    return re.compile(r"^(?:" + "|".join(pats) + r")$")


def _compile_words_matcher(lists: FillerLists) -> re.Pattern:
    """Anchored regex over the plain single-word fillers (escaped) only."""
    pats: list[str] = []
    for w in lists.words:             # plain words -> escaped
        n = normalize(w)
        if n:
            pats.append(re.escape(n))
    if not pats:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile(r"^(?:" + "|".join(pats) + r")$")


def detect(words: list[Word], cfg: FillersCfg, lists: FillerLists) -> list[CutSegment]:
    out: list[CutSegment] = []
    norm = [normalize(w.word) for w in words]
    consumed = [False] * len(words)
    pad = cfg.pad

    def clamp(i_start: int, i_end: int) -> tuple[float, float]:
        a = words[i_start].start - pad
        b = words[i_end].end + pad
        # don't eat into neighbouring words
        if i_start > 0:
            a = max(a, words[i_start - 1].end)
        if i_end < len(words) - 1:
            b = min(b, words[i_end + 1].start)
        return max(0.0, a), b

    # 1) multi-word phrases first (so their members aren't double-flagged).
    phrases = [[normalize(t) for t in ph] for ph in lists.phrases]
    phrases = [ph for ph in phrases if all(ph)]
    for ph in sorted(phrases, key=len, reverse=True):
        L = len(ph)
        i = 0
        while i + L <= len(words):
            if not any(consumed[i:i + L]) and norm[i:i + L] == ph:
                a, b = clamp(i, i + L - 1)
                if b - a > 1e-4:   # skip if neighbour clamping inverted the span
                    out.append(CutSegment(
                        id="", start=round(a, 3), end=round(b, 3),
                        type=TYPE_FILLER, action=ACTION_REMOVE, enabled=True,
                        text=" ".join(words[j].word.strip() for j in range(i, i + L))))
                for j in range(i, i + L):
                    consumed[j] = True
                i += L
            else:
                i += 1

    # 2) single tokens / mumbles.
    mumble_rx = _compile_mumble_matcher(lists)
    words_rx = _compile_words_matcher(lists)
    for i, n in enumerate(norm):
        if consumed[i] or not n:
            continue
        is_word = words_rx.match(n) is not None
        # Mumbles must be stretched (len >= 2) so a bare "а"/"э"/"м"/"ну" — which
        # is far more often a real conjunction/particle than a hesitation — stays.
        is_mumble = (len(n) >= _MIN_MUMBLE_LEN) and (mumble_rx.match(n) is not None)
        if is_word or is_mumble:
            a, b = clamp(i, i)
            if b - a > 1e-4:   # skip if neighbour clamping inverted the span
                out.append(CutSegment(
                    id="", start=round(a, 3), end=round(b, 3),
                    type=TYPE_FILLER, action=ACTION_REMOVE, enabled=True,
                    text=words[i].word.strip()))
            consumed[i] = True

    return out
