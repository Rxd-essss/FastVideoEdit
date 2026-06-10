"""Regression tests for the audit-fix pass (P0–P2)."""
from vpipe.config import MaskingCfg, ProfanityLists, SubsCfg
from vpipe.cutlist import resolve
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, TYPE_PAUSE,
                          TYPE_PROFANITY, CutList, CutSegment, Word)
from vpipe.subtitles import build_cues
from vpipe.timeline import Timeline, remap_words


def test_resolve_clips_censor_to_kept_timeline():
    # remove [2,4]; censor [3,5] straddles the cut; censor [2.4,3.6] fully inside it.
    segs = [
        CutSegment(id="r1", start=2, end=4, type=TYPE_PAUSE, action=ACTION_REMOVE, enabled=True),
        CutSegment(id="c1", start=3, end=5, type=TYPE_PROFANITY, action=ACTION_CENSOR, enabled=True),
        CutSegment(id="c2", start=2.4, end=3.6, type=TYPE_PROFANITY, action=ACTION_CENSOR, enabled=True),
    ]
    removed, censors = resolve(CutList(source="x", duration=10, segments=segs))
    assert removed == [(2.0, 4.0)]
    spans = sorted((round(c.start, 3), round(c.end, 3)) for c in censors)
    assert spans == [(4.0, 5.0)]   # c2 dropped; c1 clipped to the surviving (4,5)


def test_max_cps_extends_a_too_fast_cue():
    m = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
    word = Word("ж" * 30, 0.0, 0.4)            # 30 chars in 0.4s — far too fast
    cfg = SubsCfg(max_cps=15, min_dur=0.5, min_gap=0.05)
    cues = build_cues([word], m, cfg, MaskingCfg(), total=20.0)
    assert len(cues) == 1
    # required = 30 / 15 = 2.0s; cue end must be extended toward that.
    assert cues[0].end >= 1.9


def test_remap_words_drops_by_overlap_fraction():
    tl = Timeline([(1.0, 2.0)], duration=10)
    words = [Word("a", 0.0, 0.5),
             Word("mostlyin", 1.1, 1.9),    # 0.8 of 0.8 inside -> dropped
             Word("mostlyout", 1.8, 2.6),   # 0.2 of 0.8 inside -> kept
             Word("c", 5.0, 5.5)]
    names = [w.word for w in remap_words(words, tl)]
    assert "a" in names and "c" in names
    assert "mostlyin" not in names
    assert "mostlyout" in names
