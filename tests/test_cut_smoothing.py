"""Cut-quality fixes: seam de-click fades + word-safe hesitation boundaries.

The choppy («рвано») audio came from hard-concatenating kept segments with no
fade at the seams; partial-word clipping came from VAD hesitation cuts landing a
few ms inside real words. These tests pin both fixes.
"""
from __future__ import annotations

import pytest

from vpipe.config import HesitationsCfg
from vpipe.detect import hesitations as H
from vpipe.models import Word
from vpipe.render import _audio_seg_filter


# --- Fix A: per-seam audio fades (de-click) ----------------------------------
def test_no_fade_when_disabled():
    f = _audio_seg_filter("[0:a]", 1.0, 3.0, 1, 3, cut_fade=0.0)
    assert "afade" not in f
    assert f.startswith("[0:a]atrim=start=1.000:end=3.000,asetpts=PTS-STARTPTS")


def test_middle_segment_fades_both_sides():
    f = _audio_seg_filter("[1:a]", 5.0, 8.0, 1, 3, cut_fade=0.015)
    assert "afade=t=in:st=0:d=0.015" in f
    assert "afade=t=out:st=2.985:d=0.015" in f   # st = (8-5) - 0.015


def test_first_segment_no_fade_in():
    # i=0 -> the program's true start; only the trailing (internal) seam fades.
    f = _audio_seg_filter("[0:a]", 0.0, 2.0, 0, 3, cut_fade=0.015)
    assert "afade=t=in" not in f
    assert "afade=t=out" in f


def test_last_segment_no_fade_out():
    f = _audio_seg_filter("[0:a]", 10.0, 12.0, 2, 3, cut_fade=0.015)
    assert "afade=t=in" in f
    assert "afade=t=out" not in f


def test_single_segment_has_no_fades():
    # Only one kept piece -> no internal seams at all.
    f = _audio_seg_filter("[0:a]", 0.0, 5.0, 0, 1, cut_fade=0.015)
    assert "afade" not in f


def test_short_segment_clamps_fade_to_half():
    # A 0.02 s segment faded both sides -> d = min(0.015, 0.01) = 0.01.
    f = _audio_seg_filter("[0:a]", 1.0, 1.02, 1, 3, cut_fade=0.015)
    assert "afade=t=in:st=0:d=0.010" in f
    assert "afade=t=out:st=0.010:d=0.010" in f   # (0.02 - 0.01)


# --- Fix B: word-safe hesitation boundaries ----------------------------------
def _cfg():
    return HesitationsCfg(min_duration=0.08, max_duration=0.55,
                          pad_start=0.04, pad_end=0.04)


def _overlaps(seg, words, thr=0.012):
    return any(min(seg.end, w.end) - max(seg.start, w.start) > thr for w in words)


def test_hesitation_clamped_off_a_word(monkeypatch):
    # VAD gap [1.0,1.4] starts INSIDE the word «раз» (0.5–1.1).
    monkeypatch.setattr(H, "_get_non_speech_gaps", lambda p, c: [(1.0, 1.4)])
    words = [Word("раз", 0.5, 1.1, 0.9), Word("два", 1.5, 2.0, 0.9)]

    raw = H.detect("x.wav", 3.0, _cfg(), [])                 # no words
    safe = H.detect("x.wav", 3.0, _cfg(), [], words=words)   # word-safe

    assert raw and _overlaps(raw[0], words)        # unclamped DOES clip «раз»
    assert safe and not _overlaps(safe[0], words)  # clamped does NOT
    assert safe[0].start >= 1.1 and safe[0].end <= 1.5   # strictly inter-word


def test_genuine_interword_hesitation_kept(monkeypatch):
    # A real «эээ» Whisper dropped: gap sits fully between two words.
    monkeypatch.setattr(H, "_get_non_speech_gaps", lambda p, c: [(1.15, 1.45)])
    words = [Word("раз", 0.5, 1.1, 0.9), Word("два", 1.5, 2.0, 0.9)]
    safe = H.detect("x.wav", 3.0, _cfg(), [], words=words)
    assert len(safe) == 1
    assert not _overlaps(safe[0], words)


def test_word_clip_dropped_when_clamp_collapses(monkeypatch):
    # Gap entirely inside the word region -> clamp inverts -> dropped, not clipped.
    monkeypatch.setattr(H, "_get_non_speech_gaps", lambda p, c: [(0.7, 1.0)])
    words = [Word("слово", 0.4, 1.2, 0.9), Word("дальше", 1.6, 2.2, 0.9)]
    safe = H.detect("x.wav", 3.0, _cfg(), [], words=words)
    assert safe == []      # nothing safe to cut here -> no word-clipping cut
