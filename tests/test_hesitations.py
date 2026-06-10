"""Unit tests for the acoustic «hesitation» detector (pure selection logic).

These exercise the gap-selection / padding / dedup logic on SYNTHETIC VAD gaps
(``_get_non_speech_gaps`` is monkeypatched), so no audio / onnxruntime / torch
is touched. ``_overlaps_existing`` is also tested directly.
"""
import vpipe.detect.hesitations as hdet
from vpipe.config import Config, FillerLists, HesitationsCfg, ProfanityLists
from vpipe.detect import run_detection
from vpipe.detect.hesitations import _overlaps_existing, detect
from vpipe.models import (ACTION_REMOVE, TYPE_FILLER, TYPE_HESITATION,
                          TYPE_PAUSE, CutSegment, Segment, Transcript, Word)


def _seg(a, b, typ=TYPE_PAUSE):
    return CutSegment(id="", start=a, end=b, type=typ, action=ACTION_REMOVE,
                      enabled=True)


def _patch_gaps(monkeypatch, gaps):
    """Make _get_non_speech_gaps return a fixed synthetic gap list."""
    monkeypatch.setattr(hdet, "_get_non_speech_gaps", lambda *_a, **_k: list(gaps))


# --- _overlaps_existing ------------------------------------------------------

def test_overlap_fraction_threshold():
    existing = [_seg(2.0, 3.0)]
    # candidate [2.4, 2.8] is fully inside -> fraction 1.0 >= 0.5 -> dup
    assert _overlaps_existing(2.4, 2.8, existing, 0.5)
    # candidate [2.9, 3.4]: overlap [2.9,3.0]=0.1 of span 0.5 = 0.2 < 0.5 -> keep
    assert not _overlaps_existing(2.9, 3.4, existing, 0.5)


def test_overlap_no_touch_is_kept():
    existing = [_seg(0.0, 1.0)]
    assert not _overlaps_existing(2.0, 2.3, existing, 0.5)


def test_overlap_empty_existing():
    assert not _overlaps_existing(1.0, 1.5, [], 0.5)


def test_overlap_zero_or_inverted_span_is_dup():
    # a degenerate candidate is treated as a duplicate (never emitted)
    assert _overlaps_existing(2.0, 2.0, [], 0.5)
    assert _overlaps_existing(3.0, 2.0, [], 0.5)


# --- detect: thresholds ------------------------------------------------------

def test_detect_filters_too_short_and_too_long(monkeypatch):
    cfg = HesitationsCfg(min_duration=0.08, max_duration=0.55,
                         pad_start=0.0, pad_end=0.0)
    _patch_gaps(monkeypatch, [
        (1.0, 1.04),   # 0.04 -> below min -> drop
        (2.0, 2.30),   # 0.30 -> in range -> keep
        (3.0, 3.60),   # 0.60 -> >= max -> drop (that's a pause)
    ])
    out = detect("x.wav", duration=10.0, cfg=cfg, existing_segs=[])
    assert len(out) == 1
    s = out[0]
    assert s.type == TYPE_HESITATION
    assert s.action == ACTION_REMOVE
    assert s.enabled is True
    assert (round(s.start, 3), round(s.end, 3)) == (2.0, 2.3)
    assert s.text == "заминка 0.30с"
    assert s.id == ""   # id assigned later in run_detection


def test_detect_boundary_min_is_inclusive_max_is_exclusive(monkeypatch):
    cfg = HesitationsCfg(min_duration=0.10, max_duration=0.50,
                         pad_start=0.0, pad_end=0.0)
    _patch_gaps(monkeypatch, [
        (0.0, 0.10),    # exactly min -> kept (>=)
        (1.0, 1.50),    # exactly max -> dropped (>= max)
    ])
    out = detect("x.wav", duration=10.0, cfg=cfg, existing_segs=[])
    assert len(out) == 1
    assert round(out[0].end - out[0].start, 3) == 0.10


# --- detect: padding ---------------------------------------------------------

def test_detect_applies_inward_padding(monkeypatch):
    cfg = HesitationsCfg(min_duration=0.08, max_duration=0.55,
                         pad_start=0.04, pad_end=0.04)
    _patch_gaps(monkeypatch, [(2.0, 2.40)])   # raw 0.40
    out = detect("x.wav", duration=10.0, cfg=cfg, existing_segs=[])
    assert len(out) == 1
    s = out[0]
    assert round(s.start, 3) == 2.04
    assert round(s.end, 3) == 2.36
    # text reports the RAW gap length, not the padded one
    assert s.text == "заминка 0.40с"


def test_detect_padding_eats_whole_interval(monkeypatch):
    # pads (0.1 + 0.1 = 0.2) exceed the gap (0.15) -> nothing emitted
    cfg = HesitationsCfg(min_duration=0.08, max_duration=0.55,
                         pad_start=0.10, pad_end=0.10)
    _patch_gaps(monkeypatch, [(2.0, 2.15)])
    out = detect("x.wav", duration=10.0, cfg=cfg, existing_segs=[])
    assert out == []


def test_detect_clamps_to_duration(monkeypatch):
    cfg = HesitationsCfg(min_duration=0.08, max_duration=0.55,
                         pad_start=0.0, pad_end=0.0)
    _patch_gaps(monkeypatch, [(9.8, 10.2)])   # raw 0.40 but runs past duration
    out = detect("x.wav", duration=10.0, cfg=cfg, existing_segs=[])
    assert len(out) == 1
    assert out[0].end <= 10.0


# --- detect: dedup against existing segments ---------------------------------

def test_detect_dedups_against_existing(monkeypatch):
    cfg = HesitationsCfg(min_duration=0.08, max_duration=0.55,
                         pad_start=0.0, pad_end=0.0, overlap_threshold=0.5)
    _patch_gaps(monkeypatch, [
        (2.0, 2.30),    # sits inside an existing pause -> dropped
        (5.0, 5.30),    # clear of everything -> kept
    ])
    existing = [_seg(1.9, 2.6, TYPE_PAUSE)]
    out = detect("x.wav", duration=10.0, cfg=cfg, existing_segs=existing)
    starts = [round(s.start, 3) for s in out]
    assert starts == [5.0]


def test_detect_does_not_mutate_existing(monkeypatch):
    cfg = HesitationsCfg(pad_start=0.0, pad_end=0.0)
    _patch_gaps(monkeypatch, [(5.0, 5.30)])
    existing = [_seg(1.0, 1.5, TYPE_FILLER)]
    before = len(existing)
    detect("x.wav", duration=10.0, cfg=cfg, existing_segs=existing)
    assert len(existing) == before   # detector is additive, never appends here


# --- detect: graceful failure ------------------------------------------------

def test_detect_returns_empty_on_vad_failure(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("onnx blew up")
    monkeypatch.setattr(hdet, "_get_non_speech_gaps", boom)
    out = detect("missing.wav", duration=10.0, cfg=HesitationsCfg(),
                 existing_segs=[])
    assert out == []


# --- run_detection wiring ----------------------------------------------------

def _transcript():
    words = [Word("раз", 0.0, 0.4), Word("два", 0.5, 0.9)]
    return Transcript(language="ru", duration=10.0, model="t", audio_hash="h",
                      segments=[Segment(0.0, 0.9, "раз два", words)])


def _cfg_only_hesitations():
    # isolate the hesitation detector so the cut-list is purely its output
    cfg = Config()
    cfg.pauses.enabled = False
    cfg.fillers.enabled = False
    cfg.profanity.enabled = False
    cfg.bad_takes.enabled = False
    return cfg


def test_run_detection_skips_hesitations_without_audio_path():
    # No audio_path -> graceful skip, no crash, no hesitation segments.
    cfg = _cfg_only_hesitations()
    cl = run_detection(_transcript(), cfg, FillerLists(), ProfanityLists(),
                       source="x", llm=None, log=lambda *_: None)
    assert all(s.type != TYPE_HESITATION for s in cl.segments)


def test_run_detection_respects_disabled_flag(monkeypatch):
    _patch_gaps(monkeypatch, [(2.0, 2.30)])
    cfg = _cfg_only_hesitations()
    cfg.hesitations.enabled = False
    cl = run_detection(_transcript(), cfg, FillerLists(), ProfanityLists(),
                       source="x", llm=None, log=lambda *_: None,
                       audio_path=__file__)  # any existing path; detector won't run
    assert all(s.type != TYPE_HESITATION for s in cl.segments)


def test_run_detection_emits_and_ids_hesitations(monkeypatch):
    _patch_gaps(monkeypatch, [(2.0, 2.30), (4.0, 4.30)])
    cfg = _cfg_only_hesitations()
    cl = run_detection(_transcript(), cfg, FillerLists(), ProfanityLists(),
                       source="x", llm=None, log=lambda *_: None,
                       audio_path=__file__)   # existing file -> detector runs
    hes = [s for s in cl.segments if s.type == TYPE_HESITATION]
    assert len(hes) == 2
    assert sorted(s.id for s in hes) == ["he000", "he001"]
