"""Unit tests for scripts/benchmark_cuts.py — pure metric functions on
synthetic words/cuts, plus the cache-only I/O paths. No ffmpeg / GPU / VAD."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_cuts as bc  # noqa: E402

from vpipe.config import Config, FillerLists, ProfanityLists  # noqa: E402
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, TYPE_FILLER,  # noqa: E402
                          TYPE_HESITATION, TYPE_MANUAL, TYPE_PAUSE,
                          TYPE_PROFANITY, CutSegment, Segment, Transcript, Word)


def W(start, text="слово", dur=0.4):
    return Word(text, start, start + dur)


def C(a, b, typ=TYPE_PAUSE, action=ACTION_REMOVE, enabled=True):
    return CutSegment(id="", start=a, end=b, type=typ, action=action,
                      enabled=enabled)


# --- interval helpers ---------------------------------------------------------

def test_merge_spans_merges_and_drops_inverted():
    spans = [(3.0, 4.0), (1.0, 2.0), (1.5, 2.5), (5.0, 5.0), (7.0, 6.0)]
    assert bc.merge_spans(spans) == [(1.0, 2.5), (3.0, 4.0)]


def test_merge_spans_touching_are_joined():
    assert bc.merge_spans([(1.0, 2.0), (2.0, 3.0)]) == [(1.0, 3.0)]


def test_covered_length():
    merged = [(1.0, 2.0), (3.0, 4.0)]
    assert bc.covered_length(0.0, 5.0, merged) == 2.0
    assert bc.covered_length(1.5, 3.5, merged) == 1.0
    assert bc.covered_length(4.5, 5.0, merged) == 0.0
    assert bc.covered_length(2.0, 2.0, merged) == 0.0   # empty span


def test_remove_spans_filters_disabled_and_censor():
    cuts = [C(1.0, 2.0),
            C(1.5, 2.5),                                 # overlaps -> merged
            C(3.0, 4.0, enabled=False),                  # disabled -> out
            C(5.0, 6.0, typ=TYPE_PROFANITY, action=ACTION_CENSOR)]  # censor -> out
    assert bc.remove_spans(cuts) == [(1.0, 2.5)]


# --- SAFETY --------------------------------------------------------------------

def test_safety_no_cuts_no_violations():
    m = bc.safety_metrics([W(1.0)], [])
    assert m == {"auto_cuts": 0, "violating_cuts": 0,
                 "violation_pct": 0.0, "violations": []}


def test_safety_partial_clip_is_violation():
    words = [W(1.0, "тест", 0.5)]                       # 1.0–1.5
    cuts = [C(1.4, 2.0, typ=TYPE_PAUSE)]                # bites 100 ms of tail
    m = bc.safety_metrics(words, cuts)
    assert m["auto_cuts"] == 1
    assert m["violating_cuts"] == 1
    assert m["violation_pct"] == 100.0
    v = m["violations"][0]
    assert v["cut_type"] == TYPE_PAUSE
    assert v["word"] == "тест"
    assert abs(v["overlap_ms"] - 100.0) < 0.5


def test_safety_full_word_swallow_is_not_violation():
    # A filler cut covering the whole word removes it INTENTIONALLY.
    words = [W(1.0, "вот", 0.5)]                        # 1.0–1.5
    cuts = [C(0.95, 1.55, typ=TYPE_FILLER)]
    m = bc.safety_metrics(words, cuts)
    assert m["violating_cuts"] == 0
    assert m["violations"] == []


def test_safety_tolerance_boundary():
    words = [W(1.0, "тест", 0.5)]                       # ends at 1.5
    # overlap exactly 12 ms -> within tolerance -> safe
    m = bc.safety_metrics(words, [C(1.488, 2.0)])
    assert m["violating_cuts"] == 0
    # overlap 13 ms -> violation
    m = bc.safety_metrics(words, [C(1.487, 2.0)])
    assert m["violating_cuts"] == 1


def test_safety_residual_within_tolerance_counts_as_swallowed():
    # Cut covers all but the last 10 ms of the word -> residual sliver <= 12 ms
    # is inaudible -> treated as full removal, not clipping.
    words = [W(1.0, "тест", 0.5)]
    m = bc.safety_metrics(words, [C(0.9, 1.49)])
    assert m["violating_cuts"] == 0


def test_safety_excludes_profanity_manual_disabled_censor():
    words = [W(1.0, "тест", 0.5)]
    cuts = [C(1.2, 2.0, typ=TYPE_PROFANITY, action=ACTION_CENSOR),
            C(1.2, 2.0, typ=TYPE_PROFANITY),
            C(1.2, 2.0, typ=TYPE_MANUAL),
            C(1.2, 2.0, typ=TYPE_PAUSE, enabled=False)]
    m = bc.safety_metrics(words, cuts)
    assert m["auto_cuts"] == 0
    assert m["violating_cuts"] == 0


def test_safety_one_cut_clipping_two_words_counts_once():
    words = [W(1.0, "раз", 0.5), W(1.6, "два", 0.5)]    # 1.0–1.5, 1.6–2.1
    cuts = [C(1.4, 1.7, typ=TYPE_HESITATION)]           # clips both tails
    m = bc.safety_metrics(words, cuts)
    assert m["auto_cuts"] == 1
    assert m["violating_cuts"] == 1                     # per-cut, not per-word
    assert len(m["violations"]) == 2                    # but both words listed


# --- CLEANLINESS (а): fillers ---------------------------------------------------

_LISTS = FillerLists(mumbles=["э{2,}", "м{2,}"], words=["вот", "короче"])


def test_match_filler_words_dictionary_semantics():
    words = [W(0.0, "Привет"), W(1.0, "вот"), W(2.0, "эээ"),
             W(3.0, "э"),          # single char -> NOT a mumble (len >= 2 rule)
             W(4.0, "Короче,")]    # punctuation/case-normalized
    got = [w.word for w in bc.match_filler_words(words, _LISTS)]
    assert got == ["вот", "эээ", "Короче,"]


def test_filler_coverage_counts_covered_fraction():
    words = [W(1.0, "вот", 0.4), W(3.0, "эээ", 0.4)]
    cuts = [C(0.96, 1.44, typ=TYPE_FILLER)]             # covers "вот" fully
    m = bc.filler_coverage(words, cuts, _LISTS)
    assert m["total"] == 2
    assert m["covered"] == 1
    assert m["coverage_pct"] == 50.0


def test_filler_coverage_threshold():
    words = [W(1.0, "вот", 0.4)]                        # 1.0–1.4
    # 0.15/0.4 = 37.5% < 50% -> not covered
    m = bc.filler_coverage(words, [C(1.25, 2.0)], _LISTS)
    assert m["covered"] == 0
    # 0.25/0.4 = 62.5% >= 50% -> covered
    m = bc.filler_coverage(words, [C(1.15, 2.0)], _LISTS)
    assert m["covered"] == 1


def test_filler_coverage_no_fillers_is_100():
    m = bc.filler_coverage([W(0.0, "привет")], [], _LISTS)
    assert m == {"total": 0, "covered": 0, "coverage_pct": 100.0}


# --- CLEANLINESS (б): long pauses ------------------------------------------------

def test_long_pause_uncovered_without_cuts():
    words = [W(0.0, dur=0.4), W(2.0, dur=0.4)]          # gap 0.4–2.0 = 1.6 s
    m = bc.long_pause_residuals(words, [])
    assert m["long_gaps"] == 1
    assert m["uncovered"] == 1
    d = m["uncovered_details"][0]
    assert (d["start"], d["end"]) == (0.4, 2.0)


def test_long_pause_covered_with_padded_cut():
    words = [W(0.0, dur=0.4), W(2.0, dur=0.4)]
    # pause detector style: pads keep 0.15 s of air on each side; residual
    # 0.3 s <= 0.9 -> the long pause is considered handled.
    m = bc.long_pause_residuals(words, [C(0.55, 1.85)])
    assert m["long_gaps"] == 1
    assert m["uncovered"] == 0


def test_long_pause_short_gap_not_counted():
    words = [W(0.0, dur=0.4), W(1.2, dur=0.4)]          # gap 0.8 <= 0.9
    m = bc.long_pause_residuals(words, [])
    assert m["long_gaps"] == 0


def test_long_pause_partial_cut_still_long():
    words = [W(0.0, dur=0.4), W(3.0, dur=0.4)]          # gap 2.6 s
    # only 1.0 s removed -> residual 1.6 s > 0.9 -> still uncovered
    m = bc.long_pause_residuals(words, [C(1.0, 2.0)])
    assert m["uncovered"] == 1


# --- CLEANLINESS (в): VAD hesitation gaps ------------------------------------------

def test_hesitation_gap_window_filtering():
    gaps = [(1.0, 1.1),    # 0.10 < 0.2 -> out
            (2.0, 2.3),    # 0.30 in window
            (0.0, 0.55),   # exactly max -> out (>= max; that's a pause)
            (3.0, 3.6),    # 0.60 -> out
            (4.0, 4.2)]    # 0.20 -> exactly min -> in
    m = bc.hesitation_residuals(gaps, [])
    assert m["gaps"] == 2
    assert m["uncovered"] == 2


def test_hesitation_covered_by_cut():
    gaps = [(2.0, 2.3)]
    # 0.2/0.3 = 67% >= 50% -> covered
    m = bc.hesitation_residuals(gaps, [C(2.05, 2.25, typ=TYPE_HESITATION)])
    assert m == {"gaps": 1, "uncovered": 0}
    # 0.1/0.3 = 33% < 50% -> uncovered
    m = bc.hesitation_residuals(gaps, [C(2.0, 2.1, typ=TYPE_HESITATION)])
    assert m == {"gaps": 1, "uncovered": 1}


# --- summary --------------------------------------------------------------------

def test_cut_summary_counts_and_removed_union():
    cuts = [C(1.0, 2.0, typ=TYPE_PAUSE),
            C(1.5, 2.5, typ=TYPE_PAUSE),                # overlap -> union 1.5 s
            C(5.0, 5.2, typ=TYPE_FILLER),
            C(6.0, 6.4, typ=TYPE_PROFANITY, action=ACTION_CENSOR),  # not removed
            C(8.0, 9.0, typ=TYPE_MANUAL, enabled=False)]            # disabled
    m = bc.cut_summary(cuts, duration=10.0)
    assert m["by_type"] == {TYPE_PAUSE: 2, TYPE_FILLER: 1, TYPE_PROFANITY: 1}
    assert m["cuts_total"] == 4
    assert abs(m["removed_s"] - 1.7) < 1e-6
    assert abs(m["removed_pct"] - 17.0) < 1e-6
    assert abs(m["final_s"] - 8.3) < 1e-6


def test_cut_summary_zero_duration_guard():
    m = bc.cut_summary([C(0.0, 1.0)], duration=0.0)
    assert m["removed_pct"] == 0.0
    assert m["final_s"] == 0.0


# --- aggregate ---------------------------------------------------------------------

def test_aggregate_weighted_not_mean_of_means():
    r1 = {"duration_s": 100.0,
          "summary": {"removed_s": 10.0, "cuts_total": 10,
                      "by_type": {TYPE_PAUSE: 10}},
          "safety": {"auto_cuts": 10, "violating_cuts": 0},
          "fillers": {"total": 8, "covered": 8},
          "pauses": {"long_gaps": 3, "uncovered": 1},
          "hesitations": {"gaps": 5, "uncovered": 2}}
    r2 = {"duration_s": 50.0,
          "summary": {"removed_s": 20.0, "cuts_total": 2,
                      "by_type": {TYPE_FILLER: 2}},
          "safety": {"auto_cuts": 2, "violating_cuts": 1},
          "fillers": {"total": 2, "covered": 0},
          "pauses": {"long_gaps": 1, "uncovered": 0},
          "hesitations": None}
    agg = bc.aggregate_results([r1, r2])
    assert agg["clips"] == 2
    assert agg["duration_s"] == 150.0
    assert agg["removed_s"] == 30.0
    assert agg["removed_pct"] == 20.0
    assert agg["safety"] == {"auto_cuts": 12, "violating_cuts": 1,
                             "violation_pct": 8.33}
    assert agg["fillers"] == {"total": 10, "covered": 8, "coverage_pct": 80.0}
    assert agg["pauses"] == {"long_gaps": 4, "uncovered": 1}
    assert agg["hesitations"] == {"gaps": 5, "uncovered": 2}   # r2 = n/a, skipped
    assert agg["by_type"] == {TYPE_PAUSE: 10, TYPE_FILLER: 2}


def test_aggregate_empty_is_none():
    assert bc.aggregate_results([]) is None


# --- benchmark_clip: cache-only I/O (no ffmpeg / GPU) --------------------------------

def _tmp_cfg(tmp_path: Path) -> Config:
    return Config(paths={"cache_dir": str(tmp_path / "cache"),
                         "work_dir": str(tmp_path / "work"),
                         "out_dir": str(tmp_path / "out")})


def test_benchmark_clip_skips_without_cached_transcript(tmp_path):
    clip = tmp_path / "video.mp4"
    clip.write_bytes(b"fake video bytes" * 64)
    cfg = _tmp_cfg(tmp_path)
    res = bc.benchmark_clip(clip, cfg, FillerLists(), ProfanityLists(), vad=False)
    assert res["skipped"] is True
    assert "транскрипт" in res["reason"]


def test_benchmark_clip_skips_missing_file(tmp_path):
    res = bc.benchmark_clip(tmp_path / "nope.mp4", _tmp_cfg(tmp_path),
                            FillerLists(), ProfanityLists(), vad=False)
    assert res["skipped"] is True
    assert "не найден" in res["reason"]


def test_benchmark_clip_with_cached_transcript(tmp_path):
    clip = tmp_path / "video.mp4"
    clip.write_bytes(b"fake video bytes" * 64)
    cfg = _tmp_cfg(tmp_path)
    (tmp_path / "cache").mkdir()

    from vpipe.probe import hash_input
    h = hash_input(clip)
    words = [Word("Привет", 0.2, 0.6), Word("эээ", 0.8, 1.1),
             Word("мир", 4.0, 4.4)]                      # 2.9 s gap -> long pause
    tr = Transcript(language="ru", duration=6.0, model="large-v3",
                    audio_hash=h,
                    segments=[Segment(0.2, 4.4, "Привет эээ мир", words)])
    tr.save(tmp_path / "cache" / f"{h}.transcript.json")

    fillers = FillerLists(mumbles=["э{2,}"], words=[])
    res = bc.benchmark_clip(clip, cfg, fillers, ProfanityLists(), vad=False)

    assert res["skipped"] is False
    assert res["model"] == "large-v3"
    assert res["duration_s"] == 6.0
    assert res["hesitations"] is None                    # vad=False -> n/a
    # detection ran for real: the filler "эээ" and the long pause were cut
    assert res["summary"]["by_type"].get(TYPE_FILLER) == 1
    assert res["summary"]["by_type"].get(TYPE_PAUSE, 0) >= 1
    assert res["fillers"] == {"total": 1, "covered": 1, "coverage_pct": 100.0}
    assert res["pauses"]["long_gaps"] == 1
    assert res["pauses"]["uncovered"] == 0               # the pause cut covers it
    assert res["safety"]["violation_pct"] == 0.0         # word-safe by design
    assert res["summary"]["removed_s"] > 0


# --- report rendering ------------------------------------------------------------------

def _fake_result(name="x.mp4"):
    return {"clip": name, "name": name, "skipped": False, "reason": "",
            "notes": [], "model": "large-v3", "duration_s": 10.0,
            "summary": {"cuts_total": 1, "by_type": {TYPE_PAUSE: 1},
                        "removed_s": 1.0, "removed_pct": 10.0, "final_s": 9.0},
            "safety": {"auto_cuts": 1, "violating_cuts": 0,
                       "violation_pct": 0.0, "violations": []},
            "fillers": {"total": 2, "covered": 2, "coverage_pct": 100.0},
            "pauses": {"long_gaps": 1, "uncovered": 0, "uncovered_details": []},
            "hesitations": {"gaps": 3, "uncovered": 1}}


def test_render_markdown_smoke():
    skipped = {"clip": "y.mp4", "name": "y.mp4", "skipped": True,
               "reason": "нет кэшированного транскрипта", "notes": []}
    md = bc.render_markdown([_fake_result(), skipped], "2026-06-10 12:00")
    assert "# Бенчмарк качества резов" in md
    assert "x.mp4" in md
    assert "Итого / среднее" in md
    assert "y.mp4" in md and "Пропущенные клипы" in md
    assert "Нарушений не найдено" in md


def test_render_markdown_lists_violations():
    r = _fake_result()
    r["safety"] = {"auto_cuts": 1, "violating_cuts": 1, "violation_pct": 100.0,
                   "violations": [{"cut_type": TYPE_PAUSE, "cut_start": 1.0,
                                   "cut_end": 2.0, "word": "тест",
                                   "word_start": 1.9, "word_end": 2.3,
                                   "overlap_ms": 100.0}]}
    md = bc.render_markdown([r], "2026-06-10 12:00")
    assert "«тест»" in md
    assert "Нарушений не найдено" not in md


def test_build_json_is_serializable():
    payload = bc.build_json([_fake_result()], "2026-06-10 12:00")
    text = json.dumps(payload, ensure_ascii=False)
    back = json.loads(text)
    assert back["aggregate"]["clips"] == 1
    assert back["params"]["word_clip_tolerance_ms"] == 12.0
    assert back["clips"][0]["name"] == "x.mp4"
