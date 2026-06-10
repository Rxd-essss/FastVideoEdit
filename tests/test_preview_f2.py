"""F2 — preview endpoints' core logic (subtitles + chapters parsing).

These cover the pure pieces the new /api/preview/* endpoints rely on, without
spinning up a real Session (no ffmpeg/probe needed):
  * _parse_chapters_txt (serve.py helper) round-trips a chapters.txt file.
  * the subtitle-preview pipeline (resolve -> Timeline -> remap_words ->
    build_cues) yields cues in FINAL (post-cut) coordinates.
"""
import serve
from vpipe.config import MaskingCfg, ProfanityLists, SubsCfg
from vpipe.cutlist import resolve
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import (ACTION_REMOVE, TYPE_MANUAL, CutList, CutSegment,
                          Segment, Transcript, Word)
from vpipe.subtitles import build_cues
from vpipe.timeline import Timeline, remap_words


def test_parse_chapters_txt_mm_ss_and_h_mm_ss(tmp_path):
    p = tmp_path / "chapters.txt"
    p.write_text(
        "# comment line skipped\n"
        "\n"
        "00:00 Вступление\n"
        "01:24 Основная часть\n"
        "1:02:05 Финал с пробелами в названии\n",
        encoding="utf-8")
    out = serve._parse_chapters_txt(str(p))
    assert out == [
        {"time": 0.0, "title": "Вступление"},
        {"time": 84.0, "title": "Основная часть"},
        {"time": 3725.0, "title": "Финал с пробелами в названии"},
    ]


def test_parse_chapters_txt_missing_file_returns_empty(tmp_path):
    assert serve._parse_chapters_txt(str(tmp_path / "nope.txt")) == []


def test_parse_chapters_txt_skips_malformed(tmp_path):
    p = tmp_path / "ch.txt"
    p.write_text("notatime title\n00:10 Глава\n", encoding="utf-8")
    out = serve._parse_chapters_txt(str(p))
    assert out == [{"time": 10.0, "title": "Глава"}]


def test_subtitle_preview_pipeline_uses_final_coords():
    # Transcript: words around a cut at [2,4]. Word inside the cut is dropped;
    # words after it shift left by 2s (the removed duration) — FINAL coords.
    words = [Word("раз", 0.0, 0.5), Word("два", 0.6, 1.0),
             Word("вырезано", 2.5, 3.5),            # inside [2,4] -> dropped
             Word("потом", 5.0, 5.5), Word("конец.", 5.6, 6.0)]
    tr = Transcript(language="ru", duration=8.0, model="t", audio_hash="h",
                    segments=[Segment(0.0, 8.0, "txt", words)])
    cl = CutList(source="x", duration=8.0, segments=[
        CutSegment(id="r1", start=2.0, end=4.0, type=TYPE_MANUAL,
                   action=ACTION_REMOVE, enabled=True)])

    removed, _ = resolve(cl)
    tl = Timeline(removed, tr.duration)
    matcher = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
    remapped = remap_words(tr.all_words(), tl)
    cues = build_cues(remapped, matcher, SubsCfg(), MaskingCfg(),
                      tl.new_duration())

    assert tl.new_duration() == 6.0                # 8 - 2 removed
    text = " ".join(c.text for c in cues)
    assert "вырезано" not in text                  # word inside the cut dropped
    # "потом" was at 5.0 originally; after removing 2s before it -> 3.0 final.
    assert any(abs(c.start - 3.0) < 0.05 for c in cues)
    # all cues live within the FINAL timeline.
    assert all(0.0 <= c.start <= 6.0 and c.end <= 6.0 for c in cues)
