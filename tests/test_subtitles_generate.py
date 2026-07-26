"""Golden round-trip for vpipe.subtitles.generate — the production path that
writes the sidecar .srt/.vtt under the CUT timeline (serve consumes its dict).

No ffmpeg/GPU: generate() only remaps word timings and writes text files. We
parse the emitted .srt back and assert INVARIANTS (returned cue count matches the
file, 1-based contiguous numbering, monotonic non-overlapping times within
new_duration, the word inside a cut is gone, profanity masked) rather than exact
millisecond values, so legitimate cue-timing tuning doesn't churn the golden.

Covers subtitles.py:222-246 (generate): Timeline -> remap_words -> build_cues ->
write_srt/write_vtt/write_transcript and the returned {cues,srt,vtt,transcript}
dict — previously exercised by no test (callers stubbed generate() out).
"""
from pathlib import Path

from vpipe.config import MaskingCfg, ProfanityLists, SubsCfg
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Segment, Transcript, Word
from vpipe.subtitles import generate


def _parse_srt(text):
    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    out = []
    for b in blocks:
        idx, times, *txt = b.splitlines()
        s, e = times.split(" --> ")
        out.append((int(idx), s, e, "\n".join(txt)))
    return out


def _sec(ts):                     # "HH:MM:SS,mmm" -> float seconds
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _tr():
    words = [Word("привет", 0.0, 0.5), Word("это", 0.6, 1.0),
             Word("блядь", 1.1, 1.6), Word("вырезано", 2.5, 3.5),
             Word("дальше", 6.0, 6.6), Word("конец.", 6.7, 7.2)]
    return Transcript(language="ru", duration=8.0, model="t", audio_hash="h",
                      segments=[Segment(0.0, 8.0, "txt", words)])


def test_generate_srt_golden(tmp_path):
    matcher = ProfanityMatcher(ProfanityLists(roots=["бля"], allow=[]))
    removed = [(2.0, 4.0)]                      # drops «вырезано», shifts later words -2s
    res = generate(_tr(), removed,
                   SubsCfg(write_vtt=True, write_transcript=True),
                   MaskingCfg(), matcher, tmp_path / "clip")
    srt = _parse_srt(Path(res["srt"]).read_text(encoding="utf-8"))
    assert res["cues"] == len(srt) and len(srt) >= 1
    # 1-based contiguous numbering
    assert [b[0] for b in srt] == list(range(1, len(srt) + 1))
    # monotonic, non-overlapping, all within new_duration (8 - 2 = 6.0)
    prev_end = -1.0
    for _, s, e, _txt in srt:
        ss, ee = _sec(s), _sec(e)
        assert ss >= prev_end - 1e-6 and ee > ss and ee <= 6.0 + 1e-6
        prev_end = ee
    joined = " ".join(b[3] for b in srt)
    assert "вырезано" not in joined            # word inside the cut is gone
    assert "б***ь" in joined and "блядь" not in joined   # profanity masked in sidecar
    # optional outputs actually written + flagged in the result dict
    assert Path(res["vtt"]).exists()
    assert Path(res["vtt"]).read_text(encoding="utf-8").startswith("WEBVTT")
    assert Path(res["transcript"]).exists()
    assert "вырезано" not in Path(res["transcript"]).read_text(encoding="utf-8")


def test_generate_no_optional_outputs(tmp_path):
    matcher = ProfanityMatcher(ProfanityLists())
    res = generate(_tr(), [], SubsCfg(write_vtt=False, write_transcript=False),
                   MaskingCfg(), matcher, tmp_path / "clip")
    assert "vtt" not in res and "transcript" not in res
    assert Path(res["srt"]).exists()
