"""Tests for vpipe.export_nle — EDL + FCPXML timeline export from kept segments."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from vpipe.cutlist import resolve
from vpipe.export_nle import (_fps_base, _rational, _secs_to_tc, write_edl,
                              write_fcpxml)
from vpipe.models import ACTION_REMOVE, CutList, CutSegment
from vpipe.probe import MediaInfo
from vpipe.timeline import Timeline


def _media(fps=25.0, w=1920, h=1080, dur=12.0, path="C:/clips/source.mp4",
           has_audio=True) -> MediaInfo:
    return MediaInfo(path=path, duration=dur, fps=fps, width=w, height=h,
                     vcodec="h264", acodec="aac" if has_audio else "",
                     has_audio=has_audio, sample_rate=48000 if has_audio else 0)


# --- timecode / rational helpers --------------------------------------------
def test_secs_to_tc_integer_fps():
    assert _secs_to_tc(0.0, 25) == "00:00:00:00"
    assert _secs_to_tc(5.0, 25) == "00:00:05:00"
    assert _secs_to_tc(1.0, 25) == "00:00:01:00"
    # 1 frame at 25 fps = 0.04s
    assert _secs_to_tc(0.04, 25) == "00:00:00:01"
    # minutes / hours roll over
    assert _secs_to_tc(61.0, 30) == "00:01:01:00"
    assert _secs_to_tc(3661.0, 30) == "01:01:01:00"


def test_secs_to_tc_rounds_not_truncates():
    # 5.0 * 30 can be 149.999… in float; must land on frame 150 -> 00:00:05:00
    assert _secs_to_tc(5.0, 30) == "00:00:05:00"
    # 30 fps non-drop: frame 29 then wraps to next second
    assert _secs_to_tc(29 / 30, 30) == "00:00:00:29"
    assert _secs_to_tc(30 / 30, 30) == "00:00:01:00"


def test_secs_to_tc_ntsc_uses_integer_timebase():
    # 29.97 NON-DROP counts 30 frames per nominal second
    assert _secs_to_tc(1.0, 29.97) == "00:00:01:00"
    assert _secs_to_tc(0.0, 29.97) == "00:00:00:00"


def test_fps_base():
    assert _fps_base(25) == (25, 1)
    assert _fps_base(30) == (30, 1)
    assert _fps_base(60) == (60, 1)
    assert _fps_base(29.97) == (30000, 1001)
    assert _fps_base(23.976) == (24000, 1001)
    assert _fps_base(59.94) == (60000, 1001)
    assert _fps_base(0) == (30, 1)        # garbage falls back, never /0
    assert _fps_base(-5) == (30, 1)


def test_rational_integer_fps():
    assert _rational(0.0, 25) == "0s"
    assert _rational(1.0, 25) == "1s"            # 25/25 reduces to whole second
    assert _rational(5.0, 30) == "5s"
    # 1 frame at 25 fps -> 1/25s
    assert _rational(0.04, 25) == "1/25s"
    # 2 frames at 25 -> 2/25s (reduced; gcd(2,25)=1)
    assert _rational(0.08, 25) == "2/25s"
    # half a second at 25 -> 1/2s (12.5 -> rounds to frame 13? no: 0.5*25=12.5 -> 12)
    assert _rational(0.48, 25) == "12/25s"


def test_rational_ntsc_fps():
    # 1 second at 29.97 -> frames=round(1*30000/1001)=30 -> ticks=30*1001=30030 / 30000
    # reduced: gcd(30030,30000)=30 -> 1001/1000s
    assert _rational(1.0, 29.97) == "1001/1000s"
    assert _rational(0.0, 29.97) == "0s"


# --- EDL ---------------------------------------------------------------------
def _edl_events(text: str) -> list[str]:
    return [ln for ln in text.splitlines()
            if ln[:3].isdigit() and "AX" in ln]


def test_write_edl_basic(tmp_path):
    kept = [(0.0, 5.0), (8.0, 12.0)]
    p = tmp_path / "out.edl"
    write_edl(p, kept, _media(fps=25.0), title="FastVideoEdit")
    text = p.read_text(encoding="utf-8")

    assert text.startswith("TITLE: FastVideoEdit\n")
    assert "FCM: NON-DROP FRAME" in text
    events = _edl_events(text)
    assert len(events) == len(kept)             # one event per kept segment

    # Event 1: src 0..5, rec 0..5
    assert events[0] == ("001  AX       V     C        "
                         "00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00")
    # Event 2: src 8..12, rec butts directly after event 1 (5..9)
    assert events[1] == ("002  AX       V     C        "
                         "00:00:08:00 00:00:12:00 00:00:05:00 00:00:09:00")
    # source clip name comment present
    assert text.count("* FROM CLIP NAME: source.mp4") == 2


def test_write_edl_rec_monotonic_and_continuous(tmp_path):
    kept = [(1.0, 3.0), (4.0, 4.5), (10.0, 13.0)]
    p = tmp_path / "m.edl"
    write_edl(p, kept, _media(fps=30.0))
    events = _edl_events(p.read_text(encoding="utf-8"))
    # parse rec_in / rec_out (last two timecodes on each line)
    prev_out = None
    for ev in events:
        parts = ev.split()
        rec_in, rec_out = parts[-2], parts[-1]
        if prev_out is not None:
            assert rec_in == prev_out          # continuous: this in == prev out
        assert rec_out > rec_in                 # monotone within the event
        prev_out = rec_out


def test_write_edl_empty_kept(tmp_path):
    p = tmp_path / "e.edl"
    write_edl(p, [], _media(), title="Empty")
    text = p.read_text(encoding="utf-8")
    assert "TITLE: Empty" in text
    assert _edl_events(text) == []              # header only, no events


def test_write_edl_skips_zero_width(tmp_path):
    p = tmp_path / "z.edl"
    write_edl(p, [(0.0, 5.0), (5.0, 5.0), (8.0, 9.0)], _media(fps=25.0))
    events = _edl_events(p.read_text(encoding="utf-8"))
    assert len(events) == 2                      # zero-width segment dropped
    assert events[0].startswith("001")
    assert events[1].startswith("002")          # numbering stays contiguous


# --- FCPXML ------------------------------------------------------------------
def test_write_fcpxml_valid_xml(tmp_path):
    kept = [(0.0, 5.0), (8.0, 12.0)]
    p = tmp_path / "out.fcpxml"
    write_fcpxml(p, kept, _media(fps=25.0, w=1920, h=1080), title="MyProj")
    tree = ET.parse(p)                           # raises if invalid XML
    root = tree.getroot()
    assert root.tag == "fcpxml"
    assert root.get("version") == "1.11"

    fmt = root.find("./resources/format")
    assert fmt is not None
    assert fmt.get("width") == "1920"
    assert fmt.get("height") == "1080"
    assert fmt.get("frameDuration") == "1/25s"   # one frame at 25 fps

    asset = root.find("./resources/asset")
    assert asset is not None
    assert asset.get("duration") == "12s"        # media duration
    rep = asset.find("media-rep")
    assert rep is not None
    src = rep.get("src", "")
    # Canonical three-slash file URL; never the non-portable file://// form.
    assert src.startswith("file:///")
    assert not src.startswith("file:////")
    assert "\\" not in src                        # forward slashes only


def test_write_fcpxml_clip_count_and_timing(tmp_path):
    kept = [(0.0, 5.0), (8.0, 12.0)]
    p = tmp_path / "c.fcpxml"
    write_fcpxml(p, kept, _media(fps=25.0))
    root = ET.parse(p).getroot()
    clips = root.findall("./library/event/project/sequence/spine/asset-clip")
    assert len(clips) == len(kept)               # one clip per kept segment

    # clip 1: offset 0, start 0, dur 5
    assert clips[0].get("offset") == "0s"
    assert clips[0].get("start") == "0s"
    assert clips[0].get("duration") == "5s"
    # clip 2: offset = end of clip 1 (5s), start = source 8s, dur 4s
    assert clips[1].get("offset") == "5s"
    assert clips[1].get("start") == "8s"
    assert clips[1].get("duration") == "4s"

    # sequence duration = total kept = 5 + 4 = 9s
    seq = root.find("./library/event/project/sequence")
    assert seq.get("duration") == "9s"


def test_write_fcpxml_empty_kept(tmp_path):
    p = tmp_path / "e.fcpxml"
    write_fcpxml(p, [], _media(), title="Empty")
    root = ET.parse(p).getroot()                 # still valid XML
    clips = root.findall("./library/event/project/sequence/spine/asset-clip")
    assert clips == []
    seq = root.find("./library/event/project/sequence")
    assert seq.get("duration") == "0s"


def test_write_fcpxml_no_audio_flags(tmp_path):
    p = tmp_path / "na.fcpxml"
    write_fcpxml(p, [(0.0, 2.0)], _media(has_audio=False))
    asset = ET.parse(p).getroot().find("./resources/asset")
    assert asset.get("hasAudio") == "0"
    assert asset.get("audioSources") == "0"


def test_write_fcpxml_ntsc_frameduration(tmp_path):
    p = tmp_path / "ntsc.fcpxml"
    write_fcpxml(p, [(0.0, 2.0)], _media(fps=29.97))
    fmt = ET.parse(p).getroot().find("./resources/format")
    # one frame at 29.97 = 1001/30000s
    assert fmt.get("frameDuration") == "1001/30000s"


# --- integration with the real cutlist -> resolve -> Timeline path -----------
def test_export_from_cutlist_pipeline(tmp_path):
    """Exercise the exact chain serve.py uses: resolve -> Timeline -> kept."""
    cl = CutList(source="source.mp4", duration=12.0, segments=[
        CutSegment(id="c1", start=5.0, end=8.0, type="pause",
                   action=ACTION_REMOVE, enabled=True),
    ])
    removed, _censors = resolve(cl)
    kept = Timeline(removed, cl.duration).kept_segments()
    assert kept == [(0.0, 5.0), (8.0, 12.0)]

    edl = tmp_path / "p.edl"
    fcp = tmp_path / "p.fcpxml"
    write_edl(edl, kept, _media(fps=25.0))
    write_fcpxml(fcp, kept, _media(fps=25.0))

    assert len(_edl_events(edl.read_text(encoding="utf-8"))) == 2
    clips = ET.parse(fcp).getroot().findall(
        "./library/event/project/sequence/spine/asset-clip")
    assert len(clips) == 2


# --- frame accuracy on FRACTIONAL boundaries (P1-1 regression) ---------------
def _tc_to_frames(tc: str, fps_i: int) -> int:
    hh, mm, ss, ff = (int(x) for x in tc.split(":"))
    return ((hh * 60 + mm) * 60 + ss) * fps_i + ff


def _rat_to_frames(s: str, fps) -> int:
    from vpipe.export_nle import _fps_base
    num, den = _fps_base(fps)
    s = s.rstrip("s")
    val = (int(s.split("/")[0]) / int(s.split("/")[1])) if "/" in s else float(s)
    return round(val * num / den)


@pytest.mark.parametrize("fps", [30.0, 25.0, 29.97, 60.0, 23.976])
def test_edl_source_equals_record_on_fractional_boundaries(tmp_path, fps):
    """Each EDL event's source length must equal its record length, and events
    must butt — on millisecond-grid cut points, not just whole seconds."""
    import re
    from vpipe.export_nle import _fps_int
    kept = [(0.0, 1.017), (2.033, 3.55), (4.083, 5.967), (7.1, 8.233)]
    edl = tmp_path / "p.edl"
    write_edl(edl, kept, _media(fps=fps, dur=10.0))
    fps_i = _fps_int(fps)
    rows = re.findall(r"\d{3}\s+AX\s+V\s+C\s+(\S+) (\S+) (\S+) (\S+)",
                      edl.read_text(encoding="utf-8"))
    assert rows, "no events written"
    prev_rec_out = 0
    for si, so, ri, ro in rows:
        src = _tc_to_frames(so, fps_i) - _tc_to_frames(si, fps_i)
        rec = _tc_to_frames(ro, fps_i) - _tc_to_frames(ri, fps_i)
        assert src == rec, f"source {src}f != record {rec}f"
        assert _tc_to_frames(ri, fps_i) == prev_rec_out, "record not butted"
        prev_rec_out = _tc_to_frames(ro, fps_i)


@pytest.mark.parametrize("fps", [30.0, 25.0, 29.97, 60.0, 23.976])
def test_fcpxml_spine_gapless_on_fractional_boundaries(tmp_path, fps):
    """FCPXML clip offsets must equal the running sum of prior clip durations,
    and the sequence duration must equal the total — no sub-frame drift."""
    kept = [(0.0, 1.017), (2.033, 3.55), (4.083, 5.967), (7.1, 8.233)]
    fcp = tmp_path / "p.fcpxml"
    write_fcpxml(fcp, kept, _media(fps=fps, dur=10.0))
    root = ET.parse(fcp).getroot()
    seq = root.find("./library/event/project/sequence")
    clips = seq.findall("./spine/asset-clip")
    assert clips
    cum = 0
    for c in clips:
        off = _rat_to_frames(c.get("offset"), fps)
        assert off == cum, f"offset {off}f != cumulative {cum}f (spine gap)"
        cum += _rat_to_frames(c.get("duration"), fps)
    assert _rat_to_frames(seq.get("duration"), fps) == cum, "seq duration != sum"
