"""probe_media duration clamping (probe-perstream-duration, Wave 1).

Container format.duration is the MAX of all stream end-times, so a stream that
ends early (OBS audio outliving video, damaged/cover-art recordings) is
invisible and lets a kept segment run past that stream's EOF -- concat then
silently truncates/desyncs the tail. probe_media now clamps duration to the
SHORTEST real stream, but ONLY when both video and audio streams report a
positive duration and they differ by more than a ~1-frame deadband, so ordinary
footage (equal/absent per-stream durations, frame-quantization noise) is
unaffected byte-for-byte.

These tests exercise probe_media directly with a fake ffmpeg whose .probe()
returns hand-built ffprobe JSON -- no real ffmpeg/GPU/network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.probe import probe_media  # noqa: E402


class FakeFF:
    """Minimal ffmpeg double: probe() returns a canned ffprobe-JSON dict."""

    def __init__(self, info):
        self._info = info

    def probe(self, path):
        return self._info


def _info(fmt_dur, *, v_dur=None, a_dur=True, a_dur_val=None):
    """Build ffprobe-style JSON.

    ``v_dur``     -> video-stream 'duration' (None omits the key).
    ``a_dur``     -> include an audio stream at all (False = video only).
    ``a_dur_val`` -> audio-stream 'duration' (None omits the key).
    """
    vstream = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30/1",
    }
    if v_dur is not None:
        vstream["duration"] = str(v_dur)
    streams = [vstream]
    if a_dur:
        astream = {
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
        }
        if a_dur_val is not None:
            astream["duration"] = str(a_dur_val)
        streams.append(astream)
    return {"format": {"duration": str(fmt_dur)}, "streams": streams}


def test_duration_clamped_to_shortest_stream():
    # format says 10s, but the audio stream ends at 8s -> clamp to 8s so no
    # kept segment can run past the audio EOF and desync the render.
    ff = FakeFF(_info(10.0, v_dur=10.0, a_dur_val=8.0))
    mi = probe_media(ff, "x.mp4")
    assert mi.duration == 8.0


def test_duration_clamped_when_video_is_shortest():
    # Symmetric case: video is the short stream.
    ff = FakeFF(_info(12.0, v_dur=9.0, a_dur_val=12.0))
    mi = probe_media(ff, "x.mp4")
    assert mi.duration == 9.0


def test_duration_not_clamped_when_stream_duration_absent():
    # Audio stream omits 'duration' (common in some containers) -> the guard
    # must keep the format-level duration EXACTLY, never clamp on missing data.
    ff = FakeFF(_info(10.0, v_dur=10.0, a_dur_val=None))
    mi = probe_media(ff, "x.mp4")
    assert mi.duration == 10.0


def test_duration_not_clamped_without_audio_stream():
    # Video-only file: no second stream to compare against -> unchanged.
    ff = FakeFF(_info(10.0, v_dur=10.0, a_dur=False))
    mi = probe_media(ff, "x.mp4")
    assert mi.duration == 10.0


def test_duration_frame_deadband():
    # v=10.00, a=9.98: 0.02s gap is frame-quantization noise (< 0.05 deadband)
    # -> keep the format duration, do not clamp.
    ff = FakeFF(_info(10.0, v_dur=10.0, a_dur_val=9.98))
    mi = probe_media(ff, "x.mp4")
    assert mi.duration == 10.0


def test_duration_equal_streams_unchanged():
    # Both streams report the same length -> byte-for-byte identical result.
    ff = FakeFF(_info(10.0, v_dur=10.0, a_dur_val=10.0))
    mi = probe_media(ff, "x.mp4")
    assert mi.duration == 10.0


# --- rotation metadata (probe-rotation-swap, Wave 2) ---------------------
# ffprobe reports PRE-rotation coded width/height; ffmpeg autorotates frames at
# decode. probe_media reads the Display Matrix side-data (or legacy tags.rotate)
# and swaps width/height for a +/-90/270 rotation so downstream sees DISPLAY
# dims. 0/180 rotations keep the axes and stay byte-for-byte unchanged.


def _rot_info(*, side_data=None, tags=None):
    """Single video stream, coded 1920x1080, no audio; optional rotation meta."""
    vstream = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30/1",
    }
    if side_data is not None:
        vstream["side_data_list"] = side_data
    if tags is not None:
        vstream["tags"] = tags
    return {"format": {"duration": "10.0"}, "streams": [vstream]}


def test_rotation_90_swaps_dims():
    # Display Matrix rotation=-90 (clockwise-90 display) -> every frame 1080x1920.
    ff = FakeFF(_rot_info(side_data=[
        {"side_data_type": "Display Matrix", "rotation": -90}]))
    mi = probe_media(ff, "portrait.mp4")
    assert (mi.width, mi.height) == (1080, 1920)


def test_rotation_270_swaps_dims():
    # Positive 270 also swaps the axes (sign is irrelevant to a W/H swap).
    ff = FakeFF(_rot_info(side_data=[
        {"side_data_type": "Display Matrix", "rotation": 270}]))
    mi = probe_media(ff, "portrait.mp4")
    assert (mi.width, mi.height) == (1080, 1920)


def test_rotation_180_no_swap():
    # A 180 flip keeps the same axes -> dims unchanged.
    ff = FakeFF(_rot_info(side_data=[
        {"side_data_type": "Display Matrix", "rotation": 180}]))
    mi = probe_media(ff, "flip.mp4")
    assert (mi.width, mi.height) == (1920, 1080)


def test_legacy_tags_rotate_swaps_dims():
    # Older muxes expose rotation via stream tags.rotate, not side-data.
    ff = FakeFF(_rot_info(tags={"rotate": "90"}))
    mi = probe_media(ff, "old.mp4")
    assert (mi.width, mi.height) == (1080, 1920)


def test_no_rotation_landscape_unchanged():
    # Ordinary landscape file, no rotation metadata -> byte-for-byte unchanged.
    ff = FakeFF(_rot_info())
    mi = probe_media(ff, "land.mp4")
    assert (mi.width, mi.height) == (1920, 1080)
