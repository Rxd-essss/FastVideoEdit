"""hdr-tonemap-to-sdr (#60): HDR/10-bit (PQ/HLG) input must be tonemapped to SDR
BT.709 instead of bit-truncated to a washed-out yuv420p; SDR stays byte-for-byte
untouched (every fast-path preserved). render()-graph tests with a FakeFF, the
same pattern as test_music_duck / test_edge_fade.
(Audit 2026-07: vpipe/render.py _is_hdr + vpre tonemap, vpipe/probe.py fields.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.config import Config                                    # noqa: E402
from vpipe.models import CutList                                   # noqa: E402
from vpipe.probe import MediaInfo                                  # noqa: E402
from vpipe.render import _is_hdr, render                           # noqa: E402

_SILENT = lambda *a, **k: None  # noqa: E731


class FakeFF:
    """Records ffmpeg invocations; has_filter is configurable so the tonemap
    guard (ff.has_filter('zscale')) can be exercised both ways."""

    def __init__(self, filters=("zscale", "tonemap")):
        self.runs: list[list[str]] = []
        self._filters = set(filters)

    def has_filter(self, name):
        return name in self._filters

    def has_encoder(self, name):          # force x264 (deterministic, no nvenc)
        return False

    def probe(self, path):
        return {"format": {"duration": "10.0"}}

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.runs.append(list(args))
        try:
            Path(args[-1]).write_bytes(b"")   # args[-1] == the .part temp
        except OSError:
            pass
        return ""


def _media(color_trc="", **over):
    d = dict(path="in.mp4", duration=10.0, fps=30.0, width=1920, height=1080,
             vcodec="hevc", acodec="aac", has_audio=True, sample_rate=48000,
             color_trc=color_trc)
    d.update(over)
    return MediaInfo(**d)


def _cl():
    return CutList(source="in.mp4", duration=10.0, segments=[])


def _graph(args):
    if "-filter_complex" in args:
        return args[args.index("-filter_complex") + 1]
    return ""


# --- _is_hdr unit -------------------------------------------------------------
def test_is_hdr_matches_pq_and_hlg_only():
    assert _is_hdr(_media(color_trc="smpte2084")) is True     # PQ / HDR10
    assert _is_hdr(_media(color_trc="arib-std-b67")) is True  # HLG
    assert _is_hdr(_media(color_trc="SMPTE2084")) is True     # case-insensitive
    assert _is_hdr(_media(color_trc="bt709")) is False
    assert _is_hdr(_media(color_trc="")) is False


def test_mediainfo_color_fields_default_empty():
    m = MediaInfo(path="x", duration=1.0, fps=30.0, width=1, height=1,
                  vcodec="h264", acodec="aac", has_audio=True, sample_rate=48000)
    assert m.color_trc == "" and m.color_primaries == "" and m.pix_fmt == ""


# --- render() graph -----------------------------------------------------------
def test_hdr_source_gets_tonemap(tmp_path):
    ff = FakeFF()
    info = render(ff, _media(color_trc="smpte2084"), _cl(), Config(),
                  str(tmp_path / "out.mp4"), str(tmp_path), log=_SILENT)
    g = _graph(ff.runs[-1])
    assert "zscale=t=linear" in g and "tonemap=hable" in g
    assert "format=yuv420p" in g
    # HDR forces re-encode: the video-copy fast-path is bypassed.
    assert info["encoder"] != "copy"
    args = ff.runs[-1]
    assert args[args.index("-c:v") + 1] != "copy"


def test_sdr_source_unchanged(tmp_path):
    ff = FakeFF()
    info = render(ff, _media(color_trc="bt709"), _cl(), Config(),
                  str(tmp_path / "out.mp4"), str(tmp_path), log=_SILENT)
    g = _graph(ff.runs[-1])
    assert "zscale" not in g and "tonemap" not in g
    # No cuts, no scale, SDR -> pure remux copy fast-path intact.
    assert info["encoder"] == "copy"
    assert "-filter_complex" not in ff.runs[-1]


def test_hdr_without_zscale_warns(tmp_path):
    logs: list[str] = []
    ff = FakeFF(filters=())        # no zscale/tonemap available in this build
    info = render(ff, _media(color_trc="arib-std-b67"), _cl(), Config(),
                  str(tmp_path / "out.mp4"), str(tmp_path),
                  log=lambda *a, **k: logs.append(" ".join(str(x) for x in a)))
    g = _graph(ff.runs[-1])
    assert "zscale" not in g               # no tonemap emitted
    # Degraded, not crashed: a warning was logged and a file was produced.
    assert any("zscale" in m for m in logs)
    assert info["out"]
