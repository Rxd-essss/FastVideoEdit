# -*- coding: utf-8 -*-
"""Wave 4 (#73): FFmpegError must not embed the entire (up to ~30KB) filter
graph — cap it to a head+tail excerpt so the real ffmpeg error line stays at the
top of a readable toast. Short graphs pass through verbatim (no regression).
Fakes only — no real ffmpeg."""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe import ffmpeg_utils                                     # noqa: E402


class _FailProc:
    """Fake ffmpeg that exits non-zero with a one-line stderr diagnostic."""

    def __init__(self, cmd, *a, **k):
        self.stdout = io.StringIO("progress=end\n")
        self.stderr = io.StringIO("Error reinitializing filters!\n")
        self.returncode = 1

    def wait(self):
        return 1

    def poll(self):
        return self.returncode


def _bare_ff(monkeypatch):
    ff = ffmpeg_utils.FFmpeg.__new__(ffmpeg_utils.FFmpeg)
    ff.ffmpeg, ff.ffprobe, ff._caps = "ffmpeg", "ffprobe", {}
    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", _FailProc)
    return ff


def test_ffmpegerror_caps_graph(monkeypatch):
    # FAILS before the fix: the full 6000-char graph is appended verbatim, so
    # len(msg) > 6000 and there is no omitted-chars marker.
    ff = _bare_ff(monkeypatch)
    graph = "acrossfade=" + "x" * 6000        # >5000-char -filter_complex
    with pytest.raises(ffmpeg_utils.FFmpegError) as ei:
        ff.run(["-i", "in.mp4", "-filter_complex", graph, "out.mp4"])
    msg = str(ei.value)
    # 1) bounded — not the full wall
    assert len(msg) < 3000
    assert len(msg) < len(graph)
    # 2) truncation marker present
    assert "chars omitted" in msg
    # 3) the real ffmpeg error line comes BEFORE the graph excerpt
    assert "Error reinitializing filters!" in msg
    assert (msg.index("Error reinitializing filters!")
            < msg.index("filter_complex graph:"))


def test_ffmpegerror_short_graph_verbatim(monkeypatch):
    # A short graph (<=1200 chars) is included byte-for-byte — no regression for
    # small-render diagnostics.
    ff = _bare_ff(monkeypatch)
    graph = "scale=1280:-2,format=yuv420p"
    with pytest.raises(ffmpeg_utils.FFmpegError) as ei:
        ff.run(["-i", "in.mp4", "-filter_complex", graph, "out.mp4"])
    msg = str(ei.value)
    assert graph in msg
    assert "chars omitted" not in msg
