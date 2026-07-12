"""R2 crash fix: an oversized -filter_complex must be spilled to a temp file and
passed via -filter_complex_script so it never hits the Windows 32k argv limit.
Small graphs must pass through byte-for-byte and the temp file must always be
cleaned up. (Audit 2026-07: vpipe/ffmpeg_utils.py — long-video render crash.)"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe import ffmpeg_utils                                     # noqa: E402
from vpipe.ffmpeg_utils import _FILTERGRAPH_INLINE_MAX, _spill_filtergraph  # noqa: E402


# --- _spill_filtergraph unit --------------------------------------------------
def test_spill_large_graph_written_to_script():
    graph = "a" * (_FILTERGRAPH_INLINE_MAX + 1)
    cmd = ["ffmpeg", "-i", "x", "-filter_complex", graph, "out.mp4"]
    new, path = _spill_filtergraph(cmd)
    try:
        assert path is not None and os.path.exists(path)
        assert "-filter_complex" not in new
        assert "-filter_complex_script" in new
        assert new[new.index("-filter_complex_script") + 1] == path
        # graph round-trips exactly (no truncation / re-encoding)
        assert Path(path).read_text(encoding="utf-8") == graph
        # output path still trails the command
        assert new[-1] == "out.mp4"
    finally:
        if path and os.path.exists(path):
            os.remove(path)


def test_spill_small_graph_unchanged():
    cmd = ["ffmpeg", "-i", "x", "-filter_complex", "scale=1280:-2", "out.mp4"]
    new, path = _spill_filtergraph(cmd)
    assert path is None
    assert new is cmd                       # untouched, same object
    assert "-filter_complex" in new


def test_spill_no_filter_complex_noop():
    cmd = ["ffmpeg", "-i", "x", "-c:v", "copy", "out.mp4"]
    new, path = _spill_filtergraph(cmd)
    assert path is None and new is cmd


def test_spill_threshold_boundary_inclusive():
    # exactly the max stays inline; one over spills
    at = ["ffmpeg", "-filter_complex", "a" * _FILTERGRAPH_INLINE_MAX, "o"]
    _, p_at = _spill_filtergraph(at)
    assert p_at is None
    over = ["ffmpeg", "-filter_complex", "a" * (_FILTERGRAPH_INLINE_MAX + 1), "o"]
    _, p_over = _spill_filtergraph(over)
    try:
        assert p_over is not None
    finally:
        if p_over and os.path.exists(p_over):
            os.remove(p_over)


# --- run() integration: spill + guaranteed cleanup ----------------------------
class _FakeProc:
    def __init__(self, cmd, *a, **k):
        _FakeProc.last_cmd = list(cmd)
        self.stdout = io.StringIO("progress=end\n")
        self.stderr = io.StringIO("")
        self.returncode = 0

    def wait(self):
        return 0

    def poll(self):
        return self.returncode


def _bare_ff(monkeypatch):
    ff = ffmpeg_utils.FFmpeg.__new__(ffmpeg_utils.FFmpeg)
    ff.ffmpeg, ff.ffprobe, ff._caps = "ffmpeg", "ffprobe", {}
    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", _FakeProc)
    return ff


def test_run_spills_big_graph_and_removes_script(monkeypatch):
    ff = _bare_ff(monkeypatch)
    graph = "x" * (_FILTERGRAPH_INLINE_MAX + 50)
    ff.run(["-i", "in.mp4", "-filter_complex", graph, "out.mp4"])
    cmd = _FakeProc.last_cmd
    assert "-filter_complex_script" in cmd
    assert "-filter_complex" not in cmd
    script = cmd[cmd.index("-filter_complex_script") + 1]
    assert not os.path.exists(script)       # cleaned up in finally


def test_run_keeps_small_graph_inline(monkeypatch):
    ff = _bare_ff(monkeypatch)
    ff.run(["-i", "in.mp4", "-filter_complex", "scale=1280:-2", "out.mp4"])
    cmd = _FakeProc.last_cmd
    assert "-filter_complex" in cmd
    assert "-filter_complex_script" not in cmd
