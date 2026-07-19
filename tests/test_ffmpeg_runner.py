"""R2 crash fix: an oversized -filter_complex must be spilled to a temp file and
passed via -filter_complex_script so it never hits the Windows 32k argv limit.
Small graphs must pass through byte-for-byte and the temp file must always be
cleaned up. (Audit 2026-07: vpipe/ffmpeg_utils.py — long-video render crash.)"""
import io
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

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


# --- issue 18: no-progress watchdog (real subprocess) -------------------------
# A stalled ffmpeg used to block the worker thread (and every mutating endpoint)
# forever. _run_proc now runs a daemon watchdog that terminates a process which
# emits nothing for ``stall_timeout`` seconds and raises a clear FFmpegError.
def _write_script(body: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py", prefix="fve_wd_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def _watchdog_ff(stall_timeout):
    ff = ffmpeg_utils.FFmpeg.__new__(ffmpeg_utils.FFmpeg)
    ff.ffmpeg, ff.ffprobe, ff._caps = "ffmpeg", "ffprobe", {}
    ff.stall_timeout = stall_timeout
    return ff


# Emits two progress lines immediately, then hangs (never exits) — a stall.
_STALL_SCRIPT = (
    "import sys, time\n"
    "for i in range(2):\n"
    "    sys.stdout.write('out_time_us=%d\\n' % (i * 1000000))\n"
    "    sys.stdout.flush()\n"
    "time.sleep(15)\n"
)

# Steady sub-window progress stream that exits cleanly — a healthy encode.
_HEALTHY_SCRIPT = (
    "import sys, time\n"
    "for i in range(8):\n"
    "    sys.stdout.write('out_time_us=%d\\n' % (i * 1000000))\n"
    "    sys.stdout.flush()\n"
    "    time.sleep(0.2)\n"
    "sys.stdout.write('progress=end\\n')\n"
    "sys.stdout.flush()\n"
)


def test_stall_watchdog_kills_hung_ffmpeg():
    # FAILS before the fix: with no watchdog _run_proc waits out the 15s sleep,
    # the process exits 0 and NO FFmpegError is raised (pytest.raises fails).
    ff = _watchdog_ff(2.0)
    script = _write_script(_STALL_SCRIPT)
    try:
        start = time.monotonic()
        with pytest.raises(ffmpeg_utils.FFmpegError) as ei:
            ff._run_proc([sys.executable, script], [], 100.0, [].append, "render")
        elapsed = time.monotonic() - start
        assert "нет прогресса" in str(ei.value)
        # killed by the watchdog (~3s), NOT after the child's 15s sleep
        assert elapsed < 10
    finally:
        os.remove(script)


def test_stall_watchdog_leaves_healthy_encode_alone():
    # A steady progress stream must reset the timer and never trip the watchdog,
    # even with a tight window — no false kill, callbacks fire, exit is clean.
    ff = _watchdog_ff(3.0)
    script = _write_script(_HEALTHY_SCRIPT)
    try:
        seen = []
        ff._run_proc([sys.executable, script], [], 100.0, seen.append, "render")
        assert seen                       # progress callbacks fired
        assert seen[-1] == 1.0            # progress=end -> 1.0
    finally:
        os.remove(script)


# Finished encode: progress=end, then a long SILENT faststart moov relocation.
_TRAILER_SCRIPT = (
    "import sys, time\n"
    "sys.stdout.write('out_time_us=1000000\\n')\n"
    "sys.stdout.write('progress=end\\n')\n"
    "sys.stdout.flush()\n"
    "time.sleep(4)\n"
)


def test_stall_watchdog_disarmed_after_progress_end():
    # W6 #9: после progress=end ffmpeg может минутами МОЛЧА двигать moov-атом
    # (faststart). FAILS before the fix: окно 1.5 c истекало в тихом 4-секундном
    # трейлере, сторож убивал ГОТОВЫЙ энкод и _run_proc бросал stall-FFmpegError
    # (выше по стеку _run_atomic удалил бы .part на ~100%).
    ff = _watchdog_ff(1.5)
    script = _write_script(_TRAILER_SCRIPT)
    try:
        seen = []
        ff._run_proc([sys.executable, script], [], 100.0, seen.append, "render")
        assert seen[-1] == 1.0            # завершился чисто, без stall-ошибки
    finally:
        os.remove(script)


# --- issue 18: cancel_all + progress parsing (fakes, no real ffmpeg) ----------
def test_cancel_all_terminates_running_only():
    class _P:
        def __init__(self, alive):
            self.terminated = False
            self._alive = alive

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self.terminated = True
            self._alive = False

    alive = _P(True)
    already_done = _P(False)
    ffmpeg_utils._register_proc(alive)
    ffmpeg_utils._register_proc(already_done)
    try:
        n = ffmpeg_utils.cancel_all()
        assert n >= 1                      # the live proc was signalled
        assert alive.terminated is True
        assert already_done.terminated is False   # already exited, left alone
    finally:
        ffmpeg_utils._unregister_proc(alive)
        ffmpeg_utils._unregister_proc(already_done)


def test_run_parses_progress_fractions_and_ignores_garbage(monkeypatch):
    stream = ("out_time_us=0\n"
              "out_time_us=not_a_number\n"   # ValueError -> skipped
              "out_time_us=5000000\n"        # 5s / 10 -> 0.5
              "out_time_us=20000000\n"       # 20s / 10 -> clamps to 1.0
              "progress=end\n")             # -> 1.0

    class _PP:
        def __init__(self, cmd, *a, **k):
            self.stdout = io.StringIO(stream)
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self):
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", _PP)
    ff = ffmpeg_utils.FFmpeg.__new__(ffmpeg_utils.FFmpeg)
    ff.ffmpeg, ff.ffprobe, ff._caps = "ffmpeg", "ffprobe", {}
    seen = []
    ff.run(["-i", "in.mp4", "out.mp4"], total=10.0, on_progress=seen.append)
    assert seen == [0.0, 0.5, 1.0, 1.0]


