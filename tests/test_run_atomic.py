"""R2 output-validation: _run_atomic must reject a truncated encode (ffmpeg
exits 0 but writes a short file) instead of atomically promoting it to the final
name and reporting success. Tolerance is generous so healthy renders never trip
it, and a probe failure must never fail a good render.
(Audit 2026-07: vpipe/render.py _run_atomic — 'render silently stops partway'.)"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest                                                      # noqa: E402

from vpipe.ffmpeg_utils import FFmpegError                         # noqa: E402
from vpipe.render import _run_atomic                               # noqa: E402


class FakeFF:
    """Writes the temp output on run(); probe() returns a configurable duration
    (or raises when probe_dur is None, to simulate a broken ffprobe)."""

    def __init__(self, probe_dur, write=True):
        self._d = probe_dur
        self._write = write
        self.ran = False

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.ran = True
        if self._write:
            Path(args[-1]).write_bytes(b"\0" * 32)   # args[-1] == the .part temp

    def probe(self, path):
        if self._d is None:
            raise RuntimeError("ffprobe boom")
        return {"format": {"duration": self._d}}


def test_run_atomic_rejects_truncated_output(tmp_path):
    out = str(tmp_path / "o.mp4")
    ff = FakeFF(probe_dur=5.0)               # got 5s of an expected 60s
    with pytest.raises(FFmpegError) as e:
        _run_atomic(ff, ["-i", "x", out], out, total=60.0, desc="render")
    assert "truncated" in str(e.value)
    assert not os.path.exists(out)           # final never created
    assert not os.path.exists(out + ".part")  # .part removed


def test_run_atomic_accepts_within_tolerance(tmp_path):
    out = str(tmp_path / "o.mp4")
    ff = FakeFF(probe_dur=59.7)              # a frame short of 60s — fine
    _run_atomic(ff, ["-i", "x", out], out, total=60.0)
    assert os.path.exists(out)
    assert not os.path.exists(out + ".part")


def test_run_atomic_probe_failure_keeps_output(tmp_path):
    out = str(tmp_path / "o.mp4")
    ff = FakeFF(probe_dur=None)             # probe raises → swallowed → no-op
    _run_atomic(ff, ["-i", "x", out], out, total=60.0)
    assert os.path.exists(out)


def test_run_atomic_no_total_skips_guard(tmp_path):
    out = str(tmp_path / "o.mp4")
    ff = FakeFF(probe_dur=1.0)              # would look truncated, but total=None
    _run_atomic(ff, ["-i", "x", out], out, total=None)
    assert os.path.exists(out)


def test_run_atomic_zero_probe_does_not_reject(tmp_path):
    # A probe that returns 0.0 (unknown) must NOT be treated as truncation.
    out = str(tmp_path / "o.mp4")
    ff = FakeFF(probe_dur=0.0)
    _run_atomic(ff, ["-i", "x", out], out, total=60.0)
    assert os.path.exists(out)                # (replace-retry tests follow)


# --- #72: os.replace retry + timestamped fallback on a locked target ----------
def test_run_atomic_replace_happy(tmp_path):
    # Target not locked: os.replace succeeds on the first try, returns out_path.
    out = str(tmp_path / "o.mp4")
    saved = _run_atomic(FakeFF(probe_dur=60.0), ["-i", "x", out], out, total=60.0)
    assert saved == out
    assert os.path.exists(out) and not os.path.exists(out + ".part")


def test_run_atomic_replace_retry_then_succeeds(tmp_path, monkeypatch):
    import time
    out = str(tmp_path / "o.mp4")
    real, n = os.replace, {"c": 0}

    def flaky(src, dst):
        n["c"] += 1
        if n["c"] < 3:                        # locked for the first two attempts
            raise PermissionError("locked")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)
    saved = _run_atomic(FakeFF(probe_dur=60.0), ["-i", "x", out], out, total=60.0)
    assert saved == out                       # recovered before the fallback
    assert n["c"] == 3 and os.path.exists(out)


def test_run_atomic_replace_fallback(tmp_path, monkeypatch):
    import time
    out = str(tmp_path / "o.mp4")
    real, n = os.replace, {"c": 0}

    def flaky(src, dst):
        if dst == out:                        # real target stays locked forever
            n["c"] += 1
            raise PermissionError("locked")
        return real(src, dst)                 # the timestamped sibling is free

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)
    saved = _run_atomic(FakeFF(probe_dur=60.0), ["-i", "x", out], out, total=60.0)
    assert saved != out and saved.endswith(".mp4")
    assert os.path.exists(saved)              # finished encode kept under alt name
    assert n["c"] == 5                        # 5 locked attempts, then fallback
    assert not os.path.exists(out + ".part")  # temp consumed by the fallback
    assert not os.path.exists(out)            # original target never written
