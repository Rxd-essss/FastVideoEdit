"""Wave-1: the DeepFilterNet CLI stage is cancellable + shows a heartbeat.

When a cancel flag is threaded through, ``enhance_audio`` runs deep-filter.exe
via a REGISTERED ``Popen`` (so ``cancel_all`` can reach it), polled on a short
timeout so it honours ``should_cancel`` at a real boundary and nudges the
progress bar (the CLI emits no fraction of its own). Without a cancel flag it
keeps the legacy blocking ``subprocess.run`` path (covered by test_deepfilter).

No real ffmpeg / deep-filter / GPU: ``subprocess.Popen`` and ``time.monotonic``
are faked.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vpipe.render as render_mod                       # noqa: E402
from vpipe.config import Config                         # noqa: E402
from vpipe.render import enhance_audio                  # noqa: E402


def _cfg(tmp_path, **extra) -> Config:
    cfg = Config()
    exe = tmp_path / "fake-deep-filter.exe"
    exe.write_bytes(b"MZ")
    d = cfg.render.denoise
    d.enabled = True
    d.engine = "deepfilter"
    d.deepfilter_bin = str(exe)
    for k, v in extra.items():
        setattr(d, k, v)
    return cfg


class _ExtractFF:
    """ff stub: the wav-extraction ff.run() touches the requested output path."""

    def has_filter(self, name):
        return True

    def has_encoder(self, name):
        return False

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        if args:
            try:
                Path(args[-1]).write_bytes(b"")
            except OSError:
                pass
        return ""


class _FakePopen:
    """A deep-filter.exe stand-in whose ``communicate`` times out until it is
    either terminated or has ticked ``ticks_before_done`` times."""

    def __init__(self, *, ticks_before_done=10 ** 9, rc=0):
        self.ticks = 0
        self.ticks_before_done = ticks_before_done
        self._rc = rc
        self.returncode = None
        self.terminated = False
        self.killed = False

    def communicate(self, timeout=None):
        if self.terminated or self.ticks >= self.ticks_before_done:
            self.returncode = -15 if self.terminated else self._rc
            return ("out", "err")
        self.ticks += 1
        raise render_mod.subprocess.TimeoutExpired(cmd="deep-filter",
                                                   timeout=timeout)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def poll(self):
        return self.returncode


def _fake_clock():
    box = {"v": 0.0}

    def clock():
        box["v"] += 1.0
        return box["v"]

    return clock


def test_dfn_cancellable(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake = _FakePopen()                        # never finishes on its own
    monkeypatch.setattr(render_mod.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(render_mod.time, "monotonic", _fake_clock())
    checks = {"n": 0}

    def should_cancel():
        checks["n"] += 1
        return checks["n"] >= 1                # cancel on the first heartbeat

    with pytest.raises(render_mod.RenderCancelled):
        enhance_audio(_ExtractFF(), "in.mp4", cfg, str(tmp_path),
                      log=lambda *a: None, total=10.0,
                      should_cancel=should_cancel)
    assert fake.terminated                     # the CLI process was signalled
    # the partial DFN scratch wav must be cleaned on cancel (audit C-1)
    assert not (tmp_path / "dfn_in.wav").exists()


def test_dfn_heartbeat_ticks(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake = _FakePopen(ticks_before_done=3)     # 3 heartbeats then completes
    monkeypatch.setattr(render_mod.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(render_mod.time, "monotonic", _fake_clock())
    seen: list[float] = []
    enhance_audio(_ExtractFF(), "in.mp4", cfg, str(tmp_path),
                  log=lambda *a: None, total=10.0,
                  on_progress=seen.append,
                  should_cancel=lambda: False)
    assert seen                                # the bar actually moved
    assert all(0.0 < v < 1.0 for v in seen)    # bounded, never claims «done»
    assert seen == sorted(seen)                # non-decreasing
    assert len(set(seen)) == len(seen)         # strictly increasing


def test_dfn_no_cancel_flag_uses_blocking_run(tmp_path, monkeypatch):
    # Backward compat: without should_cancel the legacy subprocess.run path is
    # taken (Popen is never touched), so every existing deepfilter test holds.
    cfg = _cfg(tmp_path)

    def boom_popen(*a, **k):
        raise AssertionError("Popen must not be used without should_cancel")

    monkeypatch.setattr(render_mod.subprocess, "Popen", boom_popen)
    calls = {"n": 0}

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        calls["n"] += 1
        if "-o" in cmd:
            out_dir = Path(cmd[cmd.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / Path(cmd[-1]).name).write_bytes(b"RIFFok")
        return _R()

    monkeypatch.setattr(render_mod.subprocess, "run", fake_run)
    out = enhance_audio(_ExtractFF(), "in.mp4", cfg, str(tmp_path),
                        log=lambda *a: None)
    assert calls["n"] == 1
    assert out == str(tmp_path / "dfn_out" / "dfn_in.wav")
