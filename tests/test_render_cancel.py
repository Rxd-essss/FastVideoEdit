"""Wave-1: the first «Отмена» press aborts render() at the next pass boundary.

render() takes an optional should_cancel callback and raises RenderCancelled at
each pass boundary (measure / DFN / inter-pass encode) instead of letting a
killed-ffmpeg error be swallowed while the full encode still launches. No real
ffmpeg: a FakeFF records descs and can flip the cancel flag / raise mid-pass.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vpipe.render as render_mod                                    # noqa: E402
from vpipe.config import Config                                      # noqa: E402
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, TYPE_PAUSE,  # noqa: E402
                          TYPE_PROFANITY, CutList, CutSegment)
from vpipe.probe import MediaInfo                                    # noqa: E402
from vpipe.render import render                                      # noqa: E402


class FakeFF:
    """Records every run() desc; can set the cancel flag / raise on a given
    pass to simulate an Отмена that kills the live ffmpeg mid-pass."""

    def __init__(self, *, cancel_on=None, raise_on=None):
        self.descs: list[str] = []
        self.cancel_now = False
        self.cancel_on = cancel_on
        self.raise_on = raise_on

    def has_filter(self, name):
        return True

    def has_encoder(self, name):
        return False

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.descs.append(desc)
        if self.cancel_on and desc == self.cancel_on:
            self.cancel_now = True
        if self.raise_on and desc == self.raise_on:
            raise RuntimeError("ffmpeg killed by cancel")
        if args and args[-1] != "-":
            try:
                Path(args[-1]).write_bytes(b"")
            except OSError:
                pass
        return ""

    def probe(self, path):
        return {"format": {"duration": "0"}}


def _media():
    return MediaInfo(path="in.mp4", duration=10.0, fps=30.0, width=1920,
                     height=1080, vcodec="h264", acodec="aac",
                     has_audio=True, sample_rate=48000)


def _cutlist(*, cut=False, censor=False):
    segs = []
    if cut:
        segs.append(CutSegment(id="c", start=2.0, end=3.0, type=TYPE_PAUSE,
                               action=ACTION_REMOVE, enabled=True))
    if censor:
        segs.append(CutSegment(id="p", start=5.0, end=5.5, type=TYPE_PROFANITY,
                               action=ACTION_CENSOR, enabled=True))
    return CutList(source="in.mp4", duration=10.0, segments=segs)


def _loud2pass_cfg() -> Config:
    cfg = Config()
    cfg.render.denoise.loudnorm = True
    cfg.render.denoise.loudnorm_mode = "2pass"
    return cfg


def test_cancel_during_measure_aborts(tmp_path):
    # A cancel landing during the 2-pass loudnorm measurement kills that ffmpeg;
    # its error must NOT be swallowed into a one-pass fallback that then launches
    # the full encode — render() re-raises RenderCancelled instead.
    cfg = _loud2pass_cfg()
    ff = FakeFF(cancel_on="loudnorm measure", raise_on="loudnorm measure")
    out = str(tmp_path / "out.mp4")
    with pytest.raises(render_mod.RenderCancelled):
        render(ff, _media(), _cutlist(cut=True), cfg, out, str(tmp_path),
               log=lambda *a, **k: None,
               should_cancel=lambda: ff.cancel_now)
    assert "loudnorm measure" in ff.descs
    # the multi-hour encode never launched
    assert "render" not in ff.descs and "remux" not in ff.descs


def test_cancel_between_passes(tmp_path):
    # A cancel landing after the censor pass (between passes) must stop the
    # pipeline at the encode boundary — the encode is never launched.
    cfg = Config()
    ff = FakeFF(cancel_on="censor audio")
    out = str(tmp_path / "out.mp4")
    with pytest.raises(render_mod.RenderCancelled):
        render(ff, _media(), _cutlist(cut=True, censor=True), cfg, out,
               str(tmp_path), log=lambda *a, **k: None,
               should_cancel=lambda: ff.cancel_now)
    assert "censor audio" in ff.descs
    assert "render" not in ff.descs and "remux" not in ff.descs


def test_no_should_cancel_is_byte_for_byte(tmp_path):
    # Backward compat: without should_cancel render() never checks a flag and
    # runs the encode to completion exactly as before.
    cfg = Config()
    ff = FakeFF()
    out = str(tmp_path / "out.mp4")
    info = render(ff, _media(), _cutlist(cut=True), cfg, out, str(tmp_path),
                  log=lambda *a, **k: None)
    assert "render" in ff.descs
    assert info["out"] == out
