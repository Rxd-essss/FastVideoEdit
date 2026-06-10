"""Audio mastering at render time: de-esser + YouTube loudness (-14 LUFS).

Additive layer on top of the denoise feature (tests/test_denoise.py):

  * ``DenoiseCfg`` gains ``deess``/``loudnorm`` (both OFF by default).
  * ``build_apost`` — the two mastering filters are INDEPENDENT of
    ``denoise.enabled`` and append AFTER the denoise trio; ``loudnorm`` is
    always followed by ``aresample=48000`` (loudnorm upsamples to 192 kHz
    internally). Everything off -> [] (byte-for-byte audio fast-path kept).
  * ``render()`` integration — loudnorm alone routes audio through
    filter_complex while video stays copied.
  * ``serve._resolve_render_opts`` — UI opts ``denoise_deess`` /
    ``denoise_loudnorm`` -> cfg, independent of the ``denoise`` flag.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.config import Config, DenoiseCfg, load_config             # noqa: E402
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE,                 # noqa: E402
                          CutList, CutSegment)
from vpipe.probe import MediaInfo                                    # noqa: E402
from vpipe.render import build_apost, render                         # noqa: E402

import serve                                                         # noqa: E402

DEESS = "deesser=i=0.4"
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"
ARESAMPLE = "aresample=48000"


# --- DenoiseCfg defaults -----------------------------------------------------
def test_mastering_defaults_off():
    d = DenoiseCfg()
    assert d.deess is False
    assert d.loudnorm is False


def test_config_has_mastering_fields():
    cfg = Config()
    assert cfg.render.denoise.deess is False
    assert cfg.render.denoise.loudnorm is False


# --- build_apost combinations ------------------------------------------------
def _cfg(**denoise) -> Config:
    cfg = Config()
    for k, v in denoise.items():
        setattr(cfg.render.denoise, k, v)
    return cfg


def test_apost_all_off_is_empty():
    # The byte-for-byte audio copy fast-path must survive the new fields.
    assert build_apost(_cfg(enabled=False, deess=False, loudnorm=False)) == []


def test_apost_only_deess():
    assert build_apost(_cfg(deess=True)) == [DEESS]


def test_apost_only_loudnorm_restores_48k():
    # loudnorm internally resamples to 192 kHz -> aresample must follow it.
    assert build_apost(_cfg(loudnorm=True)) == [LOUDNORM, ARESAMPLE]


def test_apost_deess_then_loudnorm_order():
    # De-esser shapes the signal BEFORE the loudness pass measures it.
    assert build_apost(_cfg(deess=True, loudnorm=True)) == \
        [DEESS, LOUDNORM, ARESAMPLE]


def test_apost_full_chain_order():
    # denoise + normalize + both mastering switches: the complete pipeline.
    f = build_apost(_cfg(enabled=True, normalize=True, deess=True,
                         loudnorm=True))
    assert f == ["highpass=f=80", "afftdn=nf=-25.0", "dynaudnorm=p=0.95:m=100",
                 DEESS, LOUDNORM, ARESAMPLE]


def test_apost_denoise_plus_loudnorm_without_deess():
    f = build_apost(_cfg(enabled=True, loudnorm=True))
    assert f == ["highpass=f=80", "afftdn=nf=-25.0", LOUDNORM, ARESAMPLE]


def test_apost_aresample_only_with_loudnorm():
    # aresample is a loudnorm companion, never added on its own.
    assert ARESAMPLE not in build_apost(_cfg(enabled=True, deess=True))


def test_apost_legacy_denoise_chain_unchanged():
    # Regression: the pre-mastering chain is byte-identical to before.
    assert build_apost(_cfg(enabled=True)) == ["highpass=f=80",
                                               "afftdn=nf=-25.0"]


# --- render() integration (fake FFmpeg, same approach as test_denoise.py) ----
class FakeFF:
    def __init__(self):
        self.runs: list[list[str]] = []

    def has_filter(self, name):
        return True

    def has_encoder(self, name):          # force x264 (deterministic)
        return False

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.runs.append(list(args))
        if args:
            try:
                Path(args[-1]).write_bytes(b"")
            except OSError:
                pass


def _media(has_audio=True):
    return MediaInfo(path="in.mp4", duration=10.0, fps=30.0, width=1920,
                     height=1080, vcodec="h264", acodec="aac",
                     has_audio=has_audio, sample_rate=48000)


def _cutlist(*, cut=False):
    segs = []
    if cut:
        segs.append(CutSegment(id="c", start=2.0, end=3.0, type=TYPE_PAUSE,
                               action=ACTION_REMOVE, enabled=True))
    return CutList(source="in.mp4", duration=10.0, segments=segs)


def _final_graph(ff: FakeFF) -> str:
    args = ff.runs[-1]
    if "-filter_complex" in args:
        return args[args.index("-filter_complex") + 1]
    return ""


def _run(cfg, cl, tmp_path, *, has_audio=True):
    ff = FakeFF()
    out = str(tmp_path / "out.mp4")
    info = render(ff, _media(has_audio=has_audio), cl, cfg, out,
                  str(tmp_path), log=lambda *a, **k: None)
    return ff, info


def test_remux_loudnorm_only_filters_audio_video_copied(tmp_path):
    # Mastering WITHOUT denoise must still reach the audio (no fast-path skip).
    ff, _ = _run(_cfg(loudnorm=True), _cutlist(), tmp_path)
    g = _final_graph(ff)
    assert g == f"[0:a]{LOUDNORM},{ARESAMPLE}[outa]"
    args = ff.runs[-1]
    assert args[args.index("-c:v") + 1] == "copy"   # video untouched
    assert "aac" in args                            # audio re-encoded


def test_remux_all_off_still_pure_copy(tmp_path):
    # Regression: with the new fields at defaults the remux stays bit-exact.
    ff, info = _run(_cfg(), _cutlist(), tmp_path)
    args = ff.runs[-1]
    assert "-filter_complex" not in args
    assert "copy" in args and "-c:a" in args
    assert info["denoise"] is False


def test_cuts_mastering_applies_to_concat_output(tmp_path):
    # With cuts, mastering runs once on the FULL retimed audio (post-concat).
    ff, _ = _run(_cfg(deess=True, loudnorm=True), _cutlist(cut=True), tmp_path)
    g = _final_graph(ff)
    assert "[outv][outa_raw]" in g
    assert g.endswith(f"[outa_raw]{DEESS},{LOUDNORM},{ARESAMPLE}[outa]")


def test_video_only_ignores_mastering(tmp_path):
    # No audio track: mastering must be a no-op (no apost, no crash).
    ff, info = _run(_cfg(deess=True, loudnorm=True), _cutlist(cut=True),
                    tmp_path, has_audio=False)
    g = _final_graph(ff)
    assert "deesser" not in g and "loudnorm" not in g
    assert info["denoise"] is False


# --- serve._resolve_render_opts wiring ---------------------------------------
def _fake_session(tmp_path):
    cfg = load_config("config.yaml")
    media = SimpleNamespace(height=1080, fps=30.0)
    return SimpleNamespace(cfg=cfg, media=media, out_dir=tmp_path / "out",
                           inp=SimpleNamespace(stem="clip"))


def test_resolve_mastering_off_by_default(tmp_path):
    cfg, *_ = serve._resolve_render_opts(_fake_session(tmp_path), {})
    assert cfg.render.denoise.deess is False
    assert cfg.render.denoise.loudnorm is False


def test_resolve_mastering_independent_of_denoise(tmp_path):
    cfg, *_ = serve._resolve_render_opts(
        _fake_session(tmp_path),
        {"denoise": False, "denoise_deess": True, "denoise_loudnorm": True})
    dn = cfg.render.denoise
    assert dn.enabled is False                 # denoise stays off …
    assert dn.deess is True and dn.loudnorm is True   # … mastering still on
    assert build_apost(cfg) == [DEESS, LOUDNORM, ARESAMPLE]


def test_resolve_mastering_with_denoise(tmp_path):
    cfg, *_ = serve._resolve_render_opts(
        _fake_session(tmp_path),
        {"denoise": True, "denoise_deess": True, "denoise_loudnorm": True})
    assert build_apost(cfg) == ["highpass=f=80", "afftdn=nf=-25.0",
                                DEESS, LOUDNORM, ARESAMPLE]


def test_resolve_mastering_explicit_false(tmp_path):
    cfg, *_ = serve._resolve_render_opts(
        _fake_session(tmp_path),
        {"denoise_deess": False, "denoise_loudnorm": False})
    assert cfg.render.denoise.deess is False
    assert cfg.render.denoise.loudnorm is False
