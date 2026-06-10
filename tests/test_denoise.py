"""Audio denoise / speech-enhancement at render time.

Covers the four moving parts of the feature, additively layered on the existing
render so the censor / cuts / vertical / subtitle paths keep working:

  * ``DenoiseCfg`` defaults (OFF — audio is irreversible, user opts in).
  * ``build_apost`` — the audio post-filter list (highpass -> afftdn ->
    optional dynaudnorm), empty when disabled.
  * ``render()`` integration across all three branches (remux / no-cut
    reencode / cuts+concat), asserting the denoise filters land on the FINAL
    audio AFTER censoring, and that disabling it leaves the audio path
    byte-for-byte as before.
  * ``serve._resolve_render_opts`` — UI/queue opts -> cfg.render.denoise.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.config import Config, DenoiseCfg, load_config            # noqa: E402
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, TYPE_PROFANITY,    # noqa: E402
                          TYPE_PAUSE, CutList, CutSegment)
from vpipe.probe import MediaInfo                                    # noqa: E402
from vpipe.render import build_apost, render                         # noqa: E402

import serve                                                         # noqa: E402


# --- DenoiseCfg defaults -----------------------------------------------------
def test_denoise_default_off():
    d = DenoiseCfg()
    assert d.enabled is False                 # opt-in: audio is irreversible
    assert d.highpass_hz == 80
    assert d.nf == -25.0
    assert d.normalize is False


def test_render_cfg_has_denoise():
    cfg = Config()
    assert cfg.render.denoise.enabled is False


# --- build_apost -------------------------------------------------------------
def _cfg(**denoise) -> Config:
    cfg = Config()
    for k, v in denoise.items():
        setattr(cfg.render.denoise, k, v)
    return cfg


def test_apost_empty_when_disabled():
    assert build_apost(_cfg(enabled=False)) == []


def test_apost_basic_chain():
    f = build_apost(_cfg(enabled=True))
    assert f == ["highpass=f=80", "afftdn=nf=-25.0"]


def test_apost_custom_strength_and_highpass():
    f = build_apost(_cfg(enabled=True, highpass_hz=60, nf=-30.0))
    assert f == ["highpass=f=60", "afftdn=nf=-30.0"]


def test_apost_highpass_skipped_when_zero():
    f = build_apost(_cfg(enabled=True, highpass_hz=0))
    assert f == ["afftdn=nf=-25.0"]          # only the FFT denoiser, no highpass


def test_apost_normalize_appends_dynaudnorm():
    f = build_apost(_cfg(enabled=True, normalize=True))
    assert f == ["highpass=f=80", "afftdn=nf=-25.0", "dynaudnorm=p=0.95:m=100"]


# --- render() integration: capture the ffmpeg args via a fake FFmpeg ---------
class FakeFF:
    """Captures the last ``run`` args and fakes capability probing.

    ``censored`` controls whether Stage 1 writes a censored FLAC: we make
    ``has_filter`` true so build_censor_graph proceeds, and remember every run
    so the final-encode args can be asserted on.
    """
    def __init__(self):
        self.runs: list[list[str]] = []

    def has_filter(self, name):           # rubberband etc. — say yes
        return True

    def has_encoder(self, name):          # force the x264 path (deterministic)
        return False

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.runs.append(list(args))
        # _run_atomic writes to "<out>.part" then os.replace()s it into place,
        # so touch the trailing output path to keep that atomic move happy.
        if args:
            try:
                Path(args[-1]).write_bytes(b"")
            except OSError:
                pass


def _media(has_audio=True):
    return MediaInfo(path="in.mp4", duration=10.0, fps=30.0, width=1920,
                     height=1080, vcodec="h264", acodec="aac",
                     has_audio=has_audio, sample_rate=48000)


def _cutlist(*, cut=False, censor=False):
    segs = []
    if cut:
        segs.append(CutSegment(id="c", start=2.0, end=3.0, type=TYPE_PAUSE,
                               action=ACTION_REMOVE, enabled=True))
    if censor:
        segs.append(CutSegment(id="p", start=5.0, end=5.5, type=TYPE_PROFANITY,
                               action=ACTION_CENSOR, enabled=True))
    return CutList(source="in.mp4", duration=10.0, segments=segs)


def _final_graph(ff: FakeFF) -> str:
    """The -filter_complex of the LAST (final-encode) run, or '' if none."""
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


# branch 1: remux (no cuts, no vpost) -----------------------------------------
def test_remux_denoise_off_copies_audio(tmp_path):
    ff, info = _run(_cfg(enabled=False), _cutlist(), tmp_path)
    args = ff.runs[-1]
    assert "-filter_complex" not in args         # pure remux, audio copied
    assert "copy" in args and "-c:a" in args
    assert info["denoise"] is False


def test_remux_denoise_on_filters_original_audio(tmp_path):
    ff, info = _run(_cfg(enabled=True), _cutlist(), tmp_path)
    g = _final_graph(ff)
    # No censoring here -> denoise runs on the original audio [0:a].
    assert g == "[0:a]highpass=f=80,afftdn=nf=-25.0[outa]"
    args = ff.runs[-1]
    assert args[args.index("-c:v") + 1] == "copy"   # video still copied
    assert "-c:a" in args and "aac" in args         # audio re-encoded
    assert info["denoise"] is True


def test_remux_denoise_on_after_censor_uses_flac(tmp_path):
    # Censor only (no cuts) -> Stage 1 writes censored FLAC (input #1);
    # denoise must read [1:a], i.e. the ALREADY-censored audio.
    ff, info = _run(_cfg(enabled=True), _cutlist(censor=True), tmp_path)
    g = _final_graph(ff)
    assert g == "[1:a]highpass=f=80,afftdn=nf=-25.0[outa]"
    assert info["censored"] == 1 and info["denoise"] is True


# branch 2: no-cut reencode (rescale forces video reencode) -------------------
def test_reencode_denoise_off_passes_audio_through(tmp_path):
    ff, _ = _run(_cfg(enabled=False), _cutlist(), tmp_path)
    # Force the no-cut reencode branch via scale_h.
    ff2 = FakeFF()
    render(ff2, _media(), _cutlist(), _cfg(enabled=False),
           str(tmp_path / "o2.mp4"), str(tmp_path), scale_h=720,
           log=lambda *a, **k: None)
    g = ff2.runs[-1][ff2.runs[-1].index("-filter_complex") + 1]
    assert g == "[0:v]scale=-2:720[outv]"        # audio NOT in the graph
    assert "0:a" in ff2.runs[-1]                  # mapped straight through


def test_reencode_denoise_on_adds_audio_branch(tmp_path):
    ff = FakeFF()
    render(ff, _media(), _cutlist(), _cfg(enabled=True),
           str(tmp_path / "o.mp4"), str(tmp_path), scale_h=720,
           log=lambda *a, **k: None)
    g = ff.runs[-1][ff.runs[-1].index("-filter_complex") + 1]
    assert "[0:v]scale=-2:720[outv]" in g
    assert "[0:a]highpass=f=80,afftdn=nf=-25.0[outa]" in g


# branch 3: cuts + concat -----------------------------------------------------
def test_cuts_denoise_off_keeps_concat_outa(tmp_path):
    ff, _ = _run(_cfg(enabled=False), _cutlist(cut=True), tmp_path)
    g = _final_graph(ff)
    assert "concat=n=" in g and "[outv][outa]" in g
    assert "afftdn" not in g and "highpass" not in g


def test_cuts_denoise_on_filters_concat_output(tmp_path):
    ff, info = _run(_cfg(enabled=True), _cutlist(cut=True), tmp_path)
    g = _final_graph(ff)
    # concat emits [outa_raw], the apost chain produces the final [outa].
    assert "[outv][outa_raw]" in g
    assert g.endswith("[outa_raw]highpass=f=80,afftdn=nf=-25.0[outa]")
    assert info["denoise"] is True


def test_cuts_and_censor_denoise_after_censor(tmp_path):
    # Both a remove and a censor -> aud source is the censored FLAC [1:a].
    cl = _cutlist(cut=True, censor=True)
    ff, info = _run(_cfg(enabled=True, normalize=True), cl, tmp_path)
    g = _final_graph(ff)
    assert "[1:a]atrim=" in g                      # per-segment trims read the FLAC
    assert g.endswith(
        "[outa_raw]highpass=f=80,afftdn=nf=-25.0,dynaudnorm=p=0.95:m=100[outa]")
    assert info["censored"] == 1 and info["denoise"] is True


def test_video_only_ignores_denoise(tmp_path):
    # No audio track: denoise must be a no-op (no apost, no crash).
    ff, info = _run(_cfg(enabled=True), _cutlist(cut=True), tmp_path,
                    has_audio=False)
    g = _final_graph(ff)
    assert "afftdn" not in g and "highpass" not in g
    assert info["denoise"] is False


# --- serve._resolve_render_opts wiring ---------------------------------------
def _fake_session(tmp_path):
    cfg = load_config("config.yaml")
    media = SimpleNamespace(height=1080, fps=30.0)
    return SimpleNamespace(cfg=cfg, media=media, out_dir=tmp_path / "out",
                           inp=SimpleNamespace(stem="clip"))


def test_resolve_denoise_off_by_default(tmp_path):
    s = _fake_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(s, {})
    assert cfg.render.denoise.enabled is False


def test_resolve_denoise_on_with_params(tmp_path):
    s = _fake_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(
        s, {"denoise": True, "denoise_strength": -30,
            "denoise_normalize": True, "denoise_highpass": 60})
    dn = cfg.render.denoise
    assert dn.enabled is True
    assert dn.nf == -30.0
    assert dn.normalize is True
    assert dn.highpass_hz == 60


def test_resolve_denoise_clamps_strength(tmp_path):
    s = _fake_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(
        s, {"denoise": True, "denoise_strength": -999})
    assert cfg.render.denoise.nf == -45.0     # clamped to the safe floor
    cfg2, *_ = serve._resolve_render_opts(
        s, {"denoise": True, "denoise_strength": 100})
    assert cfg2.render.denoise.nf == -6.0     # clamped to the safe ceiling


def test_resolve_denoise_explicit_false(tmp_path):
    s = _fake_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(s, {"denoise": False})
    assert cfg.render.denoise.enabled is False
