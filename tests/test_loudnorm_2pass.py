"""Two-pass (linear) loudnorm — TECH_UPGRADE_PLAN.md section 2.2.

Additive layer on top of the mastering feature (tests/test_mastering.py):

  * ``DenoiseCfg.loudnorm_mode`` — "dynamic" (default, legacy one-pass) |
    "2pass" (measurement pass + linear loudnorm). Defaults change nothing.
  * ``parse_loudnorm_stats`` — robust extraction of the loudnorm JSON stats
    block from an ffmpeg stderr dump (last ``{...}`` with ``"input_i"``).
  * ``build_apost(cfg, loudnorm_measured=...)`` — with usable measured stats
    the loudnorm becomes the LINEAR variant (``measured_*`` + ``linear=true``);
    without them (or with a broken dict) the historical dynamic string is kept
    byte-for-byte.
  * ``measure_loudness`` / ``render()`` — the measurement pass replays the
    exact final audio path (cuts + censor + denoise/deess, no loudnorm) into
    ``-f null``; ANY failure logs honestly and falls back to dynamic (the
    render never dies because of the measurement).
  * ``serve._resolve_render_opts`` — ``loudnorm_mode`` opt is whitelisted.
  * ``FFmpeg.run`` — now returns the captured stderr window on success.

No network, no GPU, no real ffmpeg: everything below uses fakes.
"""
import io
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe import ffmpeg_utils                                        # noqa: E402
from vpipe.config import Config, DenoiseCfg, load_config              # noqa: E402
from vpipe.ffmpeg_utils import FFmpegError                            # noqa: E402
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, TYPE_PAUSE,   # noqa: E402
                          TYPE_PROFANITY, CutList, CutSegment)
from vpipe.probe import MediaInfo                                     # noqa: E402
from vpipe.render import (build_apost, build_loudnorm_measure_chain,  # noqa: E402
                          parse_loudnorm_stats, render)

import serve                                                          # noqa: E402

DYNAMIC = "loudnorm=I=-14:TP=-1.5:LRA=11"
MEASURE = DYNAMIC + ":print_format=json"
LINEAR = (DYNAMIC + ":measured_I=-27.61:measured_TP=-9.11:measured_LRA=18.06"
          ":measured_thresh=-39.20:offset=0.47:linear=true")
ARESAMPLE = "aresample=48000"

MEASURED = {"input_i": -27.61, "input_tp": -9.11, "input_lra": 18.06,
            "input_thresh": -39.20, "target_offset": 0.47}

# Realistic ffmpeg stderr from a `-f null` measurement pass (loudnorm prints
# its JSON block at the very end, values are strings — exactly like ffmpeg).
SAMPLE_STDERR = """\
ffmpeg version 8.1.1-full_build Copyright (c) 2000-2025 the FFmpeg developers
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'in.mp4':
  Duration: 00:00:10.00, start: 0.000000, bitrate: 4137 kb/s
Stream mapping:
  Stream #0:1 (aac) -> atrim:default
Output #0, null, to 'pipe:':
[Parsed_loudnorm_2 @ 000001f2a3b4c5d6]
{
\t"input_i" : "-27.61",
\t"input_tp" : "-9.11",
\t"input_lra" : "18.06",
\t"input_thresh" : "-39.20",
\t"output_i" : "-14.47",
\t"output_tp" : "-1.50",
\t"output_lra" : "16.70",
\t"output_thresh" : "-25.67",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.47"
}
[out#0/null @ 0000023] video:0KiB audio:1875KiB subtitle:0KiB other streams:0KiB
"""


# --- config defaults (backward compatibility) ---------------------------------
def test_loudnorm_mode_default_dynamic():
    assert DenoiseCfg().loudnorm_mode == "dynamic"
    assert Config().render.denoise.loudnorm_mode == "dynamic"


def test_repo_config_yaml_keeps_dynamic_default():
    # config.yaml's denoise block is commented out -> pydantic default applies.
    cfg = load_config("config.yaml")
    assert cfg.render.denoise.loudnorm_mode == "dynamic"
    assert cfg.render.denoise.loudnorm is False


# --- parse_loudnorm_stats ------------------------------------------------------
def test_parse_sample_stderr():
    stats = parse_loudnorm_stats(SAMPLE_STDERR)
    assert stats == MEASURED


def test_parse_takes_last_block():
    # A stray earlier block (e.g. from a verbose log) must not win.
    decoy = SAMPLE_STDERR.replace('"-27.61"', '"-99.00"') + SAMPLE_STDERR
    stats = parse_loudnorm_stats(decoy)
    assert stats is not None
    assert stats["input_i"] == -27.61


def test_parse_garbage_returns_none():
    assert parse_loudnorm_stats("") is None
    assert parse_loudnorm_stats(None) is None
    assert parse_loudnorm_stats("frame=  100 fps= 25 q=-1.0 size=N/A") is None
    assert parse_loudnorm_stats('{"input_i" : broken json}') is None


def test_parse_missing_keys_returns_none():
    assert parse_loudnorm_stats('{"input_i" : "-20.0", "input_tp" : "-2.0"}') is None


def test_parse_nonfinite_returns_none():
    # Pure silence: ffmpeg prints -inf — unusable for a linear pass 2.
    s = SAMPLE_STDERR.replace('"-9.11"', '"-inf"')
    assert parse_loudnorm_stats(s) is None


def test_parse_falls_back_to_earlier_valid_block():
    # Last block broken (missing key) -> the previous valid one is used.
    broken = SAMPLE_STDERR + '\n{\n\t"input_i" : "-50.0"\n}\n'
    stats = parse_loudnorm_stats(broken)
    assert stats == MEASURED


# --- build_apost with measured stats -------------------------------------------
def _cfg(**denoise) -> Config:
    cfg = Config()
    for k, v in denoise.items():
        setattr(cfg.render.denoise, k, v)
    return cfg


def test_apost_measured_builds_linear():
    f = build_apost(_cfg(loudnorm=True), loudnorm_measured=MEASURED)
    assert f == [LINEAR, ARESAMPLE]


def test_apost_no_measured_stays_dynamic():
    # Regression: the historical one-pass string, byte-for-byte.
    assert build_apost(_cfg(loudnorm=True)) == [DYNAMIC, ARESAMPLE]
    assert build_apost(_cfg(loudnorm=True), loudnorm_measured=None) == \
        [DYNAMIC, ARESAMPLE]


def test_apost_broken_measured_degrades_to_dynamic():
    bad = dict(MEASURED)
    del bad["target_offset"]
    assert build_apost(_cfg(loudnorm=True), loudnorm_measured=bad) == \
        [DYNAMIC, ARESAMPLE]
    nonnum = dict(MEASURED, input_i="oops")
    assert build_apost(_cfg(loudnorm=True), loudnorm_measured=nonnum) == \
        [DYNAMIC, ARESAMPLE]
    inf = dict(MEASURED, input_tp=float("-inf"))
    assert build_apost(_cfg(loudnorm=True), loudnorm_measured=inf) == \
        [DYNAMIC, ARESAMPLE]


def test_apost_measured_keeps_chain_order():
    f = build_apost(_cfg(enabled=True, deess=True, loudnorm=True),
                    loudnorm_measured=MEASURED)
    assert f == ["highpass=f=80", "afftdn=nf=-25.0", "deesser=i=0.4",
                 LINEAR, ARESAMPLE]


def test_apost_measured_ignored_when_loudnorm_off():
    assert build_apost(_cfg(deess=True), loudnorm_measured=MEASURED) == \
        ["deesser=i=0.4"]


# --- build_loudnorm_measure_chain ----------------------------------------------
def test_measure_chain_loudnorm_only():
    assert build_loudnorm_measure_chain(_cfg(loudnorm=True)) == MEASURE


def test_measure_chain_includes_pre_loudnorm_filters():
    c = build_loudnorm_measure_chain(
        _cfg(enabled=True, deess=True, loudnorm=True))
    assert c == f"highpass=f=80,afftdn=nf=-25.0,deesser=i=0.4,{MEASURE}"
    # The measurement pass must never master (no aresample, no linear pass).
    assert ARESAMPLE not in c and "linear=true" not in c


def test_measure_chain_does_not_mutate_cfg():
    cfg = _cfg(loudnorm=True)
    build_loudnorm_measure_chain(cfg)
    assert cfg.render.denoise.loudnorm is True   # copy, not mutation


# --- render() integration (fake FFmpeg, same approach as test_mastering.py) ----
class FakeFF:
    """Records every run; serves canned stderr to the measurement pass."""

    def __init__(self, measure_stderr=SAMPLE_STDERR, measure_raises=False):
        self.runs: list[list[str]] = []
        self.measure_stderr = measure_stderr
        self.measure_raises = measure_raises

    def has_filter(self, name):
        return True

    def has_encoder(self, name):          # force x264 (deterministic args)
        return False

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.runs.append(list(args))
        if args and args[-1] == "-":      # the `-f null -` measurement pass
            if self.measure_raises:
                raise FFmpegError("measure boom")
            return self.measure_stderr
        try:
            Path(args[-1]).write_bytes(b"")
        except OSError:
            pass
        return ""


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


def _graph(args: list[str]) -> str:
    if "-filter_complex" in args:
        return args[args.index("-filter_complex") + 1]
    return ""


def _run(cfg, cl, tmp_path, *, has_audio=True, ff=None):
    ff = ff or FakeFF()
    out = str(tmp_path / "out.mp4")
    info = render(ff, _media(has_audio=has_audio), cl, cfg, out,
                  str(tmp_path), log=lambda *a, **k: None)
    return ff, info


def _2pass_cfg(**extra) -> Config:
    return _cfg(loudnorm=True, loudnorm_mode="2pass", **extra)


def test_render_dynamic_mode_runs_no_measure_pass(tmp_path):
    # Regression: default mode -> exactly one ffmpeg run, dynamic loudnorm.
    ff, _ = _run(_cfg(loudnorm=True), _cutlist(), tmp_path)
    assert len(ff.runs) == 1
    assert f"{DYNAMIC},{ARESAMPLE}" in _graph(ff.runs[0])
    assert "linear=true" not in _graph(ff.runs[0])


def test_render_2pass_no_cuts_measures_then_linear(tmp_path):
    ff, info = _run(_2pass_cfg(), _cutlist(), tmp_path)
    assert len(ff.runs) == 2
    m, f = ff.runs[0], ff.runs[1]
    # Pass 1: audio-only measurement into -f null (video never encoded).
    assert m[-3:] == ["-f", "null", "-"]
    assert _graph(m) == f"[0:a]{MEASURE}[mout]"
    assert ["-map", "[mout]"] == m[m.index("-map"):m.index("-map") + 2]
    # Pass 2: the encode uses the LINEAR loudnorm with the measured values.
    assert _graph(f) == f"[0:a]{LINEAR},{ARESAMPLE}[outa]"
    assert f[f.index("-c:v") + 1] == "copy"       # video still copied
    assert info["denoise"] is True


def test_render_2pass_with_cuts_replays_trims_and_fades(tmp_path):
    ff, _ = _run(_2pass_cfg(), _cutlist(cut=True), tmp_path)
    assert len(ff.runs) == 2
    mg = _graph(ff.runs[0])
    # The measurement replays the EXACT final audio: trims + seam fades +
    # audio-only concat, then the pre-loudnorm chain + measurement filter.
    assert "atrim=start=0.000:end=2.000" in mg
    assert "atrim=start=3.000:end=10.000" in mg
    assert "afade=t=out" in mg and "afade=t=in" in mg     # cut_fade=0.015
    assert "concat=n=2:v=0:a=1[mraw]" in mg
    assert mg.endswith(f"[mraw]{MEASURE}[mout]")
    assert "linear=true" not in mg
    fg = _graph(ff.runs[1])
    assert f"[outa_raw]{LINEAR},{ARESAMPLE}[outa]" in fg


def test_render_2pass_measures_censored_audio(tmp_path):
    # Censoring runs first; the measurement consumes the censored FLAC ([1:a]).
    ff, _ = _run(_2pass_cfg(), _cutlist(censor=True), tmp_path)
    assert len(ff.runs) == 3                       # censor, measure, encode
    m = ff.runs[1]
    assert m[-3:] == ["-f", "null", "-"]
    assert _graph(m).startswith("[1:a]")
    assert any(str(a).endswith("censored.flac") for a in m)
    assert LINEAR in _graph(ff.runs[2])


def test_render_2pass_bad_stderr_falls_back_to_dynamic(tmp_path):
    ff = FakeFF(measure_stderr="no json here at all")
    ff, info = _run(_2pass_cfg(), _cutlist(), tmp_path, ff=ff)
    assert len(ff.runs) == 2                       # measure attempted …
    g = _graph(ff.runs[1])
    assert f"{DYNAMIC},{ARESAMPLE}" in g           # … then honest dynamic
    assert "linear=true" not in g
    assert info["out"].endswith("out.mp4")         # render survived


def test_render_2pass_measure_error_falls_back_to_dynamic(tmp_path):
    ff = FakeFF(measure_raises=True)
    ff, info = _run(_2pass_cfg(), _cutlist(cut=True), tmp_path, ff=ff)
    g = _graph(ff.runs[-1])
    assert DYNAMIC in g and "linear=true" not in g
    assert info["out"].endswith("out.mp4")


def test_render_2pass_without_loudnorm_is_noop(tmp_path):
    # loudnorm_mode alone (loudnorm=False) must not trigger anything.
    ff, _ = _run(_cfg(loudnorm=False, loudnorm_mode="2pass"), _cutlist(),
                 tmp_path)
    assert len(ff.runs) == 1
    assert "loudnorm" not in _graph(ff.runs[0])
    assert "-filter_complex" not in ff.runs[0]     # pure copy fast-path kept


def test_render_2pass_video_only_skips_measure(tmp_path):
    ff, _ = _run(_2pass_cfg(), _cutlist(cut=True), tmp_path, has_audio=False)
    assert len(ff.runs) == 1
    assert "loudnorm" not in _graph(ff.runs[0])


def test_render_2pass_unknown_mode_behaves_dynamic(tmp_path):
    ff, _ = _run(_cfg(loudnorm=True, loudnorm_mode="weird"), _cutlist(),
                 tmp_path)
    assert len(ff.runs) == 1
    assert DYNAMIC in _graph(ff.runs[0])


# --- serve._resolve_render_opts whitelist --------------------------------------
def _fake_session(tmp_path):
    cfg = load_config("config.yaml")
    media = SimpleNamespace(height=1080, fps=30.0)
    return SimpleNamespace(cfg=cfg, media=media, out_dir=tmp_path / "out",
                           inp=SimpleNamespace(stem="clip"))


def test_resolve_loudnorm_mode_default(tmp_path):
    cfg, *_ = serve._resolve_render_opts(_fake_session(tmp_path), {})
    assert cfg.render.denoise.loudnorm_mode == "dynamic"


def test_resolve_loudnorm_mode_2pass(tmp_path):
    cfg, *_ = serve._resolve_render_opts(
        _fake_session(tmp_path),
        {"denoise_loudnorm": True, "loudnorm_mode": "2pass"})
    assert cfg.render.denoise.loudnorm is True
    assert cfg.render.denoise.loudnorm_mode == "2pass"


def test_resolve_loudnorm_mode_whitelist_rejects_junk(tmp_path):
    for junk in ("evil; rm -rf", 42, None, ["2pass"], "2PASS"):
        cfg, *_ = serve._resolve_render_opts(
            _fake_session(tmp_path), {"loudnorm_mode": junk})
        assert cfg.render.denoise.loudnorm_mode == "dynamic"


def test_resolve_loudnorm_mode_dynamic_explicit(tmp_path):
    cfg, *_ = serve._resolve_render_opts(
        _fake_session(tmp_path), {"loudnorm_mode": "dynamic"})
    assert cfg.render.denoise.loudnorm_mode == "dynamic"


# --- FFmpeg.run returns the stderr window ---------------------------------------
class _FakeProc:
    def __init__(self, stderr_text):
        self.stdout = io.StringIO("progress=end\n")
        self.stderr = io.StringIO(stderr_text)
        self.returncode = 0

    def wait(self):
        return 0

    def poll(self):
        return self.returncode


def _fake_ffmpeg(monkeypatch, stderr_text):
    ff = ffmpeg_utils.FFmpeg.__new__(ffmpeg_utils.FFmpeg)
    ff.ffmpeg, ff.ffprobe, ff._caps = "ffmpeg", "ffprobe", {}
    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(stderr_text))
    return ff


def test_ffmpeg_run_returns_stderr(monkeypatch):
    ff = _fake_ffmpeg(monkeypatch, SAMPLE_STDERR)
    out = ff.run(["-i", "x", "-f", "null", "-"])
    assert parse_loudnorm_stats(out) == MEASURED


def test_ffmpeg_run_long_stderr_keeps_trailing_json(monkeypatch):
    # 100 lines of noise before the JSON: the head+tail window must still
    # contain the trailing stats block (it lives in the last ~14 lines).
    noise = "".join(f"[info] noise line {i}\n" for i in range(100))
    ff = _fake_ffmpeg(monkeypatch, noise + SAMPLE_STDERR)
    out = ff.run(["-i", "x", "-f", "null", "-"])
    assert parse_loudnorm_stats(out) == MEASURED
