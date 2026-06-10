"""DeepFilterNet 3 as an opt-in denoise engine — TECH_UPGRADE_PLAN.md 2.1.

Additive layer on the denoise feature (tests/test_denoise.py):

  * ``DenoiseCfg`` — new ``engine`` / ``deepfilter_bin`` / ``post_filter``
    fields; defaults change nothing (engine="afftdn").
  * ``_resolve_deepfilter_bin`` — absolute / repo-root-relative / cwd-relative
    / PATH lookup, never raises.
  * ``enhance_audio`` — extract 48 kHz MONO wav -> run the CLI
    (``[--pf] -D -o <outdir> <in.wav>``) -> return ``dfn_out/dfn_in.wav``;
    ANY failure (no binary, extraction error, non-zero exit, missing output)
    returns None.
  * ``build_apost`` — engine="deepfilter" skips the highpass/afftdn/dynaudnorm
    trio; deesser/loudnorm mastering stays. Measurement chain matches.
  * ``render()`` — the enhanced wav replaces the audio input ([1:a]); graceful
    fallback to the afftdn chain when the CLI is unavailable (render never
    dies, caller's cfg untouched); intermediates cleaned after success.
  * ``serve._resolve_render_opts`` — ``denoise_engine`` whitelist.

No network, no GPU, no real ffmpeg, no real deep-filter.exe: everything below
uses fakes / monkeypatched subprocess.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vpipe.render as render_mod                                     # noqa: E402
from vpipe.config import Config, DenoiseCfg, load_config              # noqa: E402
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, TYPE_PAUSE,   # noqa: E402
                          TYPE_PROFANITY, CutList, CutSegment)
from vpipe.probe import MediaInfo                                     # noqa: E402
from vpipe.render import (_resolve_deepfilter_bin, build_apost,       # noqa: E402
                          build_loudnorm_measure_chain, enhance_audio,
                          render)

import serve                                                          # noqa: E402

DYNAMIC = "loudnorm=I=-14:TP=-1.5:LRA=11"


# --- DenoiseCfg defaults (backward compatibility) ------------------------------
def test_engine_default_afftdn():
    d = DenoiseCfg()
    assert d.engine == "afftdn"                  # default keeps old behaviour
    assert d.deepfilter_bin == "tools/deep-filter.exe"
    assert d.post_filter is True
    assert Config().render.denoise.engine == "afftdn"


# --- build_apost: engine routing ------------------------------------------------
def _cfg(**denoise) -> Config:
    cfg = Config()
    for k, v in denoise.items():
        setattr(cfg.render.denoise, k, v)
    return cfg


def test_apost_afftdn_engine_unchanged():
    # Byte-for-byte the legacy chain with the default engine.
    assert build_apost(_cfg(enabled=True)) == ["highpass=f=80", "afftdn=nf=-25.0"]


def test_apost_deepfilter_skips_trio():
    # Denoising happened externally -> no ffmpeg denoise filters at all.
    assert build_apost(_cfg(enabled=True, engine="deepfilter")) == []


def test_apost_deepfilter_skips_trio_even_with_normalize():
    assert build_apost(_cfg(enabled=True, engine="deepfilter",
                            normalize=True)) == []


def test_apost_deepfilter_keeps_mastering():
    f = build_apost(_cfg(enabled=True, engine="deepfilter",
                         deess=True, loudnorm=True))
    assert f == ["deesser=i=0.4", DYNAMIC, "aresample=48000"]


def test_apost_deepfilter_engine_case_insensitive():
    assert build_apost(_cfg(enabled=True, engine="DeepFilter")) == []


def test_apost_deepfilter_without_enabled_is_noop():
    # engine only matters when the denoise toggle is on.
    assert build_apost(_cfg(enabled=False, engine="deepfilter")) == []
    f = build_apost(_cfg(enabled=False, engine="deepfilter", deess=True))
    assert f == ["deesser=i=0.4"]


def test_measure_chain_deepfilter_excludes_afftdn():
    cfg = _cfg(enabled=True, engine="deepfilter", deess=True, loudnorm=True,
               loudnorm_mode="2pass")
    chain = build_loudnorm_measure_chain(cfg)
    assert chain == "deesser=i=0.4," + DYNAMIC + ":print_format=json"
    assert "afftdn" not in chain and "highpass" not in chain


# --- _resolve_deepfilter_bin ------------------------------------------------------
def test_resolve_bin_absolute(tmp_path):
    exe = tmp_path / "deep-filter.exe"
    exe.write_bytes(b"x")
    assert _resolve_deepfilter_bin(str(exe)) == str(exe)


def test_resolve_bin_absolute_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(render_mod.shutil, "which", lambda *a, **k: None)
    assert _resolve_deepfilter_bin(str(tmp_path / "nope.exe")) is None


def test_resolve_bin_repo_root_relative(tmp_path, monkeypatch):
    (tmp_path / "tools").mkdir()
    exe = tmp_path / "tools" / "deep-filter.exe"
    exe.write_bytes(b"x")
    monkeypatch.setattr(render_mod, "_REPO_ROOT", tmp_path)
    assert _resolve_deepfilter_bin("tools/deep-filter.exe") == str(exe)


def test_resolve_bin_cwd_relative(tmp_path, monkeypatch):
    monkeypatch.setattr(render_mod, "_REPO_ROOT", tmp_path / "elsewhere")
    (tmp_path / "mytools").mkdir()
    exe = tmp_path / "mytools" / "df.exe"
    exe.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    assert _resolve_deepfilter_bin("mytools/df.exe") == str(Path("mytools/df.exe"))


def test_resolve_bin_path_stem(tmp_path, monkeypatch):
    # 'tools/deep-filter.exe' not on disk, but bare 'deep-filter' is on PATH.
    monkeypatch.setattr(render_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    hits = {"deep-filter": str(tmp_path / "onpath" / "deep-filter.exe")}

    def fake_which(name):
        return hits.get(name)

    monkeypatch.setattr(render_mod.shutil, "which", fake_which)
    assert (_resolve_deepfilter_bin("tools/deep-filter.exe")
            == hits["deep-filter"])


def test_resolve_bin_empty_and_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(render_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(render_mod.shutil, "which", lambda *a, **k: None)
    assert _resolve_deepfilter_bin("") is None
    assert _resolve_deepfilter_bin("ghost.exe") is None


# --- fakes (mirrors tests/test_denoise.py) --------------------------------------
class FakeFF:
    """Captures every ``run`` args list; touches the trailing output path so
    ``_run_atomic``'s os.replace and the wav-extraction step stay happy."""

    def __init__(self, fail_desc=None):
        self.runs: list[list[str]] = []
        self.fail_desc = fail_desc          # raise on a matching desc (tests)

    def has_filter(self, name):
        return True

    def has_encoder(self, name):
        return False                        # deterministic x264 path

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.runs.append(list(args))
        if self.fail_desc and self.fail_desc in desc:
            raise RuntimeError(f"boom in {desc}")
        if args and args[-1] == "-":      # the `-f null -` measurement pass
            return ""                     # no JSON -> dynamic fallback (fine)
        if args:
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


def _final_graph(ff: FakeFF) -> str:
    args = ff.runs[-1]
    if "-filter_complex" in args:
        return args[args.index("-filter_complex") + 1]
    return ""


def _fake_exe(tmp_path) -> str:
    exe = tmp_path / "fake-deep-filter.exe"
    exe.write_bytes(b"MZ")
    return str(exe)


def _mock_subprocess(monkeypatch, *, returncode=0, create_output=True,
                     out_bytes=b"RIFFdata"):
    """Replace subprocess.run inside vpipe.render with a recorder that mimics
    the real CLI: writes ``<out-dir>/<input-basename>`` and exits 0."""
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output=True, text=True, **kw):
        calls.append(list(cmd))
        if create_output and "-o" in cmd:
            out_dir = Path(cmd[cmd.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / Path(cmd[-1]).name).write_bytes(out_bytes)
        return SimpleNamespace(returncode=returncode, stdout="", stderr="err")

    monkeypatch.setattr(render_mod.subprocess, "run", fake_run)
    return calls


def _dfn_cfg(tmp_path, **extra) -> Config:
    return _cfg(enabled=True, engine="deepfilter",
                deepfilter_bin=_fake_exe(tmp_path), **extra)


# --- enhance_audio ----------------------------------------------------------------
def test_enhance_missing_bin_returns_none(tmp_path):
    cfg = _cfg(enabled=True, engine="deepfilter",
               deepfilter_bin=str(tmp_path / "nope.exe"))
    ff = FakeFF()
    logs = []
    out = enhance_audio(ff, "in.mp4", cfg, str(tmp_path), log=logs.append)
    assert out is None
    assert ff.runs == []                       # no wasted extraction pass
    assert any("afftdn" in m for m in logs)


def test_enhance_success_builds_correct_command(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    calls = _mock_subprocess(monkeypatch)
    ff = FakeFF()
    out = enhance_audio(ff, "src.flac", cfg, str(tmp_path),
                        log=lambda *a: None)
    # 1) extraction: 48 kHz MONO pcm wav from the given source.
    ext = ff.runs[0]
    assert ext[:2] == ["-i", "src.flac"]
    for flag, val in (("-ac", "1"), ("-ar", "48000"), ("-c:a", "pcm_s16le")):
        assert ext[ext.index(flag) + 1] == val
    assert ext[-1] == str(tmp_path / "dfn_in.wav")
    # 2) CLI: <bin> --pf -D -o <out-dir> <in.wav>  (the v0.5.6 argument format).
    assert calls == [[cfg.render.denoise.deepfilter_bin, "--pf", "-D",
                      "-o", str(tmp_path / "dfn_out"),
                      str(tmp_path / "dfn_in.wav")]]
    # 3) result: the CLI keeps the input basename inside the out dir.
    assert out == str(tmp_path / "dfn_out" / "dfn_in.wav")


def test_enhance_post_filter_off_drops_pf(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path, post_filter=False)
    calls = _mock_subprocess(monkeypatch)
    out = enhance_audio(FakeFF(), "in.mp4", cfg, str(tmp_path),
                        log=lambda *a: None)
    assert out is not None
    assert "--pf" not in calls[0] and "-D" in calls[0]


def test_enhance_nonzero_exit_returns_none(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    _mock_subprocess(monkeypatch, returncode=3)
    logs = []
    out = enhance_audio(FakeFF(), "in.mp4", cfg, str(tmp_path), log=logs.append)
    assert out is None
    assert any("afftdn" in m for m in logs)


def test_enhance_missing_output_returns_none(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    _mock_subprocess(monkeypatch, create_output=False)
    out = enhance_audio(FakeFF(), "in.mp4", cfg, str(tmp_path),
                        log=lambda *a: None)
    assert out is None


def test_enhance_empty_output_returns_none(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    _mock_subprocess(monkeypatch, out_bytes=b"")
    out = enhance_audio(FakeFF(), "in.mp4", cfg, str(tmp_path),
                        log=lambda *a: None)
    assert out is None


def test_enhance_extraction_failure_returns_none(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    calls = _mock_subprocess(monkeypatch)
    ff = FakeFF(fail_desc="extract wav")
    logs = []
    out = enhance_audio(ff, "in.mp4", cfg, str(tmp_path), log=logs.append)
    assert out is None
    assert calls == []                          # CLI never launched
    assert any("afftdn" in m for m in logs)


def test_enhance_oserror_on_launch_returns_none(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)

    def boom(*a, **k):
        raise OSError("not executable")

    monkeypatch.setattr(render_mod.subprocess, "run", boom)
    out = enhance_audio(FakeFF(), "in.mp4", cfg, str(tmp_path),
                        log=lambda *a: None)
    assert out is None


# --- render() integration -----------------------------------------------------
def _run(cfg, cl, tmp_path, *, has_audio=True, log=None):
    ff = FakeFF()
    out = str(tmp_path / "out.mp4")
    info = render(ff, _media(has_audio=has_audio), cl, cfg, out,
                  str(tmp_path), log=log or (lambda *a, **k: None))
    return ff, info


def test_render_deepfilter_cuts_uses_enhanced_wav(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    _mock_subprocess(monkeypatch)
    ff, info = _run(cfg, _cutlist(cut=True), tmp_path)
    g = _final_graph(ff)
    # The enhanced wav is input #1 and feeds the per-segment trims…
    args = ff.runs[-1]
    dfn_wav = str(tmp_path / "dfn_out" / "dfn_in.wav")
    assert args[args.index("-i", args.index("-i") + 1) + 1] == dfn_wav
    assert "[1:a]atrim=" in g
    # …and NO ffmpeg denoise filters run on top of it.
    assert "afftdn" not in g and "highpass" not in g and "dynaudnorm" not in g
    assert info["denoise"] is True


def test_render_deepfilter_consumes_censored_flac(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    _mock_subprocess(monkeypatch)
    ff, info = _run(cfg, _cutlist(cut=True, censor=True), tmp_path)
    # The extraction pass reads the CENSORED flac, not the original media.
    ext = next(r for r in ff.runs if r and str(r[-1]).endswith("dfn_in.wav"))
    assert str(ext[1]).endswith("censored.flac")
    # The graph's audio is the enhanced wav (input #1), censor already baked in.
    assert "[1:a]atrim=" in _final_graph(ff)
    assert info["censored"] == 1 and info["denoise"] is True


def test_render_deepfilter_keeps_mastering(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path, deess=True, loudnorm=True)
    _mock_subprocess(monkeypatch)
    ff, _ = _run(cfg, _cutlist(cut=True), tmp_path)
    g = _final_graph(ff)
    assert g.endswith(
        "[outa_raw]deesser=i=0.4," + DYNAMIC + ",aresample=48000[outa]")
    assert "afftdn" not in g


def test_render_deepfilter_remux_branch(tmp_path, monkeypatch):
    # No cuts, no censor, no mastering: video copied, enhanced wav muxed as aac.
    cfg = _dfn_cfg(tmp_path)
    _mock_subprocess(monkeypatch)
    ff, info = _run(cfg, _cutlist(), tmp_path)
    args = ff.runs[-1]
    assert "-filter_complex" not in args
    assert args[args.index("-c:v") + 1] == "copy"
    assert str(tmp_path / "dfn_out" / "dfn_in.wav") in args
    assert "aac" in args
    assert info["denoise"] is True and info["encoder"] == "copy"


def test_render_deepfilter_cleans_intermediates(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    _mock_subprocess(monkeypatch)
    _run(cfg, _cutlist(cut=True), tmp_path)
    assert not (tmp_path / "dfn_in.wav").exists()
    assert not (tmp_path / "dfn_out" / "dfn_in.wav").exists()
    assert not (tmp_path / "dfn_out").exists()


def test_render_deepfilter_fallback_to_afftdn(tmp_path):
    # Binary missing -> honest log, afftdn chain, render SUCCEEDS, and the
    # caller's cfg object is never mutated.
    cfg = _cfg(enabled=True, engine="deepfilter",
               deepfilter_bin=str(tmp_path / "nope.exe"))
    logs = []
    ff, info = _run(cfg, _cutlist(cut=True), tmp_path, log=logs.append)
    g = _final_graph(ff)
    assert g.endswith("[outa_raw]highpass=f=80,afftdn=nf=-25.0[outa]")
    assert info["denoise"] is True
    assert any("использую afftdn" in m for m in logs)
    assert cfg.render.denoise.engine == "deepfilter"   # caller cfg untouched


def test_render_deepfilter_exe_failure_falls_back(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    _mock_subprocess(monkeypatch, returncode=1, create_output=False)
    logs = []
    ff, info = _run(cfg, _cutlist(cut=True), tmp_path, log=logs.append)
    assert "afftdn" in _final_graph(ff)
    assert info["denoise"] is True
    assert any("использую afftdn" in m for m in logs)


def test_render_video_only_skips_deepfilter(tmp_path, monkeypatch):
    cfg = _dfn_cfg(tmp_path)
    calls = _mock_subprocess(monkeypatch)
    ff, info = _run(cfg, _cutlist(cut=True), tmp_path, has_audio=False)
    assert calls == []                          # CLI never launched
    assert info["denoise"] is False
    assert "afftdn" not in _final_graph(ff)


def test_render_afftdn_engine_never_calls_cli(tmp_path, monkeypatch):
    # Default engine: byte-for-byte the legacy path, the CLI must not run.
    cfg = _cfg(enabled=True)
    calls = _mock_subprocess(monkeypatch)
    ff, info = _run(cfg, _cutlist(cut=True), tmp_path)
    assert calls == []
    g = _final_graph(ff)
    assert g.endswith("[outa_raw]highpass=f=80,afftdn=nf=-25.0[outa]")
    assert info["denoise"] is True


def test_render_deepfilter_2pass_loudnorm_measures_enhanced(tmp_path, monkeypatch):
    # 2-pass loudnorm + deepfilter: the measurement pass must read the ENHANCED
    # wav ([1:a]) and its chain must not contain the afftdn trio.
    cfg = _dfn_cfg(tmp_path, loudnorm=True, loudnorm_mode="2pass")
    _mock_subprocess(monkeypatch)
    ff, _ = _run(cfg, _cutlist(cut=True), tmp_path)
    measure = next(r for r in ff.runs if "null" in r)
    g = measure[measure.index("-filter_complex") + 1]
    assert "[1:a]atrim=" in g
    assert "print_format=json" in g and "afftdn" not in g
    dfn_wav = str(tmp_path / "dfn_out" / "dfn_in.wav")
    assert dfn_wav in measure


# --- serve._resolve_render_opts wiring -----------------------------------------
def _fake_session(tmp_path):
    cfg = load_config("config.yaml")
    media = SimpleNamespace(height=1080, fps=30.0)
    return SimpleNamespace(cfg=cfg, media=media, out_dir=tmp_path / "out",
                           inp=SimpleNamespace(stem="clip"))


def test_resolve_engine_default(tmp_path):
    s = _fake_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(s, {"denoise": True})
    assert cfg.render.denoise.engine == "afftdn"


def test_resolve_engine_deepfilter(tmp_path):
    s = _fake_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(
        s, {"denoise": True, "denoise_engine": "deepfilter"})
    assert cfg.render.denoise.engine == "deepfilter"
    assert cfg.render.denoise.enabled is True


def test_resolve_engine_whitelist_rejects_junk(tmp_path):
    s = _fake_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(
        s, {"denoise": True, "denoise_engine": "rm -rf /"})
    assert cfg.render.denoise.engine == "afftdn"


def test_resolve_engine_ignored_when_denoise_off(tmp_path):
    s = _fake_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(
        s, {"denoise": False, "denoise_engine": "deepfilter"})
    assert cfg.render.denoise.enabled is False
    assert cfg.render.denoise.engine == "afftdn"
