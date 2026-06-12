"""Clip Maker F8 — de-click fades on the clip's TRUE edges (план §2.3.5/F8).

A Shorts clip cut from mid-phrase starts/ends with a full-amplitude waveform
step — a click. render.cut_fade fades only INTERNAL seams, so this feature adds
an afade-in over the first ~25 ms and an afade-out over the last ~25 ms of the
clip's FINAL audio, strictly AFTER the whole apost chain (incl. loudnorm — the
ordering rationale lives in vpipe.render._edge_fade_filters).

Pinned here:
  * _edge_fade_filters unit behavior: times, clamp to 0..0.2 and to half the
    output duration, 0/negative -> [].
  * render() graphs (FakeFF, same approach as test_denoise.py): clip render
    gets the fades with the right times (in at st=0, out ending at clip dur);
    the regular full render (edge_fade default 0.0) stays byte-for-byte
    fade-free, incl. the remux copy fast-path; edge_fade=0 -> no filter;
    clamped value lands in the graph; fades stack AFTER loudnorm and coexist
    with internal cut_fade seam fades; video-only sources get none.
  * serve plumbing: _run_render_pipeline forwards edge_fade to render()
    (explicit parameter — NOT a cutlist_override heuristic) and defaults to
    0.0; per-clip burn ASS files don't clobber each other across the
    /api/clips/render loop (the endpoint-level edge_fade wiring is asserted in
    test_api_clips.py).
  * ClipsCfg.edge_fade default 0.025.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.config import ClipsCfg, Config, ProfanityLists, load_config  # noqa: E402
from vpipe.detect.profanity import ProfanityMatcher                # noqa: E402
from vpipe.models import (ACTION_REMOVE, TYPE_MANUAL, TYPE_PAUSE,  # noqa: E402
                          CutList, CutSegment, Segment, Transcript, Word)
from vpipe.probe import MediaInfo                                  # noqa: E402
from vpipe.render import _edge_fade_filters, render                # noqa: E402

import serve                                                       # noqa: E402

_SILENT = lambda *a, **k: None  # noqa: E731


# --- config -------------------------------------------------------------------
def test_clips_cfg_edge_fade_default():
    assert ClipsCfg().edge_fade == 0.025
    assert Config().clips.edge_fade == 0.025


# --- _edge_fade_filters unit ----------------------------------------------------
def test_edge_fade_filters_zero_off():
    assert _edge_fade_filters(0.0, 30.0) == []
    assert _edge_fade_filters(-1.0, 30.0) == []
    assert _edge_fade_filters(0.025, 0.0) == []        # empty program


def test_edge_fade_filters_times():
    assert _edge_fade_filters(0.025, 6.0) == [
        "afade=t=in:st=0:d=0.025",
        "afade=t=out:st=5.975:d=0.025"]                # out ENDS at clip dur


def test_edge_fade_filters_clamped_to_02():
    # Absurd config value -> clamped to the sane 0.2 s maximum.
    assert _edge_fade_filters(5.0, 30.0) == [
        "afade=t=in:st=0:d=0.200",
        "afade=t=out:st=29.800:d=0.200"]


def test_edge_fade_filters_clamped_to_half_duration():
    # A 0.1 s program must not fade past its own middle.
    assert _edge_fade_filters(0.2, 0.1) == [
        "afade=t=in:st=0:d=0.050",
        "afade=t=out:st=0.050:d=0.050"]


# --- render() graphs (FakeFF — pattern from test_denoise.py) -------------------
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


def _cfg(*, cut_fade=0.0, **denoise) -> Config:
    cfg = Config()
    cfg.render.cut_fade = cut_fade
    for k, v in denoise.items():
        setattr(cfg.render.denoise, k, v)
    return cfg


def _clip_cl(*removes: tuple[float, float]) -> CutList:
    """Clip-shaped cutlist: boundary/internal REMOVEs on a 10 s source."""
    segs = [CutSegment(id=f"r{i}", start=a, end=b, type=TYPE_MANUAL,
                       action=ACTION_REMOVE, enabled=True)
            for i, (a, b) in enumerate(removes)]
    return CutList(source="in.mp4", duration=10.0, segments=segs)


def _graph(ff: FakeFF) -> str:
    args = ff.runs[-1]
    if "-filter_complex" in args:
        return args[args.index("-filter_complex") + 1]
    return ""


def _run(cfg, cl, tmp_path, *, has_audio=True, edge_fade=None, scale_h=None):
    ff = FakeFF()
    kw = {} if edge_fade is None else {"edge_fade": edge_fade}
    info = render(ff, _media(has_audio=has_audio), cl, cfg,
                  str(tmp_path / "out.mp4"), str(tmp_path),
                  log=_SILENT, scale_h=scale_h, **kw)
    return ff, info


def test_clip_render_edge_fades_with_correct_times(tmp_path):
    # Clip [2,8] of a 10 s source (boundary REMOVEs exactly as /api/clips/render
    # builds them) -> final dur 6.0: fade-in at 0, fade-out ENDING at 6.0.
    ff, _ = _run(_cfg(), _clip_cl((0.0, 2.0), (8.0, 10.0)), tmp_path,
                 edge_fade=0.025)
    g = _graph(ff)
    assert "[outv][outa_raw]" in g            # concat audio leaves as raw
    assert g.endswith(";[outa_raw]afade=t=in:st=0:d=0.025,"
                      "afade=t=out:st=5.975:d=0.025[outa]")


def test_full_render_default_has_no_edge_fades(tmp_path):
    # Same cuts, edge_fade NOT passed (every regular render call site) ->
    # graph byte-for-byte fade-free (cut_fade=0 isolates internal seams).
    ff, _ = _run(_cfg(), _clip_cl((0.0, 2.0), (8.0, 10.0)), tmp_path)
    g = _graph(ff)
    assert "afade" not in g
    assert "[outa_raw]" not in g              # concat writes [outa] directly


def test_edge_fade_zero_means_no_filter(tmp_path):
    ff, _ = _run(_cfg(), _clip_cl((0.0, 2.0), (8.0, 10.0)), tmp_path,
                 edge_fade=0.0)
    assert "afade" not in _graph(ff)


def test_edge_fade_clamped_in_graph(tmp_path):
    ff, _ = _run(_cfg(), _clip_cl((0.0, 2.0), (8.0, 10.0)), tmp_path,
                 edge_fade=5.0)
    g = _graph(ff)
    assert "afade=t=in:st=0:d=0.200" in g
    assert "afade=t=out:st=5.800:d=0.200" in g


def test_edge_fades_after_loudnorm(tmp_path):
    # The de-click contract: afade must be the LAST gain stage. loudnorm is an
    # adaptive gain stage — after a fade it would pump the edges back up;
    # after loudnorm the output is guaranteed to ramp from/to literal zero,
    # and the -14 LUFS target is untouched (2×25 ms is far below loudnorm's
    # 400 ms gating blocks).
    ff, _ = _run(_cfg(loudnorm=True), _clip_cl((0.0, 2.0), (8.0, 10.0)),
                 tmp_path, edge_fade=0.025)
    g = _graph(ff)
    assert g.endswith(";[outa_raw]loudnorm=I=-14:TP=-1.5:LRA=11,"
                      "aresample=48000,"
                      "afade=t=in:st=0:d=0.025,"
                      "afade=t=out:st=5.975:d=0.025[outa]")


def test_edge_fades_coexist_with_internal_seam_fades(tmp_path):
    # Internal cut at [5,6] -> kept (2,5)+(6,8), final dur 5.0. cut_fade keeps
    # de-clicking the internal seam per segment; edge fades land once on the
    # concat output with edge times.
    ff, _ = _run(_cfg(cut_fade=0.015), _clip_cl((0.0, 2.0), (5.0, 6.0),
                                                (8.0, 10.0)),
                 tmp_path, edge_fade=0.025)
    g = _graph(ff)
    assert "afade=t=out:st=2.985:d=0.015" in g     # seam: end of kept #0
    assert "afade=t=in:st=0:d=0.015" in g          # seam: start of kept #1
    assert g.endswith(";[outa_raw]afade=t=in:st=0:d=0.025,"
                      "afade=t=out:st=4.975:d=0.025[outa]")


def test_edge_fade_remux_branch_forces_audio_filter(tmp_path):
    # Clip == the whole file (no boundary REMOVEs, no cuts at all): the copy
    # fast-path must yield to a filtered audio path; video stays copied.
    ff, info = _run(_cfg(), _clip_cl(), tmp_path, edge_fade=0.025)
    g = _graph(ff)
    assert g == ("[0:a]afade=t=in:st=0:d=0.025,"
                 "afade=t=out:st=9.975:d=0.025[outa]")
    args = ff.runs[-1]
    assert args[args.index("-c:v") + 1] == "copy"
    assert info["denoise"] is False              # edge fade is NOT denoise


def test_remux_copy_fastpath_survives_default(tmp_path):
    # Regression: regular render, no cuts, edge_fade default -> pure remux.
    ff, _ = _run(_cfg(), _clip_cl(), tmp_path)
    assert "-filter_complex" not in ff.runs[-1]


def test_edge_fade_no_cut_reencode_branch(tmp_path):
    # Whole-file clip + rescale: audio gets its own faded branch in the graph.
    ff, _ = _run(_cfg(), _clip_cl(), tmp_path, edge_fade=0.025, scale_h=720)
    g = _graph(ff)
    assert "[0:v]scale=-2:720[outv]" in g
    assert ("[0:a]afade=t=in:st=0:d=0.025,"
            "afade=t=out:st=9.975:d=0.025[outa]") in g


def test_video_only_source_gets_no_edge_fades(tmp_path):
    ff, _ = _run(_cfg(), _clip_cl((0.0, 2.0), (8.0, 10.0)), tmp_path,
                 has_audio=False, edge_fade=0.025)
    g = _graph(ff)
    assert "afade" not in g and "-an" in ff.runs[-1]


# --- serve plumbing: explicit edge_fade parameter + per-clip burn ASS -----------
def _mk_session(tmp_path, *, duration: float = 20.0):
    n = int(duration)
    words = [Word(f"сл{i:02d}", i + 0.1, i + 0.9) for i in range(n)]
    tr = Transcript(language="ru", duration=duration, model="t", audio_hash="h",
                    segments=[Segment(0.0, duration,
                                      " ".join(w.word for w in words), words)])
    cl = CutList(source="fake.mp4", duration=duration, segments=[])
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        cfg=load_config("config.yaml"),
        inp=Path("fake.mp4"),
        media=SimpleNamespace(path="fake.mp4", duration=duration,
                              width=1920, height=1080, fps=30.0),
        ff=None, work_dir=work, out_dir=tmp_path / "out",
        matcher=ProfanityMatcher(ProfanityLists(roots=[], allow=[])),
        llm=None, transcript=tr, cutlist=cl)


def _patch_render(monkeypatch):
    calls: list[dict] = []

    def fake_render(ff, media, cl, cfg, out, work_dir, *, on_progress=None,
                    log=None, scale_h=None, fps=None, ass_path=None,
                    crop_filter=None, edge_fade=0.0):
        calls.append({"cl": cl, "ass_path": ass_path, "edge_fade": edge_fade})
        return {"out": str(out), "encoder": "fake"}

    monkeypatch.setattr(serve.render_mod, "render", fake_render)
    return calls


def _clip_override(cl: CutList, start: float, end: float, i: int = 0) -> CutList:
    clip = CutList(source=cl.source, duration=cl.duration,
                   segments=list(cl.segments))
    if start > 0:
        clip.segments.append(CutSegment(id=f"clipA{i}", start=0.0, end=start,
                                        type=TYPE_MANUAL, action=ACTION_REMOVE,
                                        enabled=True))
    if end < cl.duration:
        clip.segments.append(CutSegment(id=f"clipB{i}", start=end,
                                        end=cl.duration, type=TYPE_MANUAL,
                                        action=ACTION_REMOVE, enabled=True))
    return clip


_BASE_OPTS = {"subtitles": False, "chapters": False, "metadata": False}


def test_pipeline_forwards_edge_fade_explicitly(monkeypatch, tmp_path):
    # edge_fade reaches render() as passed — and ONLY as passed: a clip-shaped
    # cutlist_override alone (no edge_fade arg) must NOT switch fades on.
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch)
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(
        s, dict(_BASE_OPTS))
    ov = _clip_override(s.cutlist, 5.0, 15.0)
    serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                               _SILENT, _SILENT, cutlist_override=ov,
                               edge_fade=0.123)
    serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                               _SILENT, _SILENT, cutlist_override=ov)
    serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                               _SILENT, _SILENT)        # regular full render
    assert [c["edge_fade"] for c in calls] == [0.123, 0.0, 0.0]


def test_per_clip_burn_ass_not_clobbered_across_loop(monkeypatch, tmp_path):
    # F8: each clip of the render loop writes its OWN work_dir/burn_<base>.ass;
    # rendering clip02 must not overwrite clip01's file.
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch)
    for i, (a, b) in enumerate(((5.0, 11.0), (12.0, 18.0)), start=1):
        opts = {"burn_subtitles": True, "burn_style": {"karaoke": True},
                "filename": f"fake_clip{i:02d}", **_BASE_OPTS}
        cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(s, opts)
        serve._run_render_pipeline(
            s, cfg, scale_h, fps, out_dir, base, _SILENT, _SILENT,
            cutlist_override=_clip_override(s.cutlist, a, b, i - 1),
            edge_fade=0.025)
    paths = [Path(c["ass_path"]) for c in calls]
    assert [p.name for p in paths] == ["burn_fake_clip01.ass",
                                       "burn_fake_clip02.ass"]
    # after the WHOLE loop both files exist with their own clip's words
    assert all(p.is_file() for p in paths)
    t1 = paths[0].read_text(encoding="utf-8")
    t2 = paths[1].read_text(encoding="utf-8")
    assert "сл05" in t1 and "сл12" not in t1
    assert "сл12" in t2 and "сл05" not in t2
