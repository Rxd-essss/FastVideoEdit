"""F2 — Clip Maker render tract (план §2.3 пп.1-4): _run_render_pipeline with
``cutlist_override``, the ``metadata`` opt toggle, per-clip burn ASS naming and
the face-crop clip range — plus a resolve() unit locking censor clipping at the
clip boundary.

Covers the plan's F2 expectations 1-4 and 6 (5 lives in test_facecrop.py):
 1. cutlist_override renders against the PASSED cutlist; the session cutlist is
    not mutated (identity + deep snapshot compared before/after);
 2. without the override the behavior is the legacy one (render receives the
    session's own cutlist object);
 3. opts {"metadata": false} -> cfg.metadata.enabled False, the LLM metadata
    generator is NOT called and metadata.txt is NOT overwritten; without the
    key the prior (config-default) behavior holds;
 4. burn-in ASS subs are built from the override cutlist: words outside
    [start, end] are absent and the first cue starts at ~0.0; per-clip ASS file
    name (план §2.3.4) while the regular render keeps work_dir/burn.ass;
 6. a censor segment straddling the clip boundary is clipped by resolve() to
    its surviving part (boundary REMOVEs = exactly how clip cutlists are built).
No ffmpeg, no real render: vpipe.render.render is monkeypatched to a recorder.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import serve
from vpipe.config import ProfanityLists, load_config
from vpipe.cutlist import resolve
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, TYPE_MANUAL,
                          TYPE_PAUSE, TYPE_PROFANITY, CutList, CutSegment,
                          Segment, Transcript, Word)

_SILENT = lambda *a, **k: None  # noqa: E731


# --- fixtures -----------------------------------------------------------------
def _mk_session(tmp_path, *, duration: float = 20.0, cuts=()):
    """Minimal Session stand-in carrying exactly what the pipeline touches."""
    n = int(duration)
    words = [Word(f"сл{i:02d}", i + 0.1, i + 0.9) for i in range(n)]
    tr = Transcript(language="ru", duration=duration, model="t", audio_hash="h",
                    segments=[Segment(0.0, duration,
                                      " ".join(w.word for w in words), words)])
    cl = CutList(source="fake.mp4", duration=duration, segments=list(cuts))
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


def _clip_cutlist(cl: CutList, start: float, end: float, i: int = 0) -> CutList:
    """Clip cutlist exactly as /api/clips/render builds it (план §2.4):
    a copy of the live cuts + boundary REMOVEs around [start, end]."""
    clip = CutList(source=cl.source, duration=cl.duration,
                   segments=[copy.copy(seg) for seg in cl.segments])
    if start > 0:
        clip.segments.append(CutSegment(
            id=f"clipA{i}", start=0.0, end=start, type=TYPE_MANUAL,
            action=ACTION_REMOVE, enabled=True))
    if end < cl.duration:
        clip.segments.append(CutSegment(
            id=f"clipB{i}", start=end, end=cl.duration, type=TYPE_MANUAL,
            action=ACTION_REMOVE, enabled=True))
    return clip


def _patch_render(monkeypatch):
    """Replace vpipe.render.render with a recorder; returns the call list."""
    calls: list[dict] = []

    def fake_render(ff, media, cl, cfg, out, work_dir, *, on_progress=None,
                    log=None, scale_h=None, fps=None, ass_path=None,
                    crop_filter=None):
        calls.append({"cl": cl, "out": out, "ass_path": ass_path,
                      "crop_filter": crop_filter, "scale_h": scale_h,
                      "fps": fps})
        return {"out": str(out), "encoder": "fake"}

    monkeypatch.setattr(serve.render_mod, "render", fake_render)
    return calls


_BASE_OPTS = {"subtitles": False, "chapters": False, "metadata": False}


# --- 1. cutlist_override: render the passed cutlist, session untouched ---------
def test_cutlist_override_renders_passed_cutlist_session_untouched(monkeypatch,
                                                                   tmp_path):
    cut = CutSegment(id="p1", start=7.0, end=8.0, type=TYPE_PAUSE,
                     action=ACTION_REMOVE, enabled=True)
    s = _mk_session(tmp_path, cuts=[cut])
    calls = _patch_render(monkeypatch)
    clip_cl = _clip_cutlist(s.cutlist, 5.0, 15.0)
    before_obj = s.cutlist
    before_snap = json.dumps(s.cutlist.to_dict(), sort_keys=True)

    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(s, dict(_BASE_OPTS))
    res = serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                                     _SILENT, _SILENT,
                                     cutlist_override=clip_cl)
    # render() got EXACTLY the override cutlist, not the session's
    assert len(calls) == 1
    assert calls[0]["cl"] is clip_cl
    # session not mutated: same object, same content, boundary cuts NOT added
    assert s.cutlist is before_obj
    assert json.dumps(s.cutlist.to_dict(), sort_keys=True) == before_snap
    assert len(s.cutlist.segments) == 1
    # Timeline built from the override: 20с − [0,5]−[7,8]−[15,20] = 9.0с
    assert res["new_duration"] == 9.0
    assert res["old_duration"] == 20.0
    assert res["succeeded"]["render"] is True


# --- 2. no override: legacy path bit-for-bit (session cutlist used) ------------
def test_without_override_renders_session_cutlist(monkeypatch, tmp_path):
    cut = CutSegment(id="p1", start=7.0, end=8.0, type=TYPE_PAUSE,
                     action=ACTION_REMOVE, enabled=True)
    s = _mk_session(tmp_path, cuts=[cut])
    calls = _patch_render(monkeypatch)
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(s, dict(_BASE_OPTS))
    res = serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                                     _SILENT, _SILENT)
    assert calls[0]["cl"] is s.cutlist
    assert res["new_duration"] == 19.0      # 20 − 1с паузы


# --- 3. opts "metadata": resolve toggle + pipeline behavior --------------------
def test_resolve_render_opts_metadata_toggle(tmp_path):
    s = _mk_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(s, {"metadata": False})
    assert cfg.metadata.enabled is False
    cfg, *_ = serve._resolve_render_opts(s, {"metadata": True})
    assert cfg.metadata.enabled is True
    # ключа нет — прежнее поведение: значение из конфига (дефолт True)
    cfg, *_ = serve._resolve_render_opts(s, {})
    assert cfg.metadata.enabled is s.cfg.metadata.enabled is True


def test_metadata_false_skips_llm_and_preserves_metadata_txt(monkeypatch,
                                                             tmp_path):
    s = _mk_session(tmp_path)
    s.llm = object()                       # LLM «доступен» — но звать его нельзя
    _patch_render(monkeypatch)
    meta_calls: list = []
    monkeypatch.setattr(serve.metadata_mod, "generate",
                        lambda *a, **k: meta_calls.append(a) or {})
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(
        s, {"subtitles": False, "chapters": False, "metadata": False})
    sentinel = "TITLE:\nдо клипов\n"
    (out_dir / "metadata.txt").write_text(sentinel, encoding="utf-8")

    res = serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                                     _SILENT, _SILENT)
    assert meta_calls == []                                  # LLM не вызван
    assert (out_dir / "metadata.txt").read_text(encoding="utf-8") == sentinel
    assert res["metadata_path"] is None
    assert res["succeeded"]["metadata"] is True              # «выключено» ≠ «упало»


def test_metadata_without_key_keeps_prior_behavior(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    s.llm = object()
    _patch_render(monkeypatch)
    meta_calls: list = []

    def fake_generate(*a, **k):
        meta_calls.append((a, k))
        return {"title": "Т", "hook": "Х", "description": "Д", "tags": ["а"]}

    monkeypatch.setattr(serve.metadata_mod, "generate", fake_generate)
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(
        s, {"subtitles": False, "chapters": False})
    res = serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                                     _SILENT, _SILENT)
    assert len(meta_calls) == 1                  # дефолт конфига: метаданные идут
    assert res["metadata"] == "Т"
    assert (out_dir / "metadata.txt").exists()


# --- 4. burn ASS from the override cutlist + per-clip file name ----------------
def test_burn_ass_built_from_override_cutlist(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch)
    recorded: dict = {}
    real_write_ass = serve.subs_mod.write_ass

    def rec_write_ass(cues, path, style, **kw):
        recorded["cues"] = cues
        recorded["path"] = Path(path)
        return real_write_ass(cues, path, style, **kw)

    monkeypatch.setattr(serve.subs_mod, "write_ass", rec_write_ass)
    opts = {"burn_subtitles": True, "burn_style": {"karaoke": True},
            "filename": "fake_clip01", **_BASE_OPTS}
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(s, opts)
    clip_cl = _clip_cutlist(s.cutlist, 5.0, 15.0)
    res = serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                                     _SILENT, _SILENT,
                                     cutlist_override=clip_cl)
    # слова вне [5,15] отсутствуют; внутренние — на месте
    text = " ".join(c.text for c in recorded["cues"])
    assert "сл04" not in text and "сл15" not in text and "сл19" not in text
    assert "сл05" in text and "сл14" in text
    # первый кью начинается ~0.0 (клип сдвинут к нулю), всё в пределах клипа
    assert recorded["cues"][0].start < 0.2
    assert all(c.end <= 10.0 + 0.05 for c in recorded["cues"])
    # per-clip имя (план §2.3.4): work_dir/burn_<base>.ass, файл реально записан
    assert recorded["path"].name == "burn_fake_clip01.ass"
    assert recorded["path"].parent == Path(s.work_dir)
    assert recorded["path"].is_file()
    assert calls[0]["ass_path"] == str(recorded["path"])
    assert res["burned_subtitles"] is True


def test_burn_ass_name_without_override_stays_burn_ass(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch)
    opts = {"burn_subtitles": True, **_BASE_OPTS}
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(s, opts)
    serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                               _SILENT, _SILENT)
    assert calls[0]["ass_path"] == str(Path(s.work_dir) / "burn.ass")


# --- face-crop gets the clip range with an override (план §2.4) ----------------
def test_facecrop_receives_clip_range_with_override(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    _patch_render(monkeypatch)
    seen: list[tuple] = []

    def fake_detect(path, ff=None, duration=0.0, *, samples=12, start=0.0,
                    end=None, log=print):
        seen.append((start, end, duration))
        return 0.5

    monkeypatch.setattr(serve.facecrop_mod, "detect_center", fake_detect)
    opts = {"vertical": True, "vertical_center": "auto", **_BASE_OPTS}
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(s, opts)
    clip_cl = _clip_cutlist(s.cutlist, 5.0, 15.0)
    serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                               _SILENT, _SILENT, cutlist_override=clip_cl)
    assert seen == [(5.0, 15.0, 20.0)]
    # без override — дефолты detect_center (старое поведение по всему файлу)
    serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                               _SILENT, _SILENT)
    assert seen[1] == (0.0, None, 20.0)


# --- 6. resolve(): censor straddling the clip boundary is clipped --------------
def test_resolve_clips_censor_to_surviving_part_at_clip_boundary():
    # Клип [5,15] из 20с = граничные REMOVE [0,5] и [15,20] — ровно как строит
    # их /api/clips/render. Цензор, пересекающий границу клипа, обрезается к
    # выжившей части; целиком вне клипа — выпадает; внутри — нетронут.
    c_inside = CutSegment(id="c4", start=8.0, end=9.0, type=TYPE_PROFANITY,
                          action=ACTION_CENSOR, enabled=True)
    cl = CutList(source="x", duration=20.0, segments=[
        CutSegment(id="A", start=0.0, end=5.0, type=TYPE_MANUAL,
                   action=ACTION_REMOVE, enabled=True),
        CutSegment(id="B", start=15.0, end=20.0, type=TYPE_MANUAL,
                   action=ACTION_REMOVE, enabled=True),
        CutSegment(id="c1", start=4.0, end=6.0, type=TYPE_PROFANITY,
                   action=ACTION_CENSOR, enabled=True),     # через левую границу
        CutSegment(id="c2", start=14.0, end=16.0, type=TYPE_PROFANITY,
                   action=ACTION_CENSOR, enabled=True),     # через правую границу
        CutSegment(id="c3", start=16.0, end=18.0, type=TYPE_PROFANITY,
                   action=ACTION_CENSOR, enabled=True),     # целиком вне клипа
        c_inside,                                           # целиком внутри
    ])
    removed, censors = resolve(cl)
    assert removed == [(0.0, 5.0), (15.0, 20.0)]
    assert sorted((c.start, c.end) for c in censors) == [
        (5.0, 6.0), (8.0, 9.0), (14.0, 15.0)]
    # нетронутый цензор возвращается ИСХОДНЫМ объектом (контракт resolve)
    assert any(c is c_inside for c in censors)
    # обрезанные сохраняют id родителя (один выживший кусок -> без суффикса)
    assert {c.id for c in censors} == {"c1", "c2", "c4"}
