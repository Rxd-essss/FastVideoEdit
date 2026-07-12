# -*- coding: utf-8 -*-
"""Wave-4a #62(a): chapters/metadata sidecars are stem-keyed.

With sidecar_base=out/'vidA' the files land at vidA.chapters.txt /
vidA.metadata.txt — never the old fixed out_dir/chapters.txt / metadata.txt that
two videos rendered into one out_dir would overwrite. No ffmpeg: render + the
chapters/metadata generators are stubbed.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import serve
from vpipe.config import ProfanityLists, load_config
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import CutList, Segment, Transcript, Word

_SILENT = lambda *a, **k: None  # noqa: E731


def _mk_session(tmp_path, duration=20.0):
    n = int(duration)
    words = [Word(f"сл{i:02d}", i + 0.1, i + 0.9) for i in range(n)]
    tr = Transcript(language="ru", duration=duration, model="t", audio_hash="h",
                    segments=[Segment(0.0, duration,
                                      " ".join(w.word for w in words), words)])
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        cfg=load_config("config.yaml"), inp=Path("fake.mp4"),
        media=SimpleNamespace(path="fake.mp4", duration=duration,
                              width=1920, height=1080, fps=30.0),
        ff=None, work_dir=work, out_dir=out,
        matcher=ProfanityMatcher(ProfanityLists(roots=[], allow=[])),
        llm=object(), transcript=tr,
        cutlist=CutList(source="fake.mp4", duration=duration, segments=[]))


def _patch_render(monkeypatch):
    def fake_render(ff, media, cl, cfg, out, work_dir, *, on_progress=None,
                    log=None, scale_h=None, fps=None, ass_path=None,
                    crop_filter=None, edge_fade=0.0, **kw):
        return {"out": str(out), "encoder": "fake"}
    monkeypatch.setattr(serve.render_mod, "render", fake_render)


def test_chapters_metadata_stem_keyed(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    _patch_render(monkeypatch)

    def fake_chapters(tr, removed, ch_cfg, path, **kw):
        Path(path).write_text("00:00 Intro\n", encoding="utf-8")
        return {"path": str(path), "chapters": 1}

    monkeypatch.setattr(serve.chapters_mod, "generate", fake_chapters)
    monkeypatch.setattr(serve.metadata_mod, "generate",
                        lambda *a, **k: {"title": "Т", "hook": "Х",
                                         "description": "Д", "tags": ["а"]})
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(
        s, {"subtitles": False, "chapters": True, "metadata": True})
    res = serve._run_render_pipeline(s, cfg, scale_h, fps, out_dir, base,
                                     _SILENT, _SILENT,
                                     sidecar_base=out_dir / "vidA")
    assert (out_dir / "vidA.chapters.txt").exists()
    assert (out_dir / "vidA.metadata.txt").exists()
    assert not (out_dir / "chapters.txt").exists()
    assert not (out_dir / "metadata.txt").exists()
    assert res["chapters"] == str(out_dir / "vidA.chapters.txt")
    assert res["metadata_path"] == str((out_dir / "vidA.metadata.txt").resolve())
