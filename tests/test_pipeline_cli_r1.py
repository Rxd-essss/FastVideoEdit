# -*- coding: utf-8 -*-
"""W6 #7 (R1, CLI-путь): pipeline.py обязан строить сабы/главы/итог от того же
ЭФФЕКТИВНОГО (frame-snap + сливер-фильтр) катлиста, что реально исполняет
render() — как это уже делает serve.py.

fail-before: pipeline скармливал subs/chapters СЫРОЙ resolve()-набор, тогда как
render() снапит каждую границу к сетке кадров — sidecar-сабы CLI дрейфовали от
готового видео. Тяжёлые стадии (probe/transcribe/render/LLM) фейкуются;
effective_cut — НАСТОЯЩИЙ.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline                                            # noqa: E402
from vpipe.config import ProfanityLists, load_config       # noqa: E402
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE, CutList,   # noqa: E402
                          CutSegment, Segment, Transcript, Word)


def test_cli_subs_built_from_effective_cut(monkeypatch, tmp_path):
    cfg = load_config("config.yaml")
    cfg.paths.out_dir = str(tmp_path / "out")
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.work_dir = str(tmp_path / "work")
    cfg.subtitles.enabled = True
    cfg.chapters.enabled = False

    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"x")

    media = SimpleNamespace(duration=100.0, fps=25.0, width=1920, height=1080,
                            has_audio=True, describe=lambda: "1920x1080")
    monkeypatch.setattr(pipeline, "load_config", lambda p: cfg)
    monkeypatch.setattr(pipeline, "load_fillers", lambda p: None)
    monkeypatch.setattr(pipeline, "load_profanity", lambda p: ProfanityLists())
    monkeypatch.setattr(pipeline, "FFmpeg", lambda c: None)
    monkeypatch.setattr(pipeline, "probe_media", lambda ff, p: media)
    monkeypatch.setattr(pipeline, "hash_input", lambda p: "h" * 8)
    monkeypatch.setattr(pipeline, "get_client", lambda c: None)

    cache_dir = Path(cfg.paths.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    Transcript(language="ru", duration=100.0, model="t", audio_hash="h" * 8,
               segments=[Segment(0.0, 100.0, "a", [Word("a", 0.5, 1.0)])]
               ).save(cache_dir / f"{'h' * 8}.transcript.json")

    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Границы выреза СБИТЫ с сетки 25 fps: снап двигает их на 10.0 / 20.0.
    CutList(source=str(inp), duration=100.0, segments=[
        CutSegment(id="p0", start=10.017, end=20.013, type=TYPE_PAUSE,
                   action=ACTION_REMOVE, enabled=True)]
            ).save_json(out_dir / "in.cutlist.json")

    monkeypatch.setattr(pipeline.render_mod, "render",
                        lambda *a, **k: {"out": "out.mp4"})
    seen = {}

    def _subs(transcript, removed, *a, **k):
        seen["removed"] = list(removed)
        return {"cues": 0}

    monkeypatch.setattr(pipeline.subs_mod, "generate", _subs)
    monkeypatch.setattr(pipeline.summary_mod, "summarize",
                        lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv",
                        ["pipeline.py", str(inp), "--apply",
                         "--config", "config.yaml"])
    assert pipeline.main() == 0
    # ЭФФЕКТИВНЫЙ (frame-snapped) набор, а не сырой resolve():
    # fail-before здесь было [(10.017, 20.013)].
    assert seen["removed"] == [(pytest.approx(10.0), pytest.approx(20.0))]
