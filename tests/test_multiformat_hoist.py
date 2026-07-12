"""Perf #85 — the multi-format render must detect the face-center ONCE and feed
it back to every cropped format, not re-scan per format. Behavior-preserving:
the explicitly-fed center yields byte-identical crop_filters. No ffmpeg —
vpipe.render.render is a recorder, facecrop_mod.detect_center a counter.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import serve
import vpipe.facecrop as fc
from vpipe.config import ProfanityLists, load_config
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import CutList, Segment, Transcript, Word

_SILENT = lambda *a, **k: None  # noqa: E731
_BASE_OPTS = {"subtitles": False, "chapters": False, "metadata": False}


def _mk_session(tmp_path, *, width=1920, height=1080, duration=20.0):
    n = int(duration)
    words = [Word(f"сл{i:02d}", i + 0.1, i + 0.9) for i in range(n)]
    tr = Transcript(language="ru", duration=duration, model="t", audio_hash="h",
                    segments=[Segment(0.0, duration,
                                      " ".join(w.word for w in words), words)])
    cl = CutList(source="fake.mp4", duration=duration, segments=[])
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        cfg=load_config("config.yaml"), inp=Path("fake.mp4"),
        media=SimpleNamespace(path="fake.mp4", duration=duration,
                              width=width, height=height, fps=30.0),
        ff=None, work_dir=work, out_dir=tmp_path / "out",
        matcher=ProfanityMatcher(ProfanityLists(roots=[], allow=[])),
        llm=None, transcript=tr, cutlist=cl)


def _patch_render(monkeypatch):
    calls: list[dict] = []

    def fake_render(ff, media, cl, cfg, out, work_dir, *, on_progress=None,
                    log=None, scale_h=None, fps=None, ass_path=None,
                    crop_filter=None, edge_fade=0.0, **kw):
        calls.append({"out": Path(out), "crop_filter": crop_filter})
        if on_progress:
            on_progress(1.0)
        return {"out": str(out), "encoder": "fake"}

    monkeypatch.setattr(serve.render_mod, "render", fake_render)
    return calls


def test_facecrop_detected_once_across_formats(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)                     # 1920x1080 (16:9)
    calls = _patch_render(monkeypatch)
    n = {"c": 0}

    def counting_center(*a, **k):
        n["c"] += 1
        return 0.33

    monkeypatch.setattr(serve.facecrop_mod, "detect_center", counting_center)
    serve._render_formats(s, dict(_BASE_OPTS), ["9x16", "1x1", "16x9"],
                          _SILENT, _SILENT)
    # 16x9 from a 16:9 source is a duplicate (skipped); 9x16 and 1x1 render.
    assert n["c"] == 1                            # exactly ONE detection for all formats
    crops = [c["crop_filter"] for c in calls]
    assert len(crops) == 2
    # both crops are built from the SAME center 0.33 (byte-identical with the
    # old per-format auto-detection that returned 0.33 each time).
    assert crops[0] == fc.vertical_filter(1920, 1080, 0.33, (1080, 1920))
    assert crops[1] == fc.vertical_filter(1920, 1080, 0.33, (1080, 1080))


def test_explicit_center_not_hoisted(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    _patch_render(monkeypatch)
    n = {"c": 0}

    def counting_center(*a, **k):
        n["c"] += 1
        return 0.5

    monkeypatch.setattr(serve.facecrop_mod, "detect_center", counting_center)
    serve._render_formats(s, {**_BASE_OPTS, "vertical_center": "0.7"},
                          ["9x16"], _SILENT, _SILENT)
    assert n["c"] == 0                            # explicit center -> no detection
