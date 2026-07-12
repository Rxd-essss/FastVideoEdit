"""Perf #95 — _run_schematic_candidates must route through the idempotent
render_schematic_cached: a candidate whose PNG already exists in cache is served
WITHOUT relaunching headless Chrome (render_schematic). No ffmpeg/Chrome.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import serve
from vpipe.codegfx import render as cg

_SILENT = lambda *a, **k: None  # noqa: E731
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64        # valid-looking fake PNG


def _cfg_cg():
    return SimpleNamespace(codegfx_enabled=True, codegfx_chrome="",
                           codegfx_style="minimal", codegfx_size="1920x1080")


def _session(cfg_cg):
    return SimpleNamespace(
        cfg=SimpleNamespace(render=SimpleNamespace(codegfx=cfg_cg)),
        stage=lambda *a, **k: None)


def test_schematic_cache_hit_skips_chrome(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)   # redirect cache/codegfx
    fields = {"stats": [{"value": "1"}]}
    key = cg.cache_key("stat", fields, "minimal", 1920, 1080)
    png = tmp_path / f"{key}.png"
    png.write_bytes(_PNG)                            # PNG already cached

    def boom(*a, **k):
        raise AssertionError("render_schematic (Chrome) must NOT run on cache hit")

    monkeypatch.setattr(cg, "render_schematic", boom)

    cand = {"source": "schematic", "intent": "stat", "style": "minimal",
            "fields": fields}
    item = SimpleNamespace(payload=SimpleNamespace(candidates=[cand]))
    made = serve._run_schematic_candidates(_session(_cfg_cg()), [item], _SILENT)
    assert made == 1
    assert Path(cand["asset_path"]).resolve() == png.resolve()
    assert cand["preview"] == cand["asset_path"]


def test_schematic_routes_through_cached_wrapper(monkeypatch, tmp_path):
    # Serve must call the CACHED wrapper (not the raw render_schematic).
    seen = {"n": 0}

    def fake_cached(intent, fields, style, cfg=None, *, log=None):
        seen["n"] += 1
        return str(tmp_path / "out.png")

    monkeypatch.setattr(cg, "render_schematic_cached", fake_cached)
    cand = {"source": "schematic", "intent": "list", "style": "minimal",
            "fields": {"items": [{"text": "a"}]}}
    item = SimpleNamespace(payload=SimpleNamespace(candidates=[cand]))
    made = serve._run_schematic_candidates(_session(_cfg_cg()), [item], _SILENT)
    assert seen["n"] == 1 and made == 1
    assert cand["asset_path"] == str(tmp_path / "out.png")
