"""Perf #96 (UNVERIFIED, opt-in default-off) — _sweep_img_cache. With
cache_img_max_age_days == 0 nothing is evicted (exact current behavior); with a
positive age only old PNGs in the IMAGE caches (enrich_img/codegfx) go, while
fresh images and the transcript/peaks correctness caches are never touched.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import serve


def _cfg(cache: Path, max_age: int):
    return SimpleNamespace(paths=SimpleNamespace(
        cache_dir=str(cache), work_dir=str(cache / "work"),
        out_dir=str(cache / "out"), cache_img_max_age_days=max_age))


def _writer_dirs(monkeypatch, tmp_path: Path):
    """W6 #11: константы ПИСАТЕЛЕЙ -> tmp. Свип теперь метёт реальные каталоги
    imagegen/codegfx — без переадресации тест смёл бы репо-кэш разработчика."""
    from vpipe import imagegen
    from vpipe.codegfx import render as cg_render
    ig = tmp_path / "writers" / "enrich_img"
    cg = tmp_path / "writers" / "codegfx"
    monkeypatch.setattr(imagegen, "CACHE_DIR", ig)
    monkeypatch.setattr(cg_render, "CACHE_DIR", cg)
    return ig, cg


def _build(cache: Path):
    (cache / "enrich_img").mkdir(parents=True)
    (cache / "codegfx").mkdir(parents=True)
    old = cache / "enrich_img" / "old.png"
    fresh = cache / "enrich_img" / "fresh.png"
    cg_old = cache / "codegfx" / "old.png"
    tr = cache / "abc.transcript.json"
    pk = cache / "abc.peaks.json"
    for p in (old, fresh, cg_old):
        p.write_bytes(b"x")
    tr.write_text("{}", encoding="utf-8")
    pk.write_text("{}", encoding="utf-8")
    old_t = time.time() - 40 * 86400
    for p in (old, cg_old, tr, pk):               # everything old except fresh.png
        os.utime(p, (old_t, old_t))
    return old, fresh, cg_old, tr, pk


def test_img_cache_sweep_disabled_by_default(tmp_path):
    cache = tmp_path / "cache"
    old, fresh, cg_old, tr, pk = _build(cache)
    serve._sweep_img_cache(_cfg(cache, max_age=0))   # default -> no eviction
    for p in (old, fresh, cg_old, tr, pk):
        assert p.exists()


def test_img_cache_sweep_opt_in_evicts_only_old_images(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _writer_dirs(monkeypatch, tmp_path)    # писатели -> tmp (не трогаем репо)
    old, fresh, cg_old, tr, pk = _build(cache)
    serve._sweep_img_cache(_cfg(cache, max_age=30))
    assert not old.exists()                          # old enrich_img PNG gone
    assert not cg_old.exists()                       # old codegfx PNG gone
    assert fresh.exists()                            # fresh image kept
    assert tr.exists() and pk.exists()               # correctness caches untouched


def test_img_cache_sweep_covers_writer_roots(tmp_path, monkeypatch):
    """W6 #11 (#96): свип метёт ФАКТИЧЕСКИЕ каталоги писателей (imagegen —
    CWD-relative, codegfx — repo-root), а не только cfg.paths.cache_dir.
    fail-before: с кастомным cache_dir свип был тихим no-op — старые PNG
    писателей оставались навсегда."""
    ig, cg = _writer_dirs(monkeypatch, tmp_path)
    ig.mkdir(parents=True)
    cg.mkdir(parents=True)
    old_ig, old_cg = ig / "old.png", cg / "old.png"
    fresh_ig = ig / "fresh.png"
    for p in (old_ig, old_cg, fresh_ig):
        p.write_bytes(b"x")
    t = time.time() - 40 * 86400
    for p in (old_ig, old_cg):
        os.utime(p, (t, t))
    # cfg смотрит СОВСЕМ в другой (несуществующий) cache_dir
    serve._sweep_img_cache(_cfg(tmp_path / "elsewhere", max_age=30))
    assert not old_ig.exists() and not old_cg.exists()
    assert fresh_ig.exists()
