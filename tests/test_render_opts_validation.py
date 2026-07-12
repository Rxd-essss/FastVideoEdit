# -*- coding: utf-8 -*-
"""#51 — fail-fast validation of non-UI render-opts in _resolve_render_opts
(queue.json / curl reach it unvalidated): reject JSON bool quality/fps, garbage
audio_bitrate, and round odd scale_h down to even (yuv420p). Drives the shared
helper directly; no ffmpeg. The web UI only sends valid presets, so interactive
renders are unchanged — these lock the non-UI holes.
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import serve
from vpipe.config import load_config


def _mk_session(tmp_path):
    """Minimal stand-in carrying exactly what _resolve_render_opts touches."""
    cfg = load_config("config.yaml")
    media = SimpleNamespace(height=1080, fps=30.0)
    return SimpleNamespace(cfg=cfg, media=media, out_dir=tmp_path / "out",
                           inp=SimpleNamespace(stem="clip"))


# --- (#4) quality: JSON bool must not be read as QP 1 (near-lossless) ----------
def test_quality_true_ignored(tmp_path):
    s = _mk_session(tmp_path)
    default_qp = s.cfg.render.nvenc.qp
    cfg, *_ = serve._resolve_render_opts(s, {"quality": True})
    assert cfg.render.nvenc.qp == default_qp     # НЕ 1
    assert cfg.render.nvenc.qp != 1


def test_quality_int_still_applies(tmp_path):
    s = _mk_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(s, {"quality": 20})
    assert cfg.render.nvenc.qp == 20 and cfg.render.x264.crf == 20


# --- (#4) fps: JSON true/false is not an fps -----------------------------------
def test_fps_true_400(tmp_path):
    s = _mk_session(tmp_path)
    with pytest.raises(HTTPException) as ei:
        serve._resolve_render_opts(s, {"fps": True})
    assert ei.value.status_code == 400


def test_fps_false_is_source(tmp_path):
    s = _mk_session(tmp_path)
    _cfg, _scale_h, fps, *_ = serve._resolve_render_opts(s, {"fps": False})
    assert fps is None                           # False -> source fps


# --- (#6) audio_bitrate: whitelist «Nk» ----------------------------------------
def test_audio_bitrate_garbage_400(tmp_path):
    s = _mk_session(tmp_path)
    for bad in ("fast", "-320k", "320"):
        with pytest.raises(HTTPException) as ei:
            serve._resolve_render_opts(s, {"audio_bitrate": bad})
        assert ei.value.status_code == 400


def test_audio_bitrate_valid(tmp_path):
    s = _mk_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(s, {"audio_bitrate": "192k"})
    assert cfg.render.audio_bitrate == "192k"


# --- (#5) scale_h: odd height rounded down to even -----------------------------
def test_scale_h_odd_rounded(tmp_path):
    s = _mk_session(tmp_path)
    _cfg, scale_h, *_ = serve._resolve_render_opts(s, {"scale_h": 735})
    assert scale_h == 734


def test_scale_h_even_unchanged(tmp_path):
    s = _mk_session(tmp_path)
    _cfg, scale_h, *_ = serve._resolve_render_opts(s, {"scale_h": 720})
    assert scale_h == 720
