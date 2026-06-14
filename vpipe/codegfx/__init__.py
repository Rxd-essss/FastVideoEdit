# -*- coding: utf-8 -*-
"""Движок код-графики «Монтаж V2» (MONTAGE_V2_PLAN §2) — рабочая лошадь.

intent+fields+style → crisp PNG с альфой через системный headless-Chrome
(HTML→PNG, cut-out под talking-head). CPU-only, 0 VRAM, zero-upload.

Публичный контракт (зафиксирован в ``vpipe/enrich.py`` «КОНТРАКТ КОД-ГРАФИКИ»):
    from vpipe.codegfx import render_schematic
"""
from __future__ import annotations

from .render import (
    render_schematic,
    render_schematic_cached,
    resolve_chrome,
    cache_key,
)
from .validate import (
    build_payload,
    normalize_style,
    INTENTS,
    STYLES,
    STYLE_DEFAULT,
)
from .to_ass import RevealStep, reveal_steps, block_count

__all__ = [
    "render_schematic",
    "render_schematic_cached",
    "resolve_chrome",
    "cache_key",
    "build_payload",
    "normalize_style",
    "INTENTS",
    "STYLES",
    "STYLE_DEFAULT",
    "RevealStep",
    "reveal_steps",
    "block_count",
]
