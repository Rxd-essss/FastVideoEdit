"""Shared pytest markers/gates for the FastVideoEdit suite.

The ``ffmpeg`` marker gates the ONE opt-in integration tier that shells out to
the REAL ffmpeg/ffprobe binary (see tests/test_integration_render.py). It is
skipped unless ``FVE_FFMPEG_TESTS=1`` is set AND ffmpeg is discoverable on PATH,
so the default ``pytest`` run -- including every CI job (.github/workflows/ci.yml
installs NO ffmpeg and bans the real binary on purpose) -- never touches a real
binary and stays hermetic.

Run the real tier locally with:

    FVE_FFMPEG_TESTS=1 python -m pytest tests/test_integration_render.py
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ffmpeg: opt-in; runs the REAL ffmpeg binary (skipped unless "
        "FVE_FFMPEG_TESTS=1 and ffmpeg is on PATH)")


# Gate: BOTH the explicit opt-in env var AND an ffmpeg actually on PATH. Absent
# either, the whole tier is skipped -- CI (no env var, no ffmpeg) never runs it.
_RUN_FFMPEG = (os.environ.get("FVE_FFMPEG_TESTS") == "1"
               and shutil.which("ffmpeg") is not None)

requires_ffmpeg = pytest.mark.skipif(
    not _RUN_FFMPEG,
    reason="set FVE_FFMPEG_TESTS=1 with ffmpeg on PATH to run the integration tier")


# Emoji rasterisation needs a COLOUR-emoji glyph to actually render. This probes
# the real capability, not just a font FILE: Linux CI ships no emoji font, and
# Windows Server ships seguiemj.ttf yet Pillow still yields an EMPTY glyph there
# (no usable COLR/CBDT) -- both must skip. The dev's desktop Windows renders it,
# so the tests still run for real there. Prod already degrades (path -> None).
def can_rasterize_emoji() -> bool:
    from vpipe.enrich import emoji_png_path
    try:
        with tempfile.TemporaryDirectory() as d:
            return emoji_png_path("u26a1", Path(d)) is not None
    except Exception:  # noqa: BLE001 — any failure = capability absent
        return False


requires_emoji_font = pytest.mark.skipif(
    not can_rasterize_emoji(),
    reason="colour-emoji glyph does not rasterise on this host (CI) — feature unavailable")
