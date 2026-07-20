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


# Emoji rasterisation needs a COLOUR-emoji font (Segoe UI Emoji on desktop
# Windows). CI runners -- Linux and Windows Server alike -- ship none, so those
# tests skip there while still running on the dev's real machine. Production
# already degrades cleanly (emoji_png_path -> None -> overlay dropped).
def _has_emoji_font() -> bool:
    from vpipe.enrich import _emoji_font_path
    return _emoji_font_path() is not None


requires_emoji_font = pytest.mark.skipif(
    not _has_emoji_font(),
    reason="no colour-emoji font on this host (CI) — glyph rasterisation unavailable")
