"""#91: config.yaml must document the cut-quality knobs (cut_fade, min_segment),
and the documented defaults must stay in sync with RenderCfg so the docs can't
silently drift. Pure text/config assertion — no ffmpeg, CI-safe.
"""
from pathlib import Path

from vpipe.config import RenderCfg

_YAML = (Path(__file__).resolve().parents[1] / "config.yaml").read_text(
    encoding="utf-8")


def test_render_yaml_documents_cut_knobs():
    # Even commented, the keys must be present so a user can discover them.
    assert "cut_fade:" in _YAML
    assert "min_segment:" in _YAML


def test_documented_defaults_match_rendercfg():
    rc = RenderCfg()
    assert f"cut_fade: {rc.cut_fade:g}" in _YAML        # 0.015
    assert f"min_segment: {rc.min_segment:g}" in _YAML   # 0.04
