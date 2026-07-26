"""UI <-> server render-opts contract lock (audit #34, wave 3).

The render-opts contract exists TWICE with no shared schema: web/app.js
``collectRenderOpts()`` (app.js:2786-2831) assembles the payload from ~25 DOM
fields, and serve.py ``_resolve_render_opts`` (serve.py:974-1194) whitelists the
keys server-side and *silently* ignores anything it does not recognise. A key
renamed on one side without the other therefore stops reaching ffmpeg with every
backend test still green. These tests turn that silent drop RED.

What is locked here (pure backend contract; the frontend jsdom/Playwright side
is a separate follow-up, see the docstring at the bottom):

  1. Every top-level key the recorded UI payload emits is a key the server
     actually consumes  (test_every_ui_key_is_a_known_server_key).
  2. The full recorded payload resolves without error and representative
     options actually take effect  (test_recorded_payload_applies).
  3. Server defaults come from config.yaml: absent keys leave the config value
     untouched  (test_absent_keys_keep_config_yaml_defaults).
  4. Static guard: the ``$('#id')`` DOM ids collectRenderOpts reads all exist in
     web/index.html  (test_app_js_field_ids_have_no_orphans).

MAINTENANCE: ``ACCEPTED`` below MUST stay in sync with the whitelist in
_resolve_render_opts (serve.py:974). The recorded fixture
tests/fixtures/render_opts_payload.json MUST be regenerated whenever
collectRenderOpts legitimately gains or renames a field.
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import serve
from vpipe.config import load_config

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures" / "render_opts_payload.json"

# Every top-level key _resolve_render_opts consumes (serve.py:974-1194) plus the
# `formats` key the /api/render handler parses (multi-format C2). Keep in sync
# with the whitelist in serve.py:974 — a key removed there must be removed here.
ACCEPTED = {
    "encoder", "quality", "scale_h", "fps", "audio_bitrate", "censor_method",
    "subtitles", "chapters", "metadata", "formats",
    "vertical", "vertical_target", "vertical_center",
    "denoise", "denoise_strength", "denoise_highpass", "denoise_normalize",
    "denoise_engine", "denoise_deess", "denoise_loudnorm", "loudnorm_mode",
    "cut_fade", "music", "enrich", "out_dir", "filename",
    "burn_subtitles", "burn_style",
}


def _load_payload():
    p = json.loads(FIX.read_text(encoding="utf-8"))
    # Strip the human-readable provenance note; it is documentation, not a key
    # the UI emits, so it must not be checked against the server whitelist.
    p.pop("_note", None)
    return p


def _sess(tmp_path):
    return SimpleNamespace(
        cfg=load_config("config.yaml"),
        inp=Path("fake.mp4"),
        media=SimpleNamespace(duration=20.0, width=1920, height=1080, fps=30.0),
        out_dir=tmp_path / "out",
    )


def test_every_ui_key_is_a_known_server_key():
    """Renaming a key in app.js (or dropping it from the server whitelist)
    without updating the other side turns THIS red instead of silently dropping
    the option before ffmpeg."""
    payload = _load_payload()
    unknown = set(payload) - ACCEPTED
    assert not unknown, f"app.js emits keys the server ignores: {sorted(unknown)}"


def test_recorded_payload_applies(tmp_path):
    """The whole recorded payload resolves without raising, and representative
    options actually reach the config (i.e. were not silently dropped)."""
    payload = _load_payload()
    payload["music"] = {"enabled": False}  # avoid the 400 path needing a real file
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(_sess(tmp_path), payload)

    # spot-check a representative option from each server block
    assert cfg.render.encoder == payload["encoder"]                       # encoder
    assert cfg.render.denoise.enabled is True                            # denoise
    assert cfg.render.denoise.engine == payload["denoise_engine"]        # denoise engine whitelist
    assert cfg.render.denoise.loudnorm_mode == payload["loudnorm_mode"]  # loudnorm mode whitelist
    assert cfg.render.denoise.deess is True                             # mastering de-esser
    assert cfg.render.denoise.loudnorm is True                          # mastering loudnorm
    assert cfg.censor.method == payload["censor_method"]                 # censor
    assert cfg.subtitles.burn.enabled is True                           # burn-in
    assert cfg.render.enrich.enabled is True                            # enrich
    assert abs(cfg.render.cut_fade - payload["cut_fade"]) < 1e-9         # cut seam fade
    assert out_dir == (tmp_path / "out")                                 # blank out_dir -> session
    assert base.name == "клип_финал"  # filename -> stem


def test_absent_keys_keep_config_yaml_defaults(tmp_path):
    """Server defaults come from config.yaml: with an EMPTY payload the keys the
    UI would send but omitted are left at their config.yaml value (not clobbered
    by a stray override)."""
    fresh = load_config("config.yaml").render
    cfg, *_ = serve._resolve_render_opts(_sess(tmp_path), {})
    r = cfg.render
    assert r.encoder == fresh.encoder
    assert r.audio_bitrate == fresh.audio_bitrate
    assert r.denoise.loudnorm_mode == fresh.denoise.loudnorm_mode
    assert r.denoise.engine == fresh.denoise.engine
    assert r.cut_fade == fresh.cut_fade


def test_app_js_field_ids_have_no_orphans():
    """Lightweight static guard: every DOM id collectRenderOpts reads via
    $('#id') exists in web/index.html. Catches an id renamed in the template
    without updating the collector (the field would silently read undefined)."""
    js = (REPO / "web" / "app.js").read_text(encoding="utf-8")
    body = js[js.index("function collectRenderOpts"):js.index("function submitRender")]
    ids = set(re.findall(r"\$\('#(\w+)'\)", body))
    assert ids, "regex found no $('#id') reads — collectRenderOpts shape changed"
    html = (REPO / "web" / "index.html").read_text(encoding="utf-8")
    missing = [i for i in ids if f'id="{i}"' not in html and f"id='{i}'" not in html]
    assert not missing, f"collectRenderOpts reads DOM ids absent from index.html: {missing}"


# Follow-up (not required for this MVP): add package.json + a jsdom/vitest unit
# test that mounts the render modal and asserts collectRenderOpts() output keys
# equal this fixture, plus a Playwright smoke intercepting the /api/render body.
# That closes the loop on the FRONTEND side; these tests lock the SERVER side.
