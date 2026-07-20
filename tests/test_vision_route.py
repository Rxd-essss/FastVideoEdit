# -*- coding: utf-8 -*-
"""Vision-route stage (montage «variant C»): the VLM gate that vetoes/re-routes
«слоп» after looking at real frames. Hermetic — the VLM client and frame
sampling are stubbed, so no Ollama/ffmpeg is touched.

Covers:
  * veto: insert=false (and insert=true+kind=none) disables the item + note;
  * re-route: insert=true with a disagreeing kind switches payload.selected to a
    matching candidate — and NEVER invents one (leaves it if none matches);
  * re-route no-op when the selection already matches -> keep;
  * list_card/animation are veto-only (no candidate switching);
  * min_confidence gate: a low-confidence verdict leaves the item untouched;
  * best-effort: no frames -> skipped (client NOT called); transport error ->
    error, item untouched; only enabled visual items are judged;
  * serve._run_vision_route is a no-op when disabled (default) / gracefully
    skips when the model is missing;
  * llm._build_payload threads base64 images into the user message.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe import vision_route as vr                       # noqa: E402


# --- helpers -----------------------------------------------------------------
def _img_item(candidates, selected=0, enabled=True, t=10.0, quote="про реестр"):
    pl = SimpleNamespace(candidates=candidates, selected=selected,
                         style_hint="diagram", concept="структура реестра",
                         image_query_en="", asset_kind="generate")
    return SimpleNamespace(type="image", enabled=enabled, quote=quote,
                           t_start=t, status="ok", status_note="", payload=pl)


def _card_item(enabled=True):
    pl = SimpleNamespace(title="Плюсы", items=[SimpleNamespace(text="раз"),
                                               SimpleNamespace(text="два")])
    return SimpleNamespace(type="list_card", enabled=enabled, quote="плюсы",
                           t_start=5.0, status="ok", status_note="", payload=pl)


class FakeVLM:
    """Records calls; returns a scripted verdict (or raises it if it's an Exc)."""
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    def chat_json(self, system, user, schema, keep_alive=None, images=None):
        self.calls.append({"system": system, "user": user, "schema": schema,
                           "keep_alive": keep_alive, "images": images})
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict


def _cfg(**over):
    base = dict(frames=3, frame_size=512, timeout_s=90, reroute=True,
                min_confidence=0)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _no_ffmpeg(monkeypatch):
    """Never shell out to ffmpeg — hand back two dummy base64 frames."""
    monkeypatch.setattr(vr, "_sample_frames",
                        lambda *a, **k: ["ZmFrZQ==", "ZnJhbWU="])


# --- veto --------------------------------------------------------------------
def test_veto_disables_item_with_note():
    it = _img_item([{"source": "schematic", "asset_path": "/x.png"}])
    client = FakeVLM({"scene": "talking_head", "insert": False,
                      "kind": "none", "reason": "лицо, схема ни к чему"})
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg())
    assert it.enabled is False
    assert "vision:" in it.status_note and "talking_head" in it.status_note
    assert stats == {"judged": 1, "veto": 1, "convert": 0, "reroute": 0,
                     "keep": 0, "skipped": 0, "error": 0}


def test_insert_true_but_kind_none_is_a_veto():
    it = _img_item([{"source": "schematic"}])
    client = FakeVLM({"scene": "screen_share", "insert": True, "kind": "none",
                      "reason": "экран уже показывает это"})
    vr.route([it], "v.mp4", "ffmpeg", client, _cfg())
    assert it.enabled is False


# --- re-route ----------------------------------------------------------------
def test_reroute_switches_selected_to_matching_candidate():
    it = _img_item([{"source": "schematic"}, {"source": "diffusion"}], selected=0)
    client = FakeVLM({"scene": "talking_head", "insert": True, "kind": "image",
                      "reason": "лучше фото, а не схема"})
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg())
    assert it.payload.selected == 1                 # switched schematic -> diffusion
    assert it.enabled is True
    assert stats["reroute"] == 1 and stats["veto"] == 0


def test_reroute_noop_when_selection_already_matches():
    it = _img_item([{"source": "schematic"}, {"source": "diffusion"}], selected=0)
    client = FakeVLM({"scene": "code_editor", "insert": True, "kind": "diagram",
                      "reason": "схема уместна"})
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg())
    assert it.payload.selected == 0                 # already a diagram -> unchanged
    assert stats["keep"] == 1 and stats["reroute"] == 0


def test_reroute_never_invents_a_candidate():
    # only a schematic candidate exists; VLM wants an image -> cannot switch,
    # must NOT fabricate one; item stays enabled on the schematic.
    it = _img_item([{"source": "schematic"}], selected=0)
    client = FakeVLM({"scene": "broll", "insert": True, "kind": "image",
                      "reason": "фото было бы лучше"})
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg())
    assert it.payload.selected == 0 and it.enabled is True
    assert stats["keep"] == 1 and stats["reroute"] == 0


def test_reroute_off_only_vetoes():
    it = _img_item([{"source": "schematic"}, {"source": "diffusion"}], selected=0)
    client = FakeVLM({"scene": "talking_head", "insert": True, "kind": "image",
                      "reason": "фото"})
    vr.route([it], "v.mp4", "ffmpeg", client, _cfg(reroute=False))
    assert it.payload.selected == 0                 # no switch when reroute off


# --- cross-type image -> list_card (variant γ) -------------------------------
def _card_factory_stub(calls):
    def factory(item, title, bullets):
        calls.append((title, list(bullets)))
        item.type = "list_card"            # emulate the real swap
        item.payload = SimpleNamespace(title=title, items=bullets)
        return True
    return factory


def test_convert_image_to_list_card_when_vlm_asks():
    it = _img_item([{"source": "schematic"}])
    client = FakeVLM({"scene": "screen_share", "insert": True,
                      "kind": "list_card",
                      "card": {"title": "Windows vs Linux",
                               "items": ["реестр HKEY_*", "/etc файлы",
                                         "winget upgrade --all"]},
                      "reason": "лучше списком"})
    calls = []
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg(),
                     card_factory=_card_factory_stub(calls))
    assert it.type == "list_card"                    # cross-type conversion fired
    assert calls == [("Windows vs Linux",
                      ["реестр HKEY_*", "/etc файлы", "winget upgrade --all"])]
    assert stats["convert"] == 1 and stats["veto"] == 0 and stats["reroute"] == 0


def test_convert_needs_valid_card_else_keeps():
    # kind=list_card but card missing/too few bullets -> no conversion, keep
    it = _img_item([{"source": "schematic"}])
    client = FakeVLM({"scene": "screen_share", "insert": True,
                      "kind": "list_card", "card": {"title": "x", "items": ["one"]},
                      "reason": "..."})
    calls = []
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg(),
                     card_factory=_card_factory_stub(calls))
    assert it.type == "image" and calls == []        # not enough bullets -> untouched
    assert stats["convert"] == 0 and stats["keep"] == 1


def test_convert_skipped_without_factory():
    # no card_factory -> falls back to candidate reroute (none matches) -> keep
    it = _img_item([{"source": "schematic"}])
    client = FakeVLM({"scene": "screen_share", "insert": True,
                      "kind": "list_card",
                      "card": {"title": "t", "items": ["a", "b"]}, "reason": ""})
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg())  # no factory
    assert it.type == "image" and stats["keep"] == 1


def test_valid_card_helper():
    assert vr._valid_card({"card": {"title": "T", "items": ["a", "b"]}}) == \
        ("T", ["a", "b"])
    assert vr._valid_card({"card": {"title": "", "items": ["a", "b"]}}) is None
    assert vr._valid_card({"card": {"title": "T", "items": ["only"]}}) is None
    assert vr._valid_card({"kind": "diagram"}) is None       # no card key
    # caps to 4 bullets
    got = vr._valid_card({"card": {"title": "T",
                                   "items": ["1", "2", "3", "4", "5", "6"]}})
    assert got is not None and len(got[1]) == 4


def test_list_card_is_veto_only():
    it = _card_item()
    client = FakeVLM({"scene": "screen_share", "insert": True,
                      "kind": "list_card", "reason": "ок"})
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg())
    assert it.enabled is True and stats["keep"] == 1


# --- confidence gate ---------------------------------------------------------
def test_low_confidence_leaves_item_untouched():
    it = _img_item([{"source": "schematic"}])
    client = FakeVLM({"scene": "talking_head", "insert": False, "kind": "none",
                      "confidence": 20, "reason": "не уверен"})
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg(min_confidence=50))
    assert it.enabled is True                       # veto suppressed by gate
    assert stats["judged"] == 1 and stats["veto"] == 0


# --- best-effort resilience --------------------------------------------------
def test_no_frames_skips_without_calling_vlm(monkeypatch):
    monkeypatch.setattr(vr, "_sample_frames", lambda *a, **k: [])
    it = _img_item([{"source": "schematic"}])
    client = FakeVLM({"insert": False})             # would veto IF called
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg())
    assert it.enabled is True and client.calls == []
    assert stats["skipped"] == 1 and stats["judged"] == 0


def test_transport_error_leaves_item_untouched():
    it = _img_item([{"source": "schematic"}])
    client = FakeVLM(RuntimeError("ollama down"))
    stats = vr.route([it], "v.mp4", "ffmpeg", client, _cfg())
    assert it.enabled is True and stats["error"] == 1 and stats["judged"] == 0


def test_only_enabled_visual_items_are_judged():
    good = _img_item([{"source": "schematic"}])
    disabled = _img_item([{"source": "schematic"}], enabled=False)
    cta = SimpleNamespace(type="cta_subscribe", enabled=True, quote="", t_start=1.0,
                          status="ok", status_note="", payload=SimpleNamespace())
    client = FakeVLM({"scene": "s", "insert": True, "kind": "diagram", "reason": ""})
    stats = vr.route([good, disabled, cta], "v.mp4", "ffmpeg", client, _cfg())
    assert len(client.calls) == 1                   # only `good`
    assert stats["judged"] == 1


def test_empty_targets_no_client_calls():
    client = FakeVLM({"insert": True})
    stats = vr.route([], "v.mp4", "ffmpeg", client, _cfg())
    assert client.calls == [] and stats["judged"] == 0


def test_progress_callback_is_called():
    it = _img_item([{"source": "schematic"}])
    client = FakeVLM({"scene": "s", "insert": True, "kind": "diagram", "reason": ""})
    seen = []
    vr.route([it], "v.mp4", "ffmpeg", client, _cfg(),
             on_progress=lambda done, total: seen.append((done, total)))
    assert (1, 1) in seen                            # final tick fired


# --- unit: helpers -----------------------------------------------------------
def test_offsets_clamped_1_to_5():
    assert vr._offsets(0) == [0.5]                   # min 1
    assert len(vr._offsets(99)) == 5                 # max 5
    assert vr._offsets(3) == [0.5, -1.0, 2.0]


def test_reroute_image_matches_stock_for_kind_image():
    it = _img_item([{"source": "schematic"}, {"source": "stock"}], selected=0)
    assert vr._reroute_image(it, "image") == "stock"
    assert it.payload.selected == 1


# --- serve wiring ------------------------------------------------------------
def _mini_session(enabled, monkeypatch):
    import serve
    from vpipe.config import load_config
    cfg = load_config("config.yaml")
    cfg.render.vision_route.enabled = enabled
    s = SimpleNamespace(cfg=cfg, inp=Path("in.mp4"),
                        ff=SimpleNamespace(ffmpeg="ffmpeg"),
                        llm=SimpleNamespace(unload=lambda *a, **k: True,
                                            loaded_models=lambda: []),
                        stage=lambda *_a, **_k: None,
                        set_progress=lambda *_a, **_k: None)
    return serve, s


def test_run_vision_route_noop_when_disabled(monkeypatch):
    serve, s = _mini_session(False, monkeypatch)
    items = [_img_item([{"source": "schematic"}])]
    assert serve._run_vision_route(s, items, {}, log=lambda *_: None) is None
    assert items[0].enabled is True                  # untouched


def test_run_vision_route_skips_when_model_missing(monkeypatch):
    serve, s = _mini_session(True, monkeypatch)
    monkeypatch.setattr(serve, "_wait_ollama_unloaded", lambda *a, **k: None)
    fake = SimpleNamespace(available=lambda *a, **k: True,
                           has_model=lambda *a, **k: False)
    monkeypatch.setattr(serve, "get_client", lambda *a, **k: fake)
    items = [_img_item([{"source": "schematic"}])]
    assert serve._run_vision_route(s, items, {}, log=lambda *_: None) is None
    assert items[0].enabled is True                  # not judged, untouched


# --- llm images threading ----------------------------------------------------
def test_build_payload_threads_images_into_user_message():
    from vpipe.config import LlmCfg
    from vpipe.llm import OllamaClient
    c = OllamaClient(LlmCfg(model="qwen3-vl:4b-instruct"))
    p = c._build_payload("sys", "look", {"type": "object"}, 0.0,
                         images=["b64a", "b64b"])
    user = [m for m in p["messages"] if m["role"] == "user"][0]
    assert user["images"] == ["b64a", "b64b"]
    # text-only call carries no images key
    p2 = c._build_payload("sys", "hi", {"type": "object"}, 0.0)
    assert "images" not in [m for m in p2["messages"] if m["role"] == "user"][0]
