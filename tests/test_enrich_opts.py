# -*- coding: utf-8 -*-
"""P2 — настройки авто-обогащения (ENRICH_PLAN §5, cache/enrich_ui.json).

Покрывает:
  * _sanitize_enrich_opts: канонический вид с дефолтами, whitelist density/
    image_source, фильтрация types, strict=True → 400 на неверные типы и
    значения вне белых списков, strict=False → тихая замена дефолтом
    (битый файл не роняет /api/state), неизвестные ключи игнорируются,
    stocks.enabled принудительно False (Tier 2 — строго opt-in OFF, §4);
  * персист: атомарная запись + roundtrip, отсутствующий/битый/чужой json →
    None, неудача записи — best-effort (без исключения);
  * GET /api/state (ветка без сессии): enrich_opts null до настройки,
    канонический dict после.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                            # noqa: E402
from vpipe.config import load_config                    # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    c = load_config("config.yaml")
    c.paths.cache_dir = str(tmp_path / "cache")
    c.paths.out_dir = str(tmp_path / "out")
    c.paths.work_dir = str(tmp_path / "work")
    return c


@pytest.fixture()
def client(cfg, monkeypatch):
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", cfg.paths.out_dir)
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


# --- sanitize: канонический вид -------------------------------------------------
def test_sanitize_empty_gives_defaults():
    o = serve._sanitize_enrich_opts({})
    assert o == {"types": {"image": True, "animation": True,
                           "list_card": True, "cta": True},
                 "density": "normal", "image_source": "auto",
                 "user_folder": "", "stocks": {"enabled": False}}


def test_sanitize_passthrough_in_whitelist():
    o = serve._sanitize_enrich_opts({
        "types": {"image": False, "cta": False},
        "density": "aggressive", "image_source": "user_folder",
        "user_folder": "  D:/assets  "})
    assert o["types"] == {"image": False, "animation": True,
                          "list_card": True, "cta": False}
    assert o["density"] == "aggressive"
    assert o["image_source"] == "user_folder"
    assert o["user_folder"] == "D:/assets"          # trim


def test_sanitize_unknown_keys_ignored():
    o = serve._sanitize_enrich_opts({"чужое": 1, "types": {"чужой_тип": True}})
    assert "чужое" not in o
    assert set(o["types"]) == {"image", "animation", "list_card", "cta"}


def test_sanitize_stocks_forced_off_even_if_true():
    # Tier 2 (стоки) — v1.1, строго opt-in OFF: персист не имеет права хранить
    # заранее взведённый облачный тумблер.
    o = serve._sanitize_enrich_opts({"stocks": {"enabled": True}})
    assert o["stocks"] == {"enabled": False}
    o = serve._sanitize_enrich_opts({"stocks": {"enabled": True}},
                                    strict=False)
    assert o["stocks"] == {"enabled": False}


@pytest.mark.parametrize("raw", [
    "не объект",
    {"types": "all"},
    {"types": {"image": "да"}},
    {"types": {"image": 1}},                     # int — не bool
    {"density": "turbo"},
    {"density": 5},
    {"image_source": "google"},
    {"user_folder": 42},
    {"stocks": "on"},
])
def test_sanitize_strict_raises_400(raw):
    with pytest.raises(HTTPException) as e:
        serve._sanitize_enrich_opts(raw, strict=True)
    assert e.value.status_code == 400


@pytest.mark.parametrize("raw", [
    "не объект",
    {"types": "all"},
    {"types": {"image": "да"}},
    {"density": "turbo"},
    {"image_source": "google"},
    {"user_folder": 42},
    {"stocks": "on"},
])
def test_sanitize_nonstrict_falls_back_to_defaults(raw):
    o = serve._sanitize_enrich_opts(raw, strict=False)
    assert o == serve._default_enrich_opts()     # мусор тихо заменён дефолтами


def test_sanitize_nonstrict_keeps_valid_parts():
    o = serve._sanitize_enrich_opts(
        {"density": "min", "types": {"image": False, "cta": "мусор"}},
        strict=False)
    assert o["density"] == "min"                 # валидное сохранено
    assert o["types"]["image"] is False
    assert o["types"]["cta"] is True             # битое — дефолт


# --- персист ---------------------------------------------------------------------
def test_write_read_roundtrip_atomic(client, cfg):
    opts = serve._sanitize_enrich_opts({"density": "min",
                                        "types": {"image": False}})
    serve._write_enrich_opts(opts)
    p = Path(cfg.paths.cache_dir) / "enrich_ui.json"
    assert p.exists()
    assert not p.with_suffix(".json.tmp").exists()   # атомарно
    assert serve._read_enrich_opts() == opts


def test_read_missing_file_none(client):
    assert serve._read_enrich_opts() is None


@pytest.mark.parametrize("content", ["{мусор", "[]", "42", '"строка"'])
def test_read_corrupt_or_wrong_shape_none(client, cfg, content):
    p = Path(cfg.paths.cache_dir) / "enrich_ui.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    assert serve._read_enrich_opts() is None


def test_read_hand_edited_garbage_values_sanitized(client, cfg):
    # руками вписали мусор → тихая замена дефолтом, никаких 500 в /api/state
    p = Path(cfg.paths.cache_dir) / "enrich_ui.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"density": "ultra", "types": {"image": False},
                             "stocks": {"enabled": True}}),
                 encoding="utf-8")
    o = serve._read_enrich_opts()
    assert o["density"] == "normal"
    assert o["types"]["image"] is False
    assert o["stocks"] == {"enabled": False}


def test_write_failure_is_best_effort(client, monkeypatch):
    real = serve.os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(serve.os, "replace", boom)
    try:
        serve._write_enrich_opts(serve._default_enrich_opts())   # не бросает
    finally:
        monkeypatch.setattr(serve.os, "replace", real)


# --- GET /api/state (ветка без сессии) --------------------------------------------
def test_state_no_session_enrich_opts_null_then_dict(client):
    j = client.get("/api/state").json()
    assert j["no_session"] is True
    assert j["enrich_opts"] is None              # не настраивали = дефолты

    serve._write_enrich_opts(
        serve._sanitize_enrich_opts({"density": "aggressive"}))
    j = client.get("/api/state").json()
    assert j["enrich_opts"]["density"] == "aggressive"
    assert j["enrich_opts"]["stocks"] == {"enabled": False}
