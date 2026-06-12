# -*- coding: utf-8 -*-
"""B5 — редактируемый словарь филлеров: GET/PUT /api/fillers.

Покрывает:
  * GET: маппинг YAML -> API (words+phrases -> fillers, mumbles -> stretched),
    path в ответе;
  * PUT: валидация (не-список, не-строка, пустые, дубликаты с учётом
    регистра/ё, лимит 500, битый regex в stretched, длина записи) -> 400 и
    файл НЕ тронут;
  * бэкап fillers_ru.yaml.bak перед ПЕРВОЙ записью (с исходным содержимым),
    второй PUT бэкап не перезаписывает;
  * atomic запись: .tmp не остаётся, YAML парсится load_fillers, многословные
    филлеры разъезжаются в phrases, regex-метасимволы round-trip'ятся;
  * горячая перезагрузка: SESSION.fillers подменяется, и детектор филлеров
    видит новое слово БЕЗ рестарта сервера;
  * CSRF: PUT с чужим Origin -> 403, файл не изменён.

Реальный fillers_ru.yaml НЕ трогается: FILLERS_PATH monkeypatch'ится на
tmp-копию. Никакого ffmpeg/whisper/GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                            # noqa: E402
from vpipe.config import FillersCfg, load_fillers       # noqa: E402
from vpipe.detect import fillers as fdet                # noqa: E402
from vpipe.models import Word                           # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# --- fixtures -----------------------------------------------------------------
@pytest.fixture()
def fillers_file(tmp_path, monkeypatch):
    """tmp-копия реального fillers_ru.yaml + FILLERS_PATH на неё."""
    dst = tmp_path / "fillers_ru.yaml"
    dst.write_text((REPO / "fillers_ru.yaml").read_text(encoding="utf-8"),
                   encoding="utf-8")
    monkeypatch.setattr(serve, "FILLERS_PATH", dst)
    return dst


@pytest.fixture()
def client(tmp_path, fillers_file, monkeypatch):
    from vpipe.config import load_config
    cfg = load_config("config.yaml")
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.out_dir = str(tmp_path / "out")
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", cfg.paths.out_dir)
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


VALID = {"fillers": ["вот", "короче", "как бы", "на самом деле"],
         "stretched": ["э{2,}", "м{2,}", "ну+у"]}


# --- GET ------------------------------------------------------------------------
def test_get_fillers_maps_yaml(client, fillers_file):
    r = client.get("/api/fillers")
    assert r.status_code == 200
    j = r.json()
    assert j["path"] == str(fillers_file)
    assert "вот" in j["fillers"]               # words:
    assert "как бы" in j["fillers"]            # phrases: -> joined by space
    assert "э{2,}" in j["stretched"]           # mumbles:
    # ни одна запись не пустая
    assert all(isinstance(x, str) and x for x in j["fillers"] + j["stretched"])


def test_get_fillers_missing_file_empty_lists(client, fillers_file):
    fillers_file.unlink()
    j = client.get("/api/fillers").json()
    assert j["fillers"] == [] and j["stretched"] == []


# --- PUT: валидация ----------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    {},                                                   # ключей нет
    {"fillers": ["вот"]},                                 # нет stretched
    {"fillers": "вот", "stretched": []},                  # не список
    {"fillers": [], "stretched": {"э+": True}},           # не список
    {"fillers": [42], "stretched": []},                   # не строка
    {"fillers": ["  "], "stretched": []},                 # пустая после strip
    {"fillers": ["вот", "Вот"], "stretched": []},         # дубль (регистр)
    {"fillers": ["актёр", "актер"], "stretched": []},     # дубль (ё -> е)
    {"fillers": [], "stretched": ["[э"]},                 # битый regex
    {"fillers": [], "stretched": ["(э+"]},                # незакрытая группа
    {"fillers": ["х" * 201], "stretched": []},            # слишком длинно
    {"fillers": [f"w{i}" for i in range(501)], "stretched": []},   # лимит 500
])
def test_put_invalid_400_file_untouched(client, fillers_file, payload):
    before = fillers_file.read_text(encoding="utf-8")
    r = client.put("/api/fillers", json=payload)
    assert r.status_code == 400
    assert fillers_file.read_text(encoding="utf-8") == before


def test_put_limit_500_exact_ok(client, fillers_file):
    payload = {"fillers": [f"слово{i}" for i in range(500)], "stretched": []}
    assert client.put("/api/fillers", json=payload).status_code == 200


# --- PUT: запись + бэкап --------------------------------------------------------------
def test_put_roundtrip_and_yaml_shape(client, fillers_file):
    r = client.put("/api/fillers", json=VALID)
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert sorted(j["fillers"]) == sorted(VALID["fillers"])
    assert j["stretched"] == VALID["stretched"]
    assert j["path"] == str(fillers_file)

    # GET видит то же самое
    g = client.get("/api/fillers").json()
    assert sorted(g["fillers"]) == sorted(VALID["fillers"])
    assert g["stretched"] == VALID["stretched"]

    # YAML на диске честно парсится штатным загрузчиком пайплайна
    lists = load_fillers(fillers_file)
    assert lists.words == ["вот", "короче"]
    assert lists.phrases == [["как", "бы"], ["на", "самом", "деле"]]
    assert lists.mumbles == VALID["stretched"]            # метасимволы целы
    # .tmp не остался (atomic), шапка-комментарий на месте
    assert not list(fillers_file.parent.glob("*.tmp"))
    text = fillers_file.read_text(encoding="utf-8")
    assert text.startswith("# ===")
    assert "fillers_ru.yaml.bak" in text                  # шапка честно говорит о бэкапе


def test_put_backup_created_once_with_original(client, fillers_file):
    original = fillers_file.read_text(encoding="utf-8")
    bak = fillers_file.parent / (fillers_file.name + ".bak")
    assert not bak.exists()
    client.put("/api/fillers", json=VALID)
    assert bak.read_text(encoding="utf-8") == original    # бэкап = исходник
    # второй PUT бэкап НЕ перезаписывает
    client.put("/api/fillers", json={"fillers": ["типа"], "stretched": []})
    assert bak.read_text(encoding="utf-8") == original


def test_put_normalizes_inner_whitespace(client, fillers_file):
    client.put("/api/fillers", json={"fillers": ["как   бы", "  вот "],
                                     "stretched": []})
    lists = load_fillers(fillers_file)
    assert lists.words == ["вот"]
    assert lists.phrases == [["как", "бы"]]


def test_put_empty_lists_valid_yaml(client, fillers_file):
    r = client.put("/api/fillers", json={"fillers": [], "stretched": []})
    assert r.status_code == 200
    lists = load_fillers(fillers_file)
    assert lists.words == [] and lists.phrases == [] and lists.mumbles == []


# --- горячая перезагрузка словаря в живой Session ---------------------------------------
def test_put_hot_reloads_live_session(client, fillers_file, monkeypatch):
    old = load_fillers(fillers_file)
    sess = SimpleNamespace(fillers=old, task={"running": False})
    monkeypatch.setattr(serve, "SESSION", sess)
    assert "чудно" not in old.words

    r = client.put("/api/fillers",
                   json={"fillers": ["чудно", "так сказать"], "stretched": ["э+"]})
    assert r.status_code == 200
    # словарь в сессии подменён (Session кэширует FillerLists в __init__)
    assert sess.fillers is not old
    assert sess.fillers.words == ["чудно"]
    assert sess.fillers.phrases == [["так", "сказать"]]
    assert sess.fillers.mumbles == ["э+"]

    # …и детект ПОСЛЕ сохранения видит новое слово без рестарта сервера
    words = [Word(" раз", 0.0, 0.4), Word(" чудно", 0.5, 0.9),
             Word(" два", 1.0, 1.4)]
    out = fdet.detect(words, FillersCfg(), sess.fillers)
    assert [s.text for s in out] == ["чудно"]


def test_put_blocked_while_task_running(client, fillers_file, monkeypatch):
    sess = SimpleNamespace(fillers=None, task={"running": True})
    monkeypatch.setattr(serve, "SESSION", sess)
    before = fillers_file.read_text(encoding="utf-8")
    r = client.put("/api/fillers", json=VALID)
    assert r.status_code == 409                            # _guard_no_task
    assert fillers_file.read_text(encoding="utf-8") == before


# --- CSRF ---------------------------------------------------------------------------
def test_put_csrf_evil_origin_403(client, fillers_file):
    before = fillers_file.read_text(encoding="utf-8")
    r = client.put("/api/fillers", json=VALID,
                   headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert fillers_file.read_text(encoding="utf-8") == before


def test_put_local_origin_passes(client, fillers_file):
    r = client.put("/api/fillers", json=VALID,
                   headers={"Origin": "http://127.0.0.1:8000"})
    assert r.status_code == 200
