# -*- coding: utf-8 -*-
"""#35 — GET /api/chapters and GET /api/metadata read the already-persisted
generated files (work_dir/preview_chapters.txt, work_dir/preview_metadata.json)
so the Главы/Мета tabs restore after reload like Клипы (GET /api/clips). Absent
files return the empty shape (200, not 404); no session -> 409.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve


@pytest.fixture()
def client():
    return TestClient(serve.app)


def _session(tmp_path):
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(work_dir=work, task={"running": False})


def test_get_chapters_returns_parsed_list(client, monkeypatch, tmp_path):
    s = _session(tmp_path)
    monkeypatch.setattr(serve, "SESSION", s)
    (s.work_dir / "preview_chapters.txt").write_text(
        "00:00 Вступление\n01:30 Основная часть\n", encoding="utf-8")
    r = client.get("/api/chapters")
    assert r.status_code == 200
    assert r.json()["chapters"] == [
        {"time": 0.0, "title": "Вступление"},
        {"time": 90.0, "title": "Основная часть"}]


def test_get_chapters_empty_when_absent(client, monkeypatch, tmp_path):
    s = _session(tmp_path)
    monkeypatch.setattr(serve, "SESSION", s)
    r = client.get("/api/chapters")
    assert r.status_code == 200
    assert r.json() == {"chapters": []}


def test_get_metadata_returns_dict(client, monkeypatch, tmp_path):
    s = _session(tmp_path)
    monkeypatch.setattr(serve, "SESSION", s)
    meta = {"title": "Мой ролик", "description": "текст", "tags": ["a", "b"]}
    (s.work_dir / "preview_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    r = client.get("/api/metadata")
    assert r.status_code == 200
    assert r.json()["metadata"] == meta


def test_get_metadata_null_when_absent(client, monkeypatch, tmp_path):
    s = _session(tmp_path)
    monkeypatch.setattr(serve, "SESSION", s)
    r = client.get("/api/metadata")
    assert r.status_code == 200
    assert r.json() == {"metadata": None}


def test_get_chapters_metadata_no_session_409(client, monkeypatch):
    monkeypatch.setattr(serve, "SESSION", None)
    assert client.get("/api/chapters").status_code == 409
    assert client.get("/api/metadata").status_code == 409
