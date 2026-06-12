# -*- coding: utf-8 -*-
"""A8 — CSRF-защита: _origin_ok + _csrf_guard middleware (serve.py).

Покрывает:
  * чужой Origin на мутирующем /api/* -> 403 с русским detail, обработчик
    НЕ выполняется (cancel_all не вызван, флаг cancelled не выставлен);
  * локальные Origin (127.0.0.1 / localhost / [::1] / совпадающий хост,
    любой порт и схема) -> пропуск; Referer как fallback (evil -> 403,
    локальный -> ок); пустой Origin падает на Referer; без обоих заголовков
    (curl/CLI) -> пропуск;
  * GET не блокируется даже с чужим Origin (guard только POST/PUT/DELETE/PATCH),
    но PUT/DELETE/PATCH блокируются ещё ДО роутинга (403, не 404/405);
    не-/api пути guard не трогает;
  * юнит-тесты _origin_ok на сыром starlette-Request: мусорные origin
    («:::», «null», «garbage», битый IPv6) -> False; suffix-трюки
    (127.0.0.1.evil.com), punycode-IDN -> False; кейс/порт/схема-вариации
    loopback -> True;
  * DNS-rebinding: Origin http://attacker.com при Host: 127.0.0.1 -> 403
    (_origin_ok), а подлинный rebinding (Host: attacker.com + Origin
    attacker.com — _origin_ok его пропускает, это документированное разделение
    обязанностей) режется TrustedHostMiddleware с _allowed_hosts, как в проде;
  * _allowed_hosts: loopback-бинд пинит Host-список, внешний бинд -> ["*"].

Никакого реального сервера/ffmpeg: TestClient + SimpleNamespace-сессия.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                            # noqa: E402
from vpipe import ffmpeg_utils                          # noqa: E402

EVIL = "http://evil.example"
RU_DETAIL = re.compile("[а-яА-ЯёЁ]")                    # detail обязан быть русским


# --- fixtures / helpers -----------------------------------------------------------
def _idle_session() -> SimpleNamespace:
    """Минимальная сессия для /api/cancel: задача не бежит -> чистый 200."""
    return SimpleNamespace(task={"running": False, "cancelled": False})


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(serve, "SESSION", _idle_session())
    return TestClient(serve.app)


def _fake_request(headers: dict | None = None, host: str = "127.0.0.1:8000"):
    """Сырой starlette-Request с заданными заголовками (для юнит-тестов
    _origin_ok без TestClient)."""
    raw = [(b"host", host.encode())]
    for k, v in (headers or {}).items():
        raw.append((k.lower().encode(), v.encode("utf-8")))
    scope = {"type": "http", "method": "POST", "path": "/api/cancel",
             "query_string": b"", "headers": raw,
             "server": ("127.0.0.1", 8000), "scheme": "http"}
    return Request(scope)


# --- 1. чужой Origin -> 403, обработчик не выполняется -----------------------------
def test_evil_origin_403_russian_detail_handler_not_run(client, monkeypatch):
    calls = []
    monkeypatch.setattr(ffmpeg_utils, "cancel_all", lambda: calls.append(1))
    sess = SimpleNamespace(task={"running": True, "cancelled": False})
    monkeypatch.setattr(serve, "SESSION", sess)

    r = client.post("/api/cancel", headers={"Origin": EVIL})
    assert r.status_code == 403
    detail = r.json()["detail"]
    assert "CSRF" in detail
    assert RU_DETAIL.search(detail), f"detail не по-русски: {detail!r}"
    # middleware отбил запрос ДО обработчика: ни флага, ни kill ffmpeg
    assert sess.task["cancelled"] is False
    assert calls == []


def test_evil_origin_blocks_all_mutating_methods_before_routing(client):
    """Guard стоит до роутера: PUT/DELETE/PATCH на любые /api-пути -> 403,
    а не 404/405 (т.е. чужой Origin режется даже для несуществующих роутов)."""
    evil = {"Origin": EVIL}
    assert client.post("/api/cancel", headers=evil).status_code == 403
    assert client.put("/api/cutlist", headers=evil).status_code == 403
    assert client.delete("/api/no-such-route", headers=evil).status_code == 403
    assert client.patch("/api/cancel", headers=evil).status_code == 403


# --- 2. локальные Origin/Referer и их отсутствие -> пропуск ------------------------
@pytest.mark.parametrize("origin", [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:65535",        # любой порт
    "https://127.0.0.1",             # любая схема
    "http://localhost:3000",
    "http://[::1]:8000",
    "http://testserver",             # совпадает с request.url.hostname
])
def test_local_origin_passes(client, origin):
    r = client.post("/api/cancel", headers={"Origin": origin})
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_referer_fallback_evil_403_local_ok(client):
    assert client.post(
        "/api/cancel",
        headers={"Referer": EVIL + "/attack.html"}).status_code == 403
    r = client.post("/api/cancel",
                    headers={"Referer": "http://127.0.0.1:8000/"})
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_empty_origin_falls_back_to_referer(client):
    # Origin: "" — falsy -> guard читает Referer (а он чужой) -> 403
    r = client.post("/api/cancel",
                    headers={"Origin": "", "Referer": EVIL + "/x"})
    assert r.status_code == 403


def test_origin_wins_over_referer(client):
    # валидный Origin авторитетен — мусорный Referer не роняет запрос
    r = client.post("/api/cancel", headers={
        "Origin": "http://127.0.0.1:8000", "Referer": EVIL + "/x"})
    assert r.status_code == 200


def test_no_origin_no_referer_passes(client):
    """curl / собственный тулинг: браузерных заголовков нет -> не CSRF-вектор."""
    r = client.post("/api/cancel")
    assert r.status_code == 200 and r.json() == {"ok": True}


# --- 3. GET не блокируется, не-/api пути не блокируются ----------------------------
def test_get_not_blocked_even_with_evil_origin(client):
    evil = {"Origin": EVIL}
    assert client.get("/api/network", headers=evil).status_code == 200
    # GET на мутирующий путь -> честный 405 от роутера, НЕ 403 от guard
    assert client.get("/api/cancel", headers=evil).status_code == 405


def test_non_api_paths_not_guarded(client):
    # вне /api guard молчит: POST / -> 405 (нет такого роута), не 403
    assert client.post("/", headers={"Origin": EVIL}).status_code == 405


# --- 4. _origin_ok напрямую --------------------------------------------------------
@pytest.mark.parametrize("origin", [
    ":::",                            # мусор: hostname=None
    "null",                           # sandbox-iframe шлёт Origin: null
    "garbage",
    "http://[invalid",                # битый IPv6 -> urlparse кидает ValueError
    EVIL,
    "http://evil.example:80",
    "http://127.0.0.1.evil.com",      # suffix-трюк
    "http://localhost.evil.com",
    "http://xn--80ak6aa92e.com",      # punycode-IDN «аррӏе.com»
])
def test_origin_ok_rejects(origin):
    assert serve._origin_ok(_fake_request({"origin": origin})) is False


@pytest.mark.parametrize("origin", [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:1",             # порт не важен — важен hostname
    "https://localhost",
    "http://LOCALHOST:9999",          # urlparse нормализует регистр
    "http://[::1]:8000",
])
def test_origin_ok_accepts_loopback_variants(origin):
    assert serve._origin_ok(_fake_request({"origin": origin})) is True


def test_origin_ok_empty_and_missing_pass():
    assert serve._origin_ok(_fake_request({})) is True            # нет заголовков
    assert serve._origin_ok(_fake_request({"origin": ""})) is True  # пустой Origin


def test_origin_ok_same_nonloopback_host():
    """Сервер на --host my-pc.local: same-origin страница проходит, чужая нет."""
    req = _fake_request({"origin": "http://my-pc.local:8000"},
                        host="my-pc.local:8000")
    assert serve._origin_ok(req) is True
    req2 = _fake_request({"origin": "http://other-box.local:8000"},
                         host="my-pc.local:8000")
    assert serve._origin_ok(req2) is False


# --- 5. DNS-rebinding --------------------------------------------------------------
def test_rebinding_evil_origin_with_local_host_403(client):
    """Страница attacker.com шлёт POST на 127.0.0.1 (Host локальный, Origin
    чужой) — _origin_ok режет это сам."""
    r = client.post("/api/cancel", headers={
        "Host": "127.0.0.1:8000", "Origin": "http://attacker.com"})
    assert r.status_code == 403


def test_rebinding_spoofed_host_stopped_by_trusted_host(monkeypatch):
    """Подлинный rebinding: DNS attacker.com -> 127.0.0.1, браузер шлёт
    Host: attacker.com и Origin: http://attacker.com. _origin_ok тут бессилен
    (host == request.url.hostname — он сам из заголовка атакующего), поэтому
    в проде main() вешает TrustedHostMiddleware(_allowed_hosts(bind)).
    Воспроизводим прод-стек (TrustedHost снаружи guard'а) и проверяем отказ."""
    monkeypatch.setattr(serve, "SESSION", _idle_session())
    prod_like = TrustedHostMiddleware(
        serve.app, allowed_hosts=serve._allowed_hosts("127.0.0.1"))
    c = TestClient(prod_like)
    evil_host = {"Host": "attacker.com", "Origin": "http://attacker.com"}
    r = c.post("/api/cancel", headers=evil_host)
    assert r.status_code == 400                       # отбит ДО /api-обработчика

    # sanity: _origin_ok действительно пропустил бы такой запрос (разделение
    # обязанностей задокументировано в _allowed_hosts)
    assert serve._origin_ok(_fake_request(
        {"origin": "http://attacker.com"}, host="attacker.com")) is True

    # а легитимный локальный запрос через тот же стек живёт
    ok = c.post("/api/cancel", headers={"Host": "127.0.0.1:8000"})
    assert ok.status_code == 200 and ok.json() == {"ok": True}


def test_allowed_hosts_pinning():
    for bind in ("127.0.0.1", "localhost", "::1"):
        assert serve._allowed_hosts(bind) == ["127.0.0.1", "localhost", "::1"]
    assert serve._allowed_hosts("0.0.0.0") == ["*"]
    assert serve._allowed_hosts("192.168.1.50") == ["*"]
