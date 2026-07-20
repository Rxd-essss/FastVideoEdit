# -*- coding: utf-8 -*-
"""Движок код-графики (vpipe/codegfx/) — MONTAGE_V2_PLAN §2/§8.

Покрывает:
 1. Chrome-discovery (мок путей): cfg → системный Chrome → Edge → Playwright →
    None; абсолют/PATH/несуществующий; НИКОГДА не бросает.
 2. Валидатор полей по интенту (validate.build_payload): хороший payload →
    вложенный PAYLOAD; плохой/пустой/неизвестный интент → None; лимиты (≤6
    строк, ≤4 stat, ровно 2 колонки); плоская a/b/winner → win/lose ячейки.
 3. Экранирование/гигиена: управляющие символы вычищаются, длина ограничена;
    инъекция в HTML экранирует </ (анти-«закрыть <script>»).
 4. Кэш: ключ детерминирован и зависит от intent/fields/style/WH; идемпотентность
    (готовый PNG отдаётся без запуска Chrome).
 5. render_schematic: мок Chrome (happy → PNG; нет Chrome → None; битый payload →
    None; sandbox-гард Exception → None).
 6. РЕАЛЬНЫЙ рендер (если на машине есть Chrome): stat+compare+tree → непустой
    PNG 1920x1080 с альфой (RGBA); иначе skip.
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vpipe.codegfx import render as cg
from vpipe.codegfx import validate as cv
from vpipe.codegfx import to_ass as ca

_SILENT = lambda *a, **k: None  # noqa: E731
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64   # «валидный» (для _is_real_png) фейк-PNG


def _cfg(**over):
    base = dict(codegfx_enabled=True, codegfx_chrome="",
                codegfx_style="minimal", codegfx_size="1920x1080")
    base.update(over)
    return SimpleNamespace(**base)


# === 1. CHROME DISCOVERY =====================================================
def _no_system_chrome(monkeypatch):
    """Замокать так, что ни системный Chrome/Edge, ни Playwright не находятся."""
    monkeypatch.setattr(cg, "_candidate_chrome_paths", lambda: [])
    monkeypatch.setattr(cg, "_playwright_chromium", lambda: None)


def test_resolve_chrome_configured_absolute(monkeypatch, tmp_path):
    _no_system_chrome(monkeypatch)
    exe = tmp_path / "chrome.exe"
    exe.write_bytes(b"x")
    assert cg.resolve_chrome(str(exe)) == str(exe)


def test_resolve_chrome_configured_missing_falls_through(monkeypatch, tmp_path):
    """cfg-путь не найден → продолжаем автопоиск (а не падаем/возвращаем мусор)."""
    _no_system_chrome(monkeypatch)
    # сконфигурирован несуществующий абсолют → дальше системный пуст → None
    assert cg.resolve_chrome(str(tmp_path / "nope.exe"), log=_SILENT) is None


def test_resolve_chrome_configured_via_path(monkeypatch):
    _no_system_chrome(monkeypatch)
    monkeypatch.setattr(cg.shutil, "which",
                        lambda n: "/usr/bin/chromium" if n == "chromium" else None)
    assert cg.resolve_chrome("chromium") == "/usr/bin/chromium"


def test_resolve_chrome_system_candidate(monkeypatch, tmp_path):
    """Нет cfg → берём первый СУЩЕСТВУЮЩИЙ кандидат системного Chrome/Edge."""
    monkeypatch.setattr(cg, "_playwright_chromium", lambda: None)
    real = tmp_path / "chrome.exe"
    real.write_bytes(b"x")
    ghost = tmp_path / "ghost.exe"           # не существует
    monkeypatch.setattr(cg, "_candidate_chrome_paths",
                        lambda: [str(ghost), str(real)])
    assert cg.resolve_chrome("") == str(real)


def test_resolve_chrome_edge_after_chrome(monkeypatch, tmp_path):
    """Порядок: Chrome раньше Edge — если оба есть, берём первый (Chrome)."""
    monkeypatch.setattr(cg, "_playwright_chromium", lambda: None)
    chrome = tmp_path / "chrome.exe"; chrome.write_bytes(b"x")
    edge = tmp_path / "msedge.exe"; edge.write_bytes(b"x")
    monkeypatch.setattr(cg, "_candidate_chrome_paths",
                        lambda: [str(chrome), str(edge)])
    assert cg.resolve_chrome("") == str(chrome)


def test_resolve_chrome_playwright_fallback(monkeypatch):
    monkeypatch.setattr(cg, "_candidate_chrome_paths", lambda: [])
    monkeypatch.setattr(cg, "_playwright_chromium", lambda: "/pw/chromium")
    assert cg.resolve_chrome("") == "/pw/chromium"


def test_resolve_chrome_none_when_nothing(monkeypatch):
    _no_system_chrome(monkeypatch)
    assert cg.resolve_chrome("", log=_SILENT) is None


def test_resolve_chrome_playwright_internal_error_is_swallowed(monkeypatch):
    """Сбой в самом _playwright_chromium ловится ВНУТРИ него (try/except) и
    отдаёт None — resolve_chrome деградирует, а не валит задачу."""
    monkeypatch.setattr(cg, "_candidate_chrome_paths", lambda: [])
    # эмулируем сбой импорта/запуска playwright: реальная функция глотает всё.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("playwright"):
            raise RuntimeError("playwright is broken")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cg._playwright_chromium() is None          # сам глотает сбой
    assert cg.resolve_chrome("", log=_SILENT) is None  # и resolve тоже не падает


# === 2. ВАЛИДАТОР ПОЛЕЙ ======================================================
def test_validate_unknown_intent_is_none():
    assert cv.build_payload("nope", {"x": 1}, "minimal") is None


def test_validate_non_dict_fields_is_none():
    assert cv.build_payload("stat", None, "minimal") is None
    assert cv.build_payload("stat", "oops", "minimal") is None


def test_stat_requires_a_value():
    assert cv.build_payload("stat", {"stats": []}, "minimal") is None
    assert cv.build_payload("stat", {"stats": [{"label": "no value"}]},
                            "minimal") is None
    good = cv.build_payload("stat", {"stats": [{"value": "11 000",
                                               "label": "разработчиков"}],
                                     "bar": "96%"}, "minimal")
    assert good and good["type"] == "stat" and good["theme"] == "minimal"
    assert good["stats"][0]["value"] == "11 000"
    assert good["bar"] == 96                      # «96%» → int 96


def test_stat_rejects_non_numeric_value():
    # value без единой цифры — это слоп («половина», «две три»), не число: дроп.
    assert cv.build_payload("stat", {"stats": [{"value": "половина",
                                                "label": "рынка"}]},
                            "minimal") is None
    assert cv.build_payload("stat", {"stats": [{"value": "две три"},
                                               {"value": "десятками"}]},
                            "minimal") is None
    # смесь: мусор дропается, число выживает
    out = cv.build_payload("stat", {"stats": [{"value": "половина"},
                                              {"value": "21 процент",
                                               "label": "прирост"}]}, "minimal")
    assert out is not None and len(out["stats"]) == 1
    assert out["stats"][0]["value"] == "21 процент"     # цифра есть → валидно
    # число с суффиксом/символом тоже валидно
    for good in ("90%", "2017", "~2000", "11 000", "3.5×"):
        o = cv.build_payload("stat", {"stats": [{"value": good}]}, "minimal")
        assert o is not None and o["stats"][0]["value"] == good


def test_stat_caps_to_four():
    f = {"stats": [{"value": str(i)} for i in range(10)]}
    out = cv.build_payload("stat", f, "minimal")
    assert len(out["stats"]) == cv.MAX_STATS == 4


def test_compare_needs_two_columns_and_rows():
    # одна колонка → None
    assert cv.build_payload("compare", {"col_a": "Linux",
                                        "rows": [{"feature": "x", "a": "1"}]},
                            "minimal") is None
    out = cv.build_payload("compare",
                           {"col_a": "Linux", "col_b": "Windows",
                            "rows": [{"feature": "Доля", "a": "96%", "b": "4%",
                                      "winner": "a"}]}, "minimal")
    assert out["columns"] == ["Linux", "Windows"]
    cell_a, cell_b = out["rows"][0]["values"]
    assert cell_a.get("win") and cell_b.get("lose")   # winner=a → win/lose


def test_compare_winner_by_column_name():
    out = cv.build_payload("compare",
                           {"columns": ["Linux", "Windows"],
                            "rows": [{"feature": "Игры", "a": "нет", "b": "да",
                                      "winner": "Windows"}]}, "minimal")
    cell_a, cell_b = out["rows"][0]["values"]
    assert cell_b.get("win") and cell_a.get("lose")


def test_compare_caps_rows():
    rows = [{"feature": f"f{i}", "a": "1", "b": "2"} for i in range(20)]
    out = cv.build_payload("compare",
                           {"columns": ["A", "B"], "rows": rows}, "minimal")
    assert len(out["rows"]) == cv.MAX_ROWS == 6


def test_tree_needs_root_and_children():
    assert cv.build_payload("tree", {"root_label": "ROOT", "nodes": []},
                            "minimal") is None
    assert cv.build_payload("tree", {"nodes": [{"label": "a"}]},
                            "minimal") is None
    out = cv.build_payload("tree",
                           {"root_label": "HKLM",
                            "nodes": [{"label": "SOFTWARE", "desc": "проги"},
                                      {"label": "SYSTEM"}]}, "minimal")
    assert out["root"]["label"] == "HKLM"
    assert out["root"]["children"][0]["desc"] == "проги"


def test_tree_accepts_nested_root_form():
    out = cv.build_payload("tree",
                           {"root": {"label": "R",
                                     "children": [{"label": "c1"}]}}, "minimal")
    assert out["root"]["label"] == "R"
    assert out["root"]["children"][0]["label"] == "c1"


def test_list_and_process_and_timeline():
    assert cv.build_payload("list", {"items": []}, "minimal") is None
    lst = cv.build_payload("list", {"items": [{"text": "Всё есть файл",
                                              "note": "устройства"}],
                                    "ordered": True}, "minimal")
    assert lst["ordered"] is True and lst["items"][0]["note"] == "устройства"
    # process требует >= 2 шага
    assert cv.build_payload("process", {"steps": [{"title": "один"}]},
                            "minimal") is None
    proc = cv.build_payload("process",
                            {"steps": [{"title": "a"}, {"title": "b"}]},
                            "minimal")
    assert len(proc["steps"]) == 2
    # timeline требует >= 2 события
    assert cv.build_payload("timeline", {"events": [{"when": "1991"}]},
                            "minimal") is None
    tl = cv.build_payload("timeline",
                          {"events": [{"when": "1991", "what": "Linux"},
                                      {"when": "1993", "what": "Debian"}]},
                          "minimal")
    assert len(tl["events"]) == 2


def test_quote_callout_map_code():
    assert cv.build_payload("quote", {"text": ""}, "minimal") is None
    q = cv.build_payload("quote", {"text": "Всё есть файл", "by": "Unix"},
                         "minimal")
    assert q["text"] == "Всё есть файл" and q["by"] == "Unix"
    assert cv.build_payload("callout", {"term": "Ядро"}, "minimal") is None
    co = cv.build_payload("callout", {"term": "Ядро", "definition": "сердце ОС"},
                          "minimal")
    assert co["term"] == "Ядро"
    assert cv.build_payload("map", {"points": []}, "minimal") is None
    mp = cv.build_payload("map", {"region": "Европа",
                                  "points": [{"label": "Хельсинки"}]}, "minimal")
    assert mp["region"] == "Европа" and mp["points"][0]["label"] == "Хельсинки"
    assert cv.build_payload("code", {"code": "   \n  "}, "minimal") is None
    cd = cv.build_payload("code", {"code": "$ ls /etc\n"}, "minimal")
    assert "ls /etc" in cd["code"]


def test_style_normalized():
    assert cv.normalize_style("NEON") == "neon"
    assert cv.normalize_style("garbage") == cv.STYLE_DEFAULT == "minimal"
    out = cv.build_payload("quote", {"text": "x"}, "business")
    assert out["theme"] == "business"
    out2 = cv.build_payload("quote", {"text": "x"}, "garbage")
    assert out2["theme"] == "minimal"


# === 3. ЭКРАНИРОВАНИЕ / ГИГИЕНА ==============================================
def test_text_strips_control_chars_and_limits():
    dirty = "abc\x00\x07def\t\n  ghi"
    clean = cv._txt(dirty)
    assert "\x00" not in clean and "\x07" not in clean
    assert clean == "abc def ghi"
    assert len(cv._txt("x" * 999)) <= cv.TEXT_MAX


def test_build_html_escapes_script_close():
    """JSON-инъекция в <script> должна экранировать </ , иначе строка-значение
    («</script><img onerror=...>») закроет тег досрочно (XSS-вектор)."""
    payload = {"type": "quote", "theme": "minimal",
               "text": "</script><b>x</b>"}
    html = cg._build_html(payload)
    assert html is not None
    assert "</script><b>x" not in html          # сырое закрытие НЕ просочилось
    assert "<\\/script>" in html or "<\\/b>" in html


# === 4. КЭШ ==================================================================
def test_cache_key_deterministic_and_sensitive():
    f = {"stats": [{"value": "1"}]}
    k = cg.cache_key("stat", f, "minimal", 1920, 1080)
    assert k == cg.cache_key("stat", dict(f), "minimal", 1920, 1080)  # детерм.
    variants = [
        cg.cache_key("compare", f, "minimal", 1920, 1080),           # intent
        cg.cache_key("stat", {"stats": [{"value": "2"}]}, "minimal", 1920, 1080),
        cg.cache_key("stat", f, "neon", 1920, 1080),                  # style
        cg.cache_key("stat", f, "minimal", 1280, 720),               # WH
    ]
    assert k not in variants and len(set(variants)) == len(variants)


def test_cache_key_style_normalized():
    f = {"stats": [{"value": "1"}]}
    assert cg.cache_key("stat", f, "NEON", 1920, 1080) == \
           cg.cache_key("stat", f, "neon", 1920, 1080)


def test_cached_idempotent_skips_chrome(monkeypatch, tmp_path):
    """Готовый валидный PNG в кэше → render_schematic НЕ зовётся (нет Chrome)."""
    f = {"stats": [{"value": "11 000", "label": "x"}]}
    W, H = 1920, 1080
    key = cg.cache_key("stat", f, "minimal", W, H)
    (tmp_path / f"{key}.png").write_bytes(_PNG)

    def boom(*a, **k):
        raise AssertionError("render_schematic must NOT run on cache hit")
    monkeypatch.setattr(cg, "render_schematic", boom)
    out = cg.render_schematic_cached("stat", f, "minimal", _cfg(),
                                     cache_dir=tmp_path, log=_SILENT)
    assert out == str((tmp_path / f"{key}.png"))


def test_cached_invalid_payload_returns_none(monkeypatch, tmp_path):
    out = cg.render_schematic_cached("stat", {"stats": []}, "minimal", _cfg(),
                                     cache_dir=tmp_path, log=_SILENT)
    assert out is None


# === 5. render_schematic (мок Chrome) ========================================
def _fake_chrome_writer(monkeypatch, png_bytes=_PNG, exit_code=0):
    """Замокать resolve_chrome + subprocess.run так, чтобы «Chrome» писал PNG в
    путь из --screenshot=. Возвращает счётчик вызовов."""
    monkeypatch.setattr(cg, "resolve_chrome", lambda *a, **k: "/fake/chrome")
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        shot = [c for c in cmd if c.startswith("--screenshot=")][0]
        out = shot.split("=", 1)[1]
        if png_bytes is not None:
            Path(out).write_bytes(png_bytes)
        return SimpleNamespace(returncode=exit_code, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_render_happy_writes_png(monkeypatch, tmp_path):
    _fake_chrome_writer(monkeypatch)
    out = tmp_path / "stat.png"
    p = cg.render_schematic("stat", {"stats": [{"value": "1"}]}, "minimal",
                            out, _cfg(), log=_SILENT)
    assert p == str(out) and out.is_file() and out.stat().st_size > 0


def test_render_alpha_flag_present(monkeypatch, tmp_path):
    """Команда Chrome обязана нести флаг прозрачного фона (альфа cut-out)."""
    monkeypatch.setattr(cg, "resolve_chrome", lambda *a, **k: "/fake/chrome")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        shot = [c for c in cmd if c.startswith("--screenshot=")][0]
        Path(shot.split("=", 1)[1]).write_bytes(_PNG)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    cg.render_schematic("quote", {"text": "x"}, "minimal", tmp_path / "q.png",
                        _cfg(), log=_SILENT)
    assert "--default-background-color=00000000" in seen["cmd"]
    assert any(c.startswith("--window-size=1920,1080") for c in seen["cmd"])


def test_render_no_chrome_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "resolve_chrome", lambda *a, **k: None)
    assert cg.render_schematic("stat", {"stats": [{"value": "1"}]}, "minimal",
                               tmp_path / "x.png", _cfg(), log=_SILENT) is None


def test_render_bad_payload_returns_none_without_chrome(monkeypatch, tmp_path):
    """Невалидный payload отбраковывается ДО запуска Chrome."""
    def boom(*a, **k):
        raise AssertionError("chrome must not be resolved for bad payload")
    monkeypatch.setattr(cg, "resolve_chrome", boom)
    assert cg.render_schematic("stat", {"stats": []}, "minimal",
                               tmp_path / "x.png", _cfg(), log=_SILENT) is None


def test_render_chrome_empty_output_returns_none(monkeypatch, tmp_path):
    _fake_chrome_writer(monkeypatch, png_bytes=None)        # «Chrome» ничего не пишет
    assert cg.render_schematic("stat", {"stats": [{"value": "1"}]}, "minimal",
                               tmp_path / "x.png", _cfg(), log=_SILENT) is None


def test_render_chrome_garbage_output_returns_none(monkeypatch, tmp_path):
    _fake_chrome_writer(monkeypatch, png_bytes=b"not a png")  # не PNG-сигнатура
    assert cg.render_schematic("stat", {"stats": [{"value": "1"}]}, "minimal",
                               tmp_path / "x.png", _cfg(), log=_SILENT) is None


def test_render_timeout_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "resolve_chrome", lambda *a, **k: "/fake/chrome")

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 60)
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cg.render_schematic("stat", {"stats": [{"value": "1"}]}, "minimal",
                               tmp_path / "x.png", _cfg(), log=_SILENT) is None


def test_render_oserror_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "resolve_chrome", lambda *a, **k: "/fake/chrome")

    def fake_run(cmd, **kw):
        raise OSError("exec format error")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cg.render_schematic("stat", {"stats": [{"value": "1"}]}, "minimal",
                               tmp_path / "x.png", _cfg(), log=_SILENT) is None


def test_parse_size_fallback():
    assert cg._parse_size(_cfg(codegfx_size="1280x720")) == (1280, 720)
    assert cg._parse_size(_cfg(codegfx_size="garbage")) == (1920, 1080)
    assert cg._parse_size(_cfg(codegfx_size="")) == (1920, 1080)
    assert cg._parse_size(SimpleNamespace()) == (1920, 1080)   # нет атрибута


# === 6. РЕАЛЬНЫЙ РЕНДЕР (если есть Chrome) ===================================
_REAL_CHROME = cg.resolve_chrome("", log=_SILENT)
_real = pytest.mark.skipif(not _REAL_CHROME,
                           reason="нет системного Chrome/Edge — graceful skip")

_REAL_CASES = {
    "stat": {"eyebrow": "Ядро Linux", "stats": [
        {"value": "11 000", "label": "разработчиков"},
        {"value": "96", "unit": "%", "label": "серверов"}], "bar": 96,
        "source": "Linux Kernel Report"},
    "compare": {"title": "Linux vs Windows", "col_a": "Linux", "col_b": "Windows",
                "rows": [{"feature": "Доля серверов", "a": "96 %", "b": "4 %",
                          "winner": "a"},
                         {"feature": "Стоимость", "a": "бесплатно",
                          "b": "лицензия", "winner": "a"}]},
    "tree": {"eyebrow": "Реестр", "title": "Дерево веток",
             "root_label": "HKEY_LOCAL_MACHINE",
             "nodes": [{"label": "SOFTWARE", "desc": "настройки"},
                       {"label": "SYSTEM", "desc": "драйверы"}]},
}


@_real
@pytest.mark.parametrize("intent", list(_REAL_CASES))
def test_real_render_png_alpha(intent, tmp_path):
    out = tmp_path / f"{intent}.png"
    p = cg.render_schematic(intent, _REAL_CASES[intent], "minimal", out,
                            _cfg(), log=_SILENT)
    assert p == str(out), "реальный рендер должен вернуть путь"
    data = out.read_bytes()
    assert len(data) > 1000, "PNG не должен быть пустым"
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    colortype = data[25]
    assert (w, h) == (1920, 1080)
    assert colortype == 6, "ожидается RGBA (альфа cut-out над talking-head)"


# === 7. to_ass (P4, опция) — расчёт ступеней появления =======================
def test_block_count_per_intent():
    assert ca.block_count("stat",
                          {"stats": [{"value": "1"}, {"value": "2"}]}) == 2
    assert ca.block_count("list",
                          {"items": [{"text": "a"}, {"text": "b"},
                                     {"text": "c"}]}) == 3
    assert ca.block_count("quote", {"text": "x"}) == 0      # единый блок
    assert ca.block_count("stat", {"stats": []}) == 0       # невалиден → 0


def test_reveal_steps_staggered_and_bounded():
    f = {"items": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}
    steps = ca.reveal_steps("list", f, t_start=10.0, t_end=14.0)
    # head + 3 блока = 4 ступени
    assert len(steps) == 4
    assert steps[0].index == -1                  # сначала шапка
    assert [s.index for s in steps[1:]] == [0, 1, 2]
    # монотонно возрастают и не выходят за окно
    starts = [s.t_start for s in steps]
    assert starts == sorted(starts)
    assert all(s.t_start < s.t_end <= 14.0 for s in steps)


def test_reveal_steps_compresses_when_window_tight():
    """Узкое окно → шаг ужимается, но все ступени умещаются до t_end."""
    f = {"steps": [{"title": f"s{i}"} for i in range(6)]}
    steps = ca.reveal_steps("process", f, t_start=0.0, t_end=1.2)
    assert len(steps) == 7                        # head + 6
    assert all(s.t_start <= 1.2 for s in steps)
    assert steps[-1].t_start <= steps[-1].t_end


def test_reveal_steps_single_block_intent():
    steps = ca.reveal_steps("quote", {"text": "Всё есть файл"},
                            t_start=5.0, t_end=8.0)
    assert len(steps) == 1 and steps[0].index == 0
    assert steps[0].t_start == 5.0 and steps[0].t_end == 8.0


def test_reveal_steps_invalid_window_or_data():
    assert ca.reveal_steps("list", {"items": [{"text": "a"}]},
                           t_start=5.0, t_end=5.0) == []      # окно нулевое
    assert ca.reveal_steps("stat", {"stats": []},
                           t_start=0.0, t_end=3.0) == []      # данных нет


@_real
def test_real_cached_idempotent(tmp_path):
    """Второй вызов cached на тех же данных не пересоздаёт файл (mtime тот же)."""
    f = _REAL_CASES["stat"]
    p1 = cg.render_schematic_cached("stat", f, "minimal", _cfg(),
                                    cache_dir=tmp_path, log=_SILENT)
    assert p1 and Path(p1).is_file()
    mt1 = Path(p1).stat().st_mtime_ns
    p2 = cg.render_schematic_cached("stat", f, "minimal", _cfg(),
                                    cache_dir=tmp_path, log=_SILENT)
    assert p2 == p1
    assert Path(p2).stat().st_mtime_ns == mt1     # из кэша, не перерисован
