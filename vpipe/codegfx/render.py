# -*- coding: utf-8 -*-
"""Движок код-графики: intent+fields+style → PNG с альфой (MONTAGE_V2_PLAN §2).

Конвейер (доказан прототипом ``D:/tmp/montage2/code/render.py``, перенесён в
пакет): плоские валидированные ``fields`` → ``validate.build_payload`` собирает
вложенный PAYLOAD → инъекция ``<script>window.PAYLOAD=…</script>`` в
``engine.html`` → системный headless-Chrome ``--screenshot
--default-background-color=00000000`` → PNG (прозрачный фон, cut-out смещён
вправо под talking-head). CPU-only, 0 VRAM, zero-upload (всё локально).

КОНТРАКТ (зафиксирован в ``vpipe/enrich.py`` «КОНТРАКТ КОД-ГРАФИКИ» — НЕ менять):

    render_schematic(intent, fields, style, out_png, cfg=None, *, log=None)
        -> Optional[str]

Возвращает абсолютный путь PNG (==str(out_png)) при успехе, иначе ``None``
(нет Chrome / битый payload / переполнение / любой сбой) — вызывающий честно
дропает кадр («лучше пусто, чем слоп»). НИКОГДА не бросает.

Chrome-DISCOVERY (НЕ хардкод — §10): cfg.codegfx_chrome → системный Chrome
(Program Files / LOCALAPPDATA) → Edge (msedge) → Playwright-Chromium →
``None`` (graceful). Кэш-обёртка ``render_schematic_cached`` пишет в
``cache/codegfx/<sha1(intent+fields+style+WxH)>.png`` атомарно и детерминированно
(идемпотентно: есть файл — отдаём без запуска Chrome).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from . import validate

LogFn = Callable[..., None]


def _noop(*_a, **_k) -> None:
    pass


# Корень пакета (vpipe/codegfx/) и repo-root (для cwd-нейтральных путей).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
ENGINE = _HERE / "engine.html"

DEFAULT_W, DEFAULT_H = 1920, 1080
CHROME_TIMEOUT_S = 60                 # один кадр ~0.3-0.5 c; гард на зависание
CACHE_DIR = Path("cache") / "codegfx"
# Минимальный «непустой» PNG: 8-байтная сигнатура + что-то ещё. Реальный кадр
# код-графики — десятки/сотни КБ; защищаемся от 0-байтного/обрезанного вывода.
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


# --- Chrome discovery (НЕ хардкод пути — §10) --------------------------------
def _candidate_chrome_paths() -> list:
    """Список кандидатов системного Chrome/Edge по платформенным местам.
    Порядок: Chrome (Program Files x64/x86, LOCALAPPDATA, Linux/mac имена) →
    Edge (msedge). Возврат — пути-строки в порядке предпочтения (могут не
    существовать; проверка — в ``_resolve_chrome``)."""
    import os as _os
    cands: list = []
    pf = _os.environ.get("PROGRAMFILES", r"C:\Program Files")
    pf86 = _os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    local = _os.environ.get("LOCALAPPDATA", "")
    # Chrome
    cands += [
        str(Path(pf) / "Google/Chrome/Application/chrome.exe"),
        str(Path(pf86) / "Google/Chrome/Application/chrome.exe"),
    ]
    if local:
        cands.append(str(Path(local) / "Google/Chrome/Application/chrome.exe"))
    # Edge (Chromium-based — те же флаги headless --screenshot)
    cands += [
        str(Path(pf86) / "Microsoft/Edge/Application/msedge.exe"),
        str(Path(pf) / "Microsoft/Edge/Application/msedge.exe"),
    ]
    if local:
        cands.append(str(Path(local) / "Microsoft/Edge/Application/msedge.exe"))
    # POSIX-имена (Linux/mac CI) — через PATH
    cands += ["google-chrome", "google-chrome-stable", "chromium",
              "chromium-browser", "msedge"]
    return cands


def _playwright_chromium() -> Optional[str]:
    """Путь к Chromium, установленному Playwright (фолбэк, ~150-280МБ разовой
    загрузки — §10). Если playwright не стоит / браузер не скачан → None.
    НЕ запускает загрузку, лишь резолвит уже установленный бинарь."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return None
    try:
        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return path if path and Path(path).exists() else None
    except Exception:
        return None


def resolve_chrome(configured: str = "", log: LogFn = _noop) -> Optional[str]:
    """Резолв Chrome-подобного бинаря к запускаемому пути (или ``None``).
    НИКОГДА не бросает — отсутствие браузера деградирует на ``None``, кадр
    дропается, задача НЕ падает.

    Порядок (§10): ``configured`` (cfg.codegfx_chrome — абсолют / PATH) →
    системный Chrome → Edge → Playwright-Chromium → ``None``."""
    configured = str(configured or "").strip()
    if configured:
        p = Path(configured)
        if p.is_absolute():
            if p.exists():
                return str(p)
        else:
            hit = shutil.which(configured) or shutil.which(p.name)
            if hit:
                return hit
        # сконфигурированный, но не найден — продолжаем автопоиск (graceful)
        log(f"  codegfx: codegfx_chrome «{configured}» не найден — автопоиск.")
    for cand in _candidate_chrome_paths():
        p = Path(cand)
        if p.is_absolute():
            if p.exists():
                return str(p)
        else:
            hit = shutil.which(cand)
            if hit:
                return hit
    pw = _playwright_chromium()
    if pw:
        return pw
    log("  codegfx: системный Chrome/Edge/Playwright не найден — кадр дропается.")
    return None


# --- параметры canvas из cfg -------------------------------------------------
def _parse_size(cfg) -> tuple:
    """`cfg.codegfx_size` («WxH») → (W, H). Кривое/пусто → дефолт 1920x1080."""
    raw = str(getattr(cfg, "codegfx_size", "") or "").lower().strip()
    m = raw.replace(" ", "").split("x")
    if len(m) == 2:
        try:
            w, h = int(m[0]), int(m[1])
            if 320 <= w <= 7680 and 240 <= h <= 4320:
                return w, h
        except ValueError:
            pass
    return DEFAULT_W, DEFAULT_H


def _cfg_chrome(cfg) -> str:
    return str(getattr(cfg, "codegfx_chrome", "") or "")


# --- инъекция payload в engine.html -----------------------------------------
def _build_html(payload: dict) -> Optional[str]:
    """Прочитать engine.html и вставить ``window.PAYLOAD``. JSON-сериализация
    с ``ensure_ascii=False`` (кириллица как есть); ``</`` экранируется, чтобы
    строка-значение не закрыла <script> досрочно (defence-in-depth — текст
    дополнительно esc()-ится движком). Любой сбой чтения → None."""
    try:
        tpl = ENGINE.read_text(encoding="utf-8")
    except OSError:
        return None
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    inject = "<script>window.PAYLOAD=" + blob + ";</script>"
    if "<body>" in tpl:
        return tpl.replace("<body>", "<body>" + inject, 1)
    return inject + tpl


def _is_real_png(p: Path) -> bool:
    try:
        if not (p.is_file() and p.stat().st_size > len(_PNG_SIG)):
            return False
        with p.open("rb") as fh:
            return fh.read(len(_PNG_SIG)) == _PNG_SIG
    except OSError:
        return False


def _screenshot(chrome: str, html: str, out_png: Path, W: int, H: int,
                log: LogFn) -> bool:
    """Запустить headless-Chrome: HTML-файл → screenshot PNG (альфа). Пишем во
    временный .html (Chrome ест file://), снимок — в ``tmp`` рядом с целью,
    затем атомарно ``os.replace`` в out_png. Возврат — успех. НЕ бросает."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = out_png.with_name(out_png.stem + ".tmp.png")
    tmp_html: Optional[Path] = None
    try:
        fd, hp = tempfile.mkstemp(suffix=".html", prefix="codegfx_",
                                  dir=str(out_png.parent))
        tmp_html = Path(hp)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(html)
        cmd = [
            chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--no-sandbox", "--no-first-run", "--no-default-browser-check",
            "--force-device-scale-factor=1",
            "--default-background-color=00000000",   # АЛЬФА (прозрачный фон)
            "--virtual-time-budget=800",
            "--window-size=%d,%d" % (W, H),
            "--screenshot=%s" % str(tmp_png),
            tmp_html.as_uri(),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=CHROME_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log(f"  codegfx: таймаут Chrome {CHROME_TIMEOUT_S} c — кадр дропается.")
            return False
        except OSError as e:
            log(f"  codegfx: запуск Chrome не удался ({e}) — кадр дропается.")
            return False
        # Headless Chrome для --screenshot обычно exit 0, но некоторые сборки
        # возвращают ненулевой код и при этом пишут валидный PNG. Поэтому судим
        # по ФАЙЛУ, а код логируем только если файла нет.
        if not _is_real_png(tmp_png):
            tail = " | ".join((r.stderr or r.stdout or "").strip()
                              .splitlines()[-2:])
            log(f"  codegfx: Chrome не отдал валидный PNG (exit {r.returncode})"
                + (f": {tail}" if tail else "") + " — кадр дропается.")
            return False
        try:
            os.replace(tmp_png, out_png)
        except OSError as e:
            log(f"  codegfx: не удалось сохранить кадр ({e}) — кадр дропается.")
            return False
        return True
    finally:
        for junk in (tmp_html, tmp_png):
            if junk is not None:
                _unlink(junk)


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


# --- ПУБЛИЧНЫЙ КОНТРАКТ -------------------------------------------------------
def render_schematic(intent: str, fields: dict, style: str,
                     out_png, cfg=None, *, log: Optional[LogFn] = None
                     ) -> Optional[str]:
    """intent+fields+style → PNG с альфой по пути ``out_png`` (контракт §2).

    Возврат — ``str(out_png)`` при успехе, иначе ``None`` (нет Chrome / битый
    payload / переполнение / сбой). Данные движок НЕ выдумывает и НЕ валидирует
    числа (это сделал backend quote-снапом) — лишь проверяет ФОРМУ/лимиты через
    ``validate.build_payload``. НИКОГДА не бросает."""
    log = log or _noop
    out_png = Path(out_png)
    try:
        payload = validate.build_payload(intent, fields if isinstance(fields, dict)
                                         else {}, style)
        if payload is None:
            log(f"  codegfx: интент «{intent}» — данных не хватает/невалидны — кадр дропается.")
            return None
        chrome = resolve_chrome(_cfg_chrome(cfg), log)
        if not chrome:
            return None
        W, H = _parse_size(cfg)
        html = _build_html(payload)
        if html is None:
            log("  codegfx: engine.html не читается — кадр дропается.")
            return None
        ok = _screenshot(chrome, html, out_png, W, H, log)
        return str(out_png) if ok else None
    except Exception as e:                  # тотальный гард: код-графика не валит задачу
        log(f"  codegfx: неожиданный сбой ({e!r}) — кадр дропается.")
        return None


# --- кэш-обёртка (детерминированный путь cache/codegfx/<sha1>.png) -----------
def cache_key(intent: str, fields: dict, style: str, W: int, H: int) -> str:
    """SHA1 по (intent+fields+style+WxH). Детерминированно: ``sort_keys`` →
    один и тот же payload между прогонами даёт один ключ (идемпотентный кэш)."""
    try:
        fblob = json.dumps(fields, ensure_ascii=False, sort_keys=True,
                           default=str)
    except (TypeError, ValueError):
        fblob = repr(fields)
    blob = f"{intent}\x00{fblob}\x00{validate.normalize_style(style)}\x00{W}x{H}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def render_schematic_cached(intent: str, fields: dict, style: str, cfg=None, *,
                            cache_dir: Optional[Any] = None,
                            log: Optional[LogFn] = None) -> Optional[str]:
    """Как ``render_schematic``, но путь PNG выбирает САМ — ``cache/codegfx/
    <sha1>.png`` — и идемпотентен: готовый валидный PNG отдаётся без запуска
    Chrome. Удобно для стадии ``images`` / превью «Монтажа». Возврат —
    абсолютный путь PNG или ``None``."""
    log = log or _noop
    # Сначала валидируем payload — иначе кэш-ключ бессмыслен (и не плодим записи
    # на заведомо мёртвых данных).
    payload = validate.build_payload(intent, fields if isinstance(fields, dict)
                                     else {}, style)
    if payload is None:
        log(f"  codegfx: интент «{intent}» — данных не хватает/невалидны — кадр дропается.")
        return None
    W, H = _parse_size(cfg)
    base = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    # repo-root-относительный путь (как imagegen) → абсолют (path-traversal guard
    # enrich.ImagePayload обнуляет относительные пути при перезагрузке плана).
    if not base.is_absolute():
        base = _REPO_ROOT / base
    base = base.resolve()
    key = cache_key(intent, fields if isinstance(fields, dict) else {}, style, W, H)
    out_png = base / f"{key}.png"
    if _is_real_png(out_png):
        return str(out_png)                # идемпотентно
    return render_schematic(intent, fields, style, out_png, cfg, log=log)
