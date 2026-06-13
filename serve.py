#!/usr/bin/env python
"""FastVideoEdit — local web review/edit UI.

    python serve.py --video input.mp4 [--out ./out] [--config config.yaml]

Opens a local editor (http://127.0.0.1:8000) to review the AI-proposed cut list,
adjust/add cuts, preview the edit, and render — all reusing the vpipe pipeline.
Nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

for _s in (sys.stdout, sys.stderr):   # Windows consoles default to cp1251
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from typing import Literal, Optional


def _check_deps() -> None:
    missing = []
    for mod in ("yaml", "pydantic", "fastapi", "uvicorn"):
        try:
            __import__(mod)
        except ModuleNotFoundError:
            missing.append(mod)
    if missing:
        sys.stderr.write(
            "\n[FastVideoEdit] Не хватает зависимостей: " + ", ".join(missing) + "\n"
            "Похоже, запущен системный Python, а не окружение проекта (.venv).\n\n"
            "  Проще всего:   дважды кликни  run.bat   (или в PowerShell:  .\\run.ps1)\n"
            "  либо вручную:  .\\.venv\\Scripts\\Activate.ps1   затем   python serve.py\n\n")
        raise SystemExit(1)


_check_deps()

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

import anyio

from vpipe import chapters as chapters_mod
from vpipe import clips as clips_mod
from vpipe import enrich as enrich_mod
from vpipe import enrich_cards
from vpipe import enrich_llm
from vpipe import imagegen as imagegen_mod
from vpipe import facecrop as facecrop_mod
from vpipe import ffmpeg_utils
from vpipe import metadata as metadata_mod
from vpipe import netguard
from vpipe import render as render_mod
from vpipe import subtitles as subs_mod
from vpipe.config import (Config, load_config, load_fillers, load_profanity)
from vpipe.cutlist import resolve, save_txt
from vpipe.export_nle import write_edl, write_fcpxml
from vpipe.detect import run_detection
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.ffmpeg_utils import FFmpeg
from vpipe.llm import OllamaClient, get_client
from vpipe.models import (ACTION_REMOVE, CutList, CutSegment, Transcript,
                          TYPE_MANUAL)
from vpipe.probe import extract_audio, hash_input, probe_media
from vpipe.timeline import Timeline, remap_words
from vpipe import transcribe as transcribe_mod
from vpipe.transcribe import transcribe_audio
from vpipe.waveform import compute_peaks

WEB_DIR = Path(__file__).resolve().parent / "web"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts",
             ".flv", ".wmv", ".mpg", ".mpeg"}
EXT_MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
            ".webm": "video/webm", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
            ".ts": "video/mp2t", ".flv": "video/x-flv", ".wmv": "video/x-ms-wmv",
            ".mpg": "video/mpeg", ".mpeg": "video/mpeg"}
MAX_UPLOAD_BYTES = 30 * 1024**3   # 30 GB upload cap
# C3 (фоновая музыка): белый список расширений для music.path — аудио либо
# видео (из видеоконтейнера ffmpeg возьмёт звуковую дорожку).
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
MUSIC_EXT = AUDIO_EXT | VIDEO_EXT
# Авто-обогащение (§4 Tier 1 / «заменить ассет»): белый список картинок-ассетов
# для /api/browse?kind=image — ровно те форматы, что индексирует match_user_assets.
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
# Extensions /api/output is allowed to serve — exactly the artifacts the pipeline
# produces (rendered video, sidecar subs, chapters/metadata txt, NLE projects).
OUTPUT_EXT_ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".m4v",
                      ".srt", ".vtt", ".ass", ".txt", ".edl", ".fcpxml", ".json"}

# P2-#5: allowed faster-whisper model names for the UI presets (quality↔speed
# on an 8 GB card). A whitelist on the UI side — any other name -> 400. This is a
# UI-level business constraint, NOT a faster-whisper limit, so it lives here.
WHISPER_PRESETS = [
    {"key": "quality",  "label": "Качество",  "model": "large-v3",
     "hint": "≈3 ГБ VRAM · самая точная, медленнее"},
    {"key": "balanced", "label": "Баланс",    "model": "large-v3-turbo",
     "hint": "≈1,5 ГБ VRAM · быстрее, точность почти как у large-v3"},
    {"key": "speed",    "label": "Скорость",  "model": "medium",
     "hint": "≈1,5 ГБ VRAM · заметно быстрее, точность ниже"},
    {"key": "light",    "label": "Лёгкая",    "model": "small",
     "hint": "≈0,5 ГБ VRAM · минимум ресурсов, для черновика"},
]
WHISPER_ALLOWED = {p["model"] for p in WHISPER_PRESETS}

# C1: готовые стили вшитых караоке-субтитров («стиль в 1 клик» поверх пяти
# сырых полей renderModal). Как и WHISPER_PRESETS, это UI-уровневые бизнес-
# данные — поэтому живут здесь, а не в vpipe. Каждый ``style`` — ПОЛНЫЙ набор
# полей AssStyleCfg (кроме ортогонального тумблера ``enabled``); соответствие
# pydantic-модели закреплено тестами (extra="ignore" молча съел бы опечатку
# в ключе — тест сверяет точное множество ключей). Фронт получает список через
# GET /api/state — один источник истины, без дубля констант в app.js.
# Цвета — ASS &HAABBGGRR (AA=00 — непрозрачный); размеры/отступы — в пикселях
# PlayRes (равен разрешению выхода). Подбор — под talking-head Shorts.
CAPTION_PRESETS = [
    {"key": "classic", "label": "Классика",
     "hint": "Белый текст, жёлтая подсветка слова, снизу",
     "style": {"font": "Arial", "size": 52,
               "primary_color": "&H00FFFFFF", "outline_color": "&H00000000",
               "karaoke_color": "&H0000D4FF",   # #FFD400 — тёплый жёлтый
               "outline": 2.0, "shadow": 1.0,
               "position": "bottom", "karaoke": True, "margin_v": 40}},
    {"key": "neon", "label": "Неон",
     "hint": "Крупнее, бирюзовая подсветка, приподнято над низом",
     "style": {"font": "Verdana", "size": 62,
               "primary_color": "&H00FFFFFF", "outline_color": "&H00000000",
               "karaoke_color": "&H00FFE500",   # #00E5FF — бирюза
               "outline": 3.0, "shadow": 0.0,
               "position": "bottom", "karaoke": True, "margin_v": 160}},
    {"key": "minimal", "label": "Минимал",
     "hint": "Мельче и спокойнее: мягкая полупрозрачная обводка-плашка",
     "style": {"font": "Tahoma", "size": 44,
               "primary_color": "&H00FFFFFF",
               "outline_color": "&H78000000",   # чёрный ≈53% непрозрачности
               "karaoke_color": "&H006ED7F5",   # #F5D76E — приглушённое золото
               "outline": 3.0, "shadow": 0.0,
               "position": "bottom", "karaoke": True, "margin_v": 48}},
    {"key": "bold", "label": "Крупный",
     "hint": "Для просмотра без звука — большой кегль по центру кадра",
     "style": {"font": "Impact", "size": 78,
               "primary_color": "&H00FFFFFF", "outline_color": "&H00000000",
               "karaoke_color": "&H0000D4FF",
               "outline": 3.0, "shadow": 2.0,
               "position": "center", "karaoke": True, "margin_v": 40}},
]

# B5: the user-editable filler dictionary (repo root). The GET/PUT /api/fillers
# endpoints read this module global at call time so tests can repoint it at a
# tmp copy without ever touching the real file.
FILLERS_PATH = Path(__file__).resolve().parent / "fillers_ru.yaml"

APP: dict = {}   # launch-time settings reused when opening a new video
TASK_LOCK = threading.Lock()   # serialize task starts / state mutations


# --- F3: batch queue (multiple clips → transcribe + detect + render) ---------
# The single-clip editor (global SESSION) is untouched. The queue is a separate
# module-level state: jobs processed ONE AT A TIME by a daemon worker that shares
# TASK_LOCK with the editor so faster-whisper/ffmpeg never load models on the
# 8 GB GPU concurrently. A job may simply wait while the editor runs a task.
@dataclass
class QueueJob:
    """One queued clip: open → (transcribe if needed) → detect → render."""
    id: str
    path: str
    out_dir: str
    render_opts: dict = field(default_factory=dict)
    status: Literal["pending", "running", "done", "error"] = "pending"
    percent: float = 0.0
    stage: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None


QUEUE: list[QueueJob] = []
QUEUE_LOCK = threading.Lock()        # guards QUEUE list mutations
_queue_worker_thread: Optional[threading.Thread] = None
_queue_running = False               # editor↔queue GPU mutual-exclusion flag (guarded by TASK_LOCK)
_queue_cancel = threading.Event()    # set by /api/queue/stop to abort the run


class Session:
    """All mutable state for the one video being edited."""

    def __init__(self, video: str, cfg: Config, out_dir: str, use_llm: bool):
        self.cfg = cfg
        self.inp = Path(video)
        self.ff = FFmpeg(cfg.ffmpeg)
        self.media = probe_media(self.ff, self.inp)
        # A zero/unknown duration (fragmented .ts, broken container) otherwise
        # slips through to render as a cryptic «всё вырезано» — fail early & clear.
        if not (self.media.duration and self.media.duration > 0):
            raise ValueError(
                "Не удалось определить длительность видео — файл повреждён или "
                "не поддерживается. / Could not read the video duration (corrupt "
                "or unsupported file).")
        self.out_dir = Path(out_dir)
        self.cache_dir = Path(cfg.paths.cache_dir)
        # Hash the input first so the work dir can be keyed by it: same-named
        # files from different folders no longer collide on one scratch dir.
        self.audio_hash = hash_input(self.inp)
        self.work_dir = (Path(cfg.paths.work_dir) /
                         f"{self.inp.stem}-{self.audio_hash[:8]}")
        for d in (self.out_dir, self.cache_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.base = self.out_dir / self.inp.stem
        self.cutlist_path = self.out_dir / f"{self.inp.stem}.cutlist.json"

        here = Path(__file__).resolve().parent
        self.fillers = load_fillers(here / "fillers_ru.yaml")
        self.profanity = load_profanity(here / "profanity_ru.yaml")
        self.matcher = ProfanityMatcher(self.profanity)

        self.llm = get_client(cfg.llm) if use_llm else None
        if self.llm is not None and not (self.llm.available() and self.llm.has_model()):
            self.llm = None

        self.peaks: Optional[list[float]] = None
        self.last_out_dir = str(self.out_dir.resolve())
        self.transcript: Optional[Transcript] = None
        self.cutlist: Optional[CutList] = None
        self.task = {"name": None, "running": False, "percent": 0.0,
                     "stage": "", "error": None, "done": False, "results": None}
        self._ctor_fresh_detect = False   # True if __init__ ran a fresh _detect()

        # Load anything already on disk.
        cache_file = self.cache_dir / f"{self.audio_hash}.transcript.json"
        if cfg.transcribe.cache and cache_file.exists():
            self.transcript = Transcript.load(cache_file)
        if self.cutlist_path.exists():
            self.cutlist = CutList.load_json(self.cutlist_path)
        elif self.transcript is not None:
            self.cutlist = self._detect()
            self._ctor_fresh_detect = True

    # --- helpers -------------------------------------------------------------
    def _detect(self, cfg: Optional[Config] = None) -> CutList:
        """Run detection, optionally under a per-request config override (B5).

        ``cfg=None`` (ctor / transcribe / queue) -> ``self.cfg``, byte-for-byte
        the old behaviour. POST /api/detect passes the effective DEEP COPY built
        by ``_apply_detect_opts`` — ``self.cfg`` is never mutated by overrides.
        """
        assert self.transcript is not None
        eff = cfg if cfg is not None else self.cfg
        # The 16 kHz wav (extracted during transcription) feeds the acoustic
        # hesitation detector; pass it when present so VAD can run.
        wav = self.work_dir / "audio16k.wav"
        cl = run_detection(self.transcript, eff, self.fillers, self.profanity,
                           source=str(self.inp), llm=self.llm, log=lambda *_: None,
                           audio_path=wav if wav.exists() else None)
        # Preserve any manual cuts the user already drew.
        if self.cutlist is not None:
            cl.segments.extend(s for s in self.cutlist.segments if s.type == TYPE_MANUAL)
            cl.segments.sort(key=lambda s: s.start)
        cl.save_json(self.cutlist_path)
        save_txt(cl, self.out_dir / f"{self.inp.stem}.cutlist.txt")
        self.cutlist = cl
        return cl

    def set_progress(self, frac: float) -> None:
        self.task["percent"] = round(min(1.0, max(0.0, frac)) * 100, 1)

    def stage(self, msg: str) -> None:
        self.task["stage"] = msg

    def start_task(self, name: str, fn) -> None:
        # The batch queue (F3) and the editor must never load models on the 8 GB
        # GPU at once. Check BOTH the queue flag and our own running flag
        # atomically under TASK_LOCK (which also guards _queue_running), then set
        # ours — so a near-simultaneous queue start can't slip through. We do NOT
        # hold the lock during the work itself; the flags provide the exclusion.
        with TASK_LOCK:
            if _queue_running:
                raise HTTPException(409, "Очередь обрабатывает ролики — дождитесь её "
                                         "завершения или остановите очередь")
            if self.task["running"]:
                raise HTTPException(409, "Задача уже выполняется — дождитесь завершения")
            self.task = {"name": name, "running": True, "percent": 0.0,
                         "stage": "", "error": None, "done": False,
                         "results": None, "cancelled": False}

        def worker():
            try:
                fn()
                self.task["percent"] = 100.0
                self.task["done"] = True
            except Exception as e:  # noqa: BLE001
                # A user-cancelled task kills ffmpeg, which surfaces as an ugly
                # "exit 1" + ffmpeg dump. Report it as a clean cancellation so the
                # UI shows «Задача отменена», not a scary error.
                self.task["error"] = "cancelled" if self.task.get("cancelled") else str(e)
            finally:
                self.task["running"] = False

        threading.Thread(target=worker, daemon=True).start()


SESSION: Optional[Session] = None
app = FastAPI(title="FastVideoEdit")


@app.middleware("http")
async def _cache_headers(request: Request, call_next):
    """Cache policy for a local single-user editor:
    - /api/*  -> no-store: responses are session-dependent. Without this,
      /api/video (a stable URL served with etag/last-modified but no
      Cache-Control) is reused from the browser cache after switching to a
      different clip — the new clip's state loads but the OLD video/audio keeps
      playing and the waveform shows the OLD duration. no-store forces a refetch.
    - / and /static/* -> no-cache: still cached, but always revalidated, so an
      updated app.js / style.css is picked up immediately (no stale UI) while
      unchanged assets get a cheap 304 on localhost."""
    resp = await call_next(request)
    path = request.url.path
    if path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    elif path == "/" or path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def _allowed_hosts(host: str) -> list[str]:
    """Host-header allow-list for TrustedHostMiddleware.

    For the default loopback bind we pin the Host to the loopback names — this is
    what actually stops a DNS-rebinding attack (``evil.com`` rebound to 127.0.0.1
    sends ``Host: evil.com``, which is rejected before any /api handler runs;
    ``_origin_ok`` alone couldn't, since it trusted the attacker-set Host). An
    explicit non-loopback bind means the user knowingly exposed the box (they get
    the startup warning), so we don't second-guess their Host there."""
    if host in ("127.0.0.1", "localhost", "::1"):
        return ["127.0.0.1", "localhost", "::1"]
    return ["*"]


def _origin_ok(request: Request) -> bool:
    """True if a mutating request is same-origin / local. Blocks cross-site
    POSTs (CSRF / DNS-rebinding) that could browse or open arbitrary files."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return True   # non-browser client (curl/our own tooling): no CSRF vector
    try:
        from urllib.parse import urlparse
        host = urlparse(origin).hostname
    except Exception:  # noqa: BLE001
        return False
    return host == request.url.hostname or host in ("127.0.0.1", "localhost", "::1")


@app.middleware("http")
async def _csrf_guard(request: Request, call_next):
    if (request.method in ("POST", "PUT", "DELETE", "PATCH")
            and request.url.path.startswith("/api/")
            and not _origin_ok(request)):
        return JSONResponse({"detail": "Перекрёстный запрос отклонён (CSRF)."},
                            status_code=403)
    return await call_next(request)


def open_session(path: str) -> Session:
    global SESSION
    SESSION = Session(path, APP["cfg"], APP["out_dir"], APP["use_llm"])
    return SESSION


# --- P2-#4: privacy / offline persistence ------------------------------------
def _privacy_path() -> Path:
    """``cache_dir/privacy.json`` — where the persisted offline flag lives."""
    return Path(APP["cfg"].paths.cache_dir) / "privacy.json"


def _read_privacy_offline() -> bool:
    """Read the persisted offline flag. Corrupt/missing file -> ``False`` (online)."""
    try:
        data = json.loads(_privacy_path().read_text(encoding="utf-8"))
        return bool(data.get("offline", False))
    except Exception:  # noqa: BLE001 — missing / unreadable / bad JSON: default online
        return False


def _write_privacy_offline(offline: bool) -> None:
    """Persist the offline flag atomically (.tmp -> os.replace). Best-effort."""
    p = _privacy_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"offline": bool(offline)}), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass  # non-fatal: the in-process flag is already set; persistence is a bonus


# --- P2-#5: swappable local models (Whisper + LLM) persistence ---------------
def _models_path() -> Path:
    """``cache_dir/models.json`` — where the persisted model choices live."""
    return Path(APP["cfg"].paths.cache_dir) / "models.json"


def _read_models() -> dict:
    """Read persisted ``{whisper, llm}``. Corrupt/missing -> ``{}`` (cfg defaults)."""
    try:
        data = json.loads(_models_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001 — missing / unreadable / bad JSON: defaults
        pass
    return {}


def _write_models(whisper: str, llm: str) -> None:
    """Persist the current model choices atomically (.tmp -> os.replace). Best-effort."""
    p = _models_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"whisper": whisper, "llm": llm}), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass  # non-fatal: the in-process cfg is already mutated; persistence is a bonus


def _apply_saved_models(cfg: Config) -> None:
    """Apply persisted models.json choices onto ``cfg`` (called at startup).

    Only a whitelisted Whisper name and a non-empty LLM string are honoured;
    anything else is ignored so a stale/edited file can't wedge the editor.
    """
    saved = _read_models()
    w = saved.get("whisper")
    if isinstance(w, str) and w in WHISPER_ALLOWED:
        cfg.transcribe.model = w
    m = saved.get("llm")
    if isinstance(m, str) and m.strip():
        cfg.llm.model = m.strip()


# --- B5: detection parameters (per-request overrides + detect_ui.json) -------
# The UI can tune detection without touching config.yaml. The canonical option
# shape (everything optional):
#   pause_min_silence: float   (clamped 0.3..2.0)  -> pauses.min_silence
#   pause_padding:     float   (clamped 0..0.3)    -> pauses.pad_start/pad_end
#   hesitation_sensitivity: float (clamped 0..1)   -> hesitations.* (see mapping)
#   detectors: {pauses, fillers, profanity, hesitations, badtakes: bool}
# No options at all -> run_detection receives the session cfg UNCHANGED (the
# default behaviour is byte-for-byte the pre-B5 one; regression-tested).
_DETECTOR_KEYS = ("pauses", "fillers", "profanity", "hesitations", "badtakes")


def _detect_opts_path() -> Path:
    """``cache_dir/detect_ui.json`` — where the persisted detect options live."""
    return Path(APP["cfg"].paths.cache_dir) / "detect_ui.json"


def _sanitize_detect_opts(raw: dict, *, strict: bool = True) -> dict:
    """Validate + clamp UI detect options into the canonical persisted shape.

    ``strict=True`` (POST /api/detect body): a wrong TYPE is a 400 — clear
    feedback instead of silently ignoring a typo. Out-of-range NUMBERS are
    clamped, not rejected (the UI slider needn't know the server's bounds).
    ``strict=False`` (reading detect_ui.json back): a hand-edited / corrupt
    value is silently dropped — a bad file must never 500 every /api/state.
    Unknown keys are ignored in both modes (forward compatibility).
    """
    out: dict = {}

    def _num(key: str, lo: float, hi: float) -> None:
        if key not in raw or raw[key] is None:
            return
        v = raw[key]
        # bool is an int subclass — true/false is not a number here.
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(float(v))):
            if strict:
                raise HTTPException(400, f"{key}: ожидается число")
            return
        out[key] = round(min(hi, max(lo, float(v))), 3)

    _num("pause_min_silence", 0.3, 2.0)
    _num("pause_padding", 0.0, 0.3)
    _num("hesitation_sensitivity", 0.0, 1.0)

    det = raw.get("detectors")
    if det is not None:
        if not isinstance(det, dict):
            if strict:
                raise HTTPException(400, "detectors: ожидается объект {имя: true/false}")
        else:
            clean: dict = {}
            for k in _DETECTOR_KEYS:
                if k not in det:
                    continue
                v = det[k]
                if not isinstance(v, bool):
                    if strict:
                        raise HTTPException(400, f"detectors.{k}: ожидается true/false")
                    continue
                clean[k] = v
            if clean:
                out["detectors"] = clean
    return out


def _read_detect_opts() -> Optional[dict]:
    """Persisted detect options, sanitized. ``None`` if never saved / corrupt."""
    try:
        data = json.loads(_detect_opts_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing / unreadable / bad JSON
        return None
    if not isinstance(data, dict):
        return None
    return _sanitize_detect_opts(data, strict=False)


def _write_detect_opts(opts: dict) -> None:
    """Persist sanitized detect options atomically (.tmp -> os.replace).

    Best-effort like privacy.json / _save_queue: the request already holds the
    effective options in memory; persistence is a bonus, never a 500.
    """
    p = _detect_opts_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(opts, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


# --- авто-обогащение: настройки запуска (ENRICH_PLAN §5, cache/enrich_ui.json) --
_ENRICH_TYPE_KEYS = ("image", "animation", "list_card", "cta")
_ENRICH_DENSITIES = ("min", "normal", "aggressive")
# "generate" = локальная SD-генерация (ТРЕК-2 §2); "auto" тоже умеет SD (папка →
# SD → эмодзи → none). Должен совпадать с whitelist'ом sanitize_params (enrich.py).
_ENRICH_IMAGE_SOURCES = ("auto", "emoji", "user_folder", "generate")


def _enrich_opts_path() -> Path:
    """``cache_dir/enrich_ui.json`` — персист настроек «Предложить монтаж»."""
    return Path(APP["cfg"].paths.cache_dir) / "enrich_ui.json"


def _default_enrich_opts() -> dict:
    return {"types": {k: True for k in _ENRICH_TYPE_KEYS},
            "density": "normal", "image_source": "auto",
            "user_folder": "", "stocks": {"enabled": False}}


def _sanitize_enrich_opts(raw, *, strict: bool = True) -> dict:
    """Whitelist-sanitize настроек обогащения в канонический вид (паттерн B5
    _sanitize_detect_opts): ``strict=True`` (тело POST /api/enrich/suggest) —
    неверный ТИП/значение вне белого списка → 400; ``strict=False`` (чтение
    enrich_ui.json) — мусор молча заменяется дефолтом (битый файл не должен
    ронять /api/state). Неизвестные ключи игнорируются в обоих режимах.

    ``stocks.enabled`` принудительно False: стоки — Tier 2 (v1.1, строго
    opt-in OFF); до их появления персист не имеет права хранить заранее
    взведённый облачный тумблер (§4/§9 — приватность)."""
    out = _default_enrich_opts()
    if not isinstance(raw, dict):
        if strict:
            raise HTTPException(400, "Настройки обогащения: ожидается объект")
        return out

    t = raw.get("types")
    if t is not None:
        if not isinstance(t, dict):
            if strict:
                raise HTTPException(
                    400, "types: ожидается объект {тип: true/false}")
        else:
            for k in _ENRICH_TYPE_KEYS:
                if k not in t:
                    continue
                v = t[k]
                if not isinstance(v, bool):
                    if strict:
                        raise HTTPException(400, f"types.{k}: ожидается "
                                                 "true/false")
                    continue
                out["types"][k] = v

    for key, allowed in (("density", _ENRICH_DENSITIES),
                         ("image_source", _ENRICH_IMAGE_SOURCES)):
        v = raw.get(key)
        if v is None:
            continue
        if v in allowed:
            out[key] = v
        elif strict:
            raise HTTPException(400, f"{key}: допустимы "
                                     + ", ".join(allowed))

    uf = raw.get("user_folder")
    if uf is not None:
        if isinstance(uf, str):
            out["user_folder"] = uf.strip()
        elif strict:
            raise HTTPException(400, "user_folder: ожидается строка-путь")

    st = raw.get("stocks")
    if st is not None and not isinstance(st, dict) and strict:
        raise HTTPException(400, "stocks: ожидается объект {enabled: false}")
    # v1: стоков нет — что бы ни прислали/ни лежало в файле, enabled=False.
    return out


def _read_enrich_opts() -> Optional[dict]:
    """Сохранённые настройки обогащения. ``None`` — не настраивали / битый файл
    (= дефолты, паттерн detect_opts)."""
    try:
        data = json.loads(_enrich_opts_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing / unreadable / bad JSON
        return None
    if not isinstance(data, dict):
        return None
    return _sanitize_enrich_opts(data, strict=False)


def _write_enrich_opts(opts: dict) -> None:
    """Персист настроек атомарно (.tmp -> os.replace). Best-effort как
    detect_ui.json: настройки уже применены к задаче, неудача записи — не 500."""
    p = _enrich_opts_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(opts, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def _apply_detect_opts(cfg: Config, opts: Optional[dict]) -> Config:
    """Effective detection config = ``cfg`` + sanitized UI options.

    ``opts`` empty/None -> returns ``cfg`` ITSELF (same object), so the default
    path hands run_detection an unchanged config — byte-for-byte the old
    behaviour. Otherwise a DEEP COPY is modified and returned; the session cfg
    is never mutated.

    ``hesitation_sensitivity`` mapping (s in [0, 1], honest and symmetric
    around the config.yaml tuning): f = 2*s - 1 in [-1, +1], so s=0.5 (f=0)
    reproduces the configured values EXACTLY, s=0 «реже режет», s=1
    «агрессивнее». Three knobs move together:
      * min_duration  *= (1 - 0.5*f)   — s=0: only LONGER stumbles are flagged
        (1.5x the configured minimum gap); s=1: gaps half as short also count.
        Floor 0.04 s (≈1 video frame — below that a cut is imperceptible),
        ceiling max_duration (above it pauses own the gap).
      * vad_threshold += 0.15*f        — Silero speech probability needed to
        call «speech». s=0: lower threshold -> more audio counted as speech ->
        fewer non-speech gaps; s=1: higher -> more audio counted as non-speech
        -> more candidate gaps. Clamped to [0.15, 0.6] (outside that Silero
        becomes all-speech / all-noise and the gaps are garbage).
      * pad_start/pad_end *= (1 - 0.5*f) — breathing room kept at each gap
        edge. s=0: 1.5x pads (gentler, keeps more context); s=1: 0.5x pads
        (eats closer to the words). Clamped to [0, 0.12].
    ``max_duration`` is deliberately NOT touched: it is the contract line with
    the pause detector (keep max_duration <= pauses.min_silence) — moving it
    would double-flag pauses, which is dishonest «aggressiveness».
    """
    if not opts:
        return cfg
    c = cfg.model_copy(deep=True)
    if "pause_min_silence" in opts:
        c.pauses.min_silence = float(opts["pause_min_silence"])
    if "pause_padding" in opts:
        v = float(opts["pause_padding"])
        c.pauses.pad_start = v
        c.pauses.pad_end = v
    if "hesitation_sensitivity" in opts:
        f = 2.0 * float(opts["hesitation_sensitivity"]) - 1.0   # [-1 .. +1]
        h = c.hesitations
        h.min_duration = round(
            min(h.max_duration, max(0.04, h.min_duration * (1.0 - 0.5 * f))), 3)
        h.vad_threshold = round(min(0.6, max(0.15, h.vad_threshold + 0.15 * f)), 3)
        h.pad_start = round(min(0.12, max(0.0, h.pad_start * (1.0 - 0.5 * f))), 3)
        h.pad_end = round(min(0.12, max(0.0, h.pad_end * (1.0 - 0.5 * f))), 3)
    det = opts.get("detectors") or {}
    if "pauses" in det:
        c.pauses.enabled = bool(det["pauses"])
    if "fillers" in det:
        c.fillers.enabled = bool(det["fillers"])
    if "profanity" in det:
        c.profanity.enabled = bool(det["profanity"])
    if "hesitations" in det:
        c.hesitations.enabled = bool(det["hesitations"])
    if "badtakes" in det:
        # run_detection short-circuits on bad_takes.enabled BEFORE touching the
        # llm argument, so disabling bad takes here guarantees no LLM call.
        c.bad_takes.enabled = bool(det["badtakes"])
    return c


# --- P2-#6: batch-queue persistence (queue.json) -----------------------------
# The queue survives a server restart: every meaningful mutation (add/remove/
# clear) and every job status-transition (pending->running->done/error) is
# flushed to cache_dir/queue.json atomically. Progress ticks (percent/stage) are
# NOT persisted — they fire hundreds of times per job and a 'running' job is
# reset to 'pending' on load anyway. Best-effort throughout: the in-memory QUEUE
# is the source of truth, so a failed write never breaks an endpoint or the worker.

# Fields of QueueJob that can be safely restored from json. ``name`` is derived
# in _make_job_dict (from Path(path).name) and is recomputed on load, so it is
# deliberately excluded — feeding it to QueueJob(**...) would raise TypeError.
_QUEUE_JOB_FIELDS = frozenset(
    {"id", "path", "out_dir", "render_opts", "status",
     "percent", "stage", "result", "error"}
)
_VALID_STATUSES = frozenset({"pending", "running", "done", "error"})


def _queue_path() -> Path:
    """``cache_dir/queue.json`` — where the persisted batch queue lives."""
    return Path(APP["cfg"].paths.cache_dir) / "queue.json"


def _save_queue() -> None:
    """Persist QUEUE to queue.json atomically. Best-effort: never raises.

    The list snapshot is taken under QUEUE_LOCK so it is internally consistent;
    the file write happens OUTSIDE the lock so the worker is never blocked on
    disk I/O. A stale write loses nothing important — the in-memory queue wins.
    """
    with QUEUE_LOCK:
        snapshot = [_make_job_dict(j) for j in QUEUE]
    p = _queue_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        # ensure_ascii=False: stage/error carry Cyrillic; keep it human-readable.
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001 — best-effort; the queue in memory is fine
        pass


def _load_queue() -> None:
    """Load queue.json into the global QUEUE (called once at startup).

    Rules:
      * missing / corrupt / non-list file -> empty queue, never crashes;
      * a job left 'running' (server died mid-job) -> reset to 'pending' with
        percent/stage cleared, so it re-runs when the user presses «Старт»;
      * 'done'/'error' jobs are kept AS-IS (result/error preserved) so links to
        /api/output still resolve;
      * unknown keys ignored, missing keys -> dataclass defaults, invalid
        status -> 'pending';
      * entries missing required id/path/out_dir are skipped;
      * does NOT start the worker — the user starts the queue manually.
    """
    p = _queue_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing / unreadable / bad JSON
        return
    if not isinstance(raw, list):
        return

    jobs: list[QueueJob] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # Keep only known fields so QueueJob(**filtered) can never raise on an
        # unexpected/forward-compat key.
        filtered = {k: v for k, v in entry.items() if k in _QUEUE_JOB_FIELDS}

        # Required string fields — skip the whole entry if any is bad.
        if not isinstance(filtered.get("id"), str) or not filtered["id"]:
            continue
        if not isinstance(filtered.get("path"), str) or not filtered["path"]:
            continue
        if not isinstance(filtered.get("out_dir"), str) or not filtered["out_dir"]:
            continue

        # render_opts must be a dict.
        if not isinstance(filtered.get("render_opts"), dict):
            filtered["render_opts"] = {}

        # Normalize status; anything invalid/missing -> 'pending'.
        if filtered.get("status") not in _VALID_STATUSES:
            filtered["status"] = "pending"
        # A job that was mid-flight when the server died restarts from scratch.
        if filtered["status"] == "running":
            filtered["status"] = "pending"
            filtered["percent"] = 0.0
            filtered["stage"] = ""

        # Coerce percent to float (json may carry int/None/garbage).
        try:
            filtered["percent"] = float(filtered.get("percent", 0.0))
        except (TypeError, ValueError):
            filtered["percent"] = 0.0

        # stage must be a str; result Optional[dict]; error Optional[str].
        if not isinstance(filtered.get("stage"), str):
            filtered["stage"] = ""
        if filtered.get("result") is not None and not isinstance(filtered["result"], dict):
            filtered["result"] = None
        if filtered.get("error") is not None and not isinstance(filtered["error"], str):
            filtered["error"] = None

        try:
            job = QueueJob(**filtered)
        except TypeError:
            continue  # safety net: unexpected signature mismatch
        jobs.append(job)

    with QUEUE_LOCK:
        QUEUE.extend(jobs)


def _queue_pending_count() -> int:
    """How many jobs still need work (pending + running). Snapshot under
    QUEUE_LOCK. Surfaced in /api/state so the «📋 Очередь» badge can light up
    on first load — including a queue restored from queue.json while the worker
    is stopped — without first opening the modal or starting the poll."""
    with QUEUE_LOCK:
        return sum(1 for j in QUEUE if j.status in ("pending", "running"))


def _llm_host_local() -> bool:
    """True if the configured Ollama host is loopback — i.e. the transcript text
    (content of the video) stays on this machine. A user who points ``llm.host``
    at a remote Ollama is sending the transcript there; the privacy UI must say so
    honestly rather than wave every external host away as 'just a model download'."""
    try:
        import ipaddress
        from urllib.parse import urlparse
        host = urlparse(APP["cfg"].llm.host).hostname or ""
        if host in ("localhost", ""):
            return True
        return ipaddress.ip_address(host).is_loopback
    except Exception:  # noqa: BLE001 — unparseable host: don't cry wolf
        return True


def _network_summary(offline: bool, st: dict) -> str:
    """A short, HONEST Russian one-liner describing the current network posture.

    We deliberately do not claim 'нет ваших данных' for arbitrary external hosts:
    we only know what netguard counted, and a misconfigured remote LLM host would
    receive the transcript. We name the hosts and explain the usual cause."""
    ext = st.get("external_allowed", 0)
    blocked = st.get("blocked", 0)
    if offline:
        base = "Оффлайн-режим: исходящие сетевые соединения процесса блокируются."
        if blocked:
            base += f" Заблокировано: {blocked}."
        return base
    parts = []
    if ext == 0:
        parts.append("Всё локально — ни одного внешнего соединения процесса.")
    else:
        hosts = ", ".join(sorted(st.get("external_hosts", {}))) or "—"
        parts.append(f"Внешних соединений: {ext} ({hosts}) — обычно разовая "
                     f"загрузка модели Whisper. Ваше видео не отправляется.")
    if not _llm_host_local():
        parts.append("Внимание: ИИ-хост (Ollama) НЕ локальный — на него уходит "
                     "текст транскрипта. Используйте Ollama на localhost.")
    return " ".join(parts)


# --- shared render helpers (reused by /api/render AND the F3 queue worker) ----
def _resolve_render_opts(s: Session, opts: dict):
    """Translate UI/queue render-opts into (cfg, scale_h, fps, out_dir, base).

    Identical override logic for the single-clip editor and the batch queue, so
    a queued render behaves exactly like the manual one. Raises HTTPException on
    bad scale_h/fps (the queue worker turns that into a job error)."""
    cfg = s.cfg.model_copy(deep=True)
    if opts.get("encoder") in ("nvenc", "x264"):
        cfg.render.encoder = opts["encoder"]
    q = opts.get("quality")
    if isinstance(q, (int, float)):
        q = max(0, min(51, int(q)))   # H.264/HEVC QP/CRF range
        cfg.render.nvenc.qp = q
        cfg.render.nvenc.cq = q
        cfg.render.x264.crf = q
    if opts.get("audio_bitrate"):
        cfg.render.audio_bitrate = str(opts["audio_bitrate"])
    if opts.get("censor_method") in ("partial", "pitch", "lowpass", "reverse"):
        cfg.censor.method = opts["censor_method"]
    if "subtitles" in opts:
        cfg.subtitles.enabled = bool(opts["subtitles"])
    if "chapters" in opts:
        cfg.chapters.enabled = bool(opts["chapters"])
    # Clip Maker (план §2.3.2): явный тумблер LLM-метаданных. Без ключа —
    # прежнее поведение (значение из config.yaml); рендер клипов шлёт false,
    # иначе КАЖДЫЙ клип гонял бы LLM и перезаписывал общий metadata.txt.
    if "metadata" in opts:
        cfg.metadata.enabled = bool(opts["metadata"])

    # --- A: burn-in (вшитые) subtitles + style -------------------------------
    if opts.get("burn_subtitles"):
        b = cfg.subtitles.burn
        b.enabled = True
        bs = opts.get("burn_style") or {}
        if bs.get("font"):
            b.font = str(bs["font"])
        if bs.get("size"):
            try:
                b.size = max(8, min(200, int(bs["size"])))
            except (TypeError, ValueError):
                pass
        if bs.get("primary_color"):
            b.primary_color = str(bs["primary_color"])
        if bs.get("karaoke_color"):
            b.karaoke_color = str(bs["karaoke_color"])
        if bs.get("outline_color"):
            b.outline_color = str(bs["outline_color"])
        if bs.get("position") in ("bottom", "top", "center"):
            b.position = bs["position"]
        b.karaoke = bool(bs.get("karaoke", True))
        # C1 (пресеты стилей): числовые поля, которых нет среди «сырых» полей
        # UI — приходят только целым пресетом. Клампы — здравые пределы ASS;
        # мусор молча игнорируется (значение из config.yaml остаётся).
        for key, lo, hi in (("outline", 0.0, 10.0), ("shadow", 0.0, 10.0)):
            if bs.get(key) is not None:
                try:
                    setattr(b, key, max(lo, min(hi, float(bs[key]))))
                except (TypeError, ValueError):
                    pass
        if bs.get("margin_v") is not None:
            try:                                 # PlayRes-пиксели (выход ≤4K)
                b.margin_v = max(0, min(800, int(bs["margin_v"])))
            except (TypeError, ValueError):
                pass
    else:
        cfg.subtitles.burn.enabled = False

    # --- C: vertical 9:16 Shorts clip with auto face-crop --------------------
    if opts.get("vertical"):
        v = cfg.render.vertical
        v.enabled = True
        if opts.get("vertical_target"):
            v.target = str(opts["vertical_target"])
        center = opts.get("vertical_center")
        if center is None or center == "" or center == "auto":
            v.center = "auto"
        else:
            try:                                     # explicit float in [0, 1]
                v.center = f"{min(1.0, max(0.0, float(center))):.4f}"
            except (TypeError, ValueError):
                v.center = "auto"
    else:
        cfg.render.vertical.enabled = False

    # --- denoise / speech enhancement (applied AFTER profanity censoring) ----
    # OFF by default: audio is an irreversible render output, so the user opts in.
    if opts.get("denoise"):
        dn = cfg.render.denoise
        dn.enabled = True
        strength = opts.get("denoise_strength")
        if strength is not None:
            try:                                 # UI slider: afftdn noise floor dB
                dn.nf = max(-45.0, min(-6.0, float(strength)))
            except (TypeError, ValueError):
                pass
        hp = opts.get("denoise_highpass")
        if hp is not None:
            try:                                 # 0 disables the highpass stage
                dn.highpass_hz = max(0, min(300, int(hp)))
            except (TypeError, ValueError):
                pass
        if "denoise_normalize" in opts:
            dn.normalize = bool(opts["denoise_normalize"])
        # Движок: "afftdn" (ffmpeg, как раньше) | "deepfilter" (нейро, DFN3
        # через tools/deep-filter.exe; при недоступном exe рендер сам честно
        # откатывается на afftdn). Whitelist: иное значение игнорируется,
        # отсутствие ключа оставляет значение из config.yaml (дефолт afftdn).
        eng = opts.get("denoise_engine")
        if eng in ("afftdn", "deepfilter"):
            dn.engine = eng
    else:
        cfg.render.denoise.enabled = False

    # --- audio mastering: de-esser + YouTube loudness (-14 LUFS) -------------
    # INDEPENDENT of the denoise toggle (build_apost applies them even when
    # denoise is off). Same opt-in contract: absent/falsy in opts -> off.
    cfg.render.denoise.deess = bool(opts.get("denoise_deess"))
    cfg.render.denoise.loudnorm = bool(opts.get("denoise_loudnorm"))
    # Точность loudnorm: "dynamic" (однопроходный, как раньше) | "2pass"
    # (быстрый измерительный пасс + linear-нормализация). Whitelist: любое
    # другое значение игнорируется; отсутствие ключа оставляет значение из
    # config.yaml (дефолт dynamic) — поведение по умолчанию не меняется.
    lm = opts.get("loudnorm_mode")
    if lm in ("dynamic", "2pass"):
        cfg.render.denoise.loudnorm_mode = lm

    # --- cut seam smoothing (audio de-click fades) ---------------------------
    # UI sends seconds; clamp to a sane 0..0.06 s. 0 = legacy hard cuts.
    if "cut_fade" in opts and opts["cut_fade"] is not None:
        try:
            cfg.render.cut_fade = max(0.0, min(0.06, float(opts["cut_fade"])))
        except (TypeError, ValueError):
            pass

    # --- C3: фоновая музыка + авто-дакинг (sidechaincompress) ----------------
    # Контракт как у denoise: без явного opts.music музыка ВЫКЛЮЧЕНА — даже
    # если config.yaml включил её в сессии. Именно это серверно гарантирует
    # клипам Clip Maker рендер БЕЗ подложки: clips_render и Авто-пак шлют
    # сюда render_opts с music=None. Битый путь/расширение — 400 на запросе
    # (а не ошибка задачи); громкости клампятся, мусор в них игнорируется.
    mu = opts.get("music")
    if isinstance(mu, dict) and mu.get("enabled"):
        mc = cfg.render.music
        p = str(mu.get("path") or "").strip()
        if not p:
            raise HTTPException(400, "Фоновая музыка: укажи путь к аудиофайлу")
        pp = Path(p).expanduser()
        if pp.suffix.lower() not in MUSIC_EXT:
            raise HTTPException(
                400, "Фоновая музыка: неподдерживаемый формат "
                     f"«{pp.suffix or 'без расширения'}». Аудио: "
                     + ", ".join(sorted(AUDIO_EXT)) + " или видеофайл.")
        if not pp.is_file():
            raise HTTPException(400, f"Фоновая музыка: файл не найден: {p}")
        mc.enabled = True
        mc.path = str(pp.resolve())
        if mu.get("gain_db") is not None:
            try:                              # слайдер «Громкость музыки», дБ
                mc.gain_db = max(-40.0, min(0.0, float(mu["gain_db"])))
            except (TypeError, ValueError):
                pass
        if mu.get("duck_db") is not None:
            try:                              # слайдер «Приглушение при речи»
                mc.duck_db = max(-30.0, min(0.0, float(mu["duck_db"])))
            except (TypeError, ValueError):
                pass
    else:
        cfg.render.music.enabled = False

    # --- авто-обогащение (ENRICH_PLAN §5) ------------------------------------
    # Контракт как у music: без явного opts.enrich = {enabled:true} обогащение
    # ВЫКЛЮЧЕНО — даже если config.yaml включил его в сессии. Любой мусор
    # вместо объекта (строка/число/null) тоже честно выключает. min_score —
    # служебная ручка Авто-пака (score>=70 поверх enabled, без ревью); мусор
    # в ней игнорируется, значение клампится в 0..100.
    en = opts.get("enrich")
    if isinstance(en, dict) and en.get("enabled"):
        cfg.render.enrich.enabled = True
        ms = en.get("min_score")
        if ms is not None and not isinstance(ms, bool):
            try:                       # NaN → ValueError, ±inf → OverflowError
                cfg.render.enrich.min_score = max(0, min(100, int(ms)))
            except (TypeError, ValueError, OverflowError):
                pass
    else:
        cfg.render.enrich.enabled = False

    # --- scale_h: None (source) or an int in [144, 4320] ---------------------
    # When vertical is on, the exact target scale is baked into the crop filter,
    # so the generic height resize is ignored (forced to None) to avoid a double
    # scale and an off-by-a-pixel width from scale=-2.
    scale_h = opts.get("scale_h") or None        # None = source resolution
    if cfg.render.vertical.enabled:
        scale_h = None
    elif scale_h is not None:
        try:
            scale_h = int(scale_h)
        except (TypeError, ValueError):
            raise HTTPException(400, "scale_h must be an integer (144–4320)")
        if not (144 <= scale_h <= 4320):
            raise HTTPException(400, "scale_h must be between 144 and 4320")
        if scale_h == s.media.height:            # identity -> let copy fast-path win
            scale_h = None

    # --- fps: None (source) or 0 < fps <= 120 --------------------------------
    fps = opts.get("fps") or None                # None = source fps
    if fps is not None:
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            raise HTTPException(400, "fps must be a number (0 < fps <= 120)")
        if not (0 < fps <= 120):
            raise HTTPException(400, "fps must be between 0 and 120")
        if abs(fps - s.media.fps) < 0.01:        # identity -> let copy fast-path win
            fps = None

    out_dir = Path(opts.get("out_dir") or s.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path((opts.get("filename") or s.inp.stem).strip() or s.inp.stem).stem
    base = out_dir / stem
    return cfg, scale_h, fps, out_dir, base


def _final_render_dims(s: Session, scale_h,
                       vert_dims: Optional[tuple[int, int]]) -> tuple[int, int]:
    """(W, H) финального кадра — для PlayRes ASS-файлов и планировщика enrich.

    Ровно та логика, что исторически жила в ass-блоке _run_render_pipeline:
    vertical → точная цель кропа; иначе высота = scale_h (или исходная),
    ширина — пропорциональна источнику."""
    if vert_dims is not None:
        return vert_dims
    out_h = int(scale_h) if scale_h else (s.media.height or 1080)
    src_w = s.media.width or 1920
    src_h = s.media.height or 1080
    out_w = int(round(src_w * out_h / src_h)) if src_h else 1920
    return out_w, out_h


def _run_render_pipeline(s: Session, cfg, scale_h, fps, out_dir: Path,
                         base: Path, on_progress, on_stage,
                         cutlist_override: Optional[CutList] = None,
                         edge_fade: float = 0.0,
                         sidecar_base: Optional[Path] = None) -> dict:
    """Render mp4 + (optional) subtitles + chapters, returning the results dict.

    ``on_progress(frac)`` / ``on_stage(msg)`` are callbacks so the same body
    drives both the editor task (writes Session.task) and a queue job (writes
    QueueJob.percent/stage). Mirrors do_render: the mp4 is the irreplaceable
    artifact — a subtitles/chapters failure must NOT lose it.

    ``cutlist_override`` (Clip Maker, план §2.3.1): render against THIS cutlist
    instead of the session's (one Shorts clip = the live internal cuts plus
    boundary REMOVEs around [start, end]). Burn-in ASS subs and the Timeline are
    built from the SAME cutlist, and the session is never mutated. ``None``
    (the default) keeps the legacy single-render behavior bit-for-bit.

    ``edge_fade`` (Clip Maker F8): seconds of de-click afade-in/out on the TRUE
    edges of the clip's final audio. An EXPLICIT parameter (not a heuristic on
    cutlist_override): only /api/clips/render passes a non-zero value (from
    cfg.clips.edge_fade); every regular full-video render keeps the default 0.0
    and gets no edge fades.

    ``sidecar_base`` (C2 multi-format): base path for the SIDECAR subtitle files
    (.srt/.vtt) when it must differ from the mp4 base — the formats loop names
    the mp4 ``<stem>_9x16.mp4`` etc. but subtitles do not depend on the crop, so
    they are written once as plain ``<stem>.srt``. ``None`` (default) keeps the
    legacy behavior: sidecars share ``base`` with the mp4."""
    cl = cutlist_override or s.cutlist
    tr = s.transcript
    removed, _ = resolve(cl)

    # C: vertical 9:16 Shorts crop. Detect the face X-center (center='auto') once
    # here — render() stays pure and just receives the finished crop,scale filter.
    # detect_center never raises (graceful 0.5 fallback when cv2/face is absent),
    # so a probe failure cannot lose the mp4. crop_filter() returns None when the
    # source is already at/under the target aspect (then we just let it scale).
    crop_filter: Optional[str] = None
    vert_dims: Optional[tuple[int, int]] = None     # (W, H) of the vertical output
    if cfg.render.vertical.enabled:
        vcfg = cfg.render.vertical
        tw, th = facecrop_mod.parse_target(vcfg.target)
        src_w = s.media.width or 1920
        src_h = s.media.height or 1080
        if vcfg.center == "auto":
            if facecrop_mod.cv2_available():
                on_stage("Поиск лица для авто-кадра…")
            # Clip Maker (план §2.4): для клипа сэмплируем лицо ТОЛЬКО внутри
            # его диапазона (спикер мог сместиться за 26 минут). Диапазон —
            # выживший спан override-катлиста (граничные REMOVE его и задают);
            # без override дефолты detect_center = прежний проход по всему файлу.
            fc_start, fc_end = 0.0, None
            if cutlist_override is not None:
                kept = Timeline(removed, s.media.duration).kept_segments()
                if kept:
                    fc_start, fc_end = kept[0][0], kept[-1][1]
            cx = facecrop_mod.detect_center(
                s.media.path, s.ff, s.media.duration, samples=vcfg.samples,
                start=fc_start, end=fc_end)
        else:
            try:
                cx = min(1.0, max(0.0, float(vcfg.center)))
            except (TypeError, ValueError):
                cx = 0.5
        # vertical_filter always yields an exact-target crop,scale (face-aware
        # crop for landscape; cover+center-crop for already-vertical sources).
        crop_filter = facecrop_mod.vertical_filter(src_w, src_h, cx, (tw, th))
        vert_dims = (tw, th)

    # A: generate the burn-in ASS in FINAL coordinates (same pipeline as
    # /api/preview/subtitles) and hand its path to render(). PlayResX/Y match the
    # rendered resolution so style sizes/margins are pixel-accurate. A failure
    # here must NOT lose the render — fall back to no burn.
    ass_path: Optional[str] = None
    if cfg.subtitles.burn.enabled and tr is not None:
        try:
            on_stage("Подготовка вшитых субтитров…")
            tl_ass = Timeline(removed, s.media.duration)
            words_final = remap_words(tr.all_words(), tl_ass)
            cues_ass = subs_mod.build_cues(
                words_final, s.matcher, cfg.subtitles, cfg.masking,
                tl_ass.new_duration())
            # Vertical: PlayRes must match the cropped+scaled output so style
            # sizes/margins are pixel-accurate against the 9:16 frame.
            out_w, out_h = _final_render_dims(s, scale_h, vert_dims)
            # Clip Maker (план §2.3.4): per-clip имя ASS, чтобы клипы цикла не
            # перетирали один burn.ass; обычный рендер — прежнее имя бит-в-бит.
            ass_name = ("burn.ass" if cutlist_override is None
                        else f"burn_{base.name}.ass")
            ass_file = Path(s.work_dir) / ass_name
            subs_mod.write_ass(
                cues_ass, ass_file, cfg.subtitles.burn,
                karaoke=cfg.subtitles.burn.karaoke,
                words=words_final, matcher=s.matcher, mask=cfg.masking,
                play_res=(out_w, out_h))
            ass_path = str(ass_file)
        except Exception as e:  # noqa: BLE001 — burn is best-effort
            on_stage(f"Вшитые субтитры: не удалось подготовить ({e}); рендер без них.")
            ass_path = None

    # ENRICH (ENRICH_PLAN §5): оверлеи авто-обогащения — ТОЛЬКО основной рендер.
    # Клипы Clip Maker и автопак-клипы идут по cutlist_override-пути и
    # обогащение не получают (анти-скоуп §9). mp4 свят (паттерн burn-блока):
    # нет плана / несвежий hash / любой сбой подготовки → лог-warning через
    # on_stage и рендер БЕЗ обогащения, задача не падает.
    render_enrich = None
    if cfg.render.enrich.enabled and cutlist_override is None:
        try:
            plan = enrich_mod.load_enrich(_enrich_json_path(s))
            if plan is None:
                on_stage("Обогащение: план не найден (enrich.json) — "
                         "рендер без обогащения.")
            elif plan.hash != s.audio_hash:
                on_stage("Обогащение: план от другого видео (hash не совпал) — "
                         "рендер без обогащения.")
            else:
                ms = int(cfg.render.enrich.min_score or 0)
                if ms > 0:           # автопак: score>=70 ПОВЕРХ enabled (§5)
                    plan.items = [it for it in plan.items if it.score >= ms]
                tl_enr = Timeline(removed, s.media.duration)
                out_w, out_h = _final_render_dims(s, scale_h, vert_dims)
                re_plan = enrich_mod.plan_render(
                    plan, tl_enr, tr.all_words() if tr is not None else None,
                    cfg, out_w, out_h,
                    log=lambda m="": on_stage(str(m).strip() or "Обогащение…"))
                if re_plan.cards or re_plan.cta_texts:
                    # Карточки/вопрос CTA — отдельный ASS (идёт ПЕРВЫМ
                    # subtitles-фильтром, burn.ass остаётся последним, §2.2).
                    enr_ass = Path(s.work_dir) / f"enrich_{base.name}.ass"
                    enrich_cards.write_enrich_ass(
                        re_plan.cards, re_plan.cta_texts, out_w, out_h,
                        enr_ass)
                    re_plan.cards_ass = str(enr_ass)
                if re_plan.stills or re_plan.anims or re_plan.cards_ass:
                    render_enrich = re_plan
                else:
                    on_stage("Обогащение: в плане нет применимых предложений — "
                             "рендер без оверлеев.")
        except Exception as e:  # noqa: BLE001 — enrichment is best-effort
            on_stage(f"Обогащение: не удалось подготовить ({e}); рендер без него.")
            render_enrich = None

    on_stage("Рендер видео…")
    # enrich передаётся kwarg-ом только когда он есть: пустой/выключенный план
    # оставляет вызов render() бит-в-бит прежним (легаси-контракт).
    enrich_kw = {"enrich": render_enrich} if render_enrich is not None else {}
    rr = render_mod.render(
        s.ff, s.media, cl, cfg, base.with_suffix(".mp4"), s.work_dir,
        on_progress=on_progress,
        log=lambda m="": on_stage(str(m).strip() or "Рендер видео…"),
        scale_h=scale_h, fps=fps, ass_path=ass_path, crop_filter=crop_filter,
        edge_fade=edge_fade, **enrich_kw)

    sr: dict = {}
    subtitles_ok = not cfg.subtitles.enabled
    if cfg.subtitles.enabled:
        try:
            on_stage("Субтитры…")
            sr = subs_mod.generate(tr, removed, cfg.subtitles, cfg.masking,
                                   s.matcher, sidecar_base or base,
                                   log=lambda *_: None)
            subtitles_ok = True
        except Exception as e:  # noqa: BLE001 — keep the rendered mp4
            sr = {"error": str(e)}

    cr: dict = {}
    chapters_ok = not cfg.chapters.enabled
    if cfg.chapters.enabled:
        try:
            on_stage("Главы…")
            cr = chapters_mod.generate(tr, removed, cfg.chapters,
                                       out_dir / "chapters.txt", llm=s.llm,
                                       matcher=s.matcher, mask=cfg.masking,
                                       log=lambda *_: None, on_stage=on_stage)
            chapters_ok = True
        except Exception as e:  # noqa: BLE001 — keep the rendered mp4
            cr = {"error": str(e)}

    # B: YouTube metadata (title/description/tags/hook) -> out_dir/metadata.txt.
    # Same policy as chapters: a metadata failure must NOT lose the rendered mp4,
    # and it degrades silently when the LLM is off. Reuses the chapters.txt just
    # written above so the description can embed the chapter list.
    mr: dict = {}
    meta_path: Optional[Path] = None
    metadata_ok = not cfg.metadata.enabled
    if cfg.metadata.enabled and s.llm is not None:
        try:
            on_stage("Метаданные…")
            chapters_txt = out_dir / "chapters.txt"
            mr = metadata_mod.generate(
                tr, removed, cfg.metadata, s.llm,
                chapters_path=chapters_txt if chapters_txt.exists() else None,
                matcher=s.matcher, mask=cfg.masking, log=lambda *_: None)
            if mr.get("title") or mr.get("description"):
                meta_path = out_dir / "metadata.txt"
                meta_path.write_text(
                    "TITLE:\n" + mr.get("title", "") +
                    "\n\nHOOK:\n" + mr.get("hook", "") +
                    "\n\nDESCRIPTION:\n" + mr.get("description", "") +
                    "\n\nTAGS:\n" + ", ".join(mr.get("tags", [])) + "\n",
                    encoding="utf-8")
            metadata_ok = True
        except Exception as e:  # noqa: BLE001 — keep the rendered mp4
            mr = {"error": str(e)}

    tl = Timeline(removed, s.media.duration)
    return {
        "mp4": rr.get("out"), "encoder": rr.get("encoder"),
        "new_duration": round(tl.new_duration(), 1),
        "old_duration": round(s.media.duration, 1),
        "out_dir": str(out_dir.resolve()),
        "srt": sr.get("srt"), "vtt": sr.get("vtt"),
        "chapters": cr.get("path"), "n_chapters": cr.get("chapters", 0),
        "cues": sr.get("cues", 0),
        "burned_subtitles": bool(ass_path),
        "vertical": bool(crop_filter) or (cfg.render.vertical.enabled and vert_dims is not None),
        "metadata": mr.get("title", ""),
        "metadata_path": str(meta_path.resolve()) if meta_path else None,
        "succeeded": {"render": True,
                      "subtitles": subtitles_ok,
                      "chapters": chapters_ok,
                      "metadata": metadata_ok},
        "subtitles_error": sr.get("error"),
        "chapters_error": cr.get("error"),
        "metadata_error": mr.get("error"),
    }


# --- C2: мультиформат-рефрейм — один клик, несколько форматов вывода ----------
# (CapCut делает рефрейм в облаке за Pro; у нас — локально и бесплатно.)
# Поддерживается ТОЛЬКО основным рендером /api/render; очередь (F3) и Clip Maker
# (/api/clips/render) остаются одноформатными — TODO для следующей итерации.
RENDER_FORMATS = ("source", "9x16", "1x1", "16x9")
_FORMAT_LABEL = {"source": "Исходный", "9x16": "9:16", "1x1": "1:1",
                 "16x9": "16:9"}
_FORMAT_SUFFIX = {"source": "", "9x16": "_9x16", "1x1": "_1x1",
                  "16x9": "_16x9"}
# Аспект-кропы с разрешением источника (aspect_target). "9x16" сюда не входит —
# это прежний vertical-путь с каноничной целью Shorts (1080x1920 по умолчанию).
_FORMAT_ASPECT = {"1x1": (1, 1), "16x9": (16, 9)}


def _parse_formats(opts: dict) -> list[str]:
    """Validate/normalize the ``formats`` field of /api/render opts.

    * ``formats`` absent -> backward compatibility: legacy ``vertical: true``
      maps to ``["9x16"]``, otherwise ``["source"]`` (the pre-C2 behavior);
    * present -> must be a non-empty list of known formats (else 400);
      duplicates are dropped, client order is preserved.
    """
    raw = opts.get("formats")
    if raw is None:
        return ["9x16"] if opts.get("vertical") else ["source"]
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "formats: ожидается непустой список из "
                                 + ", ".join(RENDER_FORMATS))
    out: list[str] = []
    for f in raw:
        if f not in RENDER_FORMATS:
            raise HTTPException(400, f"Неизвестный формат вывода: {f!r} "
                                     f"(допустимы: {', '.join(RENDER_FORMATS)})")
        if f not in out:
            out.append(f)
    return out


def _render_formats(s: Session, opts: dict, formats: list[str],
                    on_progress, on_stage,
                    is_cancelled=lambda: False) -> dict:
    """Render the SAME cutlist in each requested output format, sequentially.

    Цикл по паттерну render_clips: каждый формат — свой полный прогон
    ``_resolve_render_opts`` + ``_run_render_pipeline`` (честное «время ×N»).
    Имена: ``<stem>.mp4`` / ``<stem>_9x16.mp4`` / ``<stem>_1x1.mp4`` /
    ``<stem>_16x9.mp4``. Общий percent = (i + frac)/N, stage «Формат i/N: …»
    (при N=1 — прежние «голые» stage/percent, бит-в-бит со старым рендером).

    Кропы: "9x16" — прежний vertical-путь (face-crop, цель 1080x1920);
    "1x1"/"16x9" — vertical-механика с целью ``aspect_target`` от размеров
    источника (face-crop по горизонтали переиспользуется). Формат, совпадающий
    с аспектом источника (16:9-кроп из 16:9-ролика, 9:16 из вертикального
    9:16-исходника, …), НЕ рендерится — в results он помечен ``skipped``
    («совпадает с исходным»). Для "9x16" аспект сверяется явно (тем же
    integer-точным правилом, что в ``aspect_target``); неизвестные размеры
    источника (0x0) совпадением не считаются — рендерим, как раньше.

    Сайдкары один раз: .srt/.vtt не зависят от кропа (пишутся как ``<stem>.srt``
    через ``sidecar_base``), chapters/metadata — тоже только при первом удачном
    прогоне; остальные форматы идут с subtitles/chapters/metadata=False.

    Результат = dict первого УСПЕШНОГО прогона (обратная совместимость с
    одноформатными потребителями: mp4/srt/chapters на верхнем уровне) плюс
    ``formats`` — массив по форматам. Упавший формат не валит остальные;
    если не удался НИ ОДИН (а попытки были) — задача падает, как раньше.
    """
    n = len(formats)
    stem = Path((opts.get("filename") or s.inp.stem).strip() or s.inp.stem).stem
    src_w = s.media.width or 0
    src_h = s.media.height or 0
    entries: list[dict] = []
    merged: Optional[dict] = None
    errors: list[str] = []
    sidecars_done = False
    for i, f in enumerate(formats):
        if is_cancelled():
            break                                  # cancel между форматами
        label = _FORMAT_LABEL[f]
        opts_i = {**opts, "filename": stem + _FORMAT_SUFFIX[f]}
        if f == "source":
            dup = False
            opts_i["vertical"] = False
        elif f == "9x16":
            # 9:16 из 9:16-источника — такой же дубль, как 16:9-кроп из
            # 16:9-ролика: сверяем аспект явно (integer-точное правило
            # aspect_target); неизвестные размеры (0x0) совпадением не считаем.
            dup = src_w > 0 and src_h > 0 and src_w * 16 == src_h * 9
            if not dup:
                opts_i["vertical"] = True          # цель/центр из opts, как раньше
        else:
            tgt = facecrop_mod.aspect_target(src_w, src_h, _FORMAT_ASPECT[f])
            dup = tgt is None
            if not dup:
                opts_i["vertical"] = True
                opts_i["vertical_target"] = f"{tgt[0]}x{tgt[1]}"
        if dup:
            entries.append({
                "format": f, "label": label, "ok": True, "skipped": True,
                "note": "совпадает с исходным форматом — пропущено "
                        "(дубль не рендерим)"})
            continue
        if sidecars_done:
            opts_i.update(subtitles=False, chapters=False, metadata=False)
        if n > 1:
            prog = (lambda fr, i=i: on_progress(
                (i + min(1.0, max(0.0, fr))) / n))
            stage = (lambda m, i=i, label=label: on_stage(
                f"Формат {i + 1}/{n}: {label} — {m}"))
            stage("подготовка…")
        else:
            prog, stage = on_progress, on_stage    # N=1: легаси бит-в-бит
        try:
            cfg, scale_h, fps, out_dir, base = _resolve_render_opts(s, opts_i)
            res = _run_render_pipeline(
                s, cfg, scale_h, fps, out_dir, base,
                on_progress=prog, on_stage=stage,
                sidecar_base=out_dir / stem)
            sidecars_done = True
            entries.append({
                "format": f, "label": label, "ok": True, "skipped": False,
                "mp4": res.get("mp4"), "encoder": res.get("encoder"),
                "vertical": res.get("vertical"),
                "burned_subtitles": res.get("burned_subtitles")})
            if merged is None:
                merged = res
        except Exception as e:  # noqa: BLE001 — упавший формат не валит остальные
            if n == 1:
                raise                              # легаси: единственный = ошибка задачи
            err = "cancelled" if is_cancelled() else str(e)
            errors.append(f"{label}: {err}")
            entries.append({"format": f, "label": label, "ok": False,
                            "error": err})
    if merged is None and errors:
        raise RuntimeError("Не удался ни один формат — " + "; ".join(errors))
    result = dict(merged) if merged is not None else {}
    result["formats"] = entries
    return result


def _make_job_dict(job: QueueJob) -> dict:
    """Serialize a QueueJob for the API (full state, polled by GET /api/queue)."""
    return {
        "id": job.id,
        "path": job.path,
        "name": Path(job.path).name,
        "out_dir": job.out_dir,
        "render_opts": job.render_opts,
        "status": job.status,
        "percent": round(job.percent, 1),
        "stage": job.stage,
        "result": job.result,
        "error": job.error,
    }


def _queue_process_one(job: QueueJob) -> None:
    """Run the full pipeline for one queued job on a TEMPORARY local Session.

    Does NOT touch the global SESSION (the editor's clip stays loaded). The
    editor and the queue must never load models concurrently on the 8 GB GPU,
    so they exclude each other two ways:
      * the editor's start_task refuses while _queue_running is True;
      * here we first wait out any in-flight editor task, then hold TASK_LOCK
        across the whole job (which also blocks the editor's start_task guard).
    """
    # No long lock here: _queue_running (set under TASK_LOCK before this runs)
    # already blocks the editor's start_task, so editor and queue never load the
    # GPU together. _chk() aborts cleanly at stage boundaries on /api/queue/stop.
    def _chk() -> None:
        if _queue_cancel.is_set():
            raise RuntimeError("Очередь остановлена")

    _chk()
    # Build a throwaway Session for this clip. The constructor probes media,
    # loads any cached transcript, and (if a transcript exists) auto-detects.
    ls = Session(job.path, APP["cfg"], job.out_dir, APP["use_llm"])

    def prog(frac: float) -> None:
        job.percent = round(min(1.0, max(0.0, frac)) * 100, 1)

    def stage(msg: str) -> None:
        job.stage = msg

    # (1) transcript: from cache (loaded in ctor) or fresh extract+transcribe.
    if ls.transcript is None:
        if not ls.media.has_audio:
            raise RuntimeError("В видео нет звуковой дорожки — транскрипция невозможна.")
        _chk()
        stage("Извлечение аудио…")
        wav = extract_audio(ls.ff, ls.inp, ls.work_dir / "audio16k.wav",
                            total=ls.media.duration,
                            on_progress=lambda f: prog(f * 0.05))
        stage("Транскрипция…")
        ls.transcript = transcribe_audio(
            wav, ls.cfg.transcribe, ls.media.duration, ls.audio_hash,
            cache_dir=ls.cache_dir,
            log=lambda m="": stage(str(m).strip() or job.stage),
            on_progress=lambda f: prog(0.05 + f * 0.35))

    # (2) detection. If the ctor LOADED an on-disk cutlist (prior editor
    #     session), re-detect so stale edits can't leak in. If the ctor just
    #     generated a FRESH cutlist from a cached transcript, skip the
    #     redundant (LLM-heavy) second pass.
    _chk()
    if not getattr(ls, "_ctor_fresh_detect", False):
        stage("Детекция вырезов…")
        ls._detect()
    prog(0.45)

    # (3) render (+ optional subtitles/chapters), reusing the editor path.
    _chk()
    cfg, scale_h, fps, out_dir, base = _resolve_render_opts(ls, job.render_opts)
    # Map the render's 0..1 onto the remaining 0.45..1.0 of the job bar.
    job.result = _run_render_pipeline(
        ls, cfg, scale_h, fps, out_dir, base,
        on_progress=lambda f: prog(0.45 + f * 0.55),
        on_stage=stage)
    prog(1.0)
    job.status = "done"


def _queue_worker() -> None:
    """Daemon loop: process pending jobs one at a time, then idle/stop."""
    global _queue_running
    while not _queue_cancel.is_set():
        with QUEUE_LOCK:
            job = next((j for j in QUEUE if j.status == "pending"), None)
            if job is not None:
                job.status = "running"
        if job is None:
            break
        # Persist the pending->running transition (outside QUEUE_LOCK, which
        # _save_queue re-acquires for its snapshot — calling it under the lock
        # would deadlock). A crash now leaves status='running' on disk, which
        # _load_queue resets back to 'pending' at next startup.
        _save_queue()
        try:
            _queue_process_one(job)
            if job.status != "done":
                job.status = "done"
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = "Очередь остановлена" if _queue_cancel.is_set() else str(e)
        # Persist the terminal transition (running->done with job.result, or
        # running->error with job.error) so results/errors survive a restart.
        _save_queue()
        time.sleep(0.1)
    # Release the GPU flag under the same lock that the editor checks it.
    with TASK_LOCK:
        _queue_running = False


def _start_queue_worker() -> None:
    """Start the queue worker, refusing (409) if the editor is mid-task. The
    editor-busy check + _queue_running set are atomic under TASK_LOCK so the
    editor and queue can never both begin GPU work."""
    global _queue_worker_thread, _queue_running
    with TASK_LOCK:
        if _queue_running:
            return
        if SESSION is not None and SESSION.task["running"]:
            raise HTTPException(409, "Редактор выполняет задачу — дождитесь её завершения")
        _queue_running = True
    _queue_cancel.clear()
    _queue_worker_thread = threading.Thread(target=_queue_worker, daemon=True)
    _queue_worker_thread.start()


# --- C5: папка-наблюдатель («кинул в папку — утром готово») -------------------
# Фоновый daemon-поток раз в WATCH_SCAN_INTERVAL секунд сканирует выбранную
# папку и ставит каждый НОВЫЙ видеофайл в существующую batch-очередь
# (транскрипция -> детекция -> рендер с дефолтными опциями), затем автостартует
# воркер очереди. «Новый» = файла нет в реестре обработанных И его (size, mtime)
# не менялись между двумя последовательными сканами — файл, который ещё
# копируется в папку, растёт между сканами и честно ждёт следующего прохода.
#
# «Новый» отсчитывается от МОМЕНТА ВКЛЮЧЕНИЯ: watch_set при включении (или
# смене папки) сидирует реестр текущим содержимым папки, поэтому включение
# наблюдателя на архиве из 50 роликов НЕ ставит их все в очередь — ровно то,
# что обещает UI («новые видео из папки встанут в очередь»).
#
# Состояние {enabled, folder} и реестр обработанных персистятся в
# cache_dir/watch.json (atomic .tmp -> os.replace, как queue.json), чтобы
# рестарт сервера не пережёвывал всю папку заново. Реестр ключуется
# os.path.normcase(абсолютный путь): Windows-пути регистронезависимы.

WATCH_SCAN_INTERVAL = 15.0           # секунд между сканами папки
WATCH_LOCK = threading.Lock()        # guards WATCH / WATCH_PROCESSED / WATCH_STATUS
WATCH: dict = {"enabled": False, "folder": None, "render_opts_preset": "current"}
WATCH_PROCESSED: dict[str, dict] = {}   # normcase(path) -> {"size": int, "mtime": float}
WATCH_STATUS: dict = {"error": None, "last_scan": None}
_watch_thread: Optional[threading.Thread] = None
_watch_stop = threading.Event()      # личный Event ТЕКУЩЕГО потока (см. _watch_apply)


def _watch_path() -> Path:
    """``cache_dir/watch.json`` — персист состояния наблюдателя + реестра."""
    return Path(APP["cfg"].paths.cache_dir) / "watch.json"


def _save_watch() -> None:
    """Persist WATCH + реестр атомарно. Best-effort: never raises (паттерн
    _save_queue — память истина, диск бонус)."""
    with WATCH_LOCK:
        snapshot = {"enabled": bool(WATCH["enabled"]),
                    "folder": WATCH["folder"],
                    "render_opts_preset": WATCH.get("render_opts_preset", "current"),
                    "processed": {k: dict(v) for k, v in WATCH_PROCESSED.items()}}
    p = _watch_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:  # noqa: BLE001 — best-effort
        pass


def _load_watch() -> None:
    """Восстановить watch.json при старте. Отсутствует/битый -> выключено.

    Реестр фильтруется по типам (паттерн _load_queue: мусорная запись не валит
    загрузку). enabled без папки невозможен — гасится."""
    try:
        raw = json.loads(_watch_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing / unreadable / bad JSON
        return
    if not isinstance(raw, dict):
        return
    folder = raw.get("folder")
    folder = folder if isinstance(folder, str) and folder else None
    processed = raw.get("processed")
    with WATCH_LOCK:
        WATCH["enabled"] = bool(raw.get("enabled", False)) and folder is not None
        WATCH["folder"] = folder
        WATCH["render_opts_preset"] = "current"
        WATCH_PROCESSED.clear()
        if isinstance(processed, dict):
            for k, v in processed.items():
                if (isinstance(k, str) and k and isinstance(v, dict)
                        and isinstance(v.get("size"), (int, float))
                        and not isinstance(v.get("size"), bool)
                        and isinstance(v.get("mtime"), (int, float))
                        and not isinstance(v.get("mtime"), bool)):
                    WATCH_PROCESSED[k] = {"size": int(v["size"]),
                                          "mtime": float(v["mtime"])}


def _watch_list_videos(folder: Path) -> dict[str, tuple[Path, int, float]]:
    """Снимок видеофайлов папки: normcase(resolve) -> (Path, size, mtime).

    Те же правила, что /api/browse: белый список VIDEO_EXT, без dot-файлов,
    только обычные файлы. Используется и сканером (scan_once), и сидированием
    реестра при включении наблюдения (watch_set) — критерий «что считается
    видео в этой папке» один на двоих. Папка недоступна -> OSError наружу.
    """
    seen: dict[str, tuple[Path, int, float]] = {}
    for e in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if e.name.startswith(".") or e.suffix.lower() not in VIDEO_EXT:
            continue
        try:
            if not e.is_file():
                continue
            st_ = e.stat()
        except OSError:
            continue   # файл исчез между iterdir и stat — подождёт следующего скана
        key = os.path.normcase(str(e.resolve()))
        seen[key] = (e, int(st_.st_size), float(st_.st_mtime))
    return seen


def scan_once(folder: Path, registry: dict[str, dict],
              pending: dict[str, tuple[int, float]]) -> list[Path]:
    """Один проход сканера — чистая list/dict-логика (без enqueue и без локов),
    чтобы тесты гоняли её напрямую.

    * ``registry`` — обработанные: normcase(path) -> {"size", "mtime"}.
    * ``pending``  — кандидаты прошлого скана: normcase(path) -> (size, mtime).

    Возвращает видеофайлы (белый список VIDEO_EXT, как /api/browse), которые
    НОВЫЕ (нет в registry, либо файл заменён — size/mtime не совпали с записью)
    И СТАБИЛЬНЫЕ (size+mtime не изменились с прошлого скана — копирование
    закончилось). Оба словаря мутируются на месте: исчезнувшие из ЭТОЙ папки
    файлы вычищаются (записи других папок не трогаем — смена папки наблюдения
    не убивает их историю), стабильные переезжают pending -> registry.

    Папка недоступна (сетевой диск отвалился) -> OSError наружу: вызывающий
    показывает статус «папка недоступна», поток живёт.
    """
    seen = _watch_list_videos(folder)

    # Удалённый из папки файл вычищаем из реестра и кандидатов — но только
    # записи ПОД этой папкой (ключи нормализованы normcase(resolve())).
    prefix = os.path.normcase(str(folder.resolve())).rstrip("\\/") + os.sep
    for d in (registry, pending):
        for key in [k for k in d if k.startswith(prefix) and k not in seen]:
            d.pop(key, None)

    new_files: list[Path] = []
    for key, (path, size, mtime) in seen.items():
        rec = registry.get(key)
        if rec is not None and rec.get("size") == size and rec.get("mtime") == mtime:
            pending.pop(key, None)
            continue                        # уже обработан и не менялся
        if pending.get(key) == (size, mtime):
            # Двухфазная стабильность: размер/время не менялись целый скан.
            pending.pop(key, None)
            registry[key] = {"size": size, "mtime": mtime}
            new_files.append(path)
        else:
            pending[key] = (size, mtime)    # ждём подтверждения следующим сканом
    return new_files


def _watch_busy_paths() -> set[str]:
    """normcase-пути, уже идущие через UI: всё в QUEUE (любой статус — done/error
    тоже: их уже обработали) + клип, открытый в редакторе. Сканер их не трогает."""
    busy: set[str] = set()
    with QUEUE_LOCK:
        for j in QUEUE:
            busy.add(os.path.normcase(j.path))
    s = SESSION
    if s is not None:
        try:
            busy.add(os.path.normcase(str(s.inp.resolve())))
        except OSError:
            pass
    return busy


def _watch_tick(folder_str: str, pending: dict[str, tuple[int, float]]) -> int:
    """Один проход наблюдателя: скан -> enqueue новых -> persist. Возвращает
    число поставленных в очередь файлов. Никогда не raise'ит (поток живёт).

    Реестр копируется под WATCH_LOCK, скан (диск, возможно сетевой) идёт БЕЗ
    лока, результат пишется обратно под локом — GET/POST /api/watch не ждут I/O.
    """
    log = logging.getLogger("fastvideoedit.watch")
    with WATCH_LOCK:
        before = {k: dict(v) for k, v in WATCH_PROCESSED.items()}
    registry = {k: dict(v) for k, v in before.items()}

    try:
        new_files = scan_once(Path(folder_str), registry, pending)
    except OSError as e:
        with WATCH_LOCK:
            WATCH_STATUS["error"] = "Папка недоступна: " + folder_str
            WATCH_STATUS["last_scan"] = time.time()
        log.warning("watch: folder unavailable %s (%s)", folder_str, e)
        return 0

    busy = _watch_busy_paths()
    enqueued = 0
    for p in new_files:
        rp = str(p.resolve())
        if os.path.normcase(rp) in busy:
            continue   # уже в очереди/редакторе — не дублируем (в реестр попал)
        job = QueueJob(id=uuid.uuid4().hex[:8], path=rp,
                       out_dir=str(APP["out_dir"]), render_opts={})
        with QUEUE_LOCK:
            QUEUE.append(job)
        enqueued += 1
        log.info("watch: добавлен в очередь %s", p.name)
    if enqueued:
        _save_queue()

    with WATCH_LOCK:
        WATCH_PROCESSED.clear()
        WATCH_PROCESSED.update(registry)
        WATCH_STATUS["error"] = None
        WATCH_STATUS["last_scan"] = time.time()
    if registry != before:
        _save_watch()
    return enqueued


def _watch_worker(stop: threading.Event) -> None:
    """Daemon-цикл наблюдателя. ``stop`` — ЛИЧНЫЙ Event этого потока (передан
    при создании), так что сигнал старому потоку не глушит новый после
    перезапуска через POST /api/watch.

    ``pending`` (кандидаты двухфазной проверки) живёт локально в потоке: смена
    папки = новый поток = чистые кандидаты, никаких гонок на общем состоянии.
    Автостарт очереди ретраится каждый проход, пока редактор занят (409)."""
    pending: dict[str, tuple[int, float]] = {}
    want_start = False
    while not stop.is_set():
        with WATCH_LOCK:
            enabled, folder = bool(WATCH["enabled"]), WATCH["folder"]
        if not enabled or not folder:
            break
        if _watch_tick(folder, pending):
            want_start = True
        if want_start and not stop.is_set():
            try:
                _start_queue_worker()
                want_start = False
            except HTTPException:
                pass   # редактор выполняет задачу — повторим через интервал
        if stop.wait(WATCH_SCAN_INTERVAL):
            break


def _watch_apply() -> None:
    """Привести фоновый поток в соответствие WATCH (вызов из main() и из
    POST /api/watch — без рестарта сервера).

    Текущий поток (если был) останавливается СВОИМ Event'ом; новый получает
    СВЕЖИЙ Event, поэтому сигнал старому никогда не убивает новый. Старый поток
    может дорабатывать последний tick параллельно со стартом нового — это
    безопасно: его pending локален, а новый поток начинает с пустых кандидатов
    (до первого enqueue минимум два скана), дублей не возникает."""
    global _watch_thread, _watch_stop
    _watch_stop.set()
    with WATCH_LOCK:
        enabled = bool(WATCH["enabled"]) and bool(WATCH["folder"])
    if not enabled:
        _watch_thread = None
        return
    _watch_stop = threading.Event()
    _watch_thread = threading.Thread(target=_watch_worker, args=(_watch_stop,),
                                     daemon=True, name="watch-folder")
    _watch_thread.start()


def _watch_state_dict() -> dict:
    """Снимок состояния для GET/POST /api/watch (всё под одним локом)."""
    with WATCH_LOCK:
        return {
            "enabled": bool(WATCH["enabled"]),
            "folder": WATCH["folder"],
            "render_opts_preset": WATCH.get("render_opts_preset", "current"),
            "processed": len(WATCH_PROCESSED),
            "error": WATCH_STATUS.get("error"),
            "last_scan": WATCH_STATUS.get("last_scan"),
        }


def S() -> Session:
    if SESSION is None:
        raise HTTPException(409, "No video opened")
    return SESSION


def _guard_no_task() -> None:
    """409 if a background task is in flight (mutating endpoints use this)."""
    if SESSION is not None and SESSION.task["running"]:
        raise HTTPException(409, "Идёт фоновая задача — дождитесь её завершения")
    if _queue_running:
        raise HTTPException(409, "Очередь обрабатывает ролики — дождитесь её "
                                 "завершения или остановите очередь")


@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/state")
def state():
    if SESSION is None:
        return {"no_session": True, "start_dir": APP.get("start_dir", str(Path.cwd())),
                "network": _network_state(),
                # P2-#6: pending count so the queue badge lights up on bootstrap
                # even with no clip open and the worker stopped (restored queue).
                "queue_running": _queue_running,
                "queue_pending": _queue_pending_count(),
                # B5: saved detection options (null = never configured).
                "detect_opts": _read_detect_opts(),
                # ENRICH §5: настройки «Предложить монтаж» (null = дефолты).
                "enrich_opts": _read_enrich_opts()}
    s = S()
    m = s.media
    return {
        "filename": s.inp.name,
        "path": str(s.inp.resolve()),   # F3: absolute path so the editor can queue this clip
        "v": s.audio_hash[:12],   # cache-bust token for /api/video & /api/peaks
        "media": {"duration": m.duration, "fps": m.fps,
                  "width": m.width, "height": m.height},
        "has_transcript": s.transcript is not None,
        "has_cutlist": s.cutlist is not None,
        "llm_ready": s.llm is not None,
        "censor_method": s.cfg.censor.method,
        "out_dir": str(s.out_dir.resolve()),
        # P2-#5: model the loaded transcript was made with (None if not yet
        # transcribed) — the UI flags when a new Whisper choice applies next run.
        "transcript_model": (getattr(s.transcript, "model", None) or None)
                            if s.transcript is not None else None,
        # A6: фактический девайс последней транскрипции ("cuda"|"cpu") — UI
        # предупреждает о тихом CPU-фоллбэке. null: нет транскрипта или старый
        # кэш без поля. Рядом — настроенный девайс для сравнения.
        "device_used": (getattr(s.transcript, "device_used", None) or None)
                       if s.transcript is not None else None,
        "device_configured": s.cfg.transcribe.device,
        "defaults": {
            "encoder": s.cfg.render.encoder,
            "quality": s.cfg.render.nvenc.qp if s.cfg.render.encoder == "nvenc" else s.cfg.render.x264.crf,
            "audio_bitrate": s.cfg.render.audio_bitrate,
            "censor_method": s.cfg.censor.method,
            "denoise": s.cfg.render.denoise.enabled,
            "denoise_strength": s.cfg.render.denoise.nf,
            "denoise_normalize": s.cfg.render.denoise.normalize,
            "denoise_engine": s.cfg.render.denoise.engine,
            "denoise_deess": s.cfg.render.denoise.deess,
            "denoise_loudnorm": s.cfg.render.denoise.loudnorm,
            "loudnorm_mode": s.cfg.render.denoise.loudnorm_mode,
            "cut_fade": s.cfg.render.cut_fade,
            # C3: фоновая музыка + авто-дакинг — посев секции renderModal.
            "music": {
                "enabled": s.cfg.render.music.enabled,
                "path": s.cfg.render.music.path or "",
                "gain_db": s.cfg.render.music.gain_db,
                "duck_db": s.cfg.render.music.duck_db,
            },
            # P2-#5: current model choices (defaults for the «⚙ Модели» modal).
            "whisper_model": s.cfg.transcribe.model,
            "llm_model": s.cfg.llm.model,
        },
        # C1: готовые стили вшитых субтитров для кнопок-пресетов renderModal.
        # Статичный список — фронт хранит его в st и НЕ дублирует значения.
        "caption_presets": CAPTION_PRESETS,
        "task": s.task,
        # B5: saved detection options from cache/detect_ui.json (null = never
        # configured -> /api/detect runs with the untouched session cfg).
        "detect_opts": _read_detect_opts(),
        # ENRICH §5: настройки «Предложить монтаж» (null = дефолты) + компактная
        # сводка плана для бутстрапа 5-й вкладки (count/stale, как clips).
        "enrich_opts": _read_enrich_opts(),
        "enrich": _enrich_state(s),
        # ТРЕК-2 §2: SD-генерация настроена? (тумблер+бинарь+модель) — UI красит
        # warning-баннер «SD не настроен — эмодзи-фолбэк» под источником картинок.
        "imagegen_ready": _sd_configured(s.cfg),
        "queue_running": _queue_running,   # F3: editor can flag a busy queue
        # P2-#6: pending count (incl. a queue restored from queue.json with the
        # worker stopped) so the «📋 Очередь» badge lights up on first load.
        "queue_pending": _queue_pending_count(),
        "network": _network_state(),       # P2-#4: zero-upload badge bootstrap
    }


def _network_state() -> dict:
    """Compact network posture for the badge (full detail at /api/network)."""
    st = netguard.stats()
    return {
        "offline": netguard.is_offline(),
        "external_allowed": st["external_allowed"],
        "blocked": st["blocked"],
    }


@app.get("/api/browse")
def browse(dir: Optional[str] = None, kind: Optional[str] = None):
    base = Path(dir).expanduser() if dir else Path(APP.get("start_dir", str(Path.cwd())))
    base = base.resolve()
    if not base.exists() or not base.is_dir():
        raise HTTPException(404, "Folder not found")
    # kind="music": файл-пикер фоновой музыки (C3) — аудио+видео из MUSIC_EXT;
    # kind="image": пикер «заменить ассет» обогащения (§4) — картинки IMAGE_EXT;
    # любой другой kind (в т.ч. "folder"/"video"/None) — прежний список видео.
    exts = {"music": MUSIC_EXT, "image": IMAGE_EXT}.get(kind, VIDEO_EXT)
    folders, files = [], []
    try:
        for e in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    folders.append(e.name)
                elif e.suffix.lower() in exts:
                    files.append({"name": e.name, "size": e.stat().st_size})
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    parent = str(base.parent) if base.parent != base else None
    return {"dir": str(base), "parent": parent, "folders": folders, "files": files}


@app.post("/api/open")
def open_video(payload: dict = Body(...)):
    _guard_no_task()
    p = Path(str(payload.get("path", ""))).expanduser()
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "File not found")
    if p.suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "Not a supported video file")
    try:
        open_session(str(p.resolve()))
    except ValueError as e:   # e.g. zero/unknown duration (corrupt container)
        raise HTTPException(400, str(e))
    return {"ok": True, "filename": p.name}


@app.post("/api/upload")
async def upload(request: Request, name: str = "video.mp4"):
    """Stream an uploaded file to disk (raw body) and return its path."""
    _guard_no_task()
    # Reject oversized uploads up-front (before streaming anything to disk).
    clen = request.headers.get("content-length")
    if clen is not None:
        try:
            if int(clen) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Файл слишком большой (>30 ГБ). "
                                         "/ File too large (>30 GB).")
        except ValueError:
            pass
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(415, "Неподдерживаемый формат файла. "
                                 "/ Unsupported file type.")
    updir = Path(APP["cfg"].paths.work_dir) / "_uploads"
    updir.mkdir(parents=True, exist_ok=True)
    dest = updir / (Path(name).name or "video.mp4")
    # Stream to a .part then atomically replace, counting bytes so a chunked /
    # Content-Length-less / malicious upload can't fill the disk past the cap.
    # A failed/aborted upload leaves no half-written file under the live name.
    tmp = dest.parent / (dest.name + ".part")
    total = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Файл слишком большой (>30 ГБ). "
                                             "/ File too large (>30 GB).")
                f.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return {"ok": True, "path": str(dest.resolve())}


@app.get("/api/video")
def video():
    # FileResponse handles HTTP Range -> 206 so the <video> can seek.
    # Send the MIME matching the container so .mkv/.webm/.mov aren't mislabeled
    # as mp4 (which breaks playback / caching for non-mp4 inputs).
    inp = S().inp
    return FileResponse(str(inp), media_type=EXT_MIME.get(inp.suffix.lower(), "video/mp4"))


@app.get("/api/peaks")
def peaks():
    # B4 (длинные ролики): прежний потолок 24 000 бакетов «размазывал» волну на
    # роликах > 40 мин (10 бакетов/с упирались в cap). Для длинных роликов
    # поднимаем потолок так, чтобы 10 бакетов/с сохранялись до ~100 мин — этого
    # хватает на весь диапазон зума редактора (макс. зум = 6× fit ≈ 9 000 px).
    # TODO(B4): честный дозапрос окна при зуме — query-параметры start/end/px у
    # /api/peaks (ffmpeg -ss/-t) и подмена данных wavesurfer на лету; сейчас не
    # нужно, т.к. стартовое разрешение покрывает доступный зум с запасом.
    s = S()
    dur = s.media.duration or 0
    max_buckets = 24000 if dur <= 2400 else 60000
    expected = max(1, min(max_buckets, int(round(dur * 10))))
    cache_file = s.cache_dir / f"{s.audio_hash}.peaks.json"
    if s.peaks is None and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            s.peaks = data.get("peaks") or []
            # Кэш старого (более низкого) разрешения — пересчитать; пустые peaks
            # (ролик без аудио) валидны при любом разрешении.
            if s.peaks and len(s.peaks) != expected:
                s.peaks = None
        except Exception:  # noqa: BLE001 — corrupt cache: recompute below
            s.peaks = None
    if s.peaks is None:
        s.peaks = compute_peaks(s.ff.ffmpeg, s.inp, dur, max_buckets=max_buckets)
        try:
            cache_file.write_text(
                json.dumps({"duration": s.media.duration, "peaks": s.peaks}),
                encoding="utf-8")
        except Exception:  # noqa: BLE001 — caching is best-effort
            pass
    return {"duration": s.media.duration, "peaks": s.peaks}


@app.get("/api/transcript")
def transcript():
    s = S()
    if s.transcript is None:
        raise HTTPException(404, "Not transcribed yet")
    # A6: "device_used" уже в to_dict() (null у старых кэшей без поля);
    # сюда добавляем только настроенный девайс — для CPU-предупреждения в UI.
    out = s.transcript.to_dict()
    out["device_configured"] = s.cfg.transcribe.device
    return out


def _join_words(words) -> str:
    """Собрать текст сегмента из его слов. Слова Whisper хранятся с ведущим
    пробелом (" слово"), поэтому прямая конкатенация даёт верные пробелы; на
    случай слова без него (другая конвенция / ручная правка) вставляем пробел
    между соседями явно. Итог — strip(), как и оригинальный segment.text."""
    parts: list[str] = []
    for w in words:
        t = w.word or ""
        if parts and t and not t[:1].isspace() and not parts[-1][-1:].isspace():
            parts.append(" ")
        parts.append(t)
    return "".join(parts).strip()


@app.put("/api/transcript/word")
def put_transcript_word(payload: dict = Body(...)):
    """Редактируемый транскрипт: правка текста ОДНОГО слова.

    Меняет только текст (тайминги слова и вырезы не трогаем — «текст=видео»
    остаётся честным), пересобирает segment.text из слов и атомарно сохраняет
    транскрипт в кэш (.tmp -> os.replace) — правка переживает перезагрузку
    страницы и повторное открытие ролика."""
    s = S()
    if s.transcript is None:
        raise HTTPException(409, "Сначала транскрибируйте — править нечего")
    _guard_no_task()
    si, wi, text = payload.get("si"), payload.get("wi"), payload.get("text")
    # bool — подкласс int в Python; true/false в JSON не считаем индексом.
    if (isinstance(si, bool) or not isinstance(si, int)
            or isinstance(wi, bool) or not isinstance(wi, int)):
        raise HTTPException(400, "Некорректные индексы слова (si/wi)")
    if not isinstance(text, str):
        raise HTTPException(400, "Некорректный текст слова")
    new = text.strip()
    if not new:
        raise HTTPException(400, "Текст слова не может быть пустым")
    if len(new) > 200:
        raise HTTPException(400, "Слишком длинный текст слова (максимум 200 символов)")
    segs = s.transcript.segments
    if not (0 <= si < len(segs)):
        raise HTTPException(400, "Сегмент не найден (si вне диапазона)")
    words = segs[si].words
    if not (0 <= wi < len(words)):
        raise HTTPException(400, "Слово не найдено (wi вне диапазона)")
    w = words[wi]
    # Слова Whisper несут ведущий пробел (" слово") — сохраняем конвенцию,
    # иначе ломается сборка текста сегмента и субтитров.
    old = w.word or ""
    prefix = old[: len(old) - len(old.lstrip())]
    w.word = prefix + new
    segs[si].text = _join_words(words)
    # Атомарная запись кэша: пишем .tmp и подменяем os.replace — обрыв/краш
    # посреди записи никогда не оставляет полу-файл под живым именем.
    # Имя tmp УНИКАЛЬНО на вызов (audit D-1): два параллельных PUT (быстрые
    # правки соседних слов) с общим .tmp на Windows ловили PermissionError на
    # os.replace → ложный 500 при уже применённой в памяти правке. С uuid каждый
    # пишет свой файл, последний replace побеждает с полным состоянием памяти.
    cache_file = s.cache_dir / f"{s.audio_hash}.transcript.json"
    tmp = cache_file.with_name(f"{cache_file.name}.{uuid.uuid4().hex}.tmp")
    try:
        s.transcript.save(tmp)
        os.replace(tmp, cache_file)
    finally:
        tmp.unlink(missing_ok=True)   # no-op после удачного replace
    return {"ok": True, "text": w.word, "segment_text": segs[si].text}


@app.get("/api/cutlist")
def get_cutlist():
    s = S()
    if s.cutlist is None:
        raise HTTPException(404, "No cut list")
    return s.cutlist.to_dict()


@app.put("/api/cutlist")
def put_cutlist(payload: dict = Body(...)):
    s = S()
    _guard_no_task()
    cl = CutList.from_dict(payload)
    cl.duration = s.media.duration
    cl.source = str(s.inp)
    s.cutlist = cl
    cl.save_json(s.cutlist_path)
    save_txt(cl, s.out_dir / f"{s.inp.stem}.cutlist.txt")
    return {"ok": True, "segments": len(cl.segments)}


@app.post("/api/detect")
def redetect(body: Optional[dict] = Body(default=None)):
    """Re-run detection, optionally with UI parameter overrides (B5).

    * Body present  -> sanitize + clamp, persist to cache/detect_ui.json
      (re-saved on every parametrized call), apply for this run.
    * Body absent   -> apply the previously saved options, if any.
    * No options either way -> ``_apply_detect_opts`` returns the session cfg
      ITSELF, so run_detection sees an UNCHANGED config — the default
      behaviour is byte-for-byte the old one (regression-tested).
    Disabled detectors are not run at all: ``badtakes=false`` short-circuits
    in run_detection before the LLM is ever touched.
    """
    s = S()
    _guard_no_task()
    if s.transcript is None:
        raise HTTPException(409, "Transcribe first")

    if body is None:
        opts = _read_detect_opts()           # saved options (None = never set)
    else:
        opts = _sanitize_detect_opts(body)   # 400 on wrong types, clamps ranges
        _write_detect_opts(opts)
    eff = _apply_detect_opts(s.cfg, opts)

    def run():
        s.stage("Детекция вырезов…")
        s._detect(cfg=eff)

    s.start_task("detect", run)
    return {"ok": True, "detect_opts": opts}


# --- B5: editable filler dictionary (fillers_ru.yaml) -------------------------
# API mapping (the YAML keeps its native three-list structure):
#   "fillers"   <-> words + phrases: a one-token entry is a single word
#                   (words:), a multi-token entry («как бы») is a consecutive
#                   word group (phrases:) — split/joined on whitespace.
#   "stretched" <-> mumbles: full-token regex patterns for stretched sounds.
_FILLERS_LIMIT = 500
_FILLERS_HEADER = """\
# =============================================================================
# Словарь русских филлеров для удаления. Файл редактируется И ИЗ UI
# (PUT /api/fillers), поэтому при сохранении из интерфейса он перезаписывается
# целиком — пользовательские комментарии в теле не сохраняются. Исходная
# версия с подробными комментариями лежит рядом: fillers_ru.yaml.bak.
#
#   mumbles — regex-паттерны растянутых звуков-заминок (ээ, ммм, нуу...);
#             паттерн матчится ЦЕЛИКОМ на произнесённый токен.
#   words   — одиночные слова-филлеры (регистр/ё/пунктуация не важны).
#   phrases — последовательности слов, удаляемые как группа.
# =============================================================================
"""


def _dump_fillers_yaml(mumbles: list[str], words: list[str],
                       phrases: list[list[str]]) -> str:
    """Serialize the three lists as tidy, valid YAML with the editable header.

    Scalars are emitted as JSON strings — a JSON string is a valid YAML
    double-quoted scalar, so regex metacharacters / quotes round-trip exactly
    through yaml.safe_load without pulling in a YAML *dumper* dependency.
    """
    def q(s: str) -> str:
        return json.dumps(s, ensure_ascii=False)

    lines = [_FILLERS_HEADER.rstrip("\n"), ""]
    lines.append("mumbles:" if mumbles else "mumbles: []")
    lines += [f"  - {q(m)}" for m in mumbles]
    lines.append("")
    lines.append("words:" if words else "words: []")
    lines += [f"  - {q(w)}" for w in words]
    lines.append("")
    lines.append("phrases:" if phrases else "phrases: []")
    lines += ["  - [" + ", ".join(q(t) for t in ph) + "]" for ph in phrases]
    return "\n".join(lines) + "\n"


def _fillers_api_lists(lists) -> dict:
    """FillerLists -> the API shape {fillers, stretched}."""
    return {
        "fillers": list(lists.words) + [" ".join(ph) for ph in lists.phrases],
        "stretched": list(lists.mumbles),
    }


def _validate_filler_strings(items, key: str) -> list[str]:
    """Common validation for both lists: list of non-empty unique strings.

    Whitespace is normalized (collapsed/stripped); duplicates are compared
    case-insensitively with ё→е folded — exactly how the detector matches, so
    «Вот» и «вот» honestly count as the same entry.
    """
    if not isinstance(items, list):
        raise HTTPException(400, f"{key}: ожидается список строк")
    if len(items) > _FILLERS_LIMIT:
        raise HTTPException(
            400, f"{key}: слишком много записей (максимум {_FILLERS_LIMIT})")
    out: list[str] = []
    seen: set[str] = set()
    for i, it in enumerate(items):
        if not isinstance(it, str):
            raise HTTPException(400, f"{key}[{i + 1}]: ожидается строка")
        s = " ".join(it.split())
        if not s:
            raise HTTPException(400, f"{key}[{i + 1}]: пустая строка")
        if len(s) > 200:
            raise HTTPException(
                400, f"{key}[{i + 1}]: слишком длинная запись (максимум 200 символов)")
        k = s.casefold().replace("ё", "е")
        if k in seen:
            raise HTTPException(400, f"{key}: дубликат «{s}»")
        seen.add(k)
        out.append(s)
    return out


@app.get("/api/fillers")
def get_fillers():
    """The editable filler dictionary in API shape (see mapping above)."""
    lists = load_fillers(FILLERS_PATH)
    return {**_fillers_api_lists(lists), "path": str(FILLERS_PATH)}


@app.put("/api/fillers")
def put_fillers(payload: dict = Body(...)):
    """Save the filler dictionary: validate -> backup once -> atomic write ->
    hot-reload into the live Session.

    CSRF: covered automatically by the _csrf_guard middleware (PUT /api/*).
    The .bak is created before the FIRST UI write only — it preserves the
    original hand-commented file forever; subsequent saves never touch it.
    """
    _guard_no_task()   # the running detect task reads SESSION.fillers
    fillers_in = _validate_filler_strings(payload.get("fillers"), "fillers")
    stretched = _validate_filler_strings(payload.get("stretched"), "stretched")
    for p in stretched:
        try:
            re.compile(p)
        except re.error as e:
            # A broken pattern would otherwise blow up the mumble matcher at
            # the next detection — refuse it here with the exact reason.
            raise HTTPException(400, f"stretched: некорректный regex «{p}» ({e})")

    words = [f for f in fillers_in if len(f.split()) == 1]
    phrases = [f.split() for f in fillers_in if len(f.split()) > 1]

    path = FILLERS_PATH
    bak = path.parent / (path.name + ".bak")
    if path.exists() and not bak.exists():
        try:
            shutil.copy2(path, bak)
        except OSError as e:
            # No backup -> no write: the user's hand-edited file is sacred.
            raise HTTPException(500, f"Не удалось создать резервную копию "
                                     f"{bak.name}: {e}")
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(_dump_fillers_yaml(stretched, words, phrases),
                   encoding="utf-8")
    os.replace(tmp, path)

    # Hot-reload: Session caches FillerLists at construction (self.fillers in
    # __init__) — without this swap only a server restart would see new words.
    # Re-read FROM DISK (not from memory) so the response proves the file
    # round-trips through load_fillers.
    new_lists = load_fillers(path)
    if SESSION is not None:
        SESSION.fillers = new_lists
    return {"ok": True, **_fillers_api_lists(new_lists), "path": str(path)}


@app.post("/api/transcribe")
def do_transcribe(body: dict = Body(default={})):
    s = S()
    _guard_no_task()
    if not s.media.has_audio:
        raise HTTPException(
            409, "В видео нет звуковой дорожки — транскрипция невозможна. "
                 "/ This video has no audio track — cannot transcribe.")

    # P2-#5: optional one-shot Whisper-model override for THIS run only. Validate
    # the name against the whitelist (400 before any work), then build a per-run
    # TranscribeCfg copy so the global APP cfg (shared by all sessions / the queue)
    # is not permanently changed — to make a choice the new default, the UI calls
    # POST /api/models instead. The cache re-transcribes on its own when the
    # requested model differs from the stored one (handled in transcribe.py).
    override = body.get("model")
    if override is not None and (not isinstance(override, str)
                                 or override not in WHISPER_ALLOWED):
        raise HTTPException(
            400, f"Недопустимая модель распознавания: «{override}». "
                 f"Разрешено: {', '.join(sorted(WHISPER_ALLOWED))}.")
    tcfg = s.cfg.transcribe
    if override is not None:
        tcfg = tcfg.model_copy(update={"model": override})

    def run():
        s.stage("Извлечение аудио…")
        # Extraction maps to 0..10%, transcription to 10..100%.
        wav = extract_audio(s.ff, s.inp, s.work_dir / "audio16k.wav",
                            total=s.media.duration,
                            on_progress=lambda f: s.set_progress(f * 0.1))
        s.stage("Транскрипция…")
        s.transcript = transcribe_audio(
            wav, tcfg, s.media.duration, s.audio_hash,
            cache_dir=s.cache_dir,
            log=lambda m="": s.stage(str(m).strip() or s.task["stage"]),
            on_progress=lambda f: s.set_progress(0.1 + f * 0.9))
        s.stage("Детекция вырезов…")
        s._detect()

    s.start_task("transcribe", run)
    return {"ok": True}


@app.post("/api/render")
def do_render(opts: dict = Body(default={})):
    s = S()
    _guard_no_task()
    if s.cutlist is None:
        raise HTTPException(409, "Nothing to render")
    if s.transcript is None:
        raise HTTPException(409, "Transcribe first")

    # C2: форматы вывода — ["source"] / ["9x16"] / … (легаси vertical=true
    # маппится внутри). Валидация форматов и общих опций (scale_h/fps) — 400
    # на запросе, а не ошибка задачи; out_dir создан до старта.
    formats = _parse_formats(opts)
    _, _, _, out_dir, _ = _resolve_render_opts(s, opts)
    s.last_out_dir = str(out_dir.resolve())

    def run():
        s.task["results"] = _render_formats(
            s, opts, formats,
            on_progress=s.set_progress, on_stage=s.stage,
            is_cancelled=lambda: bool(s.task.get("cancelled")))
        if s.task.get("cancelled"):
            # Частичные results уже сохранены — отчитываемся чистым «cancelled».
            raise RuntimeError("Задача отменена")

    s.start_task("render", run)
    return {"ok": True}


@app.post("/api/export/nle")
def export_nle(body: dict = Body(default={})):
    """Export the cut decisions as an NLE timeline project (no render).

    Produces a CMX3600 EDL or an FCPXML 1.11 project describing the *kept*
    segments laid end-to-end — importable into Premiere Pro / DaVinci Resolve /
    Final Cut to finish the edit (B-roll, colour, transitions). Pure string/XML
    generation: instant, no ffmpeg, no GPU, so it does NOT take the task slot.
    """
    s = S()
    if s.cutlist is None:
        raise HTTPException(409, "Сначала детектируйте вырезы / Nothing to export")

    fmt = str(body.get("format", "fcpxml")).lower().strip()
    if fmt not in ("fcpxml", "edl"):
        raise HTTPException(400, "Неизвестный формат экспорта (ожидается 'fcpxml' или 'edl')")

    removed, _censors = resolve(s.cutlist)
    kept = Timeline(removed, s.media.duration).kept_segments()
    title = s.inp.stem or "FastVideoEdit"

    ext = "fcpxml" if fmt == "fcpxml" else "edl"
    out_path = s.out_dir / f"{s.inp.stem}.{ext}"
    if fmt == "fcpxml":
        write_fcpxml(out_path, kept, s.media, title=title)
    else:
        write_edl(out_path, kept, s.media, title=title)

    # Results download via /api/output looks here (out_dir is always a root).
    s.last_out_dir = str(s.out_dir.resolve())
    return {"path": str(out_path), "name": out_path.name,
            "format": fmt, "segments": len(kept)}


@app.get("/api/output/{name}")
def output(name: str):
    # Serve from the editor's out dirs AND every queue job's out dir, so result
    # links work even when no clip is open in the editor or a job rendered to a
    # custom folder. Does NOT require a SESSION (queue can run standalone).
    roots: set[Path] = set()
    if SESSION is not None:
        roots.add(Path(SESSION.last_out_dir).resolve())
        roots.add(SESSION.out_dir.resolve())
    try:
        roots.add(Path(APP.get("out_dir", "./out")).resolve())
    except Exception:  # noqa: BLE001
        pass
    with QUEUE_LOCK:
        for j in QUEUE:
            try:
                roots.add(Path(j.out_dir).resolve())
            except Exception:  # noqa: BLE001
                continue
    # Only ever hand back files we actually produce. Without this an out_dir
    # pointed at a populated folder would let /api/output read ANY file there by
    # name (matters most if the server is bound to a non-loopback host).
    if Path(name).suffix.lower() not in OUTPUT_EXT_ALLOWED:
        raise HTTPException(404, "Not found")
    for root in roots:
        p = (root / name).resolve()
        if p.is_relative_to(root) and p.exists():
            return FileResponse(str(p))
    raise HTTPException(404, "Not found")


@app.post("/api/cancel")
def cancel():
    """Cancel the editor's running task. The editor and the queue never run a
    task at the same time (GPU mutual-exclusion), so killing ffmpeg here only
    ever targets the editor's own job, not a queue render."""
    s = S()
    # Snapshot + set the cancelled flag under TASK_LOCK so we can't tag a task
    # dict that start_task is concurrently replacing (which would lose the flag
    # and let cancel_all() hit a freshly-started task with a raw exit-1 dump).
    with TASK_LOCK:
        running = bool(s.task["running"])
        if running:
            # The worker's except reads this flag and reports «cancelled» itself —
            # we deliberately do NOT write task["error"] here (racy with restart).
            s.task["cancelled"] = True
    if running:
        ffmpeg_utils.cancel_all()
    return {"ok": True}


# --- P2-#4: zero-upload badge + offline mode ---------------------------------
@app.get("/api/network")
def network():
    """Live network posture: offline flag, per-host connect counters, RU summary."""
    offline = netguard.is_offline()
    st = netguard.stats()
    return {"offline": offline, "stats": st, "summary": _network_summary(offline, st)}


@app.post("/api/network/offline")
def network_offline(body: dict = Body(default={})):
    """Toggle offline mode and persist it to cache_dir/privacy.json (atomically).

    CSRF-guarded like every mutating /api/ route. The first Whisper-model download
    needs the internet — once cached, the editor runs fully offline.
    """
    enabled = bool(body.get("enabled", False))
    netguard.set_offline(enabled)
    _write_privacy_offline(enabled)
    offline = netguard.is_offline()
    st = netguard.stats()
    return {"ok": True, "offline": offline, "stats": st,
            "summary": _network_summary(offline, st)}


# --- A7: dependency health check ----------------------------------------------
def _whisper_model_cached(model: str) -> bool:
    """True when the configured Whisper model already sits in the local HF cache.

    Pure filesystem check — reuses the vpipe.transcribe first-run-download
    helpers (no network, no model load). Any error -> False, never raises.
    """
    try:
        repo_id = transcribe_mod._hf_repo_id(model)
        if repo_id is None:
            return False
        return bool(transcribe_mod._model_in_cache(
            transcribe_mod._hf_model_dir(repo_id)))
    except Exception:  # noqa: BLE001 — health must report, not crash
        return False


@app.get("/api/health")
def health():
    """Non-mutating self-diagnosis: are the external dependencies in place?

    Fast (<1.5s: path lookups + a 0.8s-capped localhost ping + an fs stat) and
    NEVER raises — every probe is individually shielded, a broken dependency is
    a ``false`` in the payload, not a 500. ``ok`` covers only the hard
    requirements (ffmpeg + ffprobe); Ollama and a cached Whisper model are
    optional conveniences the UI can warn about.
    """
    cfg = APP.get("cfg")
    ffmpeg_found, ffmpeg_path = False, None
    ffprobe_found = False
    try:
        ffmpeg_path = ffmpeg_utils.resolve_bin(
            cfg.ffmpeg.ffmpeg_bin if cfg else "ffmpeg", "ffmpeg")
        ffmpeg_found = True
    except Exception:  # noqa: BLE001 — FFmpegError or anything else: not found
        ffmpeg_found, ffmpeg_path = False, None
    try:
        ffmpeg_utils.resolve_bin(
            cfg.ffmpeg.ffprobe_bin if cfg else "ffprobe", "ffprobe")
        ffprobe_found = True
    except Exception:  # noqa: BLE001
        ffprobe_found = False
    ollama_found = False
    try:
        if cfg is not None:
            ollama_found = OllamaClient(cfg.llm).available(timeout=0.8)
    except Exception:  # noqa: BLE001 — Ollama off/unreachable: just False
        ollama_found = False
    whisper_cached = _whisper_model_cached(cfg.transcribe.model) if cfg else False
    return {
        "ok": ffmpeg_found and ffprobe_found,
        "ffmpeg": {"found": ffmpeg_found, "path": ffmpeg_path},
        "ffprobe": {"found": ffprobe_found},
        "ollama": {"found": bool(ollama_found)},
        "whisper_model_cached": bool(whisper_cached),
    }


# --- P2-#5: swappable local models (Whisper recognition + LLM) ----------------
def _llm_snapshot(cfg: Config) -> dict:
    """Read-only LLM snapshot for the UI: current model + installed list.

    Graceful when Ollama is off: ``available=False``, ``installed=[]`` — never
    raises. ``ready`` reflects whether the open editor session's LLM is wired
    (i.e. detect/chapters/metadata will actually use it).
    """
    available = False
    installed: list[str] = []
    try:
        client = OllamaClient(cfg.llm)
        available = client.available()
        if available:
            installed = client.list_models()
    except Exception:  # noqa: BLE001 — Ollama off/unreachable: graceful empty
        pass
    ready = SESSION.llm is not None if SESSION is not None else False
    return {
        "current": cfg.llm.model,
        "installed": installed,
        "available": available,
        "ready": ready,
    }


@app.get("/api/models")
def get_models():
    """Current Whisper preset + installed LLM list (graceful when Ollama is off).

    Read-only. ``whisper.transcript`` is the model the loaded transcript was made
    with (if any), so the UI can flag that a new choice applies on the NEXT run.
    """
    cfg = APP["cfg"]
    transcript_model = None
    if SESSION is not None and SESSION.transcript is not None:
        transcript_model = getattr(SESSION.transcript, "model", None) or None
    # A6 (онбординг): каждому пресету — лежит ли его модель уже в локальном
    # HF-кэше (чисто фс-проверка, БЕЗ сети) и примерный объём разовой загрузки.
    # Копии словарей: WHISPER_PRESETS не мутируем.
    dl_gb = transcribe_mod._MODEL_DOWNLOAD_GB
    presets = [{**p,
                "cached": _whisper_model_cached(p["model"]),
                "download_gb": dl_gb.get(p["model"])}
               for p in WHISPER_PRESETS]
    return {
        "whisper": {
            "current": cfg.transcribe.model,
            "presets": presets,
            "allowed": sorted(WHISPER_ALLOWED),
            "transcript": transcript_model,
        },
        "llm": _llm_snapshot(cfg),
    }


@app.post("/api/models")
def set_models(body: dict = Body(default={})):
    """Switch the Whisper preset and/or the LLM model from the UI.

    Mutates the SHARED ``APP['cfg']`` (so the choice becomes the new default for
    every session and the queue — intended), persists to models.json atomically,
    and — if the LLM changed and a session is open — rebuilds ``SESSION.llm`` with
    the same gate as the ctor (zeroed when Ollama is off or the model is missing,
    so AI features degrade gracefully). The Whisper change applies on the next
    transcription: the cache re-transcribes automatically when the stored model
    differs (handled in vpipe/transcribe.py — not duplicated here).
    """
    # Refuse while a task/queue is running: this mutates the SHARED cfg a running
    # job reads lazily (it would pick up a new Whisper model mid-batch), and the
    # LLM rebuild can null SESSION.llm out from under an in-flight detect/chapters.
    _guard_no_task()
    cfg = APP["cfg"]
    changed_whisper = changed_llm = False

    whisper = body.get("whisper")
    if whisper is not None:
        if not isinstance(whisper, str) or whisper not in WHISPER_ALLOWED:
            raise HTTPException(
                400, f"Недопустимая модель распознавания: «{whisper}». "
                     f"Разрешено: {', '.join(sorted(WHISPER_ALLOWED))}.")
        cfg.transcribe.model = whisper
        changed_whisper = True

    llm = body.get("llm")
    if llm is not None:
        llm = str(llm).strip()
        if not llm:
            raise HTTPException(400, "Имя ИИ-модели не может быть пустым.")
        cfg.llm.model = llm
        changed_llm = True

    if not changed_whisper and not changed_llm:
        raise HTTPException(400, "Не передано ни «whisper», ни «llm».")

    _write_models(cfg.transcribe.model, cfg.llm.model)

    # Rebuild the open session's LLM client when the LLM model changed, mirroring
    # Session.__init__: keep it only when Ollama is up AND the model is installed.
    llm_reason = None
    if changed_llm and SESSION is not None:
        if not APP.get("use_llm", True):
            SESSION.llm = None
            llm_reason = "disabled"   # editor launched with --no-llm
        else:
            client = get_client(cfg.llm)
            if client is not None and client.available() and client.has_model():
                SESSION.llm = client
            else:
                SESSION.llm = None
                if client is None:
                    llm_reason = "disabled"
                elif not client.available():
                    llm_reason = "ollama_off"
                else:
                    llm_reason = "model_missing"   # UI: предложить ollama pull <model>

    snap = _llm_snapshot(cfg)
    return {
        "ok": True,
        "whisper": cfg.transcribe.model,
        "llm": cfg.llm.model,
        "llm_ready": snap["ready"],
        "llm_available": snap["available"],
        "llm_installed": snap["installed"],
        "llm_reason": llm_reason,
    }


# --- F3: batch queue endpoints (separate from the single-clip editor) --------
@app.post("/api/queue/add")
def queue_add(body: dict = Body(...)):
    """Queue a clip by path with render options (same shape as /api/render)."""
    p = Path(str(body.get("path", ""))).expanduser()
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Файл не найден / File not found")
    if p.suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "Неподдерживаемый видеофайл / Unsupported video file")
    out_dir = str(Path(str(body.get("out_dir") or APP["out_dir"])).expanduser())
    render_opts = body.get("render_opts") or {}
    if not isinstance(render_opts, dict):
        render_opts = {}
    job = QueueJob(id=uuid.uuid4().hex[:8], path=str(p.resolve()),
                   out_dir=out_dir, render_opts=render_opts)
    with QUEUE_LOCK:
        QUEUE.append(job)
    _save_queue()
    return {"ok": True, "id": job.id}


@app.get("/api/queue")
def queue_list():
    with QUEUE_LOCK:
        jobs = [_make_job_dict(j) for j in QUEUE]
    return {"jobs": jobs, "running": _queue_running}


@app.post("/api/queue/start")
def queue_start():
    # _start_queue_worker refuses (409) atomically if the editor is mid-task.
    _start_queue_worker()
    return {"ok": True, "running": _queue_running}


@app.post("/api/queue/stop")
def queue_stop():
    """Abort the running queue: flag it and kill any in-flight ffmpeg. The
    current job ends in 'error: Очередь остановлена'; pending jobs stay pending."""
    _queue_cancel.set()
    ffmpeg_utils.cancel_all()
    return {"ok": True}


@app.post("/api/queue/remove")
def queue_remove(body: dict = Body(...)):
    jid = str(body.get("id", ""))
    with QUEUE_LOCK:
        before = len(QUEUE)
        QUEUE[:] = [j for j in QUEUE if not (j.id == jid and j.status != "running")]
        removed = before - len(QUEUE)
    _save_queue()
    return {"ok": True, "removed": removed}


@app.post("/api/queue/clear")
def queue_clear():
    """Drop every job that isn't currently running (pending/done/error)."""
    with QUEUE_LOCK:
        before = len(QUEUE)
        QUEUE[:] = [j for j in QUEUE if j.status == "running"]
        removed = before - len(QUEUE)
    _save_queue()
    return {"ok": True, "removed": removed}


# --- C5: watch-folder endpoints ----------------------------------------------
@app.get("/api/watch")
def watch_get():
    return _watch_state_dict()


@app.post("/api/watch")
def watch_set(body: dict = Body(...)):
    """Вкл/выкл наблюдение и/или смена папки. Мутирующий POST /api/* — CSRF
    закрыт общим _csrf_guard. Поток перезапускается сразу (_watch_apply),
    без рестарта сервера.

    Включение (или смена папки) СИДИРУЕТ реестр текущим содержимым папки:
    обрабатываются только файлы, появившиеся ПОСЛЕ включения, — как и обещает
    UI. В ответе ``seeded`` — сколько уже лежавших видео помечено «не трогать»."""
    enabled = bool(body.get("enabled", False))
    folder_raw = str(body.get("folder") or "").strip()
    folder: Optional[str] = folder_raw or None
    if enabled:
        if not folder_raw:
            raise HTTPException(400, "Укажи папку для наблюдения / "
                                     "Watch folder is required")
        p = Path(folder_raw).expanduser()
        if not p.exists() or not p.is_dir():
            raise HTTPException(404, "Папка не найдена / Folder not found")
        folder = str(p.resolve())
        # folder == out_dir -> каждый отрендеренный mp4 попадал бы обратно в
        # скан и рендерился заново — бесконечная рекурсия. Жёсткий отказ.
        try:
            out_res = str(Path(str(APP["out_dir"])).expanduser().resolve())
        except OSError:
            out_res = str(Path(str(APP["out_dir"])).expanduser())
        if os.path.normcase(folder) == os.path.normcase(out_res):
            raise HTTPException(400, "Папка наблюдения совпадает с папкой "
                                     "вывода рендера — готовые файлы попадали "
                                     "бы обратно в очередь. Выбери другую папку.")

    # C1: «новые» = появившиеся ПОСЛЕ включения. При включении (или смене папки
    # под включённым наблюдением) всё, что уже лежит в папке, сидируется в
    # реестр как обработанное — иначе первый же скан поставил бы в очередь весь
    # старый архив, а UI обещает обратное. Повторный POST с той же папкой при
    # УЖЕ включённом наблюдении НЕ сидирует: файл, ждущий двухфазного
    # подтверждения сканера, не должен «проглатываться». Файл, копирующийся
    # прямо в момент включения, попадает в реестр с промежуточным size/mtime и
    # будет подхвачен сканером как «заменённый», когда докопируется, — его
    # бросили в папку уже при включённом наблюдателе, обработать честно.
    with WATCH_LOCK:
        was_active = bool(WATCH["enabled"]) and bool(WATCH["folder"])
        prev_folder = WATCH["folder"]
    seed: dict[str, dict] = {}
    if enabled and (not was_active or
                    os.path.normcase(prev_folder or "") != os.path.normcase(folder)):
        try:
            seed = {k: {"size": size, "mtime": mtime}
                    for k, (_p, size, mtime)
                    in _watch_list_videos(Path(folder)).items()}
        except OSError:
            # Не включаем наблюдение с несидированным реестром — иначе бэклог
            # молча уехал бы в очередь при первом удачном скане.
            raise HTTPException(400, "Папка недоступна — не удалось прочитать "
                                     "её содержимое. Проверь путь и попробуй "
                                     "ещё раз.")
    with WATCH_LOCK:
        WATCH["enabled"] = enabled
        WATCH["folder"] = folder
        WATCH_PROCESSED.update(seed)
        if not enabled:
            WATCH_STATUS["error"] = None
    _save_watch()
    _watch_apply()
    out = _watch_state_dict()
    out["seeded"] = len(seed)
    return out


# --- F2: lightweight previews (no ffmpeg / no video render) -----------------
@app.post("/api/preview/subtitles")
def preview_subtitles():
    """Subtitle cues under the FINAL (post-cut) timeline — fast, no ffmpeg.

    Reuses the exact subtitle pipeline (vpipe/subtitles.py): remap words onto
    the shortened timeline, pack into cues, mask profanity. Returns cues in
    FINAL coordinates so the frontend can show them against origToFinal(time).
    Read-only — safe to call while a background task is running.
    """
    s = S()
    if s.transcript is None:
        raise HTTPException(404, "Not transcribed yet")
    if s.cutlist is None:
        raise HTTPException(404, "No cut list")
    removed, _ = resolve(s.cutlist)
    # Use media.duration (same source the frontend's origToFinal/keptSegments
    # uses via /api/state) so preview cues line up exactly with playback.
    tl = Timeline(removed, s.media.duration)
    words = remap_words(s.transcript.all_words(), tl)
    cues = subs_mod.build_cues(words, s.matcher, s.cfg.subtitles,
                              s.cfg.masking, tl.new_duration())
    return {
        "cues": [{"start": round(c.start, 3), "end": round(c.end, 3),
                  "text": c.text} for c in cues],
        "new_duration": round(tl.new_duration(), 3),
    }


def _parse_chapters_txt(path: str) -> list[dict]:
    """Parse a YouTube chapters file ('MM:SS title' / 'H:MM:SS title') into
    [{time: float, title: str}]. Skips empty/comment lines."""
    out: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ts, _, title = line.partition(" ")
        parts = ts.split(":")
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            continue
        if len(nums) == 2:
            secs = nums[0] * 60 + nums[1]
        elif len(nums) == 3:
            secs = nums[0] * 3600 + nums[1] * 60 + nums[2]
        else:
            continue
        out.append({"time": float(secs), "title": title.strip()})
    return out


@app.post("/api/preview/chapters")
def preview_chapters():
    """YouTube chapters under the FINAL timeline. Slow (LLM) → background task.

    Degrades gracefully when the LLM is off: returns {ok:false, reason:'llm_off'}
    immediately (200) WITHOUT starting a task, so the UI can explain why.
    """
    s = S()
    if s.transcript is None or s.cutlist is None:
        raise HTTPException(409, "Transcribe and detect first")
    if s.llm is None:
        return {"ok": False, "reason": "llm_off"}
    _guard_no_task()

    out_path = s.work_dir / "preview_chapters.txt"

    def run():
        s.stage("Главы…")
        removed, _ = resolve(s.cutlist)
        # Drop any stale file first: if generate() writes nothing (e.g. the whole
        # transcript was cut away), we must NOT re-parse an old run's chapters.
        try:
            out_path.unlink()
        except OSError:
            pass
        chapters_mod.generate(s.transcript, removed, s.cfg.chapters, out_path,
                              llm=s.llm, matcher=s.matcher, mask=s.cfg.masking,
                              log=lambda m="": s.stage(str(m).strip() or s.task["stage"]),
                              on_stage=lambda m="": s.stage(str(m)))
        s.task["results"] = {"chapters": _parse_chapters_txt(str(out_path))}

    s.start_task("preview_chapters", run)
    return {"ok": True}


@app.post("/api/preview/metadata")
def preview_metadata():
    """B: YouTube metadata under the FINAL timeline. Slow (LLM) → background task.

    Same shape as /api/preview/chapters: degrades gracefully when the LLM is off
    (returns {ok:false, reason:'llm_off'} immediately, WITHOUT starting a task)
    so the UI can explain why. The result lands in task['results']['metadata'].
    If preview_chapters has already run, its file is fed in so the description can
    embed the chapter list.
    """
    s = S()
    if s.transcript is None or s.cutlist is None:
        raise HTTPException(409, "Transcribe and detect first")
    if s.llm is None:
        return {"ok": False, "reason": "llm_off"}
    _guard_no_task()

    out_path = s.work_dir / "preview_metadata.json"
    chapters_path = s.work_dir / "preview_chapters.txt"

    def run():
        s.stage("Метаданные…")
        removed, _ = resolve(s.cutlist)
        result = metadata_mod.generate(
            s.transcript, removed, s.cfg.metadata, s.llm,
            chapters_path=chapters_path if chapters_path.exists() else None,
            matcher=s.matcher, mask=s.cfg.masking,
            log=lambda m="": s.stage(str(m).strip() or s.task["stage"]))
        out_path.write_text(json.dumps(result, ensure_ascii=False),
                            encoding="utf-8")
        s.task["results"] = {"metadata": result}

    s.start_task("preview_metadata", run)
    return {"ok": True}


# --- Clip Maker (план §2.4–2.5): suggest / cache / render ---------------------
def _clips_json_path(s: Session) -> Path:
    """``out/<stem>.clips.json`` — где живут кандидаты клипов (план §2.5)."""
    return s.out_dir / f"{s.inp.stem}.clips.json"


def _rank_source(cands: list) -> str:
    """Кто упорядочил кандидатов (F6): suggest() помечает каждого, значение
    одно на весь прогон — поднимаем его на верхний уровень ответа/файла."""
    return getattr(cands[0], "rank_source", "round_robin") if cands \
        else "round_robin"


def _save_clips_json(s: Session, cands: list) -> None:
    """Persist clip candidates atomically (.tmp -> os.replace), формат §2.5.

    Best-effort like the other persistence helpers (_save_queue & co): the
    task's ``results`` dict is the source of truth for the UI; this file only
    feeds ``GET /api/clips`` when the video is reopened, so a failed write must
    never lose a 2.5-minute LLM pass.
    """
    payload = {
        "version": 1,
        "hash": s.audio_hash,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": s.cfg.llm.model,
        # F6: кто задал порядок — "llm" (одновызовный re-rank) или
        # "round_robin" (фолбэк). Фронт может игнорировать.
        "rank_source": _rank_source(cands),
        "clips": [asdict(c) for c in cands],
    }
    p = _clips_json_path(s)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass  # non-fatal: candidates are already in task['results']


def _load_clips_json(s: Session) -> Optional[dict]:
    """Read ``out/<stem>.clips.json``. Missing/corrupt/wrong shape -> ``None``
    (never raises). Hash freshness is judged by the caller (GET /api/clips),
    which needs to distinguish «нет файла» from «файл от другого входа»."""
    try:
        data = json.loads(_clips_json_path(s).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing / unreadable / bad JSON
        return None
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        return None
    return data


@app.post("/api/clips/suggest")
def clips_suggest():
    """Кандидаты Shorts от локальной LLM. Медленно (окна × qwen3) → фоновая
    задача ``preview_clips``.

    Паттерн /api/preview/chapters: без LLM — мгновенный 200
    ``{ok:false, reason:'llm_off'}`` БЕЗ задачи (честный тост; фолбэк-нарезки
    без LLM нет — решение основателя). Результат кладётся в
    ``task['results']['clips']`` и кэшируется в out/<stem>.clips.json, чтобы
    повторное открытие видео заполняло панель без LLM.
    """
    s = S()
    if s.transcript is None or s.cutlist is None:
        raise HTTPException(409, "Transcribe and detect first")
    if s.llm is None:
        return {"ok": False, "reason": "llm_off"}
    _guard_no_task()

    def run():
        s.stage("Клипы…")
        cands = clips_mod.suggest(
            s.transcript, s.cutlist, s.cfg.clips, s.cfg.llm, s.llm,
            log=lambda *_: None,
            on_progress=s.set_progress, on_stage=s.stage)
        _save_clips_json(s, cands)
        s.task["results"] = {"clips": [asdict(c) for c in cands],
                             "rank_source": _rank_source(cands)}

    s.start_task("preview_clips", run)
    return {"ok": True}


@app.get("/api/clips")
def get_clips():
    """Сохранённые кандидаты (clips.json) — для восстановления панели при
    открытии файла, без LLM. Hash-валидация: файл от другого входа честно
    помечается ``stale:true`` и не показывается (план §2.4)."""
    s = S()
    data = _load_clips_json(s)
    if data is None:
        return {"clips": []}
    if data.get("hash") != s.audio_hash:
        return {"clips": [], "stale": True}
    return {"clips": data["clips"], "stale": False,
            "generated_at": data.get("generated_at"),
            "model": data.get("model"),
            # файлы до F6 писались без rank_source — тогда порядок и был
            # round-robin (MVP-сортировка)
            "rank_source": data.get("rank_source", "round_robin")}


# F7: жёсткие пределы длительности клипа при правке границ из UI (сек).
# Мягкое предупреждение «>60с — лимит Shorts» живёт на фронте; сервер
# отсекает только бессмысленное (<5с) и заведомо не-Shorts (>90с).
CLIP_SAVE_MIN = 5.0
CLIP_SAVE_MAX = 90.0


@app.post("/api/clips/save")
def clips_save(body: dict = Body(default={})):
    """F7 — правка границ кандидата из UI: обновить запись в out/<stem>.clips.json.

    ``{id, start, end}``  — новые границы (5–90 с, в пределах ролика);
    ``{id, reset: true}`` — вернуть авто-границы (запомнены при первой правке).

    dur_raw/dur_eff пересчитываются по ТЕКУЩЕМУ катлисту сессии и возвращаются
    в ответе (``{ok, clip}``) — карточка обновляет цифру без перезапроса.
    Кандидат помечается ``edited:true``; исходные границы сохраняются один раз
    в ``auto_start/auto_end`` (повторная правка их не перетирает). Топ-уровень
    файла (version/hash/generated_at/model/rank_source) не трогаем — файл
    остаётся валидным для GET /api/clips (hash-проверка) и рендера. Запись
    атомарная (.tmp -> os.replace), но в отличие от best-effort
    _save_clips_json неудача честно отдаёт 500: юзер ждёт подтверждение.
    """
    s = S()
    _guard_no_task()
    cid = body.get("id")
    if not isinstance(cid, str) or not cid:
        raise HTTPException(400, "id кандидата обязателен (строка)")
    data = _load_clips_json(s)
    if data is None:
        raise HTTPException(404, "Кандидаты не найдены — сначала «Предложить клипы»")
    if data.get("hash") != s.audio_hash:
        raise HTTPException(409, "clips.json от другого видео — правка отклонена")
    clip = next((c for c in data["clips"]
                 if isinstance(c, dict) and c.get("id") == cid), None)
    if clip is None:
        raise HTTPException(404, f"Кандидат {cid} не найден")

    duration = float(s.media.duration)
    if body.get("reset"):
        # Сброс к авто-границам; кандидат без правок — идемпотентный no-op
        # (фронт кнопку и не показывает, но пусть будет безопасно).
        start = float(clip.get("auto_start", clip["start"]))
        end = float(clip.get("auto_end", clip["end"]))
        clip["edited"] = False
    else:
        try:
            start, end = float(body.get("start")), float(body.get("end"))
        except (TypeError, ValueError):
            raise HTTPException(400, "start/end должны быть числами (секунды)")
        # NaN/Infinity валидны для json.loads — отсекаем ДО сравнений (паттерн
        # /api/clips/render: max/min с NaN тихо отдают не то).
        if not (math.isfinite(start) and math.isfinite(end)):
            raise HTTPException(400, "start/end должны быть конечными числами")
        if start < -0.001 or end > duration + 0.001:
            raise HTTPException(400, "Границы клипа вне ролика")
        start = max(0.0, start)
        end = min(duration, end)
        if not (CLIP_SAVE_MIN <= end - start <= CLIP_SAVE_MAX):
            raise HTTPException(
                400, f"Длительность клипа должна быть "
                     f"{CLIP_SAVE_MIN:.0f}–{CLIP_SAVE_MAX:.0f} секунд")
        if not clip.get("edited"):
            # Первая правка: запомнить авто-границы для «сбросить к авто».
            clip["auto_start"], clip["auto_end"] = clip["start"], clip["end"]
        clip["edited"] = True

    # dur_raw/dur_eff — по текущему live-катлисту (eff = raw − enabled-вырезы
    # внутри диапазона); катлиста ещё нет (свежеоткрытый файл) → eff == raw.
    removed = resolve(s.cutlist)[0] if s.cutlist is not None else []
    tl = Timeline(removed, duration)
    dur_raw = end - start
    clip["start"], clip["end"] = round(start, 3), round(end, 3)
    clip["dur_raw"] = round(dur_raw, 3)
    clip["dur_eff"] = round(max(0.0, dur_raw - tl.removed_overlap(start, end)), 3)

    p = _clips_json_path(s)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        raise HTTPException(500, f"Не удалось сохранить clips.json: {e}")
    return {"ok": True, "clip": clip}


# --- ENRICH_PLAN §5: авто-обогащение — план, suggest, save ----------------------
def _enrich_json_path(s: Session) -> Path:
    """``out/<stem>.enrich.json`` — где живёт план обогащения (§1.2)."""
    return s.out_dir / f"{s.inp.stem}.enrich.json"


def _save_enrich_json(s: Session, plan: "enrich_mod.EnrichPlan") -> None:
    """Persist плана из задачи suggest — best-effort (зеркало _save_clips_json):
    результат задачи уже в ``task['results']``, файл лишь кормит GET /api/enrich
    при повторном открытии; неудача записи не должна валить LLM-пасс. Строгая
    запись (500 юзеру) — только в /api/enrich/save, где юзер ждёт подтверждение."""
    try:
        enrich_mod.save_enrich(plan, _enrich_json_path(s))
    except OSError:
        pass  # non-fatal: план уже в task['results']


def _enrich_state(s: Session) -> dict:
    """Компактная сводка для GET /api/state: {count, stale} (полные items —
    GET /api/enrich). ``count`` — все предложения плана (счётчик вкладки),
    ``stale`` — план от другого аудио (hash-инвалидация, как clips)."""
    plan = enrich_mod.load_enrich(_enrich_json_path(s))
    if plan is None:
        return {"count": 0, "stale": False}
    if plan.hash != s.audio_hash:
        return {"count": 0, "stale": True}
    return {"count": len(plan.items), "stale": False}


def _run_enrich_detectors(s: Session, params: dict, log) -> list:
    """P3 (ENRICH_PLAN §7-P3): три LLM-детектора §3.1–3.3 (списки / CTA /
    иллюстрации, vpipe/enrich_llm.py) + этап ассетов §4 (P5). ``s.llm`` уже
    есть (llm_off отсечён в эндпоинте); ``params`` — полные настройки запуска,
    детекторам уходит их whitelist-подмножество (sanitize_params), а
    ``user_folder`` (Tier 1, §4) — отдельным kwarg (в params-блок плана §1.2 он
    не входит). Прогресс задачи — по детекторам (lists 45 / cta 15 /
    illustrations 30 / assets 10); сбойное окно/детектор/этап не валит задачу
    (warnings уходят в log)."""
    return enrich_llm.detect_all(
        s.transcript, s.cutlist, enrich_mod.sanitize_params(params), s.llm,
        log=log, on_progress=s.set_progress,
        user_folder=(params or {}).get("user_folder", ""))


def _sd_configured(cfg: Config) -> bool:
    """SD-генерация реально доступна: тумблер включён И бинарь sd-cli И модель
    .gguf резолвятся (паттерн opt-in DeepFilterNet). Решает, запускать ли этап
    images и показывать ли warning-баннер «SD не настроен» (ТРЕК-2 §2)."""
    ig = cfg.render.imagegen
    if not ig.imagegen_enabled:
        return False
    return bool(imagegen_mod._resolve_sd_bin(ig.imagegen_bin)
                and imagegen_mod._resolve_model(ig.imagegen_model))


def _wait_ollama_unloaded(s: Session, log, *, tries: int = 20,
                          delay: float = 0.5) -> None:
    """VRAM-менеджер (§2): выгрузить qwen3 и дождаться пустого GET /api/ps перед
    SD-этапом (8 ГБ карта не вместит LLM+SDXL). best-effort: нет Ollama / не
    выгрузилась за tries опросов — всё равно идём дальше (--max-vram подстрахует),
    задачу не валим."""
    if s.llm is None:
        return
    s.llm.unload()
    for _ in range(max(1, tries)):
        if not s.llm.loaded_models():
            return
        time.sleep(delay)
    log("  SD: Ollama не выгрузилась за отведённое время — "
        "продолжаю (sd-cli с --max-vram).")


def _run_enrich_images(s: Session, items: list, params: dict, log) -> int:
    """Этап images (ТРЕК-2 §2): сгенерировать SD-картинки точкам, помеченным
    маршрутизатором ``asset_kind="generate"``. Зовётся ПОСЛЕ детекторов и
    ПЕРЕД планировщиком. Последовательность VRAM: unload Ollama -> ждём пустого
    /api/ps -> пачка ``enrich_image_batch``. НЕ запускаем при image_source вне
    {generate, auto} или если SD не настроен (точки с generate откатятся на
    эмодзи в самом батче). Сбой этапа не валит задачу. Возврат — число
    сгенерированных картинок."""
    src = (params or {}).get("image_source", "auto")
    if src not in ("generate", "auto"):
        return 0
    gen_pts = [it for it in items
               if getattr(getattr(it, "payload", None), "asset_kind", None)
               == "generate"]
    if not gen_pts:
        return 0
    if not _sd_configured(s.cfg):
        # Маршрутизатор пометил точки на генерацию, но SD не настроен — откат на
        # эмодзи-фолбэк (батч сам это сделает по полю emoji), задача жива.
        log("  SD не настроен (бинарь/модель) — эмодзи-фолбэк для "
            f"{len(gen_pts)} картинок.")
        imagegen_mod.enrich_image_batch(gen_pts, s.cfg.render.imagegen, log)
        return 0
    s.stage(f"Монтаж: генерация {len(gen_pts)} картинок (ИИ)…")
    _wait_ollama_unloaded(s, log)
    base = 1.0 - enrich_llm.PROGRESS_WEIGHTS["assets"]

    def img_prog(frac: float) -> None:
        s.set_progress(base + enrich_llm.PROGRESS_WEIGHTS["assets"] * frac)

    try:
        return imagegen_mod.enrich_image_batch(
            gen_pts, s.cfg.render.imagegen, log, on_progress=img_prog)
    except Exception as e:  # noqa: BLE001 — сбойный SD-этап не валит задачу
        log(f"enrich: этап генерации картинок упал ({e}) — эмодзи-фолбэк")
        return 0


def _merge_enrich_user_state(s: Session, items: list) -> list:
    """Повторный suggest НЕ уничтожает работу юзера (§1: «ИИ предлагает —
    юзер решает»; CRITICAL код-ревью P2). Протокол мержа со старым планом:

    - совпавший id → переносим ``enabled``-решение юзера, а при ``edited`` —
      и правки целиком (payload + тайминги); свежие поля детектора
      (score/quote/reason) остаются новыми;
    - ручные предложения (``source:"user"``) переезжают в новый план целиком —
      детекторы их не переоткроют;
    - исчезнувшие LLM-предложения честно уходят (новый анализ);
    - план от ДРУГОГО аудио не мержится — полная hash-инвалидация,
      как в GET /api/enrich и save."""
    old = enrich_mod.load_enrich(_enrich_json_path(s))
    if old is None or old.hash != s.audio_hash:
        return items
    prev_by_id = {it.id: it for it in old.items}
    carried: set[str] = set()
    for it in items:
        prev = prev_by_id.get(it.id)
        if prev is None:
            continue
        carried.add(it.id)
        it.enabled = prev.enabled            # решение юзера — закон
        if prev.edited:                      # правки текста/таймингов — тоже
            it.payload = prev.payload
            it.t_start = prev.t_start
            it.t_end = prev.t_end
            it.edited = True
    items.extend(p for p in old.items
                 if p.source == "user" and p.id not in carried)
    return items


@app.post("/api/enrich/suggest")
def enrich_suggest(body: dict = Body(default={})):
    """Предложения авто-обогащения (§5) — фоновая задача ``enrich``.

    Паттерн /api/clips/suggest: 409 без транскрипта/катлиста; без LLM —
    мгновенный 200 ``{ok:false, reason:'llm_off'}`` БЕЗ задачи (и без персиста
    настроек — нечего применять); занятая задача/очередь → 409. Тело — настройки
    запуска (types/density/image_source/user_folder): strict-sanitize (B5),
    персист в cache/enrich_ui.json, затем фоновая задача: детекторы (P3-хук) →
    мерж с прошлым планом (_merge_enrich_user_state: повторный запуск не
    теряет enabled/edited и ручные предложения юзера) → планировщик статусов →
    out/<stem>.enrich.json (hash + cutlist_rev)."""
    s = S()
    if s.transcript is None or s.cutlist is None:
        raise HTTPException(409, "Transcribe and detect first")
    if s.llm is None:
        return {"ok": False, "reason": "llm_off"}
    _guard_no_task()

    opts = _sanitize_enrich_opts(body if isinstance(body, dict) else {},
                                 strict=True)
    _write_enrich_opts(opts)
    # Блок params плана (§1.2) — подмножество настроек (whitelist в enrich.py).
    params = enrich_mod.sanitize_params(opts)

    def run():
        s.stage("Монтаж: анализ…")
        items = _run_enrich_detectors(s, opts, log=s.stage)
        # Этап images (ТРЕК-2 §2): SD-генерация для точек asset_kind="generate".
        # ПОСЛЕ детекторов (VRAM-менеджер выгрузит Ollama), ДО мержа/планировщика:
        # генерим только свежие точки детектора, мерж затем хранит ревью юзера.
        _run_enrich_images(s, items, params, log=s.stage)
        # Повторный анализ не теряет ревью юзера: enabled/edited и ручные
        # предложения переезжают из старого плана (CRITICAL код-ревью P2).
        items = _merge_enrich_user_state(s, items)
        plan = enrich_mod.EnrichPlan(
            hash=s.audio_hash,
            # cutlist_rev — снимок enabled-вырезов НА МОМЕНТ анализа (§1.2);
            # катлист не сменится под задачей (PUT /api/cutlist держит
            # _guard_no_task), но честнее снять его внутри задачи.
            cutlist_rev=enrich_mod.compute_cutlist_rev(s.cutlist),
            model=s.cfg.llm.model, params=params, items=items)
        # Планировщик — для статусов/auto-disable, которые увидит UI (§5):
        # координаты исходника (рендер пересчитает под свой выход).
        s.stage("Монтаж: планировщик…")
        removed_now, _ = resolve(s.cutlist)
        enrich_mod.plan_render(
            plan, Timeline(removed_now, s.media.duration),
            s.transcript.all_words(), s.cfg,
            s.media.width or 1920, s.media.height or 1080,
            log=lambda *_: None)
        _save_enrich_json(s, plan)
        s.task["results"] = {"enrich": {
            "items": [it.to_dict() for it in plan.items],
            "params": params}}

    s.start_task("enrich", run)
    return {"ok": True}


@app.get("/api/enrich")
def get_enrich():
    """Сохранённый план (§5): ``{items, params, stale, cutlist_changed}``.

    Hash-валидация как у клипов: план от другого аудио → ``stale:true`` и
    пустые items (полная инвалидация). Несовпавший cutlist_rev — лишь мягкий
    баннер ``cutlist_changed:true`` (рендер всё равно корректен: ремап + дроп,
    §1.3); items при этом отдаются."""
    s = S()
    plan = enrich_mod.load_enrich(_enrich_json_path(s))
    if plan is None:
        return {"items": [], "params": None, "stale": False,
                "cutlist_changed": False}
    if plan.hash != s.audio_hash:
        return {"items": [], "params": None, "stale": True,
                "cutlist_changed": False}
    cur_rev = enrich_mod.compute_cutlist_rev(
        s.cutlist if s.cutlist is not None else [])
    return {"items": [it.to_dict() for it in plan.items],
            # load_enrich уже прогнал params через whitelist sanitize_params —
            # наружу всегда канонический вид, даже если файл правили руками.
            "params": plan.params,
            "stale": False,
            "cutlist_changed": cur_rev != plan.cutlist_rev,
            "generated_at": plan.generated_at,
            "model": plan.model}


@app.post("/api/enrich/save")
def enrich_save(body: dict = Body(default={})):
    """Мерж правок UI в план (§5): ``{items:[{id, enabled?, payload?,
    t_start?, t_end?}]}``.

    Существующий id — точечная правка: тоггл ``enabled`` (edited НЕ трогает),
    правки payload/таймингов (NaN/Infinity → 400, клампы к ролику, лимиты
    текстов §1.2 через санитайзеры enrich.py) ставят ``edited:true``. Новый id
    с валидным ``type`` — ручное предложение, ``source:"user"`` (§5); новый id
    БЕЗ type — честный 404 (опечатка, а не добавление). Нет плана → 409; план
    от другого видео → 409. Запись атомарная (.tmp -> os.replace) со СТРОГИМ
    500 — оригинальный файл при сбое остаётся целым."""
    s = S()
    _guard_no_task()
    raw_items = body.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise HTTPException(400, "items: ожидается непустой список правок")

    plan = enrich_mod.load_enrich(_enrich_json_path(s))
    if plan is None:
        raise HTTPException(409, "План обогащения не найден — сначала "
                                 "«Предложить монтаж»")
    if plan.hash != s.audio_hash:
        raise HTTPException(409, "enrich.json от другого видео — правка "
                                 "отклонена")

    duration = float(s.media.duration)
    index_of = {it.id: i for i, it in enumerate(plan.items)}
    updated: list[dict] = []
    for k, e in enumerate(raw_items):
        if not isinstance(e, dict):
            raise HTTPException(400, f"items[{k}]: ожидается объект")
        eid = e.get("id")
        if not isinstance(eid, str) or not eid:
            raise HTTPException(400, f"items[{k}]: id обязателен (строка)")
        # NaN/±Infinity валидны для json.loads и пролезают сквозь клампы
        # (паттерн /api/clips/save) — отсекаем ДО любых сравнений.
        for key in ("t_start", "t_end"):
            v = e.get(key)
            if v is None:
                continue
            if (isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(float(v))):
                raise HTTPException(400, f"items[{k}].{key}: ожидается "
                                         "конечное число (секунды)")
        if "enabled" in e and not isinstance(e["enabled"], bool):
            raise HTTPException(400, f"items[{k}].enabled: ожидается "
                                     "true/false")
        if "payload" in e and e["payload"] is not None \
                and not isinstance(e["payload"], dict):
            raise HTTPException(400, f"items[{k}].payload: ожидается объект")

        if eid in index_of:
            idx = index_of[eid]
            d = plan.items[idx].to_dict()
            touched = False                  # правка payload/таймингов → edited
            if isinstance(e.get("payload"), dict) and e["payload"]:
                d["payload"] = {**d["payload"], **e["payload"]}
                touched = True
            for key in ("t_start", "t_end"):
                if e.get(key) is not None:
                    d[key] = min(duration, max(0.0, float(e[key])))
                    touched = True
            if "enabled" in e:
                d["enabled"] = e["enabled"]   # тоггл — НЕ правка (edited не трогаем)
            if touched:
                d["edited"] = True
            # item_from_dict — единый санитайзер: клампы длительностей по типу,
            # лимиты текстов §1.2, NaN-гарды внутри payload. id/type из d —
            # неизменны. Тип валиден (взят из существующего item) → не None.
            it = enrich_mod.item_from_dict(d)
            plan.items[idx] = it
        else:
            t = e.get("type")
            if t not in enrich_mod.ENR_TYPES:
                raise HTTPException(404, f"Предложение {eid} не найдено")
            d = dict(e)
            d["source"] = "user"             # ручное предложение (§5)
            for key in ("t_start", "t_end"):
                if e.get(key) is not None:
                    d[key] = min(duration, max(0.0, float(e[key])))
            it = enrich_mod.item_from_dict(d)
            plan.items.append(it)
            index_of[it.id] = len(plan.items) - 1
        updated.append(it.to_dict())

    try:
        enrich_mod.save_enrich(plan, _enrich_json_path(s))
    except OSError as e:
        raise HTTPException(500, f"Не удалось сохранить enrich.json: {e}")
    return {"ok": True, "items": updated}


def _render_clip_job(s: Session, *, start: float, end: float, idx: int,
                     n: int, fname: str, render_opts: dict, set_pct) -> dict:
    """Рендер ОДНОГО клипа [start, end] — общий кирпич clips_render и Авто-пака.

    Весь per-clip пайплайн (план §2.4) в одном месте: стадия/процент
    «Клип i/N…», копия ЖИВОГО катлиста + 2 граничных REMOVE вокруг диапазона,
    edge_fade из cfg.clips (F8: де-клик истинных краёв клипа — только для
    клипов, обычный рендер живёт с дефолтом 0.0; кламп к 0–0.2 — внутри
    render._edge_fade_filters), _resolve_render_opts + _run_render_pipeline
    (``cutlist_override`` — сессия не мутируется) и try/except «упавший клип
    не валит остальные».

    Возвращает ``{"ok": True, **res}`` либо ``{"ok": False, "error": …}``
    («cancelled», если ошибка вызвана отменой); специфику записи результата
    (id/hook/filename) добавляют вызывающие. ``set_pct`` — колбэк общего
    прогресса цикла 0..1: у clips_render это s.set_progress, у Авто-пака —
    отрезок весов стадии «clips».
    """
    s.stage(f"Клип {idx + 1}/{n}: рендер…")
    set_pct(idx / n)
    # Катлист клипа = копия живых вырезов + 2 граничных REMOVE (§2.4).
    cl = s.cutlist
    clip_cl = CutList(source=cl.source, duration=cl.duration,
                      segments=[copy.copy(seg) for seg in cl.segments])
    if start > 0:
        clip_cl.segments.append(CutSegment(
            id=f"clipA{idx}", start=0.0, end=start,
            type=TYPE_MANUAL, action=ACTION_REMOVE, enabled=True))
    if end < cl.duration:
        clip_cl.segments.append(CutSegment(
            id=f"clipB{idx}", start=end, end=cl.duration,
            type=TYPE_MANUAL, action=ACTION_REMOVE, enabled=True))
    try:
        cfg, scale_h, fps, out_dir, base = _resolve_render_opts(
            s, {**render_opts, "filename": fname})
        try:
            ef = float(getattr(cfg.clips, "edge_fade", 0.0) or 0.0)
        except (TypeError, ValueError):
            ef = 0.0
        res = _run_render_pipeline(
            s, cfg, scale_h, fps, out_dir, base,
            on_progress=lambda f: set_pct((idx + min(1.0, max(0.0, f))) / n),
            on_stage=lambda m: s.stage(f"Клип {idx + 1}/{n}: {m}"),
            cutlist_override=clip_cl, edge_fade=ef)
        return {"ok": True, **res}
    except Exception as e:  # noqa: BLE001 — упавший клип не валит остальные
        return {"ok": False,
                "error": "cancelled" if s.task.get("cancelled") else str(e)}


@app.post("/api/clips/render")
def clips_render(body: dict = Body(default={})):
    """Рендер выбранных клипов — ОДНА фоновая задача ``render_clips`` (план §2.4).

    НЕ через F3-очередь (зафиксированное решение): очередь строит новую Session
    per job и пере-детектирует катлист — выбрасывая кураторские правки
    enable/disable. Здесь — цикл по клипам в живой сессии: на каждый клип копия
    live-катлиста + 2 граничных REMOVE вокруг [start, end], рендер через
    ``_run_render_pipeline(cutlist_override=…)`` (сессия не мутируется).

    Серверные гарантии: ``chapters``/``metadata`` принудительно false
    независимо от клиента (иначе каждый клип гонял бы LLM и перетирал общий
    metadata.txt); упавший клип не валит остальные; /api/cancel останавливает
    цикл между клипами (частичные результаты сохраняются);
    percent = (i+f)/N, stage «Клип i/N: …».
    """
    s = S()
    _guard_no_task()
    if s.cutlist is None:
        raise HTTPException(409, "Nothing to render")
    if s.transcript is None:
        raise HTTPException(409, "Transcribe first")

    raw = body.get("clips")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "Не передано ни одного клипа "
                                 "(ожидается clips: [{start, end}])")
    render_opts = body.get("render_opts") or {}
    if not isinstance(render_opts, dict):
        render_opts = {}
    # Сервер принудительно глушит главы/метаданные для клипов (план §2.4) —
    # независимо от того, что прислал клиент. music=None — клипы Shorts НИКОГДА
    # не получают фоновую подложку (C3): _resolve_render_opts при не-dict
    # music принудительно выключает cfg.render.music. enrich=None — обогащение
    # Shorts-клипов вне скоупа v1 (ENRICH_PLAN §9); cutlist_override-путь его
    # и так не применяет, это второй (явный) предохранитель.
    render_opts = {**render_opts, "chapters": False, "metadata": False,
                   "music": None, "enrich": None}

    duration = float(s.media.duration)
    clips_in: list[dict] = []
    for i, c in enumerate(raw):
        if not isinstance(c, dict):
            raise HTTPException(400, f"Клип №{i + 1}: ожидается объект "
                                     "{start, end}")
        try:
            start, end = float(c.get("start")), float(c.get("end"))
        except (TypeError, ValueError):
            raise HTTPException(400, f"Клип №{i + 1}: start/end должны быть "
                                     "числами (секунды)")
        # NaN/±Infinity валидны для json.loads и пролезают сквозь клампы:
        # max(0.0, nan) → 0.0, min(duration, nan) → duration (Python отдаёт
        # первый аргумент при ложном сравнении) — получился бы клип на весь
        # файл. Отсекаем ДО клампов.
        if not (math.isfinite(start) and math.isfinite(end)):
            raise HTTPException(400, f"Клип №{i + 1}: start/end должны быть "
                                     "конечными числами")
        start = max(0.0, start)
        end = min(duration, end)
        if not end - start > 0:
            raise HTTPException(400, f"Клип №{i + 1}: пустой диапазон")
        name = c.get("filename")
        if name is not None and not isinstance(name, str):
            raise HTTPException(400, f"Клип №{i + 1}: filename должен быть строкой")
        clips_in.append({"start": start, "end": end,
                         "filename": ((name or "").strip()
                                      or f"{s.inp.stem}_clip{i + 1:02d}")})

    # Fail fast: битые ОБЩИЕ опции (scale_h/fps) дают 400 на запросе, а не
    # ошибку задачи; заодно out_dir создан и ссылки /api/output смотрят туда.
    _, _, _, out_dir0, _ = _resolve_render_opts(s, render_opts)
    s.last_out_dir = str(out_dir0.resolve())

    n = len(clips_in)

    def run():
        results: list[dict] = []
        for i, c in enumerate(clips_in):
            if s.task.get("cancelled"):
                break                               # cancel между клипами
            job = _render_clip_job(s, start=c["start"], end=c["end"],
                                   idx=i, n=n, fname=c["filename"],
                                   render_opts=render_opts,
                                   set_pct=s.set_progress)
            results.append({"filename": c["filename"], **job})
        s.task["results"] = {"clips": results}
        if s.task.get("cancelled"):
            # Частичные results уже сохранены выше; воркер start_task отчитается
            # чистым «cancelled» — как у обычного отменённого рендера.
            raise RuntimeError("Задача отменена")

    s.start_task("render_clips", run)
    return {"ok": True, "count": n}


# --- C4: «Авто-пак» — сырец → готовый ролик + пак Shorts одной кнопкой ---------
# Одна фоновая задача, склеенная из ГОТОВЫХ кирпичей: транскрипция (если нет),
# детекция (если катлист пуст), мультиформат-рендер основного ролика (C2),
# suggest (если нет свежего clips.json) и точный цикл рендера клипов
# (/api/clips/render). Против 4 ручных облачных шагов CapCut — локально и
# бесплатно. Веса стадий для общего percent (нормируются по активным стадиям,
# чтобы пропущенная транскрипция не оставляла «дыру» в прогрессе):
_AUTOPACK_WEIGHTS = {"transcribe": 0.35, "detect": 0.10, "enrich": 0.12,
                     "main": 0.25, "suggest": 0.10, "clips": 0.20}
# ENRICH §5: в автопаке применяются ТОЛЬКО предложения со score >= 70 поверх
# enabled — консервативный порог, автопак идёт без ревью-вкладки.
AUTOPACK_ENRICH_MIN_SCORE = 70


def _autopack_top_clips(cands: list, duration: float, top_k: int) -> list[dict]:
    """Первые ``top_k`` ВАЛИДНЫХ кандидатов (порядок списка = порядок re-rank).

    Принимает и ``asdict(ClipCandidate)`` свежего suggest, и записи из
    clips.json — диску доверять нельзя (файл мог быть правлен руками), поэтому
    битые записи (не-объект, нечисловые/NaN границы, пустой диапазон) молча
    пропускаются — паттерн _load_queue: мусор не валит задачу.
    """
    out: list[dict] = []
    for c in cands:
        if len(out) >= top_k:
            break
        if not isinstance(c, dict):
            continue
        try:
            start, end = float(c.get("start")), float(c.get("end"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        start = max(0.0, start)
        end = min(float(duration), end)
        if not end - start > 0:
            continue
        out.append({"id": str(c.get("id") or f"c{len(out) + 1:02d}"),
                    "start": start, "end": end,
                    "hook": str(c.get("hook_phrase") or "")})
    return out


@app.post("/api/autopack")
def autopack(body: dict = Body(default={})):
    """C4 «Авто-пак»: сырец → готовый ролик (+ пак Shorts) ОДНОЙ фоновой задачей.

    Тело: ``{top_k: 1–10 (клампится, дефолт 3), formats: [...] для ОСНОВНОГО
    ролика (как у /api/render, дефолт ["source"]), clips: bool (дефолт true),
    enrich: bool (дефолт false — ENRICH §5: применить СВЕЖИЙ план обогащения
    к основному ролику со score >= 70 поверх enabled; плана нет/несвежий →
    warning, детекторы автопак не зовёт), render_opts: {...} (как у
    /api/render)}``.

    Стадии (каждая видна в SSE stage, percent — по весам _AUTOPACK_WEIGHTS):
      а) транскрипция, если её нет (кэш → стадия пропускается);
      б) детекция, если катлист ПУСТ — существующий НЕ передетекчивается
         (юзер мог править вырезы руками);
      б2) enrich=true и свежий план: пометка «применяю план (score>=70)» —
         сам план применит рендер основного ролика (opts.enrich);
      в) рендер основного ролика через _render_formats (C2, мультиформат);
      г) clips=true: suggest — но свежий clips.json (hash совпал)
         переиспользуется без LLM; LLM выключен → warning, основной остаётся;
      д) рендер топ-K клипов — точный цикл /api/clips/render (per-clip ass,
         edge_fade, chapters/metadata принудительно false).

    Надёжность: упавший suggest (Ollama умерла) → частичный успех
    (``ok:true`` + warnings, clips:{error}), упавший клип не валит остальные;
    /api/cancel срабатывает между КАЖДОЙ стадией и между клипами/форматами,
    сохраняя уже сделанное в ``task['results']``.
    """
    s = S()
    _guard_no_task()

    # --- валидация тела: 400 на запросе, а не ошибка задачи -------------------
    raw_k = body.get("top_k", 3)
    if (isinstance(raw_k, bool) or not isinstance(raw_k, (int, float))
            or not math.isfinite(float(raw_k))):
        raise HTTPException(400, "top_k: ожидается число 1–10")
    top_k = max(1, min(10, int(raw_k)))
    do_clips = bool(body.get("clips", True))
    formats = _parse_formats({"formats": body.get("formats")})
    render_opts = body.get("render_opts") or {}
    if not isinstance(render_opts, dict):
        raise HTTPException(400, "render_opts: ожидается объект")

    need_transcribe = s.transcript is None
    if need_transcribe and not s.media.has_audio:
        raise HTTPException(
            409, "В видео нет звуковой дорожки — транскрипция невозможна. "
                 "/ This video has no audio track — cannot transcribe.")
    # «Пуст» = нет вовсе ИЛИ ноль сегментов; непустой катлист — святое
    # (кураторские правки), его не передетекчиваем.
    need_detect = s.cutlist is None or not s.cutlist.segments

    # Fail fast: битые общие опции (scale_h/fps) → 400 здесь, не ошибка задачи;
    # заодно out_dir создан и ссылки /api/output смотрят туда (паттерн render).
    _, _, _, out_dir0, _ = _resolve_render_opts(s, render_opts)
    s.last_out_dir = str(out_dir0.resolve())

    # Решения о клипах — ДО старта: от них зависят веса прогресса. Свежий
    # clips.json (hash-валидация как в GET /api/clips) переиспользуется —
    # 2.5-минутный LLM-проход не гоняется зря.
    cached = _load_clips_json(s) if do_clips else None
    cached_fresh = bool(cached and cached.get("hash") == s.audio_hash
                        and cached.get("clips"))
    will_suggest = do_clips and not cached_fresh and s.llm is not None
    will_render_clips = do_clips and (cached_fresh or s.llm is not None)

    # ENRICH §5: стадия выполняется ТОЛЬКО при body.enrich=true И существующем
    # СВЕЖЕМ плане (hash совпал). Детекторы автопак НЕ зовёт — нет плана →
    # warning и пропуск (запуск анализа из автопака решит P3).
    do_enrich = bool(body.get("enrich", False))
    enrich_plan = enrich_mod.load_enrich(_enrich_json_path(s)) \
        if do_enrich else None
    will_enrich = bool(enrich_plan is not None
                       and enrich_plan.hash == s.audio_hash)

    plan = [k for k, on in (("transcribe", need_transcribe),
                            ("detect", need_detect),
                            ("enrich", will_enrich),
                            ("main", True),
                            ("suggest", will_suggest),
                            ("clips", will_render_clips)) if on]
    total_w = sum(_AUTOPACK_WEIGHTS[k] for k in plan)
    spans: dict[str, tuple[float, float]] = {}
    off = 0.0
    for k in plan:
        w = _AUTOPACK_WEIGHTS[k] / total_w
        spans[k] = (off, w)
        off += w

    def sub(key: str):
        """on_progress стадии: её 0..1 → свой отрезок общего percent."""
        o, w = spans[key]
        return lambda f: s.set_progress(o + min(1.0, max(0.0, f)) * w)

    # Сервер принудительно глушит главы/метаданные для клипов (паттерн
    # /api/clips/render) — иначе каждый клип гонял бы LLM и тёр metadata.txt.
    # music=None — подложка (C3) только в ОСНОВНОМ ролике, клипы без музыки;
    # enrich=None — обогащение Shorts-клипов вне скоупа v1 (ENRICH_PLAN §9).
    clip_opts = {**render_opts, "chapters": False, "metadata": False,
                 "music": None, "enrich": None}
    # Основной ролик: обогащение управляется автопаком, а не сырыми
    # render_opts клиента — явный ключ в обе стороны (вкл со score-порогом /
    # выкл, если стадия не запланирована).
    main_opts = {**render_opts,
                 "enrich": ({"enabled": True,
                             "min_score": AUTOPACK_ENRICH_MIN_SCORE}
                            if will_enrich else {"enabled": False})}
    cl_duration = float(s.media.duration)

    def run():
        warnings: list[str] = []
        skipped: list[str] = []
        results: dict = {"warnings": warnings, "skipped": skipped}
        # Сразу в task: отмена/падение на любой стадии сохраняет уже сделанное.
        s.task["results"] = results

        def _chk() -> None:
            if s.task.get("cancelled"):
                raise RuntimeError("Задача отменена")

        # --- (а) транскрипция -------------------------------------------------
        if need_transcribe:
            _chk()
            p = sub("transcribe")
            s.stage("Извлечение аудио…")
            wav = extract_audio(s.ff, s.inp, s.work_dir / "audio16k.wav",
                                total=s.media.duration,
                                on_progress=lambda f: p(f * 0.1))
            _chk()
            s.stage("Транскрипция…")
            s.transcript = transcribe_audio(
                wav, s.cfg.transcribe, s.media.duration, s.audio_hash,
                cache_dir=s.cache_dir,
                log=lambda m="": s.stage(str(m).strip() or s.task["stage"]),
                on_progress=lambda f: p(0.1 + f * 0.9))
            p(1.0)
        else:
            skipped.append("Транскрипция: уже есть (кэш) — пропущена")

        # --- (б) детекция -----------------------------------------------------
        _chk()
        if need_detect:
            s.stage("Детекция вырезов…")
            s._detect()
            sub("detect")(1.0)
        else:
            skipped.append("Детекция: катлист уже есть — не передетекчиваем")

        # --- (б2) обогащение основного ролика (ENRICH §5) ----------------------
        # Стадия лишь фиксирует решение и счётчик: сам план применит
        # _run_render_pipeline стадии «main» (opts.enrich в main_opts).
        if do_enrich:
            if will_enrich:
                _chk()
                n_apply = sum(
                    1 for it in enrich_plan.items
                    if it.enabled and it.score >= AUTOPACK_ENRICH_MIN_SCORE)
                s.stage(f"Обогащение: применяю план "
                        f"(score ≥ {AUTOPACK_ENRICH_MIN_SCORE}, "
                        f"предложений: {n_apply})…")
                results["enrich"] = {"applied": True,
                                     "min_score": AUTOPACK_ENRICH_MIN_SCORE,
                                     "count": n_apply}
                sub("enrich")(1.0)
            elif enrich_plan is not None:
                warnings.append("Обогащение пропущено: план от другого видео "
                                "(несвежий hash) — запустите «Предложить "
                                "монтаж» заново")
                results["enrich"] = {"applied": False}
            else:
                warnings.append("Обогащение пропущено: нет плана — сначала "
                                "«Предложить монтаж»")
                results["enrich"] = {"applied": False}

        # --- (в) основной ролик (мультиформат C2) ------------------------------
        _chk()
        res_main = _render_formats(
            s, dict(main_opts), formats,
            on_progress=sub("main"),
            on_stage=lambda m: s.stage(f"Основной ролик: {m}"),
            is_cancelled=lambda: bool(s.task.get("cancelled")))
        results["main"] = res_main
        removed_now, _ = resolve(s.cutlist)
        totals = {"duration_before": round(float(s.media.duration), 1),
                  "duration_after": res_main.get("new_duration"),
                  "cuts": len(removed_now),
                  "clips_rendered": 0}
        results["totals"] = totals
        _chk()

        # --- (г) подбор клипов --------------------------------------------------
        cands = None       # None = рендерить нечего; list = кандидаты в порядке re-rank
        if not do_clips:
            skipped.append("Клипы: выключены (clips=false)")
        elif cached_fresh:
            skipped.append("Подбор клипов: использован сохранённый clips.json")
            cands = cached["clips"]
        elif s.llm is None:
            # Зафиксированное решение (план §2.4): фолбэк-нарезки без LLM нет.
            warnings.append("ИИ выключен — клипы не предложены")
            results["clips"] = []
        else:
            _chk()
            s.stage("Клипы: подбор…")
            try:
                fresh = clips_mod.suggest(
                    s.transcript, s.cutlist, s.cfg.clips, s.cfg.llm, s.llm,
                    log=lambda *_: None,
                    on_progress=sub("suggest"), on_stage=s.stage)
            except Exception as e:  # noqa: BLE001 — Ollama умерла: частичный успех
                _chk()              # отмена «через» suggest — честный cancelled
                warnings.append(f"Подбор клипов не удался: {e}")
                results["clips"] = {"error": str(e)}
            else:
                sub("suggest")(1.0)
                _save_clips_json(s, fresh)   # панель клипов оживёт без LLM
                cands = [asdict(c) for c in fresh]
                if not cands:
                    warnings.append("ИИ не нашёл подходящих клипов")
                    results["clips"] = []
                    cands = None

        # --- (д) рендер топ-K клипов — общий цикл с /api/clips/render ----------
        if cands is not None:
            top = _autopack_top_clips(cands, cl_duration, top_k)
            n = len(top)
            clip_results: list[dict] = []
            results["clips"] = clip_results
            p = sub("clips")
            for i, c in enumerate(top):
                if s.task.get("cancelled"):
                    break                       # cancel между клипами
                fname = f"{s.inp.stem}_clip{i + 1:02d}"
                job = _render_clip_job(s, start=c["start"], end=c["end"],
                                       idx=i, n=n, fname=fname,
                                       render_opts=clip_opts, set_pct=p)
                entry = {"ok": job["ok"], "id": c["id"], "filename": fname,
                         "hook": c["hook"]}
                if job["ok"]:
                    entry["mp4"] = job.get("mp4")
                else:
                    entry["error"] = job.get("error")
                clip_results.append(entry)
            totals["clips_rendered"] = sum(1 for x in clip_results if x.get("ok"))

        _chk()
        results["ok"] = True   # частичный успех (warnings) — тоже успех

    s.start_task("autopack", run)
    return {"ok": True, "top_k": top_k, "formats": formats, "clips": do_clips}


@app.get("/api/events")
async def events(request: Request):
    s = S()

    def _payload() -> str:
        t = s.task
        return json.dumps({k: t[k] for k in
                           ("name", "running", "percent", "stage",
                            "error", "done", "results", "cancelled")})

    async def gen():
        last = None
        while True:
            if await request.is_disconnected():
                break
            payload = _payload()
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if not s.task["running"]:
                # Always flush one final snapshot so the client sees the
                # terminal state (done/error/results) before we close.
                final = _payload()
                if final != last:
                    yield f"data: {final}\n\n"
                break
            await anyio.sleep(0.2)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --- A7: silence Windows client-disconnect noise -------------------------------
class _ConnectionResetFilter(logging.Filter):
    """Drop ONLY the ConnectionResetError / WinError 10054 noise.

    On Windows, uvicorn + the asyncio proactor loop dump a full traceback
    («ConnectionResetError [WinError 10054]», often via
    ``_ProactorBasePipeTransport._call_connection_lost``) every time the browser
    aborts an in-flight response — which happens constantly while seeking the
    <video> element (each seek kills the previous /api/video range request).
    That is normal client behaviour, not a server fault, so it is pure console
    spam.

    The filter is deliberately NARROW: it matches the exception type in
    ``exc_info`` and the exact marker strings in the formatted message, and
    nothing else — any other error on these loggers (real socket faults, h11
    protocol errors, unhandled exceptions in tasks) still gets through, so we
    never hide a genuine problem.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # True -> keep record
        exc = record.exc_info[1] if isinstance(record.exc_info, tuple) and \
            len(record.exc_info) > 1 else None
        if isinstance(exc, ConnectionResetError):
            return False
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — unformattable record: keep it
            return True
        if "ConnectionResetError" in msg or "WinError 10054" in msg:
            return False
        return True


def _install_connection_reset_filter() -> None:
    """Attach the filter to the two loggers that emit the disconnect spam.

    Logger-level filters survive uvicorn's dictConfig (it replaces handlers but
    never strips existing filters), so installing before ``uvicorn.run`` works.
    """
    flt = _ConnectionResetFilter()
    for name in ("uvicorn.error", "asyncio"):
        logging.getLogger(name).addFilter(flt)


def main() -> int:
    ap = argparse.ArgumentParser(description="FastVideoEdit web editor")
    ap.add_argument("--video", default=None, help="video to open (optional — you can pick one in the UI)")
    ap.add_argument("--start", default=None, help="start folder for the file browser")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="блокировать любые исходящие соединения в интернет (Whisper/Ollama локально продолжают работать)")
    args = ap.parse_args()

    # P2-#4: arm the outbound-connection guard BEFORE any Whisper/LLM/HF work, so
    # every connect (and any blocking) is accounted for from the very first call.
    netguard.install()

    cfg = load_config(args.config)
    if args.out:
        cfg.paths.out_dir = args.out
    APP["cfg"] = cfg
    APP["out_dir"] = cfg.paths.out_dir
    APP["use_llm"] = not args.no_llm

    # P2-#5: apply persisted model choices (models.json) to cfg BEFORE opening the
    # session, so the user's last Whisper/LLM selection is the default for this run.
    _apply_saved_models(cfg)

    # P2-#6: restore the persisted batch queue (queue.json). A job left 'running'
    # (server died mid-render) is reset to 'pending'; done/error jobs keep their
    # result/error. The worker is NOT auto-started — the user presses «Старт».
    _load_queue()

    # C5: папка-наблюдатель — восстановить watch.json и, если включён, поднять
    # фоновый сканер (реестр обработанных не даст пережевать папку заново).
    _load_watch()
    _watch_apply()

    # Startup offline state: persisted flag from privacy.json, overridden by --offline.
    start_offline = _read_privacy_offline() or args.offline
    netguard.set_offline(start_offline)
    if start_offline:
        print("  Приватность: ОФФЛАЙН-режим — исходящие соединения в интернет заблокированы.")
    APP["start_dir"] = (args.start or
                        (str(Path(args.video).resolve().parent) if args.video else str(Path.home())))

    if args.video:
        if not Path(args.video).exists():
            print(f"Video not found: {args.video}")
            return 2
        try:
            open_session(args.video)
        except ValueError as e:
            print(str(e))
            return 2

    # Sweep stale *.part scraps left by an interrupted render/upload.
    try:
        for d in (Path(cfg.paths.work_dir), Path(cfg.paths.out_dir)):
            if d.exists():
                for p in d.rglob("*.part"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
    except Exception:  # noqa: BLE001 — best-effort cleanup, never block startup
        pass

    # Pin the Host header (DNS-rebinding defence). Added here, not at module
    # scope, so the TestClient (Host: testserver) isn't rejected by the suite.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts(args.host))

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    url = f"http://{args.host}:{args.port}"
    if args.host not in ("127.0.0.1", "localhost"):
        print("\n  ВНИМАНИЕ: сервер слушает на " + args.host + ", а не на localhost.")
        print("  Эндпоинты /api/browse и /api/open позволяют просматривать и")
        print("  открывать ПРОИЗВОЛЬНЫЕ файлы на этой машине — любой, кто видит")
        print("  этот адрес в сети, получит такой же доступ. Используйте только")
        print("  в доверенной сети.")
        print("  WARNING: bound to a non-local host — arbitrary file browse/open"
              " is exposed to the network.")
    print(f"\n  FastVideoEdit editor -> {url}")
    if SESSION is not None:
        print(f"  video: {SESSION.inp.name}  ({SESSION.media.describe()})")
        print(f"  transcript: {'loaded' if SESSION.transcript else 'NOT yet — click Транскрибировать'}"
              f"   LLM: {'ready' if SESSION.llm else 'off'}\n")
    else:
        print(f"  no video yet — open one in the UI (tab «Файлы»). Browsing from: {APP['start_dir']}\n")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    # A7: mute the «ConnectionResetError [WinError 10054]» tracebacks the
    # proactor loop prints whenever the browser drops a video stream mid-read.
    _install_connection_reset_filter()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
