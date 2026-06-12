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
import math
import os
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
    def _detect(self) -> CutList:
        assert self.transcript is not None
        # The 16 kHz wav (extracted during transcription) feeds the acoustic
        # hesitation detector; pass it when present so VAD can run.
        wav = self.work_dir / "audio16k.wav"
        cl = run_detection(self.transcript, self.cfg, self.fillers, self.profanity,
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


def _run_render_pipeline(s: Session, cfg, scale_h, fps, out_dir: Path,
                         base: Path, on_progress, on_stage,
                         cutlist_override: Optional[CutList] = None) -> dict:
    """Render mp4 + (optional) subtitles + chapters, returning the results dict.

    ``on_progress(frac)`` / ``on_stage(msg)`` are callbacks so the same body
    drives both the editor task (writes Session.task) and a queue job (writes
    QueueJob.percent/stage). Mirrors do_render: the mp4 is the irreplaceable
    artifact — a subtitles/chapters failure must NOT lose it.

    ``cutlist_override`` (Clip Maker, план §2.3.1): render against THIS cutlist
    instead of the session's (one Shorts clip = the live internal cuts plus
    boundary REMOVEs around [start, end]). Burn-in ASS subs and the Timeline are
    built from the SAME cutlist, and the session is never mutated. ``None``
    (the default) keeps the legacy single-render behavior bit-for-bit."""
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
            if vert_dims is not None:
                # Vertical: PlayRes must match the cropped+scaled output so style
                # sizes/margins are pixel-accurate against the 9:16 frame.
                out_w, out_h = vert_dims
            else:
                out_h = int(scale_h) if scale_h else (s.media.height or 1080)
                src_w = s.media.width or 1920
                src_h = s.media.height or 1080
                out_w = int(round(src_w * out_h / src_h)) if src_h else 1920
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

    on_stage("Рендер видео…")
    rr = render_mod.render(
        s.ff, s.media, cl, cfg, base.with_suffix(".mp4"), s.work_dir,
        on_progress=on_progress,
        log=lambda m="": on_stage(str(m).strip() or "Рендер видео…"),
        scale_h=scale_h, fps=fps, ass_path=ass_path, crop_filter=crop_filter)

    sr: dict = {}
    subtitles_ok = not cfg.subtitles.enabled
    if cfg.subtitles.enabled:
        try:
            on_stage("Субтитры…")
            sr = subs_mod.generate(tr, removed, cfg.subtitles, cfg.masking,
                                   s.matcher, base, log=lambda *_: None)
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
                "queue_pending": _queue_pending_count()}
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
            # P2-#5: current model choices (defaults for the «⚙ Модели» modal).
            "whisper_model": s.cfg.transcribe.model,
            "llm_model": s.cfg.llm.model,
        },
        "task": s.task,
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
def browse(dir: Optional[str] = None):
    base = Path(dir).expanduser() if dir else Path(APP.get("start_dir", str(Path.cwd())))
    base = base.resolve()
    if not base.exists() or not base.is_dir():
        raise HTTPException(404, "Folder not found")
    folders, files = [], []
    try:
        for e in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    folders.append(e.name)
                elif e.suffix.lower() in VIDEO_EXT:
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
    s = S()
    cache_file = s.cache_dir / f"{s.audio_hash}.peaks.json"
    if s.peaks is None and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            s.peaks = data.get("peaks") or []
        except Exception:  # noqa: BLE001 — corrupt cache: recompute below
            s.peaks = None
    if s.peaks is None:
        s.peaks = compute_peaks(s.ff.ffmpeg, s.inp, s.media.duration)
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
    return s.transcript.to_dict()


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
def redetect():
    s = S()
    _guard_no_task()
    if s.transcript is None:
        raise HTTPException(409, "Transcribe first")

    def run():
        s.stage("Детекция вырезов…")
        s._detect()

    s.start_task("detect", run)
    return {"ok": True}


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

    # Effective config + params = base config + UI overrides (shared with queue).
    cfg, scale_h, fps, out_dir, base = _resolve_render_opts(s, opts)
    s.last_out_dir = str(out_dir.resolve())

    def run():
        s.task["results"] = _run_render_pipeline(
            s, cfg, scale_h, fps, out_dir, base,
            on_progress=s.set_progress,
            on_stage=s.stage)

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
    return {
        "whisper": {
            "current": cfg.transcribe.model,
            "presets": WHISPER_PRESETS,
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
        s.task["results"] = {"clips": [asdict(c) for c in cands]}

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
            "model": data.get("model")}


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
    # независимо от того, что прислал клиент.
    render_opts = {**render_opts, "chapters": False, "metadata": False}

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

    cl = s.cutlist
    n = len(clips_in)

    def run():
        results: list[dict] = []
        for i, c in enumerate(clips_in):
            if s.task.get("cancelled"):
                break                               # cancel между клипами
            s.stage(f"Клип {i + 1}/{n}: рендер…")
            s.set_progress(i / n)
            # Катлист клипа = копия живых вырезов + 2 граничных REMOVE (§2.4).
            clip_cl = CutList(source=cl.source, duration=cl.duration,
                              segments=[copy.copy(seg) for seg in cl.segments])
            if c["start"] > 0:
                clip_cl.segments.append(CutSegment(
                    id=f"clipA{i}", start=0.0, end=c["start"],
                    type=TYPE_MANUAL, action=ACTION_REMOVE, enabled=True))
            if c["end"] < cl.duration:
                clip_cl.segments.append(CutSegment(
                    id=f"clipB{i}", start=c["end"], end=cl.duration,
                    type=TYPE_MANUAL, action=ACTION_REMOVE, enabled=True))
            try:
                opts_i = {**render_opts, "filename": c["filename"]}
                cfg, scale_h, fps, out_dir, base = _resolve_render_opts(s, opts_i)
                res = _run_render_pipeline(
                    s, cfg, scale_h, fps, out_dir, base,
                    on_progress=lambda f, i=i: s.set_progress(
                        (i + min(1.0, max(0.0, f))) / n),
                    on_stage=lambda m, i=i: s.stage(f"Клип {i + 1}/{n}: {m}"),
                    cutlist_override=clip_cl)
                results.append({"ok": True, "filename": c["filename"], **res})
            except Exception as e:  # noqa: BLE001 — упавший клип не валит остальные
                err = "cancelled" if s.task.get("cancelled") else str(e)
                results.append({"ok": False, "filename": c["filename"],
                                "error": err})
        s.task["results"] = {"clips": results}
        if s.task.get("cancelled"):
            # Частичные results уже сохранены выше; воркер start_task отчитается
            # чистым «cancelled» — как у обычного отменённого рендера.
            raise RuntimeError("Задача отменена")

    s.start_task("render_clips", run)
    return {"ok": True, "count": n}


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
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
