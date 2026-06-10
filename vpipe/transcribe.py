"""Stage 2 — transcription via faster-whisper (CUDA, CPU fallback) with caching
and a CUDA-OOM model-downgrade ladder."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from .config import TranscribeCfg
from .models import Segment, Transcript, Word

# Only genuine memory-exhaustion messages trigger the smaller-model GPU ladder.
# Library/driver problems (missing cuDNN/cuBLAS DLLs, version mismatch) are NOT
# fixed by a smaller model, so those break straight to the CPU fallback instead
# of pointlessly reloading every model size on the GPU.
_OOM_HINTS = ("out of memory", "cuda_error_out_of_memory",
              "alloc_failed", "status_alloc_failed", "cudamalloc")


def bootstrap_cuda_dlls() -> None:
    """On Windows, make the pip-installed cuBLAS/cuDNN DLLs loadable.

    The CTranslate2 wheel links these dynamically but does not bundle them; the
    nvidia-*-cu12 wheels drop them under site-packages/nvidia/**/bin. We add
    those dirs to the DLL search path so faster-whisper finds them.
    """
    if sys.platform != "win32":
        return
    try:
        import site
        roots: list[Path] = [Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
        try:
            for sp in site.getsitepackages():
                roots.append(Path(sp) / "nvidia")
        except Exception:
            pass
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for bindir in root.glob("*/bin"):
                key = str(bindir).lower()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    os.add_dll_directory(str(bindir))
                    os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
                except OSError:
                    pass
    except Exception:
        pass


# --- first-run model-download notice -----------------------------------------
# faster-whisper silently downloads the model from Hugging Face on first use —
# the #1 "it hangs" complaint. Before constructing WhisperModel we check the
# local HF cache (filesystem only, ZERO network — offline-safe) and, if the
# model is missing, announce the one-time download and report progress every
# couple of seconds from a watcher thread. When the model is already cached the
# behaviour is byte-for-byte unchanged (not a single extra message).

# Approximate one-time download sizes in decimal GB (model.bin + tokenizer).
_MODEL_DOWNLOAD_GB = {
    "tiny": 0.08, "tiny.en": 0.08,
    "base": 0.15, "base.en": 0.15,
    "small": 0.5, "small.en": 0.5,
    "distil-small.en": 0.4,
    "medium": 1.5, "medium.en": 1.5,
    "distil-medium.en": 0.8,
    "large-v1": 3.1, "large-v2": 3.1, "large-v3": 3.1, "large": 3.1,
    "distil-large-v2": 1.5, "distil-large-v3": 1.5, "distil-large-v3.5": 1.5,
    "large-v3-turbo": 1.6, "turbo": 1.6,
}


def _hf_repo_id(size: str) -> Optional[str]:
    """Map a faster-whisper model size to its HF repo id. Local lookup only."""
    try:
        if "/" in size:  # already a repo id, e.g. "Systran/faster-whisper-..."
            return size
        from faster_whisper.utils import _MODELS  # lazy: keeps --help fast
        return _MODELS.get(size)
    except Exception:
        return None


def _hf_model_dir(repo_id: str,
                  cache_root: Optional[str | Path] = None) -> Optional[Path]:
    """Local HF-cache directory of the repo (it may not exist yet)."""
    try:
        if cache_root is None:
            from huggingface_hub import constants  # transitively installed
            cache_root = constants.HF_HUB_CACHE
        return Path(cache_root) / ("models--" + repo_id.replace("/", "--"))
    except Exception:
        return None


def _model_in_cache(repo_dir: Optional[Path]) -> bool:
    """True when a snapshot with model.bin already sits in the local HF cache.

    Pure filesystem check — no network, so offline mode is never broken.
    On any unexpected error we assume "cached" so behaviour stays as before.
    """
    try:
        if repo_dir is None:
            return True  # cannot tell -> stay silent, change nothing
        snaps = repo_dir / "snapshots"
        if not snaps.is_dir():
            return False
        # .exists() resolves the blob symlink; a broken link counts as missing.
        return any(p.exists() for p in snaps.glob("*/model.bin"))
    except Exception:
        return True


def _dir_gb(path: Path) -> float:
    """Recursive directory size in decimal gigabytes (best effort)."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / 1e9


def _start_download_watch(size: str, log, *, interval: float = 2.0,
                          cache_root: Optional[str | Path] = None,
                          ) -> Optional[Callable[[], None]]:
    """Announce a first-run model download and watch its progress.

    If the model is missing from the local HF cache: log a Russian one-time
    notice with the approximate size, then start a daemon thread that measures
    the model's cache directory every ``interval`` seconds and reports the
    downloaded gigabytes through ``log`` (-> stage -> SSE -> UI).

    Returns a stop() callable to invoke right after WhisperModel() returns
    (or raises). Returns None — and emits nothing — when the model is already
    cached, offline mode is on, or anything at all goes wrong.
    """
    try:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            return None  # offline mode never downloads — do not promise one
        repo_id = _hf_repo_id(size)
        if repo_id is None:
            return None
        repo_dir = _hf_model_dir(repo_id, cache_root)
        if repo_dir is None or _model_in_cache(repo_dir):
            return None

        approx = _MODEL_DOWNLOAD_GB.get(size)
        approx_txt = f"~{approx:g} ГБ, " if approx else ""
        log(f"Скачиваю модель Whisper «{size}» "
            f"({approx_txt}однократно — дальше работает офлайн)…")

        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(interval):
                try:
                    gb = _dir_gb(repo_dir)
                    if gb <= 0:
                        continue  # download has not materialised on disk yet
                    tail = f" из ~{approx:g} ГБ" if approx else ""
                    log(f"Скачиваю модель Whisper «{size}»… {gb:.1f} ГБ{tail}")
                except Exception:
                    continue  # the watcher must never break a real download

        t = threading.Thread(target=watch, daemon=True,
                             name=f"whisper-download-watch-{size}")
        t.start()

        def stopper() -> None:
            stop.set()
            t.join(timeout=interval + 1.0)

        return stopper
    except Exception:
        return None  # dirt-proof: any failure -> behave exactly as before


def _is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return any(h in msg for h in _OOM_HINTS)


# --- system-proxy workaround --------------------------------------------------
# A Windows system proxy (e.g. socks4://127.0.0.1:10808 from a local proxy
# client) breaks WhisperModel() even when the model is fully cached and
# HF_HUB_OFFLINE=1: huggingface_hub builds its HTTP client BEFORE the cache
# check and dies on "Unknown scheme for proxy URL 'socks4://…'". Found by the
# 2026-06 pilot run on this very machine. We detect that error class and retry
# once with proxy env neutralised (NO_PROXY=* + cleared *_PROXY) — the cached
# load then succeeds without any network at all.
_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                   "http_proxy", "https_proxy", "all_proxy")


def _is_proxy_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "proxy" in msg and ("unknown scheme" in msg or "socks" in msg
                               or "unsupported" in msg)


def _load_model(size: str, device: str, ctype: str, log):
    """Construct WhisperModel; on a system-proxy error retry with proxies off."""
    from faster_whisper import WhisperModel  # imported late so --help stays fast
    try:
        return WhisperModel(size, device=device, compute_type=ctype)
    except Exception as e:  # noqa: BLE001
        if not _is_proxy_error(e):
            raise
        log("  системный прокси мешает huggingface_hub — повторяю без прокси…")
        saved = {v: os.environ.pop(v) for v in _PROXY_ENV_VARS if v in os.environ}
        saved_no = {v: os.environ.get(v) for v in ("NO_PROXY", "no_proxy")}
        os.environ["NO_PROXY"] = "*"
        try:
            return WhisperModel(size, device=device, compute_type=ctype)
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(
                "Системный прокси Windows (socks4/5) несовместим с huggingface_hub. "
                "Отключите прокси-клиент на время первой загрузки модели, задайте "
                "HTTP_PROXY=http://…, либо укажите модель ЛОКАЛЬНЫМ ПУТЁМ в "
                "config.yaml (transcribe.model) — локальный путь сеть не трогает. "
                f"Исходная ошибка: {e2}") from e2
        finally:
            os.environ.update(saved)
            for v, val in saved_no.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val


def _model_ladder(cfg: TranscribeCfg) -> list[str]:
    ladder = [cfg.model]
    for m in cfg.fallback_models:
        if m not in ladder:
            ladder.append(m)
    return ladder


def _run_once(audio_path: str, size: str, device: str, ctype: str,
              cfg: TranscribeCfg, duration: float, audio_hash: str,
              log, on_progress=None) -> Transcript:
    log(f"  loading whisper '{size}' on {device} ({ctype}) ...")
    # First run only: announce the silent HF download + progress watcher.
    # No-op (None) when the model is already in the local cache.
    stop_watch = _start_download_watch(size, log)
    try:
        model = _load_model(size, device, ctype, log)
    finally:
        if stop_watch is not None:
            try:
                stop_watch()
            except Exception:
                pass  # never let the watcher mask the real outcome
    try:
        log(f"  transcribe attempt: model='{size}' device={device} ({ctype})")
        seg_iter, info = model.transcribe(
            audio_path,
            language=cfg.language,
            beam_size=cfg.beam_size,
            word_timestamps=True,
            # Disable carry-over of prior text: long-form decoding otherwise drifts
            # and shifts word boundaries, causing audio/caption desync.
            condition_on_previous_text=False,
            vad_filter=cfg.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=cfg.vad_min_silence_ms),
        )
        segments: list[Segment] = []
        # Force evaluation HERE so a lazy CUDA OOM raises inside this try.
        for s in seg_iter:
            words = [Word(w.word, float(w.start), float(w.end),
                          float(w.probability or 0.0))
                     for w in (s.words or [])
                     if w.start is not None and w.end is not None]
            segments.append(Segment(float(s.start), float(s.end),
                                    s.text.strip(), words))
            if on_progress and duration:
                on_progress(min(1.0, s.end / duration))
        return Transcript(language=info.language or cfg.language,
                          duration=duration, model=size,
                          requested_model=cfg.model,
                          audio_hash=audio_hash, segments=segments)
    finally:
        del model
        _free_gpu()


def transcribe_audio(audio_path: str | Path, cfg: TranscribeCfg,
                     duration: float, audio_hash: str,
                     cache_dir: Optional[str | Path] = None,
                     log=print, on_progress=None) -> Transcript:
    """Transcribe ``audio_path``; cache by ``audio_hash``; fall back on OOM.

    Order: requested device down the model ladder (large-v3 -> medium -> small),
    then, if all GPU attempts fail, CPU on the same ladder.
    """
    cache_path: Optional[Path] = None
    if cfg.cache and cache_dir:
        cache_path = Path(cache_dir) / f"{audio_hash}.transcript.json"
        if cache_path.exists():
            try:
                cached = Transcript.load(cache_path)
            except Exception as e:  # noqa: BLE001 — corrupt cache: re-transcribe
                log(f"  cache unreadable ({str(e)[:80]}); re-transcribing")
                cached = None
            if cached is not None:
                req = getattr(cached, "requested_model", "") or ""
                if req:
                    # New cache: reuse when the REQUESTED model matches — even if
                    # the stored result is an OOM fallback (requested large-v3,
                    # got medium). Otherwise large-v3 would re-OOM every run.
                    hit = (req == cfg.model)
                else:
                    # Legacy cache (no requested_model): reuse if the actual model
                    # is a legitimate point on this run's ladder (an OOM fallback).
                    hit = getattr(cached, "model", None) in _model_ladder(cfg)
                if hit:
                    log(f"  cache hit: {cache_path.name} "
                        f"(model={getattr(cached, 'model', None)})")
                    return cached
                log(f"  cache for '{req or getattr(cached, 'model', None)}' != "
                    f"requested '{cfg.model}'; re-transcribing")

    bootstrap_cuda_dlls()
    audio_path = str(audio_path)
    ladder = _model_ladder(cfg)
    last_err: Optional[Exception] = None

    # Phase 1: requested device (typically cuda) down the ladder.
    if cfg.device != "cpu":
        for size in ladder:
            try:
                tr = _run_once(audio_path, size, cfg.device, cfg.compute_type,
                               cfg, duration, audio_hash, log, on_progress)
                return _cache_and_return(tr, cache_path, log)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_oom(e):
                    log(f"  !! GPU issue on '{size}' ({str(e)[:80]}); falling back ...")
                    continue
                log(f"  !! error on '{size}' ({str(e)[:120]}); trying CPU ...")
                break  # non-OOM GPU error: jump to CPU

    # Phase 2: CPU fallback.
    for size in ladder:
        try:
            tr = _run_once(audio_path, size, "cpu", "int8",
                           cfg, duration, audio_hash, log, on_progress)
            return _cache_and_return(tr, cache_path, log)
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"  !! CPU error on '{size}' ({str(e)[:100]})")
            continue

    # Offline mode + a model that isn't in the local HF cache fails with a raw
    # huggingface_hub trace on every ladder rung. Surface a clear, actionable
    # message instead (the privacy badge's offline toggle is the fix).
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError(
            "Модель Whisper не найдена в локальном кэше, а офлайн-режим включён. "
            "Выключите офлайн-режим (бейдж приватности «🔒») для разовой загрузки "
            "модели, затем включите его снова. / Whisper model not in the local "
            "cache and offline mode is on — disable offline mode once to download it.")
    raise RuntimeError(f"Transcription failed on all fallbacks. Last error: {last_err}")


def _cache_and_return(tr: Transcript, cache_path: Optional[Path], log) -> Transcript:
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tr.save(cache_path)
        log(f"  cached transcript -> {cache_path.name}")
    return tr


def _free_gpu() -> None:
    import gc
    gc.collect()
    try:  # torch is optional; only present in some envs
        import torch  # type: ignore
        torch.cuda.empty_cache()
    except Exception:
        pass
