"""Thin wrapper around ffmpeg / ffprobe: discovery, capability probing, and a
progress-aware runner."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .config import FfmpegCfg


class FFmpegError(RuntimeError):
    pass


# A many-cut / censor-dense -filter_complex overflows the Windows 32767-char
# CreateProcess limit (~196 chars per kept segment). Any graph longer than this
# is written to a temp file and passed via -filter_complex_script so it never
# touches the command line. Small graphs pass through byte-for-byte unchanged so
# short renders keep their exact argv (and every FakeFF test still sees an inline
# -filter_complex). Keep this ABOVE the largest single-statement graph the
# enrich/music fast-paths build, or those would start spilling too (harmless but
# would change their argv).
_FILTERGRAPH_INLINE_MAX = 8000


def _spill_filtergraph(cmd: "list[str]") -> "tuple[list[str], Optional[str]]":
    """If cmd carries an oversized inline -filter_complex, spill it to a temp
    file and swap in -filter_complex_script. Returns (new_cmd, temp_path|None).

    Only the local command is rewritten; the caller's ``args`` list is never
    touched, so the FFmpegError diagnostic and FakeFF tests still read the
    inline graph from ``args``.
    """
    try:
        i = cmd.index("-filter_complex")
    except ValueError:
        return cmd, None
    graph = cmd[i + 1] if i + 1 < len(cmd) else ""
    if len(graph) <= _FILTERGRAPH_INLINE_MAX:
        return cmd, None
    fd, path = tempfile.mkstemp(suffix=".ffscript", prefix="fve_fc_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(graph)
    out = list(cmd)
    out[i:i + 2] = ["-filter_complex_script", path]
    return out, path


# --- running-process registry -----------------------------------------------
# Track every live ffmpeg subprocess so a UI/server can cancel an in-flight
# render (see cancel_all()). Guarded by a lock because run() is called from
# background worker threads.
_running_procs: "set[subprocess.Popen]" = set()
_running_lock = threading.Lock()


def _register_proc(proc: "subprocess.Popen") -> None:
    with _running_lock:
        _running_procs.add(proc)


def _unregister_proc(proc: "subprocess.Popen") -> None:
    with _running_lock:
        _running_procs.discard(proc)


def cancel_all() -> int:
    """Terminate every tracked, still-running ffmpeg process.

    Returns the number of processes that were signalled. Safe to call from any
    thread; each run() removes its own process in a finally block.
    """
    with _running_lock:
        procs = list(_running_procs)
    n = 0
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
                n += 1
        except Exception:
            pass
    return n


def _winget_candidates(name: str) -> list[str]:
    """Common locations a winget-installed ffmpeg lands in on Windows."""
    out: list[str] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        links = Path(local) / "Microsoft" / "WinGet" / "Links" / f"{name}.exe"
        out.append(str(links))
        pkgs = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if pkgs.exists():
            for hit in pkgs.glob(f"Gyan.FFmpeg*/**/bin/{name}.exe"):
                out.append(str(hit))
    return out


def resolve_bin(configured: str, name: str) -> str:
    """Resolve an ffmpeg/ffprobe binary to a runnable path."""
    # 1) explicit absolute path in config
    if os.path.isabs(configured) and Path(configured).exists():
        return configured
    # 2) on PATH
    found = shutil.which(configured) or shutil.which(name)
    if found:
        return found
    # 3) known winget locations (PATH may not be refreshed in this process)
    for cand in _winget_candidates(name):
        if Path(cand).exists():
            return cand
    raise FFmpegError(
        f"Could not find '{name}'. Install it (winget install Gyan.FFmpeg) or set "
        f"ffmpeg.{name}_bin to an absolute path in config.yaml.")


class FFmpeg:
    def __init__(self, cfg: FfmpegCfg):
        self.ffmpeg = resolve_bin(cfg.ffmpeg_bin, "ffmpeg")
        self.ffprobe = resolve_bin(cfg.ffprobe_bin, "ffprobe")
        self._caps: dict[str, set[str]] = {}
        # Seconds of total silence on both ffmpeg pipes before the no-progress
        # watchdog kills a presumed-hung render (see _run_proc). 0 disables it.
        self.stall_timeout = float(getattr(cfg, "stall_timeout", 180.0) or 0.0)
        # Seconds of total silence on both ffmpeg pipes before the no-progress
        # watchdog kills a presumed-hung render (see _run_proc). 0 disables it.
        self.stall_timeout = float(getattr(cfg, "stall_timeout", 180.0) or 0.0)

    # --- capability probing --------------------------------------------------
    def _list(self, kind: str) -> set[str]:
        if kind in self._caps:
            return self._caps[kind]
        flag = {"encoders": "-encoders", "filters": "-filters"}[kind]
        try:
            r = subprocess.run([self.ffmpeg, "-hide_banner", flag],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            names: set[str] = set()
            for line in r.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] and not parts[0].startswith("-"):
                    # encoders/filters lines: "<flags> <name> <desc...>"
                    names.add(parts[1])
            self._caps[kind] = names
        except Exception:
            self._caps[kind] = set()
        return self._caps[kind]

    def has_encoder(self, name: str) -> bool:
        return name in self._list("encoders")

    def has_filter(self, name: str) -> bool:
        return name in self._list("filters")

    # --- ffprobe -------------------------------------------------------------
    def probe(self, path: str | Path) -> dict:
        r = subprocess.run(
            [self.ffprobe, "-v", "error", "-show_format", "-show_streams",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise FFmpegError(f"ffprobe failed for {path}:\n{r.stderr.strip()}")
        return json.loads(r.stdout)

    # --- runner --------------------------------------------------------------
    def run(self, args: list[str], total: Optional[float] = None,
            on_progress: Optional[Callable[[float], None]] = None,
            desc: str = "ffmpeg") -> str:
        """Run ffmpeg with ``args`` (no leading 'ffmpeg'); report progress.

        ``total`` is the expected output duration in seconds; ``on_progress``
        receives a 0..1 fraction. Raises FFmpegError with the stderr tail.

        Returns the captured stderr window (first ~20 + last ~40 lines) on
        success — enough for callers that parse trailing report blocks (e.g.
        the ``loudnorm=...:print_format=json`` stats of the 2-pass loudness
        measurement). Existing callers that ignore the return value are
        unaffected.
        """
        cmd = [self.ffmpeg, "-hide_banner", "-nostdin", "-y",
               "-progress", "pipe:1", "-nostats", *args]
        # Spill an oversized -filter_complex to a temp file (Windows 32k argv
        # limit). No-op for normal graphs; the temp file is removed in the outer
        # finally below even on error/cancel.
        cmd, _fc_script = _spill_filtergraph(cmd)
        try:
            return self._run_proc(cmd, args, total, on_progress, desc)
        finally:
            if _fc_script:
                try:
                    os.remove(_fc_script)
                except OSError:
                    pass

    def _run_proc(self, cmd: list[str], args: list[str], total: Optional[float],
                  on_progress: Optional[Callable[[float], None]], desc: str) -> str:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        _register_proc(proc)

        # No-progress watchdog: if ffmpeg goes silent on BOTH pipes for
        # ``stall_timeout`` seconds it is presumed hung (an NVENC/TDR lockup, a
        # dropped network-drive input, a starved filtergraph) and terminated, so
        # the worker thread — and with it every mutating endpoint _guard_no_task
        # would 409 — stops waiting forever. Armed only when a positive window
        # AND a known ``total`` are set, so capability probes (-encoders/-filters)
        # and any zero-total call are never watched. ANY line on either pipe
        # resets the timer (progress OR stderr, so a caller that passes ``total``
        # without ``on_progress`` — e.g. the loudnorm measure pass — is not
        # false-killed), and a healthy encode ticks sub-second, so the generous
        # default never trips a real render.
        last_progress = [time.monotonic()]
        stalled = [False]
        win = getattr(self, "stall_timeout", 0.0) or 0.0

        def _watchdog() -> None:
            while proc.poll() is None:
                if time.monotonic() - last_progress[0] > win:
                    stalled[0] = True
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    return
                time.sleep(min(2.0, win / 2.0))

        # Keep the first ~20 lines (config/setup errors land here) AND the last
        # ~40 lines (the actual failure) so diagnostics are not truncated to one
        # end of a long ffmpeg log.
        stderr_head: list[str] = []
        stderr_tail: list[str] = []

        def drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                last_progress[0] = time.monotonic()
                if len(stderr_head) < 20:
                    stderr_head.append(line)
                stderr_tail.append(line)
                if len(stderr_tail) > 40:
                    del stderr_tail[0]

        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()

        if win > 0 and total:
            threading.Thread(target=_watchdog, daemon=True).start()

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                last_progress[0] = time.monotonic()
                line = line.strip()
                if line.startswith("out_time_us=") and total and on_progress:
                    try:
                        us = int(line.split("=", 1)[1])
                        on_progress(min(1.0, max(0.0, (us / 1e6) / total)))
                    except (ValueError, ZeroDivisionError):
                        pass
                elif line == "progress=end" and on_progress:
                    on_progress(1.0)
            proc.wait()
            t.join(timeout=1.0)
        finally:
            _unregister_proc(proc)

        # A watchdog kill terminates the process, so proc.returncode is non-zero;
        # surface the stall as its own clear error BEFORE the generic exit-code
        # dump below, which would otherwise mask it with a truncated stderr tail.
        if stalled[0]:
            raise FFmpegError(
                f"{desc}: нет прогресса дольше {int(win)} c — процесс остановлен "
                f"(возможно, зависание кодировщика или потерян вход).")

        if proc.returncode != 0:
            head = "".join(stderr_head).strip()
            tail = "".join(stderr_tail).strip()
            # Avoid duplicating the log when it was short enough to fit entirely
            # in both buffers.
            if tail.startswith(head):
                detail = tail
            else:
                detail = f"{head}\n  ...\n{tail}"
            parts = [f"{desc} failed (exit {proc.returncode}). ffmpeg said:\n{detail}"]
            if "-filter_complex" in args:
                try:
                    graph = args[args.index("-filter_complex") + 1]
                    # Cap the reconstructed graph: a long many-cut render has an
                    # ~18-30KB graph that would otherwise BECOME task['error'],
                    # flood the SSE payload and push the real ffmpeg line (kept
                    # FIRST in `detail`) off the auto-dismissing toast. A
                    # head+tail excerpt keeps it diagnosable; the full graph is
                    # also on disk as the spilled .ffscript during the run.
                    if len(graph) > 1200:
                        graph = (graph[:600]
                                 + f"\n… [{len(graph) - 1200} chars omitted] …\n"
                                 + graph[-600:])
                    parts.append(f"\nfilter_complex graph:\n{graph}")
                except (IndexError, ValueError):
                    pass
            raise FFmpegError("".join(parts))

        # Success: hand back the stderr window for trailing-report parsers
        # (harmless duplication of a few lines is possible when the log is
        # 20..40 lines long — parsers take the LAST match, so it's safe).
        head = "".join(stderr_head)
        tail = "".join(stderr_tail)
        return tail if tail.startswith(head) else head + tail


# --- ffprobe helpers ---------------------------------------------------------
def parse_fps(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        n, d = rate.split("/")
        d = float(d)
        return float(n) / d if d else 0.0
    return float(rate)
