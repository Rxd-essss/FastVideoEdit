"""Process-wide outbound-connection guard — the engine behind the «zero-upload» promise.

FastVideoEdit runs entirely on the local machine: Whisper on the GPU, Ollama on
``localhost``, ffmpeg locally. The only outbound connection in normal operation is
faster-whisper / huggingface_hub fetching a Whisper model *once* (and only if it is
not already cached). This module makes that promise **provable and enforceable**:

* it accounts every outbound TCP ``connect`` the process makes, split into LOCAL
  (loopback / localhost / AF_UNIX — e.g. the Ollama call) and EXTERNAL (everything
  else, e.g. ``huggingface.co``), so the UI can show «0 внешних соединений»;
* in **offline mode** it *blocks* every EXTERNAL connect before it leaves the box,
  so even a hypothetical «phone-home» cannot exfiltrate the user's video.

Mechanism: we monkey-patch ``socket.socket.connect`` and ``socket.socket.connect_ex``
at the class level. ``http.client`` / ``urllib`` / ``requests`` / ``huggingface_hub``
all ultimately route through these, so a single hook covers every outbound TCP
connection THIS PYTHON PROCESS opens (Whisper's HF fetch, the Ollama call). We do
**not** touch ``bind`` / ``listen`` / ``accept`` — the editor's own web server (incoming
connections from the browser) is completely unaffected.

SCOPE / honest boundary (the UI copy must not over-promise): this guards the Python
process only. Separate *subprocesses* — ffmpeg, the Ollama server — have their own
sockets we don't see; in practice ffmpeg is only ever handed LOCAL file paths and
Ollama is expected on ``localhost``. DNS (``getaddrinfo``) is also not intercepted, so
in offline mode a name *lookup* may still leave the box before the (blocked) connect —
no content leaks that way, and the ``HF_HUB_OFFLINE`` env vars below stop Whisper's
fetch cleanly. So: "we block this process's outbound network sockets", NOT "no packet
of any kind can ever leave the machine".

Only the Python standard library is used: ``socket``, ``ipaddress``, ``threading``,
``os``, ``errno``.
"""
from __future__ import annotations

import errno
import ipaddress
import os
import socket as _socket
import threading
from typing import Callable

# --- module state -----------------------------------------------------------
_installed: bool = False
_lock = threading.Lock()            # guards _stats AND _offline (single mutex)
_offline: bool = False
_orig_connect: Callable | None = None
_orig_connect_ex: Callable | None = None
_stats: dict = {
    "local": 0,                     # loopback / localhost / AF_UNIX connects
    "external_allowed": 0,          # external connects actually established
    "blocked": 0,                   # external connects refused while offline
    "external_hosts": {},           # {host: count} for established externals
    "blocked_hosts": {},            # {host: count} for refused externals
}


# --- address classification -------------------------------------------------
def _classify(address) -> str:
    """Return ``"local"`` for loopback / localhost / AF_UNIX, else ``"external"``.

    AF_UNIX socket addresses are a plain ``str`` (a filesystem path), not a
    ``(host, port)`` tuple — those never leave the machine, so they are local.
    For ``(host, port[, flow, scope])`` we treat ``localhost`` / ``""`` and any
    loopback IP literal (``127.0.0.0/8``, ``::1``) as local; a non-IP hostname
    (e.g. ``huggingface.co``) is external by definition.
    """
    if isinstance(address, str):        # AF_UNIX path
        return "local"
    try:
        host = address[0]
    except (TypeError, IndexError):     # unexpected address shape — be safe
        return "local"
    if host in ("localhost", "", None):
        return "local"
    try:
        return "local" if ipaddress.ip_address(host).is_loopback else "external"
    except ValueError:
        # Not an IP literal -> a hostname that would need DNS -> external.
        return "external"


def _host_of(address) -> str:
    return address if isinstance(address, str) else str(address[0])


# --- hooked socket methods --------------------------------------------------
def _hooked_connect(self, address):
    kind = _classify(address)
    with _lock:
        if kind == "local":
            _stats["local"] += 1
        elif _offline:
            host = _host_of(address)
            _stats["blocked"] += 1
            _stats["blocked_hosts"][host] = _stats["blocked_hosts"].get(host, 0) + 1
            raise OSError(
                f"FastVideoEdit: оффлайн-режим — исходящее соединение к {host} заблокировано"
            )
        else:
            host = _host_of(address)
            _stats["external_allowed"] += 1
            _stats["external_hosts"][host] = _stats["external_hosts"].get(host, 0) + 1
    return _orig_connect(self, address)


def _hooked_connect_ex(self, address):
    kind = _classify(address)
    with _lock:
        if kind == "local":
            _stats["local"] += 1
        elif _offline:
            host = _host_of(address)
            _stats["blocked"] += 1
            _stats["blocked_hosts"][host] = _stats["blocked_hosts"].get(host, 0) + 1
            # connect_ex returns an errno instead of raising. ENETUNREACH is the
            # cross-platform «network unreachable» code from the stdlib.
            return errno.ENETUNREACH
        else:
            host = _host_of(address)
            _stats["external_allowed"] += 1
            _stats["external_hosts"][host] = _stats["external_hosts"].get(host, 0) + 1
    return _orig_connect_ex(self, address)


# --- public API -------------------------------------------------------------
def install() -> None:
    """Idempotently patch ``socket.socket.connect`` / ``.connect_ex``.

    Safe to call more than once (e.g. across test setups) — the wrapper is never
    stacked twice, and the original methods are saved exactly once.
    """
    global _installed, _orig_connect, _orig_connect_ex
    if _installed:
        return
    _orig_connect = _socket.socket.connect
    _orig_connect_ex = _socket.socket.connect_ex
    _socket.socket.connect = _hooked_connect
    _socket.socket.connect_ex = _hooked_connect_ex
    _installed = True


def set_offline(enabled: bool) -> None:
    """Toggle offline mode (block all EXTERNAL connects) + HF offline env vars.

    The env vars are belt-and-braces: faster-whisper / huggingface_hub then report
    «model not cached» immediately instead of hanging on a socket error mid-download.
    """
    global _offline
    enabled = bool(enabled)
    # Flip the flag and the env vars together under the lock so a concurrent
    # reader never sees offline=True with the HF env not yet set (or vice versa).
    with _lock:
        _offline = enabled
        if enabled:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "0"


def is_offline() -> bool:
    with _lock:
        return _offline


def stats() -> dict:
    """Thread-safe deep-ish copy of the counters (nested dicts copied too)."""
    with _lock:
        return {
            "local": _stats["local"],
            "external_allowed": _stats["external_allowed"],
            "blocked": _stats["blocked"],
            "external_hosts": dict(_stats["external_hosts"]),
            "blocked_hosts": dict(_stats["blocked_hosts"]),
        }


def reset() -> None:
    """Zero all counters (used by tests for isolation; never resets the offline flag)."""
    with _lock:
        _stats["local"] = 0
        _stats["external_allowed"] = 0
        _stats["blocked"] = 0
        _stats["external_hosts"].clear()
        _stats["blocked_hosts"].clear()
