"""Proxy-bug workaround in transcription (2026-06 pilot finding).

A Windows system proxy (socks4:// in the registry / env) makes WhisperModel()
fail even with a fully cached model. ``_load_model`` must detect that error
class, retry once with proxy env neutralised (NO_PROXY=*), and restore the
environment afterwards; non-proxy errors must propagate untouched.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

from vpipe.transcribe import _is_proxy_error, _load_model

PROXY_MSG = "Unknown scheme for proxy URL 'socks4://127.0.0.1:10808'"


def _fake_faster_whisper(monkeypatch, model_cls):
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = model_cls
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)


def test_is_proxy_error_matches_real_message():
    assert _is_proxy_error(ValueError(PROXY_MSG))
    assert _is_proxy_error(RuntimeError("Unsupported proxy scheme socks5h"))
    assert not _is_proxy_error(RuntimeError("CUDA out of memory"))
    assert not _is_proxy_error(RuntimeError("connection refused"))  # no 'proxy'


def test_proxy_error_retries_with_neutralised_env(monkeypatch):
    calls = []

    class Model:
        def __init__(self, size, device=None, compute_type=None):
            calls.append({v: os.environ.get(v)
                          for v in ("HTTP_PROXY", "ALL_PROXY", "NO_PROXY")})
            if len(calls) == 1:
                raise ValueError(PROXY_MSG)

    _fake_faster_whisper(monkeypatch, Model)
    monkeypatch.setenv("HTTP_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.setenv("ALL_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.delenv("NO_PROXY", raising=False)

    logs = []
    m = _load_model("large-v3", "cuda", "int8_float16", logs.append)

    assert m is not None and len(calls) == 2
    # Retry ran with proxies cleared and NO_PROXY=*.
    assert calls[1]["HTTP_PROXY"] is None
    assert calls[1]["ALL_PROXY"] is None
    assert calls[1]["NO_PROXY"] == "*"
    # Environment restored afterwards.
    assert os.environ["HTTP_PROXY"] == "socks4://127.0.0.1:10808"
    assert os.environ["ALL_PROXY"] == "socks4://127.0.0.1:10808"
    assert "NO_PROXY" not in os.environ
    assert any("прокси" in s for s in logs)


def test_proxy_error_twice_raises_actionable_message(monkeypatch):
    class Model:
        def __init__(self, *a, **k):
            raise ValueError(PROXY_MSG)

    _fake_faster_whisper(monkeypatch, Model)
    with pytest.raises(RuntimeError) as ei:
        _load_model("large-v3", "cuda", "int8_float16", lambda *_: None)
    msg = str(ei.value)
    assert "прокси" in msg.lower() and "локальн" in msg.lower()


def test_non_proxy_error_propagates_without_retry(monkeypatch):
    calls = []

    class Model:
        def __init__(self, *a, **k):
            calls.append(1)
            raise RuntimeError("CUDA out of memory")

    _fake_faster_whisper(monkeypatch, Model)
    with pytest.raises(RuntimeError, match="out of memory"):
        _load_model("large-v3", "cuda", "int8_float16", lambda *_: None)
    assert len(calls) == 1  # no retry for non-proxy errors


def test_env_restored_even_when_retry_fails(monkeypatch):
    class Model:
        def __init__(self, *a, **k):
            raise ValueError(PROXY_MSG)

    _fake_faster_whisper(monkeypatch, Model)
    monkeypatch.setenv("HTTPS_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.setenv("NO_PROXY", "localhost")
    with pytest.raises(RuntimeError):
        _load_model("large-v3", "cuda", "int8_float16", lambda *_: None)
    assert os.environ["HTTPS_PROXY"] == "socks4://127.0.0.1:10808"
    assert os.environ["NO_PROXY"] == "localhost"
