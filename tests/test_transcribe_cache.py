"""Opt-in (default-OFF) WhisperModel cache — vpipe.transcribe (#90).

Fully mocked: ``_load_model`` and ``_start_download_watch`` are monkeypatched so
no CUDA / CTranslate2 / Hugging Face network is touched. Verifies that the
DEFAULT (cache off) path still constructs the model per call, the opt-in path
reuses a resident model, and ``release_whisper_cache`` evicts it.
"""
import vpipe.transcribe as T
from vpipe.config import TranscribeCfg


class _FakeInfo:
    language = "ru"


class _FakeModel:
    def transcribe(self, *a, **k):
        # Empty segment iterator + an info stub — no audio/GPU needed.
        return iter([]), _FakeInfo()


def _install(monkeypatch):
    """Patch model construction (counted) + silence the download watcher, and
    start from a clean module cache."""
    calls = {"n": 0}

    def fake_load(size, device, ctype, log):
        calls["n"] += 1
        return _FakeModel()

    monkeypatch.setattr(T, "_load_model", fake_load)
    monkeypatch.setattr(T, "_start_download_watch", lambda *a, **k: None)
    T._MODEL_CACHE.clear()
    return calls


def _run(cfg):
    return T._run_once("a.wav", "tiny", "cpu", "int8", cfg,
                       duration=1.0, audio_hash="h", log=lambda *a, **k: None)


def test_no_cache_constructs_each_call(monkeypatch):
    calls = _install(monkeypatch)
    cfg = TranscribeCfg(cache_model=False)
    _run(cfg)
    _run(cfg)
    assert calls["n"] == 2            # one cold build per call (legacy behaviour)
    assert T._MODEL_CACHE == {}       # nothing retained when the flag is off


def test_cache_reuses_when_enabled(monkeypatch):
    calls = _install(monkeypatch)
    cfg = TranscribeCfg(cache_model=True)
    _run(cfg)
    _run(cfg)
    assert calls["n"] == 1                          # 2nd run reused the resident model
    assert list(T._MODEL_CACHE) == [("tiny", "cpu", "int8")]
    T._MODEL_CACHE.clear()                          # don't leak into other tests


def test_release_evicts(monkeypatch):
    calls = _install(monkeypatch)
    cfg = TranscribeCfg(cache_model=True)
    _run(cfg)
    assert T._MODEL_CACHE                            # populated after a cached run
    T.release_whisper_cache()
    assert T._MODEL_CACHE == {}                      # evicted
    _run(cfg)
    assert calls["n"] == 2                           # had to reconstruct after release
