"""P2-#4 — zero-upload guard + offline mode.

Covers the parts that don't open real network sockets:
  * netguard._classify — loopback / localhost / AF_UNIX -> local; hostnames /
    public IPs -> external; IPv6 ::1 -> local.
  * the hooked connect / connect_ex counters, with the ORIGINAL connect stubbed
    out so no real TCP is attempted: local always passes, external is allowed
    online and BLOCKED (raise / ENETUNREACH) offline.
  * set_offline / is_offline toggling the HF offline env vars.
  * install() idempotency.
  * serve.py REST surface: GET /api/network, POST /api/network/offline (+ atomic
    privacy.json persistence + corrupt-file tolerance), and the /api/state
    network bootstrap field — all via FastAPI TestClient.

TestClient uses an in-process ASGI transport (no real sockets), so the installed
guard never interferes with the test HTTP calls themselves.
"""
import errno
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe import netguard                       # noqa: E402
from vpipe.config import load_config             # noqa: E402
import serve                                     # noqa: E402


# --- fixtures ----------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_guard():
    """Reset counters + force online before each test; restore env after."""
    netguard.install()                           # idempotent — class hook in place
    netguard.set_offline(False)
    netguard.reset()
    saved_env = {k: os.environ.get(k) for k in
                 ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")}
    yield
    netguard.set_offline(False)
    netguard.reset()
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture()
def stub_orig(monkeypatch):
    """Replace the saved ORIGINAL connect/connect_ex with recorders so the hooks
    never touch a real socket. Returns the list of addresses that 'connected'."""
    calls = []

    def fake_connect(self, address):
        calls.append(address)
        return None

    def fake_connect_ex(self, address):
        calls.append(address)
        return 0

    monkeypatch.setattr(netguard, "_orig_connect", fake_connect)
    monkeypatch.setattr(netguard, "_orig_connect_ex", fake_connect_ex)
    return calls


@pytest.fixture()
def client(tmp_path, monkeypatch):
    cfg = load_config("config.yaml")
    cfg.paths.cache_dir = str(tmp_path / "cache")
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", str(tmp_path / "out"))
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "SESSION", None)
    return TestClient(serve.app)


# --- _classify ---------------------------------------------------------------
@pytest.mark.parametrize("addr,expected", [
    (("127.0.0.1", 11434), "local"),       # Ollama default
    (("localhost", 8000), "local"),
    (("127.5.6.7", 80), "local"),          # 127.0.0.0/8
    (("::1", 443), "local"),               # IPv6 loopback
    (("", 0), "local"),
    ("/tmp/some.sock", "local"),           # AF_UNIX path (str)
    (("huggingface.co", 443), "external"),
    (("8.8.8.8", 53), "external"),
    (("api.openai.com", 443), "external"),
    (("2001:4860:4860::8888", 443), "external"),
])
def test_classify(addr, expected):
    assert netguard._classify(addr) == expected


# --- counters: local always passes ------------------------------------------
def test_local_connect_counts_and_passes(stub_orig):
    sock = object()
    netguard._hooked_connect(sock, ("127.0.0.1", 11434))
    netguard._hooked_connect(sock, ("localhost", 8000))
    s = netguard.stats()
    assert s["local"] == 2
    assert s["external_allowed"] == 0
    assert s["blocked"] == 0
    assert stub_orig == [("127.0.0.1", 11434), ("localhost", 8000)]


# --- external online: allowed + counted -------------------------------------
def test_external_online_allowed(stub_orig):
    sock = object()
    netguard._hooked_connect(sock, ("huggingface.co", 443))
    s = netguard.stats()
    assert s["external_allowed"] == 1
    assert s["blocked"] == 0
    assert s["external_hosts"] == {"huggingface.co": 1}
    assert stub_orig == [("huggingface.co", 443)]


# --- external offline: blocked (raise) --------------------------------------
def test_external_offline_blocks_connect(stub_orig):
    netguard.set_offline(True)
    sock = object()
    with pytest.raises(OSError) as ei:
        netguard._hooked_connect(sock, ("huggingface.co", 443))
    assert "оффлайн" in str(ei.value).lower()
    assert "huggingface.co" in str(ei.value)
    s = netguard.stats()
    assert s["blocked"] == 1
    assert s["external_allowed"] == 0
    assert s["blocked_hosts"] == {"huggingface.co": 1}
    assert stub_orig == []                  # original connect NEVER called


# --- offline never blocks loopback (Ollama keeps working) -------------------
def test_offline_allows_local(stub_orig):
    netguard.set_offline(True)
    sock = object()
    netguard._hooked_connect(sock, ("127.0.0.1", 11434))   # Ollama
    s = netguard.stats()
    assert s["local"] == 1
    assert s["blocked"] == 0
    assert stub_orig == [("127.0.0.1", 11434)]


# --- connect_ex variant ------------------------------------------------------
def test_connect_ex_offline_returns_errno(stub_orig):
    netguard.set_offline(True)
    sock = object()
    rc = netguard._hooked_connect_ex(sock, ("huggingface.co", 443))
    assert rc == errno.ENETUNREACH
    assert rc != 0
    assert netguard.stats()["blocked"] == 1
    assert stub_orig == []


def test_connect_ex_online_passes(stub_orig):
    sock = object()
    rc = netguard._hooked_connect_ex(sock, ("huggingface.co", 443))
    assert rc == 0
    assert netguard.stats()["external_allowed"] == 1
    assert stub_orig == [("huggingface.co", 443)]


def test_connect_ex_local_passes(stub_orig):
    sock = object()
    rc = netguard._hooked_connect_ex(sock, ("127.0.0.1", 11434))
    assert rc == 0
    assert netguard.stats()["local"] == 1


# --- set_offline + env vars --------------------------------------------------
def test_set_offline_env_vars():
    netguard.set_offline(True)
    assert netguard.is_offline() is True
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
    netguard.set_offline(False)
    assert netguard.is_offline() is False
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


# --- install idempotency -----------------------------------------------------
def test_install_idempotent():
    import socket
    netguard.install()
    hooked = socket.socket.connect
    netguard.install()                 # second call must NOT re-wrap
    assert socket.socket.connect is hooked
    assert socket.socket.connect is netguard._hooked_connect


def test_reset_clears_counters(stub_orig):
    sock = object()
    netguard._hooked_connect(sock, ("huggingface.co", 443))
    assert netguard.stats()["external_allowed"] == 1
    netguard.reset()
    s = netguard.stats()
    assert s["external_allowed"] == 0
    assert s["external_hosts"] == {}


def test_stats_is_a_copy():
    s1 = netguard.stats()
    s1["local"] = 999
    s1["external_hosts"]["x"] = 5
    s2 = netguard.stats()
    assert s2["local"] == 0
    assert s2["external_hosts"] == {}


# --- serve.py REST surface ---------------------------------------------------
def test_api_network_default_online(client):
    r = client.get("/api/network")
    assert r.status_code == 200
    j = r.json()
    assert j["offline"] is False
    assert "stats" in j and "external_allowed" in j["stats"]
    assert isinstance(j["summary"], str) and j["summary"]


def test_api_offline_toggle_and_persist(client, tmp_path):
    r = client.post("/api/network/offline", json={"enabled": True})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["offline"] is True
    assert netguard.is_offline() is True
    # persisted to cache_dir/privacy.json
    pf = Path(serve.APP["cfg"].paths.cache_dir) / "privacy.json"
    assert pf.exists()
    assert json.loads(pf.read_text(encoding="utf-8"))["offline"] is True
    # and back off
    r2 = client.post("/api/network/offline", json={"enabled": False})
    assert r2.json()["offline"] is False
    assert json.loads(pf.read_text(encoding="utf-8"))["offline"] is False


def test_read_privacy_offline_corrupt_defaults_false(client):
    pf = Path(serve.APP["cfg"].paths.cache_dir)
    pf.mkdir(parents=True, exist_ok=True)
    (pf / "privacy.json").write_text("{ this is not json", encoding="utf-8")
    assert serve._read_privacy_offline() is False


def test_read_privacy_offline_roundtrip(client):
    serve._write_privacy_offline(True)
    assert serve._read_privacy_offline() is True
    serve._write_privacy_offline(False)
    assert serve._read_privacy_offline() is False


def test_api_state_has_network_field(client):
    # no session -> still exposes the network bootstrap for the badge
    r = client.get("/api/state")
    assert r.status_code == 200
    j = r.json()
    assert "network" in j
    assert set(j["network"]) >= {"offline", "external_allowed", "blocked"}


def test_network_summary_phrasing():
    online0 = serve._network_summary(False, {"external_allowed": 0, "blocked": 0,
                                             "external_hosts": {}})
    assert "локально" in online0.lower()
    onlineN = serve._network_summary(False, {"external_allowed": 1, "blocked": 0,
                                             "external_hosts": {"huggingface.co": 1}})
    assert "huggingface.co" in onlineN
    off = serve._network_summary(True, {"external_allowed": 0, "blocked": 2,
                                        "external_hosts": {}})
    assert "оффлайн" in off.lower()
    assert "2" in off
