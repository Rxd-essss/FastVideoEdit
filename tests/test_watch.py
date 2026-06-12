"""C5 — папка-наблюдатель («кинул в папку — утром готово»).

Pure coverage, no real ffmpeg/whisper and no live watcher threads:
  * scan_once — чистая функция сканера: новый файл (двухфазная стабильность),
    растущий файл, уже в реестре, заменённый, исчезнувший, чужая папка в
    реестре, недоступная папка -> OSError.
  * _watch_tick — enqueue в QUEUE (без воркера), skip файлов, уже идущих через
    UI/очередь, недоступная папка -> статус-ошибка без падения.
  * GET/POST /api/watch — валидация (нет папки / не существует / == out_dir),
    persist в watch.json + roundtrip через _load_watch, перезапуск потока
    замокан (_watch_apply).
  * _watch_apply — реальный старт/стоп daemon-потока по Event.
"""
import json
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe.config import load_config


# --- fixtures ---------------------------------------------------------------
@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """APP с изолированным cache/out + чистые QUEUE и watch-состояние."""
    cfg = load_config("config.yaml")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg.paths.cache_dir = str(cache_dir)
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", str(out_dir))
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "QUEUE", [])
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "WATCH",
                        {"enabled": False, "folder": None,
                         "render_opts_preset": "current"})
    monkeypatch.setattr(serve, "WATCH_PROCESSED", {})
    monkeypatch.setattr(serve, "WATCH_STATUS", {"error": None, "last_scan": None})
    monkeypatch.setattr(serve, "_watch_thread", None)
    return cache_dir, out_dir


@pytest.fixture()
def client(wired, monkeypatch):
    """TestClient с замоканным перезапуском потока (эндпоинты — без тредов)."""
    calls = []
    monkeypatch.setattr(serve, "_watch_apply", lambda: calls.append(1))
    c = TestClient(serve.app)
    c.apply_calls = calls
    return c


@pytest.fixture()
def in_dir(tmp_path):
    d = tmp_path / "inbox"
    d.mkdir()
    return d


def _mk(path: Path, size: int = 100, mtime: float = 1000.0) -> Path:
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def _key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


# --- scan_once: чистая логика сканера ----------------------------------------
def test_scan_new_file_two_phase(in_dir):
    """Новый файл подтверждается только ВТОРЫМ сканом с тем же size/mtime."""
    f = _mk(in_dir / "a.mp4")
    registry, pending = {}, {}
    assert serve.scan_once(in_dir, registry, pending) == []      # скан 1: кандидат
    assert pending == {_key(f): (100, 1000.0)}
    assert registry == {}

    got = serve.scan_once(in_dir, registry, pending)             # скан 2: стабилен
    assert [p.name for p in got] == ["a.mp4"]
    assert registry == {_key(f): {"size": 100, "mtime": 1000.0}}
    assert pending == {}


def test_scan_growing_file_waits(in_dir):
    """Файл ещё копируется (размер растёт) — не отдаётся, пока не стабилен."""
    f = _mk(in_dir / "big.mp4", size=10, mtime=1000.0)
    registry, pending = {}, {}
    assert serve.scan_once(in_dir, registry, pending) == []
    _mk(in_dir / "big.mp4", size=20, mtime=1001.0)               # вырос
    assert serve.scan_once(in_dir, registry, pending) == []      # снова ждать
    assert pending == {_key(f): (20, 1001.0)}
    got = serve.scan_once(in_dir, registry, pending)             # стабилен
    assert [p.name for p in got] == ["big.mp4"]


def test_scan_skips_registry_match(in_dir):
    """Файл из реестра с теми же size/mtime не переобрабатывается."""
    f = _mk(in_dir / "done.mp4")
    registry = {_key(f): {"size": 100, "mtime": 1000.0}}
    pending = {}
    assert serve.scan_once(in_dir, registry, pending) == []
    assert serve.scan_once(in_dir, registry, pending) == []
    assert pending == {}


def test_scan_replaced_file_reprocessed(in_dir):
    """Файл с тем же именем, но другим size/mtime — заново, через двухфазность."""
    f = _mk(in_dir / "v.mp4", size=50, mtime=900.0)
    registry = {_key(f): {"size": 10, "mtime": 100.0}}           # старая версия
    pending = {}
    assert serve.scan_once(in_dir, registry, pending) == []      # фаза 1
    got = serve.scan_once(in_dir, registry, pending)             # фаза 2
    assert [p.name for p in got] == ["v.mp4"]
    assert registry[_key(f)] == {"size": 50, "mtime": 900.0}


def test_scan_vanished_file_purged(in_dir):
    """Удалённый из папки файл вычищается из реестра и кандидатов."""
    f = _mk(in_dir / "gone.mp4")
    registry = {_key(f): {"size": 100, "mtime": 1000.0}}
    pending = {_key(f): (100, 1000.0)}
    f.unlink()
    assert serve.scan_once(in_dir, registry, pending) == []
    assert registry == {}
    assert pending == {}


def test_scan_keeps_other_folder_entries(in_dir, tmp_path):
    """Записи реестра из ДРУГОЙ папки не трогаются (смена папки наблюдения
    не убивает историю прежней)."""
    other = os.path.normcase(str(tmp_path / "elsewhere" / "old.mp4"))
    registry = {other: {"size": 5, "mtime": 1.0}}
    serve.scan_once(in_dir, registry, {})
    assert other in registry


def test_scan_ignores_non_video_and_dotfiles(in_dir):
    _mk(in_dir / "notes.txt")
    _mk(in_dir / "clip.mp4.part")          # расширение .part — не видео
    _mk(in_dir / ".hidden.mp4")            # dot-файл, как в /api/browse
    (in_dir / "sub.mp4").mkdir()           # папка с видео-именем
    registry, pending = {}, {}
    assert serve.scan_once(in_dir, registry, pending) == []
    assert serve.scan_once(in_dir, registry, pending) == []
    assert pending == {} and registry == {}


def test_scan_missing_folder_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        serve.scan_once(tmp_path / "no_such_dir", {}, {})


# --- _watch_tick: скан -> enqueue -> persist ----------------------------------
def test_watch_tick_enqueues_new_file(wired, in_dir):
    cache_dir, out_dir = wired
    f = _mk(in_dir / "night.mp4")
    pending = {_key(f): (100, 1000.0)}     # «прошлый скан» уже видел файл
    assert serve._watch_tick(str(in_dir), pending) == 1
    assert len(serve.QUEUE) == 1
    job = serve.QUEUE[0]
    assert Path(job.path) == f.resolve()
    assert job.out_dir == str(out_dir)
    assert job.render_opts == {}           # дефолтные опции рендера
    assert job.status == "pending"
    # реестр обновлён и персистнут вместе с очередью
    assert _key(f) in serve.WATCH_PROCESSED
    assert (cache_dir / "watch.json").exists()
    assert (cache_dir / "queue.json").exists()
    assert serve.WATCH_STATUS["error"] is None


def test_watch_tick_skips_files_already_in_queue(wired, in_dir):
    """Файл, уже идущий через UI/очередь, не дублируется — но в реестр попадает
    (наблюдатель не подхватит его и позже)."""
    _, out_dir = wired
    f = _mk(in_dir / "manual.mp4")
    serve.QUEUE.append(serve.QueueJob(id="ui1", path=str(f.resolve()),
                                      out_dir=str(out_dir)))
    pending = {_key(f): (100, 1000.0)}
    assert serve._watch_tick(str(in_dir), pending) == 0
    assert len(serve.QUEUE) == 1           # дубля нет
    assert _key(f) in serve.WATCH_PROCESSED


def test_watch_tick_unavailable_folder_sets_error(wired, tmp_path):
    """Папка недоступна (сетевой диск) — статус-ошибка, без падения и enqueue."""
    assert serve._watch_tick(str(tmp_path / "unplugged"), {}) == 0
    assert serve.QUEUE == []
    assert "недоступна" in (serve.WATCH_STATUS["error"] or "")
    # папка вернулась — ошибка снимается
    ok = tmp_path / "unplugged"
    ok.mkdir()
    serve._watch_tick(str(ok), {})
    assert serve.WATCH_STATUS["error"] is None


# --- эндпоинты GET/POST /api/watch --------------------------------------------
def test_watch_get_defaults(client):
    r = client.get("/api/watch")
    assert r.status_code == 200
    j = r.json()
    assert j["enabled"] is False
    assert j["folder"] is None
    assert j["render_opts_preset"] == "current"
    assert j["processed"] == 0
    assert j["error"] is None


def test_watch_post_enable_valid_folder(client, wired, in_dir):
    cache_dir, _ = wired
    r = client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    assert r.status_code == 200
    j = r.json()
    assert j["enabled"] is True
    assert Path(j["folder"]) == in_dir.resolve()
    assert client.apply_calls          # поток перезапущен (замокан)
    data = json.loads((cache_dir / "watch.json").read_text(encoding="utf-8"))
    assert data["enabled"] is True
    assert Path(data["folder"]) == in_dir.resolve()
    assert data["render_opts_preset"] == "current"


def test_watch_post_enable_requires_folder(client):
    r = client.post("/api/watch", json={"enabled": True})
    assert r.status_code == 400
    r = client.post("/api/watch", json={"enabled": True, "folder": "  "})
    assert r.status_code == 400


def test_watch_post_missing_folder_404(client, tmp_path):
    r = client.post("/api/watch", json={"enabled": True,
                                        "folder": str(tmp_path / "nope")})
    assert r.status_code == 404


def test_watch_post_file_as_folder_404(client, in_dir):
    f = _mk(in_dir / "file.mp4")
    r = client.post("/api/watch", json={"enabled": True, "folder": str(f)})
    assert r.status_code == 404


def test_watch_post_folder_equals_out_dir_400(client, wired):
    """folder == out_dir -> рекурсия рендеров, жёсткий 400."""
    _, out_dir = wired
    r = client.post("/api/watch", json={"enabled": True, "folder": str(out_dir)})
    assert r.status_code == 400
    assert serve.WATCH["enabled"] is False


def test_watch_post_disable_keeps_folder(client, wired, in_dir):
    client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    r = client.post("/api/watch", json={"enabled": False, "folder": str(in_dir)})
    assert r.status_code == 200
    j = r.json()
    assert j["enabled"] is False
    assert j["folder"] == str(in_dir)      # путь сохранён для удобства UI
    # выключенному наблюдателю папку не валидируем (можно сохранить будущую)
    r2 = client.post("/api/watch", json={"enabled": False, "folder": "X:\\later"})
    assert r2.status_code == 200


# --- C1: включение на непустой папке НЕ обрабатывает бэклог --------------------
def test_watch_post_enable_seeds_existing_files(client, wired, in_dir):
    """Включение сидирует реестр текущим содержимым папки: старые файлы
    считаются обработанными, в очередь встают только появившиеся ПОСЛЕ."""
    old1 = _mk(in_dir / "archive1.mp4", size=11, mtime=100.0)
    old2 = _mk(in_dir / "archive2.mkv", size=22, mtime=200.0)
    _mk(in_dir / "notes.txt")              # не видео — не сидируется
    _mk(in_dir / ".hidden.mp4")            # dot-файл — не сидируется
    r = client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    assert r.status_code == 200
    assert r.json()["seeded"] == 2
    assert serve.WATCH_PROCESSED[_key(old1)] == {"size": 11, "mtime": 100.0}
    assert serve.WATCH_PROCESSED[_key(old2)] == {"size": 22, "mtime": 200.0}
    # Воркер с таким реестром не ставит бэклог в очередь — сколько ни сканируй.
    pending = {}
    assert serve._watch_tick(str(in_dir), pending) == 0
    assert serve._watch_tick(str(in_dir), pending) == 0
    assert serve.QUEUE == []
    # …а файл, появившийся ПОСЛЕ включения, проходит как обычно (двухфазно).
    _mk(in_dir / "fresh.mp4")
    assert serve._watch_tick(str(in_dir), pending) == 0   # фаза 1: кандидат
    assert serve._watch_tick(str(in_dir), pending) == 1   # фаза 2 -> очередь
    assert [Path(j.path).name for j in serve.QUEUE] == ["fresh.mp4"]


def test_watch_enable_seed_persisted(client, wired, in_dir):
    """Сидированный реестр сразу уезжает в watch.json — рестарт сервера не
    пережуёт бэклог."""
    cache_dir, _ = wired
    f = _mk(in_dir / "old.mp4")
    client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    data = json.loads((cache_dir / "watch.json").read_text(encoding="utf-8"))
    assert _key(f) in data["processed"]


def test_watch_repost_same_folder_does_not_reseed(client, wired, in_dir):
    """Повторный POST с той же папкой при УЖЕ включённом наблюдении не сидирует:
    файл, ждущий двухфазного подтверждения сканера, не «проглатывается»."""
    client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    f = _mk(in_dir / "inflight.mp4")
    r = client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    assert r.status_code == 200
    assert r.json()["seeded"] == 0
    assert _key(f) not in serve.WATCH_PROCESSED   # сканер подхватит его сам


def test_watch_reenable_seeds_again(client, wired, in_dir):
    """Выкл -> файл упал в папку -> вкл: на момент включения файл УЖЕ лежит,
    значит сидируется как обработанный («только новые» — честно от момента
    включения)."""
    client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    client.post("/api/watch", json={"enabled": False, "folder": str(in_dir)})
    f = _mk(in_dir / "while_off.mp4")
    r = client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    assert r.json()["seeded"] == 1
    assert _key(f) in serve.WATCH_PROCESSED


def test_watch_folder_change_seeds_new_folder(client, wired, in_dir, tmp_path):
    """Смена папки под включённым наблюдением сидирует НОВУЮ папку."""
    other = tmp_path / "inbox2"
    other.mkdir()
    g = _mk(other / "preexisting.mp4")
    client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    r = client.post("/api/watch", json={"enabled": True, "folder": str(other)})
    assert r.json()["seeded"] == 1
    assert _key(g) in serve.WATCH_PROCESSED


def test_watch_enable_seed_failure_400(client, wired, in_dir, monkeypatch):
    """Папку не прочитать при сидировании -> 400 и наблюдение НЕ включается:
    иначе несидированный реестр молча отправил бы бэклог в очередь."""
    def boom(folder):
        raise OSError("unreadable")
    monkeypatch.setattr(serve, "_watch_list_videos", boom)
    r = client.post("/api/watch", json={"enabled": True, "folder": str(in_dir)})
    assert r.status_code == 400
    assert serve.WATCH["enabled"] is False
    assert serve.WATCH_PROCESSED == {}


# --- persist roundtrip ---------------------------------------------------------
def test_watch_persist_roundtrip(wired, in_dir):
    """_save_watch -> _load_watch восстанавливает состояние и реестр."""
    serve.WATCH.update(enabled=True, folder=str(in_dir))
    serve.WATCH_PROCESSED["k1"] = {"size": 10, "mtime": 5.0}
    serve._save_watch()

    serve.WATCH.update(enabled=False, folder=None)
    serve.WATCH_PROCESSED.clear()
    serve._load_watch()
    assert serve.WATCH["enabled"] is True
    assert serve.WATCH["folder"] == str(in_dir)
    assert serve.WATCH_PROCESSED == {"k1": {"size": 10, "mtime": 5.0}}


def test_load_watch_corrupt_and_garbage(wired):
    cache_dir, _ = wired
    (cache_dir / "watch.json").write_text("{broken", encoding="utf-8")
    serve._load_watch()                    # не падает, состояние не тронуто
    assert serve.WATCH["enabled"] is False

    (cache_dir / "watch.json").write_text(json.dumps({
        "enabled": True, "folder": "D:/inbox",
        "processed": {"good": {"size": 1, "mtime": 2.0},
                      "bad1": {"size": "x", "mtime": 2.0},
                      "bad2": "not-a-dict",
                      "bad3": {"size": True, "mtime": 2.0}},
    }), encoding="utf-8")
    serve._load_watch()
    assert serve.WATCH["enabled"] is True
    assert serve.WATCH["folder"] == "D:/inbox"
    assert serve.WATCH_PROCESSED == {"good": {"size": 1, "mtime": 2.0}}


def test_load_watch_enabled_without_folder_disabled(wired):
    cache_dir, _ = wired
    (cache_dir / "watch.json").write_text(
        json.dumps({"enabled": True, "folder": None}), encoding="utf-8")
    serve._load_watch()
    assert serve.WATCH["enabled"] is False


# --- _watch_apply: реальный поток стартует и останавливается -------------------
def test_watch_apply_starts_and_stops_thread(wired, in_dir, monkeypatch):
    monkeypatch.setattr(serve, "_watch_stop", threading.Event())
    serve.WATCH.update(enabled=True, folder=str(in_dir))
    serve._watch_apply()
    t = serve._watch_thread
    try:
        assert t is not None and t.is_alive()
        assert t.daemon
        # выключение через _watch_apply сигналит ЛИЧНЫЙ Event потока
        serve.WATCH["enabled"] = False
        serve._watch_apply()
        t.join(timeout=5)
        assert not t.is_alive()
        assert serve._watch_thread is None
    finally:
        serve._watch_stop.set()
        if t is not None:
            t.join(timeout=5)


def test_watch_worker_exits_when_disabled(wired):
    """Воркер с выключенным наблюдением выходит сразу (без 15с-ожиданий)."""
    start = time.monotonic()
    serve._watch_worker(threading.Event())
    assert time.monotonic() - start < 2.0
