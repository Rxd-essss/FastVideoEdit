"""Редактируемый транскрипт — PUT /api/transcript/word (фича «текст = видео»).

Покрывает бэкенд-часть без ffmpeg/whisper/GPU (FastAPI TestClient + фейковая
сессия, как в test_models_api.py):
  * happy-path: ответ {ok, text, segment_text}, ведущий пробел Whisper
    сохранён, segment.text пересобран, кэш-файл на диске обновлён АТОМАРНО
    (.tmp не остаётся), GET /api/transcript возвращает правку;
  * тайминги/prob слов не меняются (правка текста ≠ правка монтажа);
  * валидация: bad si/wi -> 400 (включая bool/float/None), пустой/длинный/
    нестроковый text -> 400, нет транскрипта/сессии -> 409, занятая задача
    или работающая очередь -> 409;
  * _join_words: обе конвенции (" слово" Whisper и «голое» слово).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                        # noqa: E402
from vpipe.models import Segment, Transcript, Word  # noqa: E402

HASH = "ab" * 20


def make_transcript() -> Transcript:
    return Transcript(
        language="ru", duration=5.0, model="large-v3", audio_hash=HASH,
        segments=[
            Segment(0.0, 2.5, "Привет мир", [
                Word(" Привет", 0.0, 1.0, 0.99),
                Word(" мир", 1.0, 2.5, 0.98)]),
            # Первое слово БЕЗ ведущего пробела — обе конвенции должны жить.
            Segment(2.5, 5.0, "Это тест", [
                Word("Это", 2.5, 3.0, 0.97),
                Word(" тест", 3.0, 5.0, 0.96)]),
        ])


@pytest.fixture()
def sess(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    tr = make_transcript()
    tr.save(cache / f"{HASH}.transcript.json")   # стартовый кэш — как после транскрипции
    # A6: GET /api/transcript теперь отдаёт device_configured из s.cfg — у
    # реальной Session cfg есть всегда, повторяем это в фейке.
    cfg = SimpleNamespace(transcribe=SimpleNamespace(device="cuda"))
    return SimpleNamespace(transcript=tr, cache_dir=cache, audio_hash=HASH,
                           cfg=cfg, task={"running": False})


@pytest.fixture()
def client(sess, monkeypatch):
    monkeypatch.setattr(serve, "SESSION", sess)
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


def put_word(client, **body):
    return client.put("/api/transcript/word", json=body)


# --- happy path ---------------------------------------------------------------
def test_edit_word_happy_path(client, sess):
    r = put_word(client, si=0, wi=1, text="всем")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["text"] == " всем"                  # ведущий пробел Whisper сохранён
    assert j["segment_text"] == "Привет всем"    # segment.text пересобран из слов
    assert sess.transcript.segments[0].words[1].word == " всем"
    assert sess.transcript.segments[0].text == "Привет всем"


def test_edit_word_updates_cache_on_disk(client, sess):
    assert put_word(client, si=0, wi=0, text="Здравствуй").status_code == 200
    p = sess.cache_dir / f"{HASH}.transcript.json"
    data = json.loads(p.read_text(encoding="utf-8"))   # читаем С ДИСКА
    assert data["segments"][0]["words"][0]["w"] == " Здравствуй"
    assert data["segments"][0]["text"] == "Здравствуй мир"
    # атомарность: временный файл не остаётся
    assert not (sess.cache_dir / f"{HASH}.transcript.json.tmp").exists()


def test_edit_word_survives_get_transcript(client):
    put_word(client, si=0, wi=1, text="всем")
    j = client.get("/api/transcript").json()
    assert j["segments"][0]["words"][1]["w"] == " всем"
    assert j["segments"][0]["text"] == "Привет всем"


def test_edit_word_no_leading_space_convention_kept(client):
    r = put_word(client, si=1, wi=0, text="Вот")
    j = r.json()
    assert j["text"] == "Вот"                    # пробела не было — не добавляем
    assert j["segment_text"] == "Вот тест"       # join всё равно ставит пробел между словами


def test_edit_word_strips_user_whitespace(client):
    r = put_word(client, si=0, wi=1, text="  всем  ")
    assert r.status_code == 200
    assert r.json()["text"] == " всем"


def test_edit_word_timings_untouched(client, sess):
    before = [(w.start, w.end, w.prob) for w in sess.transcript.segments[0].words]
    put_word(client, si=0, wi=1, text="всем")
    after = [(w.start, w.end, w.prob) for w in sess.transcript.segments[0].words]
    assert before == after
    assert sess.transcript.segments[0].start == 0.0
    assert sess.transcript.segments[0].end == 2.5


def test_text_exactly_200_ok(client):
    assert put_word(client, si=0, wi=0, text="ы" * 200).status_code == 200


# --- валидация: 400 -----------------------------------------------------------
@pytest.mark.parametrize("si,wi", [(-1, 0), (2, 0), (99, 0), (0, -1), (0, 2), (0, 99)])
def test_bad_indices_400(client, si, wi):
    assert put_word(client, si=si, wi=wi, text="x").status_code == 400


@pytest.mark.parametrize("si,wi", [
    ("0", 0), (None, 0), (0, "1"), (0, None),
    (True, 0), (0, False),       # bool — подкласс int, но валидным индексом не считается
    (1.5, 0), (0, 0.0),          # float из JSON — тоже не индекс
])
def test_non_int_indices_400(client, si, wi):
    assert put_word(client, si=si, wi=wi, text="x").status_code == 400


@pytest.mark.parametrize("text", ["", "   ", None, 42])
def test_bad_text_400(client, text):
    assert put_word(client, si=0, wi=0, text=text).status_code == 400


def test_text_too_long_400(client, sess):
    assert put_word(client, si=0, wi=0, text="ы" * 201).status_code == 400
    assert sess.transcript.segments[0].words[0].word == " Привет"   # не тронуто


def test_missing_fields_400(client):
    assert client.put("/api/transcript/word", json={}).status_code == 400


# --- занятость / отсутствие состояния: 409 ------------------------------------
def test_busy_task_409(client, sess):
    sess.task["running"] = True
    r = put_word(client, si=0, wi=0, text="x")
    assert r.status_code == 409
    assert sess.transcript.segments[0].words[0].word == " Привет"   # не тронуто


def test_queue_running_409(client, monkeypatch):
    monkeypatch.setattr(serve, "_queue_running", True)
    assert put_word(client, si=0, wi=0, text="x").status_code == 409


def test_no_transcript_409(client, sess):
    sess.transcript = None
    assert put_word(client, si=0, wi=0, text="x").status_code == 409


def test_no_session_409(sess, monkeypatch):
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    c = TestClient(serve.app)
    assert c.put("/api/transcript/word",
                 json={"si": 0, "wi": 0, "text": "x"}).status_code == 409


# --- _join_words --------------------------------------------------------------
def test_join_words_whisper_convention():
    words = [Word(" Привет", 0.0, 1.0), Word(" мир", 1.0, 2.0)]
    assert serve._join_words(words) == "Привет мир"


def test_join_words_inserts_space_when_missing():
    words = [Word("Привет", 0.0, 1.0), Word("мир", 1.0, 2.0)]
    assert serve._join_words(words) == "Привет мир"


def test_join_words_empty():
    assert serve._join_words([]) == ""
