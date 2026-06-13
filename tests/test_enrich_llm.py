# -*- coding: utf-8 -*-
"""P3 — LLM-детекторы авто-обогащения (vpipe/enrich_llm.py, ENRICH_PLAN §3,
§7-P3): detect_all() со строго замоканной LLM (MockLLM-паттерн test_clips).

Покрывает список §7-P3 полностью:
 1. перечисления: quote-снап точный и фаззи (SequenceMatcher>=0.75 ±30 слов),
    дроп пункта при промахе (<2 выживших — дроп списка), анти-дробление
    (25 слов от конца предыдущего / 60 c от первого — отрез; вложенный — дроп),
    нормализация text_short/title (<=60, капитализация, без точки), <=6
    пунктов, дедуп окон по IoU>=0.5, score эвристикой кода, ОРИГИНАЛЬНЫЕ
    word-индексы/секунды в плане;
 2. CTA: все поля схем в required (R4: optional молча выкидывается), один
    вызов на весь effective-текст с маркерами [N|м:сс], снап к границе
    сегмента, дроп t<60 c и хвоста, мусорный type/comment без вопроса — дроп,
    лимит 2 на 10 мин с приоритетом «1 subscribe + 1 comment», дедуп <120 c,
    EFFECTIVE-текст с вырезанным ретейком (тест из плана: дубль не уходит в
    промпт, filtered word_idx мапится в ОРИГИНАЛЬНЫЙ);
 3. иллюстрации: клампы индексов к окну, снап старта к началу whisper-сегмента,
    кламп длительности 2.5–4.0, style-фолбэк photo, пустой/русский query → без
    авто-ассета, окно-лимит 4 точки;
 4. one-bad-window не валит детектор, сбойный детектор не валит пасс;
 5. выключенный в params тип НЕ зовёт LLM (счётчик вызовов), llm_off/пустой
    транскрипт → [];
 6. keep_alive=300 между вызовами и 0 на ПОСЛЕДНЕМ вызове всего пасса;
    прогресс-веса детекторов (lists 45 / cta 15 / illustrations 30 / assets 10).
Без реальной LLM и Ollama: мок — голый объект с chat_json().
"""
from __future__ import annotations

import pytest

from vpipe import enrich_llm
from vpipe.enrich import (ENR_CTA_COMMENT, ENR_CTA_SUBSCRIBE, ENR_IMAGE,
                          ENR_LIST_CARD, EnrichItem, default_params)
from vpipe.enrich_llm import detect_all
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE, CutList, CutSegment,
                          Segment, Transcript, Word)

_SILENT = lambda *a, **k: None  # noqa: E731

WD = 0.5          # шаг слов, сек (start = i*WD, end = +0.4)
SEG_WORDS = 10    # слов в whisper-сегменте


class MockLLM:
    """Строгий мок (паттерн test_clips): голый объект с chat_json()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat_json(self, system, user, schema, keep_alive=None):
        self.calls.append({"system": system, "user": user, "schema": schema,
                           "keep_alive": keep_alive})
        if not self._responses:
            return {}
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _tok(i: int) -> str:
    return f"ток{i:03d}"


def _build_tr(n_words: int, *, gaps: dict | None = None,
              rename: dict | None = None) -> Transcript:
    """Транскрипт из n_words уникальных слов, сегменты по SEG_WORDS.

    ``gaps[i]`` — добавить паузу (сек) ПЕРЕД словом i (для теста 60 c);
    ``rename[i]`` — подменить текст слова (для теста ретейка).
    """
    gaps = gaps or {}
    rename = rename or {}
    segs: list[Segment] = []
    cur = 0.0
    words_all: list[Word] = []
    for i in range(n_words):
        cur += gaps.get(i, 0.0)
        words_all.append(Word(rename.get(i, _tok(i)),
                              round(cur, 3), round(cur + 0.4, 3)))
        cur += WD
    for s0 in range(0, n_words, SEG_WORDS):
        ws = words_all[s0:s0 + SEG_WORDS]
        segs.append(Segment(ws[0].start, ws[-1].end,
                            " ".join(w.word for w in ws), ws))
    return Transcript(language="ru", duration=round(cur, 3), model="t",
                      audio_hash="h", segments=segs)


def _cl(tr: Transcript, cuts=()) -> CutList:
    return CutList(source="x.mp4", duration=tr.duration, segments=list(cuts))


def _params(**types) -> dict:
    p = default_params()
    p["types"].update(types)
    return p


def _only(kind: str) -> dict:
    base = {"image": False, "animation": False, "list_card": False,
            "cta": False}
    base[kind] = True
    return _params(**base)


def _q(i: int, k: int = 3) -> str:
    """Дословная цитата из k слов начиная со слова i."""
    return " ".join(_tok(j) for j in range(i, i + k))


def _lists_resp(items, intro=50, title="Плюсы реестра."):
    return {"lists": [{"intro_quote": _q(intro), "title_short": title,
                       "items": items}]}


def _item(i: int, text: str = "пункт") -> dict:
    return {"text_short": text, "quote": _q(i)}


# === 1. перечисления ==============================================================
def test_list_exact_snap_original_coords_and_normalization():
    tr = _build_tr(300)
    llm = MockLLM([_lists_resp([_item(60, "первый пункт."),
                                _item(70, "второй пункт")])])
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert len(llm.calls) == 1                      # 300 слов = одно окно 400
    assert len(out) == 1
    it = out[0]
    assert isinstance(it, EnrichItem) and it.type == ENR_LIST_CARD
    assert it.id.startswith("enr_")
    # ОРИГИНАЛЬНЫЕ word-индексы и секунды (§1.2)
    assert it.word_start == 50
    assert it.t_start == pytest.approx(25.0)        # слово 50 @ 25.0 c
    assert it.quote == _q(50)
    pl = it.payload
    assert [ci.word_idx for ci in pl.items] == [60, 70]
    assert [ci.t_word for ci in pl.items] == [pytest.approx(30.0),
                                              pytest.approx(35.0)]
    # нормализация КОДОМ: капитализация, без точки; title без точки
    assert [ci.text for ci in pl.items] == ["Первый пункт", "Второй пункт"]
    assert pl.title == "Плюсы реестра"
    assert pl.mode == "scrim"
    # score — эвристика кода: 40*2/2 + 8*2 + 12 (intro найден) = 68
    assert it.score == 68
    assert "2 пунктов" in it.reason


def test_list_fuzzy_snap_recovers_distorted_quote():
    tr = _build_tr(300)
    # модель исказила первое слово цитаты (R4: «перенняется») — точный снап
    # промахивается, фаззи (>=0.75, ±30 слов от якоря-intro) находит
    bad_quote = f"{_tok(60)}х {_tok(61)} {_tok(62)}"
    llm = MockLLM([_lists_resp([{"text_short": "первый", "quote": bad_quote},
                                _item(70, "второй")])])
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert len(out) == 1
    assert [ci.word_idx for ci in out[0].payload.items] == [60, 70]


def test_list_item_snap_miss_dropped_and_list_below_two_dropped():
    tr = _build_tr(300)
    # один пункт со снап-промахом → дроп пункта; выжил 1 < 2 → дроп списка
    llm = MockLLM([_lists_resp([_item(60, "первый"),
                                {"text_short": "мимо",
                                 "quote": "жираф закат вулкан"}])])
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert out == []


def test_list_antifragment_25_words_cuts_tail():
    tr = _build_tr(300)
    # пункт 3 в 100-(70+3)=27 словах от конца предыдущего (>25) — отрез хвоста
    llm = MockLLM([_lists_resp([_item(60, "а"), _item(70, "б"),
                                _item(100, "в"), _item(110, "г")])])
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert len(out) == 1
    assert [ci.word_idx for ci in out[0].payload.items] == [60, 70]


def test_list_antifragment_60s_cuts_tail():
    # слова идут подряд по индексам, но перед словом 75 пауза 65 c → пункт 75
    # дальше 60 c от первого пункта при разрыве всего в 2 слова (<25)
    tr = _build_tr(300, gaps={75: 65.0})
    llm = MockLLM([_lists_resp([_item(60, "а"), _item(70, "б"),
                                _item(75, "в")], intro=50)])
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert len(out) == 1
    assert [ci.word_idx for ci in out[0].payload.items] == [60, 70]


def test_list_nested_item_dropped_not_cut():
    tr = _build_tr(300)
    # пункт «внутри» предыдущего (61 < 60+3) — дроп самого пункта, хвост живёт
    llm = MockLLM([_lists_resp([_item(60, "а"), _item(61, "вложенный"),
                                _item(70, "б")])])
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert len(out) == 1
    assert [ci.word_idx for ci in out[0].payload.items] == [60, 70]


def test_list_caps_six_items_and_text_limit():
    tr = _build_tr(300)
    long_text = "очень длинный пункт " * 6                  # > 60 символов
    items = [_item(50 + 5 * j, long_text) for j in range(8)]
    llm = MockLLM([_lists_resp(items, intro=40)])
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert len(out) == 1
    pl = out[0].payload
    assert len(pl.items) == 6                               # <=6 ЖЁСТКО
    assert all(len(ci.text) <= 60 for ci in pl.items)
    assert all(not ci.text.endswith(".") for ci in pl.items)


def test_list_window_dedup_iou():
    tr = _build_tr(440)                  # 2 окна: [0,400) и [360,440)
    resp = _lists_resp([_item(370, "а"), _item(380, "б")], intro=365)
    llm = MockLLM([resp, resp])          # один и тот же список в обоих окнах
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert len(llm.calls) == 2
    assert len(out) == 1                 # IoU=1.0 >= 0.5 → слит
    assert [ci.word_idx for ci in out[0].payload.items] == [370, 380]


def test_lists_prompt_plain_text_no_markers_and_schema_required():
    tr = _build_tr(120)
    llm = MockLLM([{"lists": []}])
    detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    call = llm.calls[0]
    assert call["system"] == enrich_llm._LISTS_SYSTEM
    assert call["schema"] is enrich_llm._LISTS_SCHEMA
    assert "[0|" not in call["user"]                 # §3.1: текст БЕЗ маркеров
    assert _tok(0) in call["user"] and _tok(119) in call["user"]
    # проверенные формулировки v2 (не «улучшать»): текстовый образец смыслового
    # списка внутри инструкции; JSON few-shot отсутствует
    assert "он централизованный… он быстрый… он типизированный" \
        in enrich_llm._LISTS_SYSTEM
    assert "title_short" in enrich_llm._LISTS_SYSTEM
    item = enrich_llm._LISTS_SCHEMA["properties"]["lists"]["items"]
    assert item["required"] == ["intro_quote", "title_short", "items"]
    assert item["properties"]["items"]["items"]["required"] == \
        ["text_short", "quote"]


# === 2. CTA ======================================================================
def _cta(typ: str, wi: int, q: str = "", reason: str = "r") -> dict:
    return {"type": typ, "word_idx": wi, "comment_question": q,
            "reason": reason}


def test_cta_schema_all_required_and_prompt_invariants():
    item = enrich_llm._CTA_SCHEMA["properties"]["ctas"]["items"]
    # ВСЕ поля в required — R4: optional-поля модель молча выкидывает
    assert item["required"] == ["type", "word_idx", "comment_question",
                                "reason"]
    assert "ЗАПРЕЩЕНО: первые 60 секунд" in enrich_llm._CTA_SYSTEM
    assert "пустая строка" in enrich_llm._CTA_SYSTEM
    assert "Не больше 5 элементов" in enrich_llm._CTA_SYSTEM


def test_cta_single_call_snap_to_segment_end_and_payloads():
    tr = _build_tr(600)                                  # 300 c
    llm = MockLLM([{"ctas": [
        _cta("subscribe", 150),                          # t≈79.9
        _cta("comment", 450, q="Какой дистрибутив выбрал?")]}])  # t≈229.9
    out = detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    assert len(llm.calls) == 1                           # ОДИН вызов на весь текст
    assert [it.type for it in out] == [ENR_CTA_SUBSCRIBE, ENR_CTA_COMMENT]
    sub, com = out
    # снап к концу предложения: слово 150 → сегмент 15, его последнее слово 159
    assert sub.word_start == sub.word_end == 159
    assert sub.t_start == pytest.approx(159 * WD + 0.4)  # конец слова 159
    assert sub.t_end == pytest.approx(sub.t_start + 4.0)  # duration_s sub_like
    assert sub.payload.variant == "sub_like"
    assert sub.quote == tr.segments[15].text
    assert com.payload.question == "Какой дистрибутив выбрал?"
    assert com.word_start == 459
    assert sub.score == 60 and com.score == 70           # эвристика кода


def test_cta_drop_before_60s_and_tail_and_junk():
    tr = _build_tr(300)                                  # 150 c, хвост-гард 130
    llm = MockLLM([{"ctas": [
        _cta("subscribe", 20),                           # t≈14.9 < 60 → дроп
        _cta("subscribe", 290),                          # t≈149.9 > 130 → дроп
        _cta("comment", 150, q=""),                      # без вопроса → дроп
        _cta("like", 150),                               # type вне множества
        {"type": "subscribe", "word_idx": "150",
         "comment_question": "", "reason": "r"},         # строка-индекс
        {"type": "subscribe", "word_idx": True,
         "comment_question": "", "reason": "r"},         # bool
        "не словарь",
    ]}])
    out = detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    assert out == []


def test_cta_weak_generic_question_dropped_kept_topical():
    """Вырожденный общий вопрос-попрошайка («пишите/станет интересной», без
    «?») считается «нет вопроса» и дропается; живой тематический вопрос с «?»
    остаётся. Если общий был единственным comment — гарантия добирает шаблон."""
    tr = _build_tr(1200)                                 # 600 c
    llm = MockLLM([{"ctas": [
        _cta("comment", 300, q="Если эта тема станет интересной, пишите"),  # общий
        _cta("comment", 800, q="Какой дистрибутив выбрали и почему?"),      # живой
    ]}])
    out = detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    qs = [it.payload.question for it in out if it.type == ENR_CTA_COMMENT]
    assert "Какой дистрибутив выбрали и почему?" in qs
    assert all("станет интересной" not in q for q in qs)


def test_cta_weak_question_helper_keeps_questions_with_qmark():
    # «?» -> это вопрос, даже если есть слово-триггер: не дропаем
    assert enrich_llm._is_weak_question("Какой дистрибутив выбрали, пишите?") \
        is False
    assert enrich_llm._is_weak_question("Что думаете?") is False
    # без «?» и общий паттерн -> вырожденный
    assert enrich_llm._is_weak_question(
        "Если эта тема станет интересной, пишите") is True
    assert enrich_llm._is_weak_question("Оставь лайк под видео") is True
    # без «?» но осмысленный (предмет есть, паттерна нет) -> НЕ дропаем
    assert enrich_llm._is_weak_question(
        "Расскажите про свой опыт с Linux") is False


def test_cta_question_trimmed_to_120():
    tr = _build_tr(300)
    long_q = "почему " * 30                              # > 120 символов
    llm = MockLLM([{"ctas": [_cta("comment", 150, q=long_q)]}])
    out = detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    assert len(out) == 1
    assert len(out[0].payload.question) <= 120


def test_cta_density_two_per_10min_one_of_each_type():
    tr = _build_tr(1200)                                 # 600 c
    llm = MockLLM([{"ctas": [
        _cta("subscribe", 300),                          # t≈155
        _cta("comment", 600, q="Вопрос по теме момента?"),  # t≈305
        _cta("subscribe", 900),                          # t≈455 — второй sub
    ]}])                                                 # в 10-мин окне → дроп
    out = detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    assert sorted(it.type for it in out) == [ENR_CTA_COMMENT,
                                             ENR_CTA_SUBSCRIBE]
    sub = next(it for it in out if it.type == ENR_CTA_SUBSCRIBE)
    assert sub.word_start == 309                         # выжил первый subscribe


def test_cta_dedup_within_120s():
    tr = _build_tr(1200)
    llm = MockLLM([{"ctas": [
        _cta("subscribe", 300),                          # t≈155
        _cta("comment", 400, q="Вопрос?"),               # t≈205: <120 c от sub
    ]}])
    out = detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    # comment (score 70) обрабатывается первым → дедуп выкидывает МОДЕЛЬНЫЙ
    # subscribe@155 (он в <120 c). Гарантия пакета добирает subscribe ШАБЛОНОМ
    # в задней трети — но он обязан стоять ≥120 c от выжившего comment@205.
    com = next(it for it in out if it.type == ENR_CTA_COMMENT)
    assert com.payload.question == "Вопрос?"              # выжил модельный comment
    assert com.t_start == pytest.approx(409 * WD + 0.4)  # сегмент слова 400 → 409
    for it in out:
        if it.type == ENR_CTA_SUBSCRIBE:
            assert abs(it.t_start - com.t_start) >= 120.0  # не модельный @155


def test_cta_subscribe_only_gets_fallback_comment():
    """Требование «призыв в комментарии» гарантировано: модель отдала только
    subscribe → код синтезирует шаблонный comment-CTA в задней трети, не
    нарушая гардов (дедуп ≥120 c от subscribe)."""
    tr = _build_tr(1200)                                 # 600 c
    llm = MockLLM([{"ctas": [_cta("subscribe", 300)]}])  # t≈155, только sub
    out = detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    assert sorted(it.type for it in out) == [ENR_CTA_COMMENT, ENR_CTA_SUBSCRIBE]
    com = next(it for it in out if it.type == ENR_CTA_COMMENT)
    assert com.payload.question == enrich_llm._FALLBACK_QUESTION
    assert com.score == 70
    # фолбэк ≥120 c от принятого subscribe (t≈155) и в задней трети
    sub = next(it for it in out if it.type == ENR_CTA_SUBSCRIBE)
    assert abs(com.t_start - sub.t_start) >= 120.0
    assert com.t_start > tr.duration * 0.5


def test_cta_empty_after_guards_no_forced_fallback():
    """На пустом результате (всё за гардами/мусор) фолбэк НЕ навязывается —
    насильно вставлять CTA в ролик без валидного места неправильно."""
    tr = _build_tr(300)                                  # 150 c
    llm = MockLLM([{"ctas": [_cta("subscribe", 20)]}])   # t≈14.9 < 60 → дроп
    out = detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    assert out == []


def test_cta_effective_text_excludes_cut_retake_and_maps_to_original():
    """Тест из плана (§3/§7-P3): R4 поймал CTA в вырезанном дубле 12:12 —
    детектор работает по EFFECTIVE-тексту, дубль в промпт не попадает, а
    filtered word_idx мапится обратно в ОРИГИНАЛЬНЫЙ индекс/секунды."""
    # слова 80..119 — ретейк, юзер вырезал [40 c, 60 c)
    tr = _build_tr(300, rename={i: f"дубль{i:03d}" for i in range(80, 120)})
    cut = CutSegment(id="r1", start=40.0, end=60.0, type=TYPE_PAUSE,
                     action=ACTION_REMOVE, enabled=True)
    llm = MockLLM([{"ctas": [_cta("subscribe", 150)]}])
    out = detect_all(tr, _cl(tr, [cut]), _only("cta"), llm, log=_SILENT)
    user = llm.calls[0]["user"]
    assert "дубль" not in user                       # вырезанный дубль не ушёл в LLM
    assert "[150|" in user                           # маркеры — FILTERED-индексы
    # filtered 150 = original 190 (40 вырезанных слов до него): сегмент 19,
    # его последнее слово 199 — ОРИГИНАЛЬНЫЕ индекс и секунды в плане
    assert len(out) == 1
    assert out[0].word_start == 199
    assert out[0].t_start == pytest.approx(199 * WD + 0.4)


def test_cta_markers_thinned_for_long_video(monkeypatch):
    tr = _build_tr(300)
    monkeypatch.setattr(enrich_llm, "CTA_LONG_S", 100.0)   # «45 мин» для теста
    llm = MockLLM([{"ctas": []}])
    detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    user = llm.calls[0]["user"]
    assert "[50|" in user                            # шаг 50 — маркер есть
    assert "[25|" not in user                        # шага 25 больше нет


# === 3. иллюстрации ==============================================================
def _pt(a: int, b: int, *, concept="реестр Windows",
        q="windows registry diagram", style="diagram") -> dict:
    return {"word_start": a, "word_end": b, "concept": concept,
            "image_query_en": q, "style": style}


def test_ill_snap_to_segment_start_and_duration_clamp():
    tr = _build_tr(300)
    llm = MockLLM([{"points": [_pt(15, 16)]}])
    out = detect_all(tr, _cl(tr), _only("image"), llm, log=_SILENT)
    assert len(out) == 1
    it = out[0]
    assert it.type == ENR_IMAGE
    # старт снапнут к началу whisper-сегмента слова 15 (сегмент 1 @ 5.0 c)
    assert it.t_start == pytest.approx(tr.segments[1].start)
    assert it.word_start == 10                        # первое слово сегмента
    assert it.word_end == 16
    # dur = конец слова 16 (8.4) − 5.0 = 3.4 → в клампе 2.5–4.0
    assert it.t_end - it.t_start == pytest.approx(3.4)
    assert it.payload.style_hint == "diagram"
    assert it.payload.image_query_en == "windows registry diagram"
    # image_source=auto + diagram + валидный английский query → SD-генерация
    # (ТРЕК-2 §2): помечен asset_kind="generate", промпт переписан в text-free.
    assert it.payload.asset_kind == "generate"
    assert it.payload.gen_prompt_en == (
        "abstract conceptual illustration, no text, windows registry diagram")
    assert it.payload.position == "top_right"
    assert it.score == 70                             # 55 + 15 за валидный query


def test_ill_duration_clamped_to_max():
    tr = _build_tr(300)
    llm = MockLLM([{"points": [_pt(15, 40)]}])        # сырых 15.4 c → кламп 4.0
    out = detect_all(tr, _cl(tr), _only("image"), llm, log=_SILENT)
    assert out[0].t_end - out[0].t_start == pytest.approx(4.0)


def test_ill_clamps_indices_to_window():
    tr = _build_tr(300)
    llm = MockLLM([{"points": [_pt(-5, 999)]}])       # клампы к окну [0, 300)
    out = detect_all(tr, _cl(tr), _only("image"), llm, log=_SILENT)
    assert len(out) == 1
    assert out[0].word_start == 0                     # сегмент слова 0
    assert out[0].word_end == 299


def test_ill_style_fallback_and_russian_query_means_no_asset():
    tr = _build_tr(300)
    llm = MockLLM([{"points": [
        _pt(15, 20, style="logo"),                    # вне множества → photo
        _pt(150, 155, q="сервер делл", style="photo"),  # русский → без ассета
        _pt(250, 255, q="   ", style="photo"),        # пустой → без ассета
    ]}])
    out = detect_all(tr, _cl(tr), _only("image"), llm, log=_SILENT)
    assert len(out) == 3
    assert out[0].payload.style_hint == "photo"
    assert out[1].payload.image_query_en == ""
    assert out[2].payload.image_query_en == ""
    # русский/пустой query → SD не зовём; без emoji_map → asset_kind="none".
    assert out[1].payload.asset_kind == "none"
    assert out[2].payload.asset_kind == "none"
    # точка 0: photo + валидный английский query (дефолт _pt) → SD-генерация
    # напрямую, query как есть (без diagram-переписывания).
    assert out[0].payload.asset_kind == "generate"
    assert out[0].payload.gen_prompt_en == "windows registry diagram"
    assert out[1].score == 55                         # без query — без бонуса


def test_ill_icon_style_never_generates_falls_back_to_emoji_or_none():
    """style=icon → SD НЕ зовём (логотип/значок SD не нарисует, §2). Без
    emoji_map → asset_kind="none" (предложение без ассета)."""
    tr = _build_tr(300)
    llm = MockLLM([{"points": [_pt(15, 20, q="windows logo", style="icon")]}])
    out = detect_all(tr, _cl(tr), _only("image"), llm, log=_SILENT)
    assert len(out) == 1
    assert out[0].payload.style_hint == "icon"
    assert out[0].payload.asset_kind == "none"        # icon → не generate
    assert out[0].payload.gen_prompt_en == ""


def test_ill_emoji_source_skips_sd_routing(monkeypatch):
    """image_source=emoji → SD-маршрут выключен даже для photo с валидным
    английским query: точка уходит в эмодзи-фолбэк (тут emoji_map подменён)."""
    tr = _build_tr(300)
    monkeypatch.setattr(enrich_llm, "_load_emoji_map",
                        lambda *_a, **_k: {"registry": "u1f4c1"})
    llm = MockLLM([{"points": [_pt(15, 20, q="windows registry", style="photo",
                                   concept="реестр registry")]}])
    out = detect_all(tr, _cl(tr),
                     {**_only("image"), "image_source": "emoji"},
                     llm, log=_SILENT)
    assert len(out) == 1
    assert out[0].payload.asset_kind == "emoji"       # SD не звался
    assert out[0].payload.emoji == "u1f4c1"


def test_ill_generate_source_routes_photo_to_sd():
    """image_source=generate → photo с английским query помечается на SD
    (asset_kind="generate"), без папки юзера."""
    tr = _build_tr(300)
    llm = MockLLM([{"points": [_pt(15, 20, q="dell server", style="photo")]}])
    out = detect_all(tr, _cl(tr),
                     {**_only("image"), "image_source": "generate"},
                     llm, log=_SILENT)
    assert len(out) == 1
    assert out[0].payload.asset_kind == "generate"
    assert out[0].payload.gen_prompt_en == "dell server"


def test_ill_window_limit_four_points():
    tr = _build_tr(300)
    llm = MockLLM([{"points": [_pt(10 + 40 * j, 15 + 40 * j)
                               for j in range(6)]}])
    out = detect_all(tr, _cl(tr), _only("image"), llm, log=_SILENT)
    assert len(out) == 4                              # окно-лимит 4 точки


def test_ill_junk_points_dropped():
    tr = _build_tr(300)
    llm = MockLLM([{"points": [
        {"word_start": "10", "word_end": 15, "concept": "к",
         "image_query_en": "q", "style": "photo"},    # строка-индекс
        _pt(20, 25, concept="   "),                   # пустой концепт
        "не словарь",
        _pt(50, 55),                                  # валидная
    ]}])
    out = detect_all(tr, _cl(tr), _only("image"), llm, log=_SILENT)
    assert len(out) == 1 and out[0].word_end == 55


def test_ill_prompt_markers_and_anticringe_verbatim():
    tr = _build_tr(120)
    llm = MockLLM([{"points": []}])
    detect_all(tr, _cl(tr), _only("image"), llm, log=_SILENT)
    call = llm.calls[0]
    assert call["system"] == enrich_llm._ILL_SYSTEM
    assert call["schema"] is enrich_llm._ILL_SCHEMA
    assert "[0|0:00]" in call["user"]                 # маркеры каждые 10 слов
    assert "[10|0:05]" in call["user"]
    # анти-кринж строка R5 §5 — дословно из плана
    assert "Никаких людей, рукопожатий и офисов." in enrich_llm._ILL_SYSTEM
    assert "(«сервер Dell»)" in enrich_llm._ILL_SYSTEM
    item = enrich_llm._ILL_SCHEMA["properties"]["points"]["items"]
    assert item["required"] == ["word_start", "word_end", "concept",
                                "image_query_en", "style"]


# === 4. сбои окон/детекторов ====================================================
def test_one_bad_window_does_not_lose_lists_pass():
    tr = _build_tr(440)                               # 2 окна списков
    logs: list[str] = []
    llm = MockLLM([RuntimeError("boom"),
                   _lists_resp([_item(370, "а"), _item(380, "б")], intro=365)])
    out = detect_all(tr, _cl(tr), _only("list_card"), llm, log=logs.append)
    assert len(llm.calls) == 2
    assert len(out) == 1                              # второе окно выжило
    assert any("boom" in m for m in logs)


def test_failed_detector_does_not_lose_pass():
    tr = _build_tr(300)
    # списки упали целиком (окно), CTA вернул мусор, иллюстрации работают
    llm = MockLLM([RuntimeError("lists down"),
                   {"ctas": "мусор"},
                   {"points": [_pt(150, 155)]}])
    out = detect_all(tr, _cl(tr), _params(), llm, log=_SILENT)
    assert [it.type for it in out] == [ENR_IMAGE]


def test_llm_none_or_empty_transcript_returns_empty():
    tr = _build_tr(100)
    assert detect_all(tr, _cl(tr), _params(), None, log=_SILENT) == []
    empty = Transcript(language="ru", duration=0.0, model="t",
                       audio_hash="h", segments=[])
    assert detect_all(empty, _cl(empty), _params(), MockLLM([]),
                      log=_SILENT) == []
    assert detect_all(tr, None, _params(), MockLLM([]), log=_SILENT) == []


# === 5. выключенный тип не зовёт LLM ============================================
def test_disabled_types_skip_llm_calls_entirely():
    tr = _build_tr(300)
    # только списки: 1 окно → ровно 1 вызов
    llm = MockLLM([{"lists": []}])
    detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert len(llm.calls) == 1
    # только CTA: ровно 1 вызов
    llm = MockLLM([{"ctas": []}])
    detect_all(tr, _cl(tr), _only("cta"), llm, log=_SILENT)
    assert len(llm.calls) == 1
    # всё выключено: ноль вызовов, пустой план
    llm = MockLLM([])
    out = detect_all(tr, _cl(tr),
                     _params(image=False, animation=False,
                             list_card=False, cta=False), llm, log=_SILENT)
    assert out == [] and llm.calls == []


def test_animation_flag_has_no_detector_and_changes_nothing():
    tr = _build_tr(300)
    llm = MockLLM([{"lists": []}, {"ctas": []}, {"points": []}])
    detect_all(tr, _cl(tr), _params(animation=False), llm, log=_SILENT)
    assert len(llm.calls) == 3                        # animation НЕ имеет детектора


# === 6. keep_alive и прогресс-веса ==============================================
def test_keep_alive_300_between_and_0_on_last_call_of_pass():
    tr = _build_tr(300)                               # 1+1+1 вызов
    llm = MockLLM([{"lists": []}, {"ctas": []}, {"points": []}])
    detect_all(tr, _cl(tr), _params(), llm, log=_SILENT)
    assert [c["keep_alive"] for c in llm.calls] == [300, 300, 0]


def test_keep_alive_last_call_shifts_when_types_disabled():
    tr = _build_tr(300)
    llm = MockLLM([{"lists": []}])                    # только списки → их вызов
    detect_all(tr, _cl(tr), _only("list_card"), llm, log=_SILENT)
    assert [c["keep_alive"] for c in llm.calls] == [0]  # последний = 0
    llm = MockLLM([{"lists": []}, {"points": []}])    # списки + иллюстрации
    detect_all(tr, _cl(tr), _params(cta=False), llm, log=_SILENT)
    assert [c["keep_alive"] for c in llm.calls] == [300, 0]


def test_progress_weights_sum_and_detector_milestones():
    assert sum(enrich_llm.PROGRESS_WEIGHTS.values()) == pytest.approx(1.0)
    assert enrich_llm.PROGRESS_WEIGHTS == {"lists": 0.45, "cta": 0.15,
                                           "illustrations": 0.30,
                                           "assets": 0.10}
    tr = _build_tr(300)
    llm = MockLLM([{"lists": []}, {"ctas": []}, {"points": []}])
    seen: list[float] = []
    detect_all(tr, _cl(tr), _params(), llm, log=_SILENT,
               on_progress=seen.append)
    # вехи детекторов: 0.45 (после списков), 0.60 (после CTA), 0.90 (после
    # иллюстраций — этап assets придёт в P5), финал 1.0; прогресс монотонен
    for mark in (0.0, 0.45, 0.60, 0.90, 1.0):
        assert any(abs(p - mark) < 1e-9 for p in seen), (mark, seen)
    assert seen == sorted(seen)
    assert seen[-1] == 1.0


def test_progress_reaches_full_even_with_all_types_disabled():
    tr = _build_tr(100)
    seen: list[float] = []
    out = detect_all(tr, _cl(tr),
                     _params(image=False, animation=False,
                             list_card=False, cta=False),
                     MockLLM([]), log=_SILENT, on_progress=seen.append)
    assert out == []
    assert seen[-1] == 1.0
