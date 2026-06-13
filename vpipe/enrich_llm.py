"""LLM-детекторы авто-обогащения (ENRICH_PLAN §3, §7-P3).

Три ОТДЕЛЬНЫХ вызова-детектора, НЕ мега-промпт (вердикт R4 §4): перечисления
(§3.1, окна 400 слов), CTA (§3.2, ОДИН вызов на весь ролик), точки иллюстраций
(§3.3, окна 600 слов). Каркас — vpipe/clips.py: окна ``segment_windows``,
per-window try/except («one bad window must not lose the pass» — и сбойный
ДЕТЕКТОР тоже не валит пасс), плоские схемы БЕЗ enum/min/max со ВСЕМИ полями в
``required`` (R4: optional-поля модель молча выкидывает — доказано на
comment_question), ``OllamaClient.chat_json``, температура 0 (config.yaml).

Вход всех детекторов — EFFECTIVE-текст: поток ``transcript.all_words()``,
отфильтрованный по выжившим интервалам cutlist по правилу remap_words
(слово на >50% в вырезе — вон): R4 поймал CTA в вырезанном дубле (12:12).
Маппинг filtered→original индексов держит код; в план пишутся ОРИГИНАЛЬНЫЕ
word-индексы и ОРИГИНАЛЬНЫЕ секунды (§1.2) — ремап в финальные делает
планировщик/рендер.

Промпты — ДОСЛОВНО проверенные v2 из R4 (D:/tmp/enrich/r4_llm.md + probe-скрипты;
снап цитат 24/25, parse 9/9 strict). НЕ «улучшать» формулировки. JSON few-shot
ЗАПРЕЩЁН (qwen3:8b попугайничает вплоть до копирования text_short); текстовый
образец-паттерн ВНУТРИ инструкции — работает (разблокировал смысловые списки).

Все числовые границы валидирует КОД, не модель (qwen игнорирует числовые
запреты — доказано R4 дважды: subscribe@0:36 пережил явный запрет 60 с).
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import Callable, Optional

from .clips import _norm_token, _valid_int
from .cutlist import resolve
from .enrich import (CARD_ITEM_TEXT_MAX, CARD_ITEMS_MAX, CARD_TITLE_MAX,
                     CTA_QUESTION_MAX, ENR_CTA_COMMENT, ENR_CTA_SUBSCRIBE,
                     ENR_IMAGE, ENR_LIST_CARD, IMAGE_DUR_DEF, IMAGE_DUR_MAX,
                     IMAGE_DUR_MIN, EnrichItem, _trim_text, item_from_dict)
from .llm import segment_windows
from .models import CutList, Transcript, Word
from .timeline import Timeline

LogFn = Callable[..., None]


def _noop(*_a, **_k) -> None:
    pass


# --- размеры окон / маркеры (§3.1–3.3, §3.4 — числа R4, не менять без проб) ----
KEEP_ALIVE_BETWEEN = 300       # сек; тёплый старт между вызовами, 0 на ПОСЛЕДНЕМ
LISTS_WINDOW = 400             # §3.1: окна 400 слов, overlap 40, текст БЕЗ маркеров
LISTS_OVERLAP = 40
ILL_WINDOW = 600               # §3.3: окна 600 слов
ILL_MARK_EVERY = 10            # маркеры [N|м:сс] каждые 10 слов
ILL_MAX_PER_WINDOW = 4         # окно-лимит 4 точки (модель упирается в потолок — R4)
CTA_MARK_EVERY = 25            # §3.2: маркеры каждые 25 слов…
CTA_MARK_EVERY_LONG = 50       # …>45 мин — прореживание до 50 (num_ctx 16384)
CTA_LONG_S = 45 * 60.0
CTA_TEMPERATURE = 0.4          # подъём temp ТОЛЬКО на CTA-вызове: разнообразие
                               # тематического вопроса (списки/иллюстрации — 0)

# --- пост-обработка КОДОМ (остаточные болезни R4 лечатся только кодом) ----------
FUZZY_RATIO = 0.75             # фаззи-снап SequenceMatcher >= 0.75…
FUZZY_RADIUS = 30              # …по окну ±30 слов вокруг последнего якоря
ITEM_MAX_GAP_WORDS = 25        # анти-дробление: пункт дальше 25 слов — отрез хвоста
LIST_SPAN_MAX_S = 60.0         # пункты одного списка в пределах 60 с
LIST_IOU_DUP = 0.5             # слияние/дедуп списков между окнами (паттерн _is_time_dup)
CTA_MIN_T_S = 60.0             # дроп CTA раньше 60-й секунды (effective-таймлиния)
CTA_TAIL_S = 20.0              # …и позже (конец − 20 с)
CTA_DENSITY_WINDOW_S = 600.0   # плотность: не больше 2 CTA в любом 10-мин окне,
CTA_DENSITY_MAX = 2            # приоритет «1 subscribe + 1 comment» по score
CTA_DEDUP_GAP_S = 120.0        # дедуп по близости < 120 c

# --- score: эвристика КОДА, не модели (§3.1) ------------------------------------
_SCORE_SNAP_W = 40             # полнота снапа (snapped/total)
_SCORE_ITEM_W = 8              # количество выживших пунктов (<=6)
_SCORE_INTRO_BONUS = 12        # наличие найденного intro
_CTA_SCORE = {"subscribe": 60, "comment": 70}  # comment несёт уникальный вопрос
_ILL_SCORE_BASE = 55
_ILL_SCORE_QUERY = 15          # есть вменяемый английский image_query_en

# Прогресс задачи по детекторам (§3.4): lists 45 / cta 15 / illustrations 30 /
# assets 10. Этап assets (подбор ассетов, §4 Tier 0/1) приходит в P5 — пока его
# вес проскакивается мгновенно (детекторы честно доводят до 0.9, конец — 1.0).
PROGRESS_WEIGHTS = {"lists": 0.45, "cta": 0.15,
                    "illustrations": 0.30, "assets": 0.10}

_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)

# Распознаём ВЫРОЖДЕННЫЙ comment-вопрос (qwen3:8b на temp>0 иногда выдаёт
# вместо тематического вопроса общую попрошайку «если тема станет интересной,
# пишите» / «оставь лайк» — это ровно то, что промпт запрещает). Фильтр
# НАМЕРЕННО узкий: бьём только бессодержательные шаблоны-попрошайки, а живой
# императив с предметом («Напиши, какой дистрибутив выбрал и зачем») — НЕ
# трогаем (там есть предмет и вопросительное слово). Эвристика КОДА (модель
# числа/запреты игнорирует — модульный докстринг). template-фолбэк ниже
# страхует, если в итоге ни одного годного comment не осталось.
_WEAK_Q = re.compile(
    r"станет интересн|"                            # «если тема станет интересной…»
    r"что (?:вы |ты )?думаешь|что (?:вы )?думаете|"  # «что думаете?»
    r"^\W*согла[сш]|"                              # «согласны?»
    r"^\W*пиши(?:те)?\s+в?\s*коммент|"             # голое «пишите в комментах»
    r"оставь(?:те)?\s+(?:лайк|коммент)|"           # «оставь лайк/коммент»
    r"жду ваши коммент|делитесь в коммент",
    re.IGNORECASE)
# Вопросительное слово/предмет = вопрос содержательный, даже без «?».
_HAS_INTERROGATIVE = re.compile(
    r"\b(как(?:ой|ая|ое|ие|им)?|что|чем|где|куда|почему|зачем|сколько|"
    r"кто|когда|стоит ли|а вы|а ты)\b", re.IGNORECASE)


def _is_weak_question(q: str) -> bool:
    """True, если comment_question — бессодержательная попрошайка без предмета.

    Настоящий тематический вопрос почти всегда содержит «?» ИЛИ вопросительное
    слово/предмет; вырожденная попрошайка — нет. Дропаем ТОЛЬКО когда нет «?»,
    нет вопросительного слова И сработал общий шаблон — чтобы не зарубить живой
    вопрос («Напиши, какой дистрибутив выбрал» содержит «какой» -> остаётся)."""
    if "?" in q or _HAS_INTERROGATIVE.search(q):
        return False
    return bool(_WEAK_Q.search(q))


# === промпты и схемы — ДОСЛОВНО v2 из R4 (см. модульный докстринг) ==============
# §3.1 — детектор перечислений (probe1b_quotes.py + поле title_short из плана).
_LISTS_SYSTEM = (
    "Ты — монтажёр обучающих видео. Тебе дают фрагмент расшифровки устной "
    "речи (человек говорит в камеру, пунктуация может отсутствовать).\n\n"
    "Найди ПЕРЕЧИСЛЕНИЯ — места, где говорящий ПОДРЯД называет несколько "
    "однотипных пунктов:\n"
    "- явные: «во-первых… во-вторых», «первый… второй», «всего N "
    "остановок/путей/способов»;\n"
    "- структурные: «сначала… потом… дальше… и в конце»;\n"
    "- смысловые БЕЗ слов-маркеров: подряд названы несколько свойств, плюсов, "
    "минусов или причин одного предмета. Пример смыслового: «чем он хорош? он "
    "централизованный… он быстрый… он типизированный» — это перечисление из "
    "трёх пунктов (Централизованный / Быстрый / Типизированный).\n\n"
    "Жёсткие правила:\n"
    "- одна подводка = ОДИН список со ВСЕМИ его пунктами; НЕ дроби одно "
    "перечисление на несколько списков;\n"
    "- пункт — это элемент списка целиком, а не очередной кусок фразы;\n"
    "- в перечислении минимум 2 пункта, обычно 3-6;\n"
    "- НЕ выдумывай: каждый пункт реально произнесён в тексте.\n\n"
    "Для каждого перечисления верни:\n"
    "- intro_quote — ДОСЛОВНАЯ цитата (3-6 слов) фразы-подводки, после "
    "которой начинается перечисление;\n"
    "- title_short — заголовок карточки 2-4 слова («Плюсы реестра»), пустая "
    "строка если не очевиден;\n"
    "- items — пункты строго в порядке произнесения:\n"
    "  - text_short — суть пункта, сжатая до 2-5 слов для карточки на экране "
    "(НЕ дословно: убери вводные слова, начни с заглавной буквы);\n"
    "  - quote — ДОСЛОВНЫЕ первые 3-6 слов отрезка речи, где произносится "
    "этот пункт (точно как в тексте).\n\n"
    "Если перечислений нет — верни пустой список."
)

_LISTS_USER_TMPL = (
    "Расшифровка:\n{text}\n\n"
    "Верни JSON: {{\"lists\": [{{\"intro_quote\": \"…\", \"title_short\": \"…\", "
    "\"items\": [{{\"text_short\": \"…\", \"quote\": \"…\"}}]}}]}}"
)

_LISTS_SCHEMA = {
    "type": "object",
    "properties": {
        "lists": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intro_quote": {"type": "string"},
                    "title_short": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text_short": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["text_short", "quote"],
                        },
                    },
                },
                "required": ["intro_quote", "title_short", "items"],
            },
        }
    },
    "required": ["lists"],
}

# §3.2 — детектор CTA (probe2b_cta.py, v2: comment_question в required,
# запрет первых 60 секунд).
_CTA_SYSTEM = (
    "Ты — продюсер YouTube-канала. Тебе дают ПОЛНУЮ расшифровку видео "
    "(человек говорит в камеру). В тексте расставлены маркеры вида [N|м:сс] — "
    "номер СЛЕДУЮЩЕГО слова и время; слова между маркерами нумеруются подряд.\n\n"
    "Найди лучшие места для призывов к действию (CTA):\n"
    "- type=\"subscribe\" — 1-3 места для ненавязчивого значка «подпишись + "
    "лайк». Ставь его сразу ПОСЛЕ сильного момента: ценный вывод, вау-факт, "
    "конец полезного блока — зритель только что получил пользу. ЗАПРЕЩЕНО: "
    "первые 60 секунд видео (вступление — НЕ сильный момент), середина "
    "недосказанной мысли. Для subscribe поле comment_question — пустая "
    "строка \"\".\n"
    "- type=\"comment\" — ОБЯЗАТЕЛЬНО предложи 1-2 призыва написать в "
    "комментарии. comment_question — КОНКРЕТНЫЙ короткий вопрос зрителю, "
    "который вытекает из того, о чём говорится именно в этом месте видео. "
    "Вопрос живой, разговорный, как задал бы сам автор. Не общий («что "
    "думаете?», «пишите в комментарии»), а по теме момента — называй в "
    "вопросе конкретный предмет спора/выбора из речи. Для спорных, "
    "субъективных тем и тем «А vs B» это КРИТИЧНО: именно такой вопрос "
    "разгоняет обсуждение. Ориентиры по стилю (НЕ копируй дословно, "
    "придумай свой по теме ЭТОГО видео): «А вы на чём сидите и почему?», "
    "«Какой вариант выбрали бы вы и для каких задач?». Запрещены вопросы "
    "без предмета («что думаете?», «согласны?»).\n\n"
    "Для каждого CTA: word_idx — номер слова, ПОСЛЕ которого показать призыв "
    "(конец фразы, не середина); reason — коротко по-русски, почему именно "
    "здесь. Не больше 5 элементов всего, но хотя бы один из них — type=comment. "
    "Если автор сам в этом месте уже просит лайк/комментарий — это хорошее "
    "место, можно использовать."
)

_CTA_USER_TMPL = (
    "Полная расшифровка с маркерами:\n{text}\n\n"
    "Верни JSON: {{\"ctas\": [{{\"type\": \"subscribe|comment\", \"word_idx\": N, "
    "\"comment_question\": \"…\", \"reason\": \"…\"}}]}}"
)

_CTA_SCHEMA = {
    "type": "object",
    "properties": {
        "ctas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "word_idx": {"type": "integer"},
                    "comment_question": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["type", "word_idx", "comment_question", "reason"],
            },
        }
    },
    "required": ["ctas"],
}

# §3.3 — детектор иллюстраций (probe3_illustrations.py + анти-кринж строка
# R5 §5 дословно из плана).
_ILL_SYSTEM = (
    "Ты — монтажёр обучающих видео. Тебе дают фрагмент расшифровки "
    "(человек говорит в камеру). Маркеры вида [N|м:сс] — номер СЛЕДУЮЩЕГО "
    "слова и время его произнесения; слова между маркерами нумеруются "
    "подряд.\n\n"
    "Найди моменты, где на экран уместно вывести картинку-иллюстрацию:\n"
    "- названа конкретная сущность: продукт, компания, технология, программа;\n"
    "- объясняется концепция, которую проще показать схемой;\n"
    "- звучит яркое число или статистика.\n\n"
    "Для каждой точки верни:\n"
    "- word_start, word_end — номера первого и последнего слова отрезка, пока "
    "картинка видна (отрезок 5-20 слов, начинается там, где сущность "
    "произносится);\n"
    "- concept — что показать, коротко по-русски;\n"
    "- image_query_en — поисковый запрос для картинки, АНГЛИЙСКИЙ, 2-6 слов;\n"
    "- style — одно из: photo (фото предмета/места), diagram (схема/график), "
    "icon (логотип/значок).\n\n"
    "ВАЖНО про плотность: НЕ ЧАЩЕ одной картинки на 60-90 секунд речи. "
    "Во фрагменте около 4 минут — значит, не больше 3-4 точек. Выбирай только "
    "самые сильные моменты, остальные пропусти. Если сильных нет — верни "
    "пустой список.\n\n"
    "Запрос — про КОНКРЕТНЫЙ объект из речи («сервер Dell»), а не про "
    "абстракцию («успех», «бизнес»). Нет конкретного объекта — пропусти "
    "точку. Никаких людей, рукопожатий и офисов."
)

_ILL_USER_TMPL = (
    "Расшифровка с маркерами:\n{text}\n\n"
    "Верни JSON: {{\"points\": [{{\"word_start\": N, \"word_end\": N, "
    "\"concept\": \"…\", \"image_query_en\": \"…\", \"style\": "
    "\"photo|diagram|icon\"}}]}}"
)

_ILL_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word_start": {"type": "integer"},
                    "word_end": {"type": "integer"},
                    "concept": {"type": "string"},
                    "image_query_en": {"type": "string"},
                    "style": {"type": "string"},
                },
                "required": ["word_start", "word_end", "concept",
                             "image_query_en", "style"],
            },
        }
    },
    "required": ["points"],
}


# === EFFECTIVE-поток слов ========================================================
@dataclass
class _EffStream:
    """Слова вне enabled-вырезов + маппинги обратно в оригинал.

    ``words[p]`` — ОРИГИНАЛЬНЫЕ объекты Word (секунды — оригинальная
    таймлиния); ``orig[p]`` — их индекс в ``transcript.all_words()``.
    """
    words: list[Word] = field(default_factory=list)
    orig: list[int] = field(default_factory=list)       # filtered -> original idx
    norm: list[str] = field(default_factory=list)       # нормализованные токены
    all_words: list[Word] = field(default_factory=list)  # ВСЕ слова (== all_words())
    seg_of: list[int] = field(default_factory=list)     # original idx -> сегмент
    seg_first: dict = field(default_factory=dict)       # сегмент -> первый orig idx
    seg_last: dict = field(default_factory=dict)        # сегмент -> последний orig idx
    tl: Timeline = None  # type: ignore[assignment]


def _build_eff(transcript: Transcript, cutlist: CutList) -> _EffStream:
    """EFFECTIVE-поток: правило remap_words (>50% слова в вырезе — дроп).

    Порядок слов воспроизводит ``Transcript.all_words()`` БАЙТ-В-БАЙТ
    (стабильная сортировка по start того же сегментного обхода) — иначе
    оригинальные word-индексы плана разъехались бы с UI/планировщиком.
    """
    pairs: list[tuple[Word, int]] = []
    for si, seg in enumerate(transcript.segments):
        for w in seg.words:
            pairs.append((w, si))
    pairs.sort(key=lambda p: p[0].start)
    removed, _censors = resolve(cutlist)
    duration = float(cutlist.duration or transcript.duration)
    tl = Timeline(removed, duration)
    eff = _EffStream(tl=tl)
    eff.all_words = [w for w, _si in pairs]
    eff.seg_of = [si for _w, si in pairs]
    for i, (_w, si) in enumerate(pairs):
        eff.seg_first.setdefault(si, i)
        eff.seg_last[si] = i
    for i, (w, _si) in enumerate(pairs):
        dur = max(0.0, w.end - w.start)
        if dur <= 0.0:
            if tl.inside(0.5 * (w.start + w.end)):
                continue
        elif tl.removed_overlap(w.start, w.end) > 0.5 * dur:
            continue
        eff.words.append(w)
        eff.orig.append(i)
    eff.norm = [_norm_token(w.word) for w in eff.words]
    return eff


def _fmt_mmss(t: float) -> str:
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


def _plain_text(eff: _EffStream, lo: int, hi: int) -> str:
    """Текст окна БЕЗ маркеров (§3.1: минус ~140 токенов на окно)."""
    return " ".join(w.word.strip() for w in eff.words[lo:hi])


def _marked_text(eff: _EffStream, lo: int, hi: int, every: int) -> str:
    """Текст с маркерами ``[N|м:сс]``: N — FILTERED-индекс СЛЕДУЮЩЕГО слова,
    время — EFFECTIVE (после вырезов) секунды этого слова (то «видео», которое
    реально увидит зритель — правила «не в первые 60 с» работают по нему)."""
    parts: list[str] = []
    for p in range(lo, hi):
        if (p - lo) % every == 0:
            t = eff.tl.remap_clamped(eff.words[p].start)
            parts.append(f"[{p}|{_fmt_mmss(t)}]")
        parts.append(eff.words[p].word.strip())
    return " ".join(parts)


def _win_cfg(size: int, overlap: int) -> SimpleNamespace:
    return SimpleNamespace(max_segments_per_call=size, segment_overlap=overlap)


# === снап цитат (паттерн _snap_start_word clips.py:499-525) ======================
def _quote_tokens(quote) -> list[str]:
    if not isinstance(quote, str):
        return []
    return [t for t in (_norm_token(x) for x in quote.split()) if t]


def _snap_exact(quote, eff: _EffStream, lo: int,
                hi: int) -> Optional[tuple[int, int]]:
    """Первые 3–5 нормализованных токенов цитаты в eff.norm[lo:hi] -> (pos, k)."""
    toks = _quote_tokens(quote)
    if not toks:
        return None
    for k in range(min(5, len(toks)), min(3, len(toks)) - 1, -1):
        target = toks[:k]
        for p in range(lo, hi - k + 1):
            if eff.norm[p:p + k] == target:
                return p, k
    return None


def _snap_fuzzy(quote, eff: _EffStream, lo: int, hi: int,
                anchor: Optional[int]) -> Optional[tuple[int, int]]:
    """Фаззи-снап: SequenceMatcher >= 0.75 по окну ±30 слов вокруг якоря
    (последний успешный снап: intro или предыдущий пункт — пункты идут
    подряд). Без якоря — по всему окну. Лечит искажённое моделью слово
    в цитате («перенняется» вместо «переносится», R4 w2)."""
    toks = _quote_tokens(quote)
    if not toks:
        return None
    k = min(5, len(toks))
    if k < 3:
        # цитата 1-2 токена (нарушение промпта моделью): фаззи ≥0.75 в окне
        # ±30 слов даёт ложные совпадения ("он"≈"оно") — лучше дроп пункта.
        return None
    target = " ".join(toks[:k])
    if anchor is None:
        a, b = lo, hi - k
    else:
        a = max(lo, anchor - FUZZY_RADIUS)
        b = min(hi - k, anchor + FUZZY_RADIUS)
    best_p, best_r = None, FUZZY_RATIO - 1e-9
    for p in range(a, b + 1):
        r = SequenceMatcher(None, target,
                            " ".join(eff.norm[p:p + k])).ratio()
        if r > best_r:
            best_p, best_r = p, r
    return (best_p, k) if best_p is not None else None


def _snap(quote, eff: _EffStream, lo: int, hi: int,
          anchor: Optional[int]) -> Optional[tuple[int, int]]:
    """Точный снап -> фаззи -> None (промах = дроп пункта, §3.1)."""
    hit = _snap_exact(quote, eff, lo, hi)
    if hit is not None:
        return hit
    return _snap_fuzzy(quote, eff, lo, hi, anchor)


def _norm_short(text, limit: int) -> str:
    """Нормализация text_short/title КОДОМ (§3.1): схлоп пробелов, жёсткий
    лимит по границе слова, капитализация, без точки в конце."""
    s = _trim_text(text, limit)
    s = s.rstrip(".…").rstrip()
    if s:
        s = s[0].upper() + s[1:]
    return s


# === §3.1 детектор перечислений ==================================================
def _card_from_list(lst, eff: _EffStream, lo: int, hi: int) -> Optional[dict]:
    """Один сырой список модели -> кандидат-карточка (или None).

    Вся пост-обработка КОДОМ: снап цитат (промах -> дроп пункта),
    анти-дробление (вложенные/обратные — дроп; >25 слов от конца предыдущего
    или >60 c от первого — отрез хвоста), <=6 пунктов, <2 выживших -> дроп,
    score — эвристика кода (полнота снапа x пункты x наличие intro).
    """
    if not isinstance(lst, dict):
        return None
    raw_items = lst.get("items") if isinstance(lst.get("items"), list) else []
    intro_quote = lst.get("intro_quote") \
        if isinstance(lst.get("intro_quote"), str) else ""
    title = _norm_short(lst.get("title_short"), CARD_TITLE_MAX)
    intro = _snap(intro_quote, eff, lo, hi, None)
    anchor = intro[0] if intro is not None else None

    n_total = 0
    snapped: list[tuple[int, int, str]] = []      # (pos, k, text)
    for r in raw_items:
        if not isinstance(r, dict):
            continue
        n_total += 1
        text = _norm_short(r.get("text_short"), CARD_ITEM_TEXT_MAX)
        if not text:
            continue                               # пустой текст — пункт бесполезен
        hit = _snap(r.get("quote"), eff, lo, hi, anchor)
        if hit is None:
            continue                               # промах снапа -> дроп пункта
        pos, k = hit
        snapped.append((pos, k, text))
        anchor = pos

    surv: list[tuple[int, int, str]] = []
    for pos, k, text in snapped:
        if surv:
            ppos, pk, _t = surv[-1]
            if pos < ppos + pk:
                continue                           # вложен/ушёл назад — дроп пункта
            if pos - (ppos + pk) > ITEM_MAX_GAP_WORDS:
                break                              # «уехал в чужой блок» — отрез
            if eff.words[pos].start - eff.words[surv[0][0]].start \
                    > LIST_SPAN_MAX_S:
                break                              # пункты одного списка — в 60 c
        surv.append((pos, k, text))
    surv = surv[:CARD_ITEMS_MAX]
    if len(surv) < 2:
        return None

    first_pos = intro[0] if intro is not None else surv[0][0]
    last_pos, last_k, _lt = surv[-1]
    end_pos = min(last_pos + last_k - 1, len(eff.words) - 1)
    score = round(_SCORE_SNAP_W * len(snapped) / max(1, n_total)
                  + _SCORE_ITEM_W * len(surv)
                  + (_SCORE_INTRO_BONUS if intro is not None else 0))
    return {
        "type": ENR_LIST_CARD,
        "score": max(0, min(100, score)),
        "word_start": eff.orig[first_pos],
        "word_end": eff.orig[end_pos],
        "t_start": eff.words[first_pos].start,
        "t_end": eff.words[end_pos].end,
        "quote": _trim_text(intro_quote, 120),
        "reason": ("перечисление из %d пунктов" % len(surv))
                  + (f": {title}" if title else ""),
        "payload": {"title": title, "mode": "scrim",
                    "items": [{"text": text,
                               "word_idx": eff.orig[pos],
                               "t_word": eff.words[pos].start}
                              for pos, _k, text in surv]},
    }


def _dedup_cards(cands: list[dict]) -> list[dict]:
    """Слияние/дедуп списков между окнами по IoU>=0.5 диапазонов (паттерн
    ``_is_time_dup``). «Слияние» = выживает лучший экземпляр: полный список
    бьёт фрагмент (больше пунктов), затем score, затем ранний."""
    def iou(a: dict, b: dict) -> float:
        inter = min(a["t_end"], b["t_end"]) - max(a["t_start"], b["t_start"])
        if inter <= 0:
            return 0.0
        union = max(a["t_end"], b["t_end"]) - min(a["t_start"], b["t_start"])
        return inter / union if union > 0 else 0.0

    order = sorted(cands, key=lambda c: (-len(c["payload"]["items"]),
                                         -c["score"], c["t_start"]))
    kept: list[dict] = []
    for c in order:
        if any(iou(c, k) >= LIST_IOU_DUP for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda c: c["t_start"])
    return kept


def _detect_lists(eff: _EffStream, llm, ka_next, log: LogFn,
                  prog) -> list[dict]:
    cards: list[dict] = []
    wins = segment_windows(len(eff.words), _win_cfg(LISTS_WINDOW, LISTS_OVERLAP))
    n_win = len(wins)
    for wi, (lo, hi) in enumerate(wins):
        prog(PROGRESS_WEIGHTS["lists"] * wi / max(1, n_win))
        log(f"Монтаж: перечисления {wi + 1}/{n_win}…")
        user = _LISTS_USER_TMPL.format(text=_plain_text(eff, lo, hi))
        try:
            data = llm.chat_json(_LISTS_SYSTEM, user, _LISTS_SCHEMA,
                                 keep_alive=ka_next())
        except Exception as e:  # noqa: BLE001 — одно сбойное окно не валит детектор
            log(f"  enrich: окно списков [{lo}:{hi}] пропущено ({e})")
            continue
        raw = data.get("lists") if isinstance(data, dict) else None
        for lst in (raw if isinstance(raw, list) else []):
            c = _card_from_list(lst, eff, lo, hi)
            if c is not None:
                cards.append(c)
    return _dedup_cards(cards)


# === §3.2 детектор CTA ===========================================================
@contextmanager
def _cta_temperature(llm, temp: float):
    """Временно поднять temperature клиента на CTA-вызов и вернуть назад.

    Работает только для боевого ``OllamaClient`` (мутируемый ``cfg.temperature``).
    Тестовый ``MockLLM`` ``cfg`` не имеет — для него no-op, детерминизм тестов
    не трогаем. Восстанавливаем исходное значение даже при исключении.
    """
    cfg = getattr(llm, "cfg", None)
    if cfg is None or not hasattr(cfg, "temperature"):
        yield
        return
    saved = cfg.temperature
    try:
        cfg.temperature = temp
        yield
    finally:
        cfg.temperature = saved


def _detect_cta(eff: _EffStream, transcript: Transcript, llm, ka_next,
                log: LogFn) -> list[dict]:
    n = len(eff.words)
    if n == 0:
        return []
    eff_dur = eff.tl.new_duration()
    every = CTA_MARK_EVERY_LONG if eff_dur > CTA_LONG_S else CTA_MARK_EVERY
    log("Монтаж: CTA…")
    user = _CTA_USER_TMPL.format(text=_marked_text(eff, 0, n, every))
    # CTA — единственный детектор, которому нужно РАЗНООБРАЗИЕ, а не точный
    # снап: на temp=0 qwen3:8b детерминированно залипает в общий вопрос без
    # предмета («Если эта тема станет интересной, пишите»). Лёгкий подъём до
    # 0.4 на ОДНОМ этом вызове разблокирует осмысленный тематический вопрос
    # (живой Prod9: «Какой дистрибутив Linux вы выбрали и почему?»). Тесты на
    # MockLLM температуру не читают (нет .cfg) — детерминизм тестов цел.
    with _cta_temperature(llm, CTA_TEMPERATURE):
        try:
            data = llm.chat_json(_CTA_SYSTEM, user, _CTA_SCHEMA,
                                 keep_alive=ka_next())
        except Exception as e:  # noqa: BLE001 — сбойный детектор не валит пасс
            log(f"  enrich: CTA-детектор пропущен ({e})")
            return []
    raw = data.get("ctas") if isinstance(data, dict) else None

    cand: list[dict] = []
    for r in (raw if isinstance(raw, list) else []):
        if not isinstance(r, dict):
            continue
        typ = r.get("type")
        if typ not in ("subscribe", "comment"):
            continue                               # type вне множества -> дроп
        wi = r.get("word_idx")
        if not _valid_int(wi):
            continue
        wi = max(0, min(n - 1, wi))
        question = _trim_text(r.get("comment_question"), CTA_QUESTION_MAX)
        if typ == "comment" and (not question or _is_weak_question(question)):
            continue                               # без вопроса / общий -> дроп
            # (template-фолбэк ниже гарантирует, что comment-CTA не пропадёт)
        # Снап к концу предложения: граница whisper-сегмента (±25 слов маркера
        # хватает для значка, который висит секунды; снап убирает «середину фразы»).
        orig_i = eff.orig[wi]
        si = eff.seg_of[orig_i]
        last_i = eff.seg_last[si]
        t_orig = eff.all_words[last_i].end
        t_eff = eff.tl.remap_clamped(t_orig)
        if t_eff < CTA_MIN_T_S or t_eff > eff_dur - CTA_TAIL_S:
            continue                               # первые 60 c / последние 20 c
        cand.append({
            "typ": typ, "t_eff": t_eff,
            "d": {
                "type": (ENR_CTA_SUBSCRIBE if typ == "subscribe"
                         else ENR_CTA_COMMENT),
                "score": _CTA_SCORE[typ],
                "word_start": last_i, "word_end": last_i,
                "t_start": t_orig, "t_end": 0.0,   # t_end выставит item_from_dict
                "quote": _trim_text(transcript.segments[si].text, 120),
                "reason": _trim_text(r.get("reason"), 200),
                "payload": ({"variant": "sub_like"} if typ == "subscribe"
                            else {"question": question}),
            },
        })

    # Дедуп <120 c + плотность <=2 на 10 мин с приоритетом «1 subscribe +
    # 1 comment» (по score: comment > subscribe — он несёт уникальный вопрос).
    cand.sort(key=lambda c: (-c["d"]["score"], c["t_eff"]))
    kept: list[dict] = []
    for c in cand:
        if any(abs(c["t_eff"] - k["t_eff"]) < CTA_DEDUP_GAP_S for k in kept):
            continue
        near = [k for k in kept
                if abs(c["t_eff"] - k["t_eff"]) < CTA_DENSITY_WINDOW_S]
        if len(near) >= CTA_DENSITY_MAX:
            continue
        if any(k["typ"] == c["typ"] for k in near):
            continue                               # второй такой же тип в окне
        kept.append(c)

    # Гарантия пакета «1 subscribe + 1 comment»: на этом ролике (живой Prod9
    # Linux/Windows) модель ставит оба subscribe во вступление/концовку, и
    # ОБА срезают временные гарды (60 c / хвост 20 c), а comment без подъёма
    # температуры дегенерирует в общий вопрос. Если детектор отработал
    # нормально (kept не пуст — ролик «живой», место под призыв есть), но
    # типа comment ИЛИ subscribe среди принятых нет — синтезируем шаблонный,
    # не нарушая гардов времени/дедупа/плотности. На пустом kept (мусор / всё
    # за гардами) фолбэк НЕ навязываем: насильно вставлять CTA в ролик, где
    # даже один настоящий CTA не прошёл, неправильно. comment важнее (это и
    # есть требование пользователя) — добираем его первым.
    if kept:
        for typ in ("comment", "subscribe"):
            if not any(k["typ"] == typ for k in kept):
                fb = _fallback_cta(typ, eff, transcript, kept, eff_dur)
                if fb is not None:
                    kept.append(fb)

    kept.sort(key=lambda c: c["t_eff"])
    return [c["d"] for c in kept]


# Шаблонный вопрос-фолбэк: открытый, но с предметом-плейсхолдером, который
# работает для типового «обзор/сравнение/мнение» ролика без знания темы. Это
# страховка последней инстанции; основной путь — осмысленный вопрос модели.
_FALLBACK_QUESTION = "А что в итоге выбрали вы и почему? Расскажите в комментариях"


def _fallback_cta(typ: str, eff: _EffStream, transcript: Transcript,
                  kept: list[dict], eff_dur: float) -> Optional[dict]:
    """Один гарантированный CTA типа ``typ``, если модель его не дала.

    Ставим в задней трети ролика (типовое место и для «обсуждаемого» вопроса,
    и для «подпишись после пользы») на границу whisper-сегмента, соблюдая те
    же гарды, что и для модельных CTA: t∈[60, конец−20], дедуп ≥120 c от уже
    принятых, плотность ≤2/10 мин, не два одинаковых типа в 10-мин окне. Если
    безопасного места нет — None (лучше ничего, чем CTA внахлёст)."""
    lo_t = CTA_MIN_T_S
    hi_t = eff_dur - CTA_TAIL_S
    if hi_t <= lo_t:
        return None
    target_t = max(lo_t, min(hi_t, eff_dur * 0.72))  # задняя треть

    def slot_ok(t_eff: float) -> bool:
        if any(abs(t_eff - k["t_eff"]) < CTA_DEDUP_GAP_S for k in kept):
            return False
        near = [k for k in kept
                if abs(t_eff - k["t_eff"]) < CTA_DENSITY_WINDOW_S]
        if len(near) >= CTA_DENSITY_MAX:
            return False
        return not any(k["typ"] == typ for k in near)

    # Кандидаты-границы сегментов в окне [60, конец−20], отсортированы по
    # близости к target_t; берём первый, проходящий гарды.
    seg_ends: list[tuple[float, int, int]] = []  # (t_eff, seg_idx, last_orig)
    seen_seg: set[int] = set()
    for p in range(len(eff.words)):
        si = eff.seg_of[eff.orig[p]]
        if si in seen_seg:
            continue
        seen_seg.add(si)
        last_i = eff.seg_last[si]
        t_eff = eff.tl.remap_clamped(eff.all_words[last_i].end)
        if lo_t <= t_eff <= hi_t:
            seg_ends.append((t_eff, si, last_i))
    if not seg_ends:
        return None
    seg_ends.sort(key=lambda x: abs(x[0] - target_t))
    for t_eff, si, last_i in seg_ends:
        if not slot_ok(t_eff):
            continue
        d = {
            "type": (ENR_CTA_SUBSCRIBE if typ == "subscribe"
                     else ENR_CTA_COMMENT),
            "score": _CTA_SCORE[typ],
            "word_start": last_i, "word_end": last_i,
            "t_start": eff.all_words[last_i].end, "t_end": 0.0,
            "quote": _trim_text(transcript.segments[si].text, 120),
        }
        if typ == "subscribe":
            d["reason"] = ("подписка после полезного блока (шаблон: модель "
                           "ставила значок только во вступление/концовку)")
            d["payload"] = {"variant": "sub_like"}
        else:
            d["reason"] = ("призыв в комментарии (шаблон: модель не "
                           "предложила осмысленный вопрос)")
            d["payload"] = {"question": _FALLBACK_QUESTION}
        return {"typ": typ, "t_eff": t_eff, "d": d}
    return None


# === §3.3 детектор иллюстраций ===================================================
def _detect_illustrations(eff: _EffStream, transcript: Transcript, llm,
                          ka_next, log: LogFn, prog) -> list[dict]:
    out: list[dict] = []
    wins = segment_windows(len(eff.words), _win_cfg(ILL_WINDOW, 0))
    n_win = len(wins)
    base = PROGRESS_WEIGHTS["lists"] + PROGRESS_WEIGHTS["cta"]
    for wi, (lo, hi) in enumerate(wins):
        prog(base + PROGRESS_WEIGHTS["illustrations"] * wi / max(1, n_win))
        log(f"Монтаж: иллюстрации {wi + 1}/{n_win}…")
        user = _ILL_USER_TMPL.format(
            text=_marked_text(eff, lo, hi, ILL_MARK_EVERY))
        try:
            data = llm.chat_json(_ILL_SYSTEM, user, _ILL_SCHEMA,
                                 keep_alive=ka_next())
        except Exception as e:  # noqa: BLE001 — одно сбойное окно не валит детектор
            log(f"  enrich: окно иллюстраций [{lo}:{hi}] пропущено ({e})")
            continue
        raw = data.get("points") if isinstance(data, dict) else None
        n_kept = 0
        for r in (raw if isinstance(raw, list) else []):
            if n_kept >= ILL_MAX_PER_WINDOW:
                break                              # окно-лимит 4 точки
            if not isinstance(r, dict):
                continue
            a, b = r.get("word_start"), r.get("word_end")
            if not (_valid_int(a) and _valid_int(b)):
                continue
            a = max(lo, min(hi - 1, a))            # клампы к окну
            b = max(a, min(hi - 1, b))
            concept = _trim_text(r.get("concept"), 80)
            if not concept:
                continue                           # без концепта точка бессмысленна
            q = r.get("image_query_en")
            q = " ".join(q.split()) if isinstance(q, str) else ""
            if not q or _CYRILLIC.search(q):
                q = ""                             # пустой/русский запрос -> без
                                                   # авто-ассета (asset_kind none)
            style = r.get("style")
            if style not in ("photo", "diagram", "icon"):
                style = "photo"                    # style-фолбэк
            orig_a, orig_b = eff.orig[a], eff.orig[b]
            si = eff.seg_of[orig_a]
            t0 = transcript.segments[si].start     # снап старта к началу сегмента
            dur = eff.all_words[orig_b].end - t0
            dur = IMAGE_DUR_DEF if dur <= 0 else \
                min(IMAGE_DUR_MAX, max(IMAGE_DUR_MIN, dur))
            out.append({
                "type": ENR_IMAGE,
                "score": _ILL_SCORE_BASE + (_ILL_SCORE_QUERY if q else 0),
                "word_start": eff.seg_first[si], "word_end": orig_b,
                "t_start": t0, "t_end": t0 + dur,
                "quote": _trim_text(
                    " ".join(w.word for w in eff.words[a:b + 1]), 120),
                "reason": f"иллюстрация: {concept}",
                "payload": {"concept": concept, "image_query_en": q,
                            "style_hint": style, "asset_kind": "none",
                            "position": "top_right"},
            })
            n_kept += 1
    # Плотность сверх окна-лимита здесь НЕ режем — потолки планировщика
    # (<=2/мин и общие) срежут по score (§3.3: «он срежет»).
    return out


# === сборка ======================================================================
def detect_all(transcript: Optional[Transcript], cutlist: Optional[CutList],
               params: Optional[dict], llm, log: LogFn = _noop,
               on_progress=None) -> list[EnrichItem]:
    """Полный LLM-пасс обогащения (§3): списки -> CTA -> иллюстрации.

    - выключенный в ``params["types"]`` тип НЕ вызывается вообще (§3.4);
    - keep_alive=300 между вызовами эпика, 0 на ПОСЛЕДНЕМ (паттерн clips);
    - per-window и per-детектор try/except: сбой не валит пасс (warnings в log);
    - прогресс по детекторам: lists 45 / cta 15 / illustrations 30 / assets 10
      (этап assets придёт в P5 — его вес проскакивается в конце);
    - элементы строятся через ``item_from_dict`` (клампы длительностей §1.2,
      id enr_*); координаты — ОРИГИНАЛЬНЫЕ word-индексы/секунды.
    """
    prog = on_progress if on_progress is not None else (lambda _f: None)
    if llm is None or transcript is None or not transcript.segments \
            or cutlist is None:
        prog(1.0)
        return []
    types = (params or {}).get("types") or {}
    run_lists = bool(types.get("list_card", True))
    run_cta = bool(types.get("cta", True))
    run_ill = bool(types.get("image", True))

    eff = _build_eff(transcript, cutlist)
    if not eff.words:
        prog(1.0)
        return []

    # План вызовов известен заранее -> знаем ПОСЛЕДНИЙ (keep_alive=0 на нём,
    # VRAM под Whisper/рендер — §3.4).
    n_calls = ((len(segment_windows(len(eff.words),
                                    _win_cfg(LISTS_WINDOW, LISTS_OVERLAP)))
                if run_lists else 0)
               + (1 if run_cta else 0)
               + (len(segment_windows(len(eff.words), _win_cfg(ILL_WINDOW, 0)))
                  if run_ill else 0))
    made = 0

    def ka_next() -> int:
        nonlocal made
        made += 1
        return 0 if made >= n_calls else KEEP_ALIVE_BETWEEN

    dicts: list[dict] = []
    prog(0.0)
    if run_lists:
        try:
            dicts.extend(_detect_lists(eff, llm, ka_next, log, prog))
        except Exception as e:  # noqa: BLE001 — сбойный детектор не валит пасс
            log(f"enrich: детектор перечислений упал ({e}) — пропускаю")
    prog(PROGRESS_WEIGHTS["lists"])
    if run_cta:
        try:
            dicts.extend(_detect_cta(eff, transcript, llm, ka_next, log))
        except Exception as e:  # noqa: BLE001
            log(f"enrich: CTA-детектор упал ({e}) — пропускаю")
    prog(PROGRESS_WEIGHTS["lists"] + PROGRESS_WEIGHTS["cta"])
    if run_ill:
        try:
            dicts.extend(_detect_illustrations(eff, transcript, llm, ka_next,
                                               log, prog))
        except Exception as e:  # noqa: BLE001
            log(f"enrich: детектор иллюстраций упал ({e}) — пропускаю")
    prog(1.0 - PROGRESS_WEIGHTS["assets"])
    # Этап assets (вес 10%): подбор ассетов user_folder/эмодзи приходит в P5 —
    # пока предложения честно уходят с asset_kind="none".
    prog(1.0)

    items: list[EnrichItem] = []
    for d in dicts:
        it = item_from_dict(d, log)
        if it is not None:
            items.append(it)
    return items
