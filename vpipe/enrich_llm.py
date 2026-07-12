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

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from .clips import _norm_token, _valid_int
from .cutlist import resolve
from .enrich import (CANDIDATES_MAX, CARD_ITEM_TEXT_MAX, CARD_ITEMS_MAX,
                     CARD_TITLE_MAX, CTA_QUESTION_MAX, EMOJI_SVG_DIR,
                     ENR_ANIMATION, ENR_CTA_COMMENT, ENR_CTA_SUBSCRIBE,
                     ENR_IMAGE, ENR_LIST_CARD, IMAGE_DUR_DEF, IMAGE_DUR_MAX,
                     IMAGE_DUR_MIN, SCHEMATIC_INTENTS, SCHEMATIC_STYLE_DEF,
                     SCHEMATIC_STYLES, VISUAL_SOURCES, EnrichItem, _trim_text,
                     item_from_dict, sanitize_candidate)
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
CTA_WINDOW = 3200              # слов на ОДИН CTA-вызов: ~≤0.8*num_ctx с маркерами
                               # (>~21 мин ролик режется на окна; короче — 1 окно)
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

# §3.3 — детектор-КЛАССИФИКАТОР иллюстраций (MONTAGE_V2 §1, codegfx a_code.md §2.1).
# Диспетчер: на каждую сильную точку — ОДИН источник + intent (для schematic).
# Узкий экстрактор по intent (заполнение fields с quote-снапом) — отдельный
# вызов на стадии кандидатов (ниже). Анти-кринж строка R5 §5 — дословно.
_ILL_SYSTEM = (
    "Ты — монтажёр обучающих видео. Тебе дают фрагмент расшифровки "
    "(человек говорит в камеру). Маркеры вида [N|м:сс] — номер СЛЕДУЮЩЕГО "
    "слова и время его произнесения; слова между маркерами нумеруются "
    "подряд.\n\n"
    "Найди моменты, где на экран уместно что-то вывести, и реши, ЧТО именно "
    "монтажёр поставил бы в каждом — выбери ОДИН источник:\n"
    "- schematic — КОД-СХЕМА с настоящим текстом из речи: данные, структура, "
    "сравнение, числа/статистика, список/шаги, термин, код/команда. Это "
    "главный источник объясняющего ролика (его НЕ нарисовать фото);\n"
    "- photo — РЕАЛЬНАЯ физическая сцена/объект/место (дата-центр, макро "
    "клавиатуры, человек за ноутбуком) — фотореалистичный кадр;\n"
    "- icon — бренд/логотип/значок (только когда явно назван логотип);\n"
    "- none — абстракция/лозунг без данных и без конкретного предмета "
    "(«успех», «удобство») — иллюстрировать нечего.\n\n"
    "Для schematic дополнительно выбери intent (тип схемы):\n"
    "- stat — звучит число/статистика («11 тысяч разработчиков», «96% серверов»);\n"
    "- compare — «A против B», таблица свойств («Linux vs Windows»);\n"
    "- tree — иерархия/структура (реестр, дерево каталогов, оргструктура);\n"
    "- list — перечисление подряд («во-первых… во-вторых», философия Unix);\n"
    "- process — шаги-инструкция («открыть PowerShell → wsl --install → …»);\n"
    "- timeline — даты/этапы во времени («в 1991 Торвальдс…»);\n"
    "- callout — определение термина («Ядро (kernel) — это…»);\n"
    "- code — терминал/конфиг/команда (`ls /etc`, `wsl --install`);\n"
    "- quote — дословная цитата/тезис, который надо выделить.\n"
    "Для photo/icon/none поле intent — пустая строка \"\".\n\n"
    "Для каждой точки верни:\n"
    "- word_start, word_end — номера первого и последнего слова отрезка, пока "
    "визуал виден (отрезок 5-20 слов, начинается там, где сущность "
    "произносится);\n"
    "- concept — что показать, коротко по-русски;\n"
    "- source — schematic | photo | icon | none;\n"
    "- intent — тип схемы из списка выше (или \"\" для не-schematic);\n"
    "- image_query_en — для photo: АНГЛИЙСКИЙ запрос про КОНКРЕТНЫЙ объект "
    "сцены, 2-6 слов; для остальных — пустая строка \"\".\n\n"
    "ВАЖНО про плотность: НЕ ЧАЩЕ одного визуала на 60-90 секунд речи. "
    "Во фрагменте около 4 минут — значит, не больше 3-4 точек. Выбирай только "
    "самые сильные моменты, остальные пропусти. Если сильных нет — верни "
    "пустой список.\n\n"
    "Запрос photo — про КОНКРЕТНЫЙ объект из речи («сервер Dell»), а не про "
    "абстракцию («успех», «бизнес»). Нет конкретного объекта — source=none. "
    "Никаких людей, рукопожатий и офисов."
)

_ILL_USER_TMPL = (
    "Расшифровка с маркерами:\n{text}\n\n"
    "Верни JSON: {{\"points\": [{{\"word_start\": N, \"word_end\": N, "
    "\"concept\": \"…\", \"source\": \"schematic|photo|icon|none\", "
    "\"intent\": \"…\", \"image_query_en\": \"…\"}}]}}"
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
                    "source": {"type": "string"},
                    "intent": {"type": "string"},
                    "image_query_en": {"type": "string"},
                },
                "required": ["word_start", "word_end", "concept",
                             "source", "intent", "image_query_en"],
            },
        }
    },
    "required": ["points"],
}


# === РОУТЕР источника визуала (MONTAGE_V2 §1) ====================================
# Взаимоисключающий: каждая точка получает РОВНО один источник. Решение модели
# (source/intent) — основа; КОД страхует дешёвыми эвристиками поверх (политика
# репо: модель ошибается типом — числа/структуру детектим регэкспом).
#
# Сигналы «это данные/схема, НЕ фото» (форс schematic, даже если модель сказала
# photo): числа/%, слова структуры/сравнения/списка/шагов/термина/кода. Это
# ровно те Prod9-концепты, что диффузия превращала в кракозябры (§0, c_diff §3).
# \b перед цифрой: «96%»/«11 тысяч» считаем, «ток100»/«fable5» (буква+цифра) — нет.
_NUM_RE = re.compile(r"\b\d")
_PCT_RE = re.compile(r"%|процент", re.IGNORECASE)
# keyword → intent, в которую форсим точку (первое совпадение по concept/quote).
_INTENT_SIGNALS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bvs\b|против|сравн|или\b.*\bили\b", re.IGNORECASE), "compare"),
    (re.compile(r"дерев|структур|иерарх|ветк|каталог|реестр|оргструктур",
                re.IGNORECASE), "tree"),
    (re.compile(r"шаг|инструкц|сначала|потом|установ|команд", re.IGNORECASE),
     "process"),
    (re.compile(r"философи|принцип|во-первых|перечисл|список", re.IGNORECASE),
     "list"),
    (re.compile(r"\b1991\b|истори|хронолог|\bэтап", re.IGNORECASE), "timeline"),
    (re.compile(r"термин|определени|\bядро\b|\bkernel\b", re.IGNORECASE),
     "callout"),
    (re.compile(r"`|\bls\b|\.exe|/etc|конфиг|терминал|синтаксис", re.IGNORECASE),
     "code"),
    (re.compile(r"статистик|процент|%|тысяч|миллион", re.IGNORECASE), "stat"),
)
# Сигналы «явный бренд/значок» (узко — только когда назван логотип). НЕ как
# фолбэк иллюстрации (§1: эмодзи под нож).
_ICON_RE = re.compile(r"логотип|значок|иконк|\bbrand\b|бренд", re.IGNORECASE)


# Детектор-словарь источника (что вернула модель в _ILL_SCHEMA.source). Внутренне
# photo == diffusion-кандидат; роутер работает в этом словаре, _build_candidates
# материализует (photo→diffusion). schematic/icon/none — как в VISUAL_SOURCES.
_DETECT_SOURCES = ("schematic", "photo", "icon", "none")


def _route_source(source: str, intent: str, concept: str, quote: str,
                  query_en: str) -> tuple[str, str]:
    """Решить (source, intent) для точки по сигналам речи (взаимоисключающий §1).

    Работает в ДЕТЕКТОР-словаре ``_DETECT_SOURCES`` (schematic/photo/icon/none);
    ``_build_candidates`` материализует photo→diffusion-кандидат.

    1. Модельный source/intent — основа (валидируем по множествам).
    2. КОД-страховка: числа/%/структурные слова → форс schematic + intent даже
       если модель сказала photo (модель путает данные с фото — c_diff §3).
    3. photo требует НЕпустой английский query (без предмета SD не зовём — §5);
       нет query → none (лучше пусто, чем слоп).
    4. icon — только при явном слове-значке; иначе схлоп в schematic/none.
    Возврат: (source ∈ _DETECT_SOURCES, intent ∈ SCHEMATIC_INTENTS|"")."""
    src = source if source in _DETECT_SOURCES else ""
    it = intent if intent in SCHEMATIC_INTENTS else ""
    blob = f"{concept} {quote}"

    # (2) числа/структура → данные, не фото. Форсим schematic + подбираем intent.
    has_data = bool(_NUM_RE.search(blob) or _PCT_RE.search(blob))
    forced_intent = ""
    for rx, name in _INTENT_SIGNALS:
        if rx.search(blob):
            forced_intent = name
            break
    if (has_data or forced_intent) and src in ("photo", "", "none"):
        src = "schematic"
        if not it:
            it = forced_intent or "stat"      # число без явного типа → stat

    # (4) icon только при явном слове-значке; иначе не icon.
    if src == "icon" and not _ICON_RE.search(blob):
        src = "schematic" if (has_data or forced_intent) else "none"
        it = it or forced_intent

    # (1/3) schematic без intent — добираем по сигналу, иначе callout (термин —
    # самый безопасный «текст на экране»). photo без английского query → none.
    if src == "schematic" and not it:
        it = forced_intent or "callout"
    if src == "photo":
        q = " ".join((query_en or "").split())
        if not q or _CYRILLIC.search(q):
            src, it = "none", ""
        else:
            it = ""
    if src not in _DETECT_SOURCES:
        src = "none"
    if src != "schematic":
        it = ""
    return src, it


# === узкие экстракторы schematic-fields (codegfx a_code.md §2.2) =================
# ОДИН прицельный LLM-вызов на schematic-точку: плоская схема ИМЕННО этого intent,
# ВСЕ поля required (R4: optional молча выкидывается), числа/значения с quote-
# якорем. КОД снапит quote к транскрипту (_snap) и валидирует числа — модель
# данные НЕ выдумывает. <2 валидных значения → точка дропается (§8 анти-слоп).
SCHEMATIC_VALUES_MIN = 2          # <2 снапнутых значений на schematic → дроп точки
_SCHEMATIC_INTRO_QUOTE = (
    "Для КАЖДОГО значения дай поле quote — ДОСЛОВНЫЕ 3-6 слов из речи, где это "
    "произнесено (точно как в тексте). Не выдумывай: бери только реально "
    "сказанные числа/слова.")

_EXTRACT_SYSTEM = {
    "stat": (
        "Ты — монтажёр. Тебе дают фрагмент речи. Извлеки числа/статистику для "
        "карточки-инфографики. " + _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни eyebrow (подводка 2-4 слова) и stats — список значений: value "
        "(само число, как в речи), unit (единица: «%», «ГБ», пусто), label "
        "(что это число, 1-3 слова), quote (якорь)."),
    "compare": (
        "Ты — монтажёр. Тебе дают фрагмент речи со сравнением A против B. "
        "Извлеки таблицу. " + _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни title (например «Linux vs Windows»), col_a, col_b (заголовки "
        "колонок) и rows — строки: feature (свойство), a (значение слева), b "
        "(значение справа), winner («a»/«b»/«»), quote (якорь)."),
    "tree": (
        "Ты — монтажёр. Тебе дают фрагмент речи об иерархии/структуре. Извлеки "
        "дерево ПЛОСКИМ списком. " + _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни title, root_label (корень дерева) и nodes — узлы: label (имя), "
        "desc (пояснение 1-3 слова), parent (имя родителя или \"\" для верхних), "
        "quote (якорь)."),
    "list": (
        "Ты — монтажёр. Тебе дают фрагмент речи с перечислением. " +
        _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни title (заголовок 2-4 слова), ordered (true если порядок важен) и "
        "items — пункты: text (суть 2-5 слов), note (короткое пояснение или "
        "\"\"), quote (якорь)."),
    "process": (
        "Ты — монтажёр. Тебе дают фрагмент речи с инструкцией-шагами. " +
        _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни title и steps — шаги по порядку: title (что сделать 2-4 слова), "
        "desc (детали), quote (якорь)."),
    "timeline": (
        "Ты — монтажёр. Тебе дают фрагмент речи с датами/этапами. " +
        _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни title и events — события по порядку: when (дата/этап), what "
        "(что произошло), quote (якорь)."),
    "callout": (
        "Ты — монтажёр. Тебе дают фрагмент речи с определением термина. " +
        _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни term (термин), definition (определение одной фразой) и quote "
        "(якорь, где термин определён)."),
    "code": (
        "Ты — монтажёр. Тебе дают фрагмент речи о команде/конфиге/терминале. " +
        _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни title, lang (язык/оболочка: bash, powershell, …), code "
        "(САМ текст команды/конфига, как в речи) и quote (якорь)."),
    "quote": (
        "Ты — монтажёр. Тебе дают фрагмент речи с ярким тезисом. " +
        _SCHEMATIC_INTRO_QUOTE + "\n"
        "Верни text (сам тезис ДОСЛОВНО), by (автор/источник или \"\") и quote "
        "(якорь — те же слова тезиса в тексте)."),
}

# Плоские required-схемы по intent (ВСЕ поля required — R4). Числа/значения
# несут quote; КОД снапит и валидирует. Список-полей собирает движок (codegfx).
def _row_schema(*fields: str) -> dict:
    return {"type": "object",
            "properties": {f: {"type": "string"} for f in fields},
            "required": list(fields)}


_EXTRACT_SCHEMA = {
    "stat": {"type": "object", "properties": {
        "eyebrow": {"type": "string"},
        "stats": {"type": "array",
                  "items": _row_schema("value", "unit", "label", "quote")}},
        "required": ["eyebrow", "stats"]},
    "compare": {"type": "object", "properties": {
        "title": {"type": "string"}, "col_a": {"type": "string"},
        "col_b": {"type": "string"},
        "rows": {"type": "array",
                 "items": _row_schema("feature", "a", "b", "winner", "quote")}},
        "required": ["title", "col_a", "col_b", "rows"]},
    "tree": {"type": "object", "properties": {
        "title": {"type": "string"}, "root_label": {"type": "string"},
        "nodes": {"type": "array",
                  "items": _row_schema("label", "desc", "parent", "quote")}},
        "required": ["title", "root_label", "nodes"]},
    "list": {"type": "object", "properties": {
        "title": {"type": "string"}, "ordered": {"type": "boolean"},
        "items": {"type": "array",
                  "items": _row_schema("text", "note", "quote")}},
        "required": ["title", "ordered", "items"]},
    "process": {"type": "object", "properties": {
        "title": {"type": "string"},
        "steps": {"type": "array",
                  "items": _row_schema("title", "desc", "quote")}},
        "required": ["title", "steps"]},
    "timeline": {"type": "object", "properties": {
        "title": {"type": "string"},
        "events": {"type": "array",
                   "items": _row_schema("when", "what", "quote")}},
        "required": ["title", "events"]},
    "callout": {"type": "object", "properties": {
        "term": {"type": "string"}, "definition": {"type": "string"},
        "quote": {"type": "string"}},
        "required": ["term", "definition", "quote"]},
    "code": {"type": "object", "properties": {
        "title": {"type": "string"}, "lang": {"type": "string"},
        "code": {"type": "string"}, "quote": {"type": "string"}},
        "required": ["title", "lang", "code", "quote"]},
    "quote": {"type": "object", "properties": {
        "text": {"type": "string"}, "by": {"type": "string"},
        "quote": {"type": "string"}},
        "required": ["text", "by", "quote"]},
}
# intent → (имя списка строк, ключ-якорь quote в каждой строке). Для intent без
# списка (callout/code/quote) — список пуст, якорь — в самом объекте.
_EXTRACT_LIST_FIELD = {
    "stat": "stats", "compare": "rows", "tree": "nodes",
    "list": "items", "process": "steps", "timeline": "events",
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
    # Окна, чтобы промпт не превысил num_ctx на длинных роликах: маркеры несут
    # АБСОЛЮТНЫЙ filtered-индекс, поэтому word_idx из окон совместимы с общими
    # гардами/дедупом ниже. Короткий/средний ролик = РОВНО одно окно [0:n]
    # (поведение без изменений). Модель держим тёплой (ka_next → 300; финальный
    # unload — в detect_all).
    wins = (segment_windows(n, _win_cfg(CTA_WINDOW, 0))
            if n > CTA_WINDOW else [(0, n)])
    cand: list[dict] = []
    for lo, hi in wins:
        user = _CTA_USER_TMPL.format(text=_marked_text(eff, lo, hi, every))
        # CTA — единственный детектор, которому нужно РАЗНООБРАЗИЕ, а не точный
        # снап: на temp=0 qwen3:8b детерминированно залипает в общий вопрос без
        # предмета («Если эта тема станет интересной, пишите»). Лёгкий подъём до
        # 0.4 на ЭТОМ вызове разблокирует осмысленный тематический вопрос
        # (живой Prod9: «Какой дистрибутив Linux вы выбрали и почему?»). Тесты на
        # MockLLM температуру не читают (нет .cfg) — детерминизм тестов цел.
        with _cta_temperature(llm, CTA_TEMPERATURE):
            try:
                data = llm.chat_json(_CTA_SYSTEM, user, _CTA_SCHEMA,
                                     keep_alive=ka_next())
            except Exception as e:  # noqa: BLE001 — сбойное окно не валит пасс
                log(f"  enrich: CTA-окно [{lo}:{hi}] пропущено ({e})")
                continue
        raw = data.get("ctas") if isinstance(data, dict) else None
        for r in (raw if isinstance(raw, list) else []):
            if not isinstance(r, dict):
                continue
            typ = r.get("type")
            if typ not in ("subscribe", "comment"):
                continue                           # type вне множества -> дроп
            wi = r.get("word_idx")
            if not _valid_int(wi):
                continue
            wi = max(0, min(n - 1, wi))
            question = _trim_text(r.get("comment_question"), CTA_QUESTION_MAX)
            if typ == "comment" and (not question or _is_weak_question(question)):
                continue                           # без вопроса / общий -> дроп
                # (template-фолбэк ниже гарантирует, что comment-CTA не пропадёт)
            # Снап к концу предложения: граница whisper-сегмента (±25 слов маркера
            # хватает для значка, который висит секунды; снап убирает «середину фразы»).
            orig_i = eff.orig[wi]
            si = eff.seg_of[orig_i]
            last_i = eff.seg_last[si]
            t_orig = eff.all_words[last_i].end
            t_eff = eff.tl.remap_clamped(t_orig)
            if t_eff < CTA_MIN_T_S or t_eff > eff_dur - CTA_TAIL_S:
                continue                           # первые 60 c / последние 20 c
            cand.append({
                "typ": typ, "t_eff": t_eff,
                "d": {
                    "type": (ENR_CTA_SUBSCRIBE if typ == "subscribe"
                             else ENR_CTA_COMMENT),
                    "score": _CTA_SCORE[typ],
                    "word_start": last_i, "word_end": last_i,
                    "t_start": t_orig, "t_end": 0.0,  # t_end выставит item_from_dict
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


# === валидация чисел schematic-значений (codegfx a_code.md §2.3) =================
# КОД, не модель (политика репо): значение-число обязано РЕАЛЬНО встречаться в
# окне транскрипта (нормализовано: «11 тысяч»→11000, «96 процентов»→96). Не нашли
# → значение-число дроп. Нечисловые значения (слова) проходят (их страхует снап).
_NUM_TOKEN_RE = re.compile(r"\d[\d\s.,]*")
_RU_SCALE = {"тысяч": 1_000, "тыс": 1_000, "миллион": 1_000_000,
             "млн": 1_000_000, "миллиард": 1_000_000_000, "млрд": 1_000_000_000}


def _nums_in(text: str) -> set[int]:
    """Множество целых чисел, встречающихся в тексте (с учётом «11 тысяч» →
    11000, «96 процентов» → 96). Дробную часть отбрасываем (для сверки величин).
    Пробелы/точки/запятые внутри числа схлопываются («11 000»→11000)."""
    out: set[int] = set()
    low = text.lower()
    for m in _NUM_TOKEN_RE.finditer(low):
        digits = re.sub(r"[\s.,]", "", m.group())
        if not digits:
            continue
        try:
            base = int(digits)
        except ValueError:
            continue
        out.add(base)
        # «11 тысяч» — масштаб в следующем слове
        tail = low[m.end():m.end() + 16].lstrip()
        for kw, mult in _RU_SCALE.items():
            if tail.startswith(kw):
                out.add(base * mult)
                break
    return out


def _value_in_window(value: str, eff: _EffStream, lo: int, hi: int) -> bool:
    """Значение schematic-строки валидно для окна (a_code §2.3). Нечисловое →
    True (текст страхует quote-снап). Числовое → его целое обязано встречаться
    среди чисел окна (нормализовано) — иначе выдумка модели, дроп."""
    nums = _nums_in(value)
    if not nums:
        return True
    window_text = " ".join(w.word for w in eff.words[lo:hi])
    win_nums = _nums_in(window_text)
    return bool(nums & win_nums)


def _clean_extracted_fields(intent: str, data: dict, eff: _EffStream,
                            lo: int, hi: int) -> tuple[dict, int]:
    """Сырой ответ узкого экстрактора → (плоские fields, число валидных значений).

    КОД-валидация (a_code §2.3): для каждой строки списка снапим её quote к окну
    (_snap, промах → строка дроп) И сверяем числа (выдуманное число → строка
    дроп). intent без списка (callout/code/quote) — валидируем один quote-якорь.
    Возврат fields БЕЗ служебного quote (его в payload не кладём — он лишь для
    валидации); n_valid — сколько значений прошло (порог SCHEMATIC_VALUES_MIN)."""
    if not isinstance(data, dict):
        return {}, 0
    list_field = _EXTRACT_LIST_FIELD.get(intent)
    fields: dict = {}
    n_valid = 0
    anchor: Optional[int] = None
    if list_field is None:
        # callout/code/quote: один объект с якорем-quote.
        quote = data.get("quote")
        if _snap(quote, eff, lo, hi, None) is None:
            return {}, 0                       # якорь не снапнулся → выдумка, дроп
        for k, v in data.items():
            if k == "quote":
                continue
            if isinstance(v, str) and v.strip():
                fields[k] = _trim_text(v, CARD_ITEM_TEXT_MAX * 4)
        # значимость: есть ли непустое смысловое поле
        n_valid = sum(1 for k, v in fields.items() if v)
        return fields, (1 if n_valid else 0)

    # scalar-поля верхнего уровня (eyebrow/title/col_a/col_b/ordered/…)
    for k, v in data.items():
        if k == list_field:
            continue
        if isinstance(v, bool):
            fields[k] = v
        elif isinstance(v, str) and v.strip():
            fields[k] = _trim_text(v, CARD_ITEM_TEXT_MAX * 2)

    rows_in = data.get(list_field)
    rows_out: list[dict] = []
    for r in (rows_in if isinstance(rows_in, list) else []):
        if not isinstance(r, dict):
            continue
        quote = r.get("quote")
        hit = _snap(quote, eff, lo, hi, anchor)
        if hit is None:
            continue                           # quote не снапнулся → строка дроп
        anchor = hit[0]
        # числовые значения строки обязаны встречаться в окне (анти-выдумка)
        bad_num = False
        row: dict = {}
        for k, v in r.items():
            if k == "quote":
                continue
            if isinstance(v, bool):
                row[k] = v
            elif isinstance(v, str) and v.strip():
                sv = _trim_text(v, CARD_ITEM_TEXT_MAX)
                if not _value_in_window(sv, eff, lo, hi):
                    bad_num = True
                    break
                row[k] = sv
        if bad_num or not row:
            continue
        rows_out.append(row)
        n_valid += 1
        if len(rows_out) >= CARD_ITEMS_MAX:
            break
    fields[list_field] = rows_out
    return fields, n_valid


def _extract_schematic(intent: str, eff: _EffStream, lo: int, hi: int, llm,
                       keep_alive: int, log: LogFn) -> Optional[dict]:
    """Узкий экстрактор fields для одной schematic-точки (a_code §2.2). Один
    прицельный LLM-вызов по плоской схеме intent, КОД-валидация значений
    (_clean_extracted_fields). <SCHEMATIC_VALUES_MIN валидных → None (дроп —
    лучше пусто, чем выдумка). Сбой вызова → None (точка останется без schematic-
    кандидата, останется none — пасс не валим)."""
    system = _EXTRACT_SYSTEM.get(intent)
    schema = _EXTRACT_SCHEMA.get(intent)
    if system is None or schema is None:
        return None
    user = ("Фрагмент речи:\n" + _plain_text(eff, lo, hi)
            + "\n\nВерни JSON строго по схеме (все поля обязательны).")
    try:
        data = llm.chat_json(system, user, schema, keep_alive=keep_alive)
    except Exception as e:  # noqa: BLE001 — сбойный экстрактор не валит пасс
        log(f"  enrich: экстрактор {intent} пропущен ({e})")
        return None
    fields, n_valid = _clean_extracted_fields(intent, data, eff, lo, hi)
    if n_valid < SCHEMATIC_VALUES_MIN and intent not in ("callout", "quote"):
        return None                            # <2 значений → дроп (анти-слоп §8)
    if n_valid < 1:
        return None
    return fields


# === арт-дирекшн-промпт диффузии (MONTAGE_V2 §5, c_diff §2б) =====================
# LLM пишет БОГАТЫЙ промпт (subject+свет+линза+grade); фото-суффикс/негатив — в
# imagegen.py. Здесь — лёгкий код-обогатитель голого query_en, когда отдельного
# LLM-вызова за арт-дирекшном делать не хочется (дёшево, без сети). Доказано
# (c_diff §1/§2): голый query = слоп; «cinematic… depth of field… photorealistic»
# = кадр. Суффикс STYLE_SUFFIX добавит imagegen — здесь только subject-обогащение.
_ARTDIR_SUFFIX = (", cinematic lighting, shallow depth of field, "
                  "photorealistic, professional photography, ultra detailed")


def _art_direction_prompt(query_en: str) -> str:
    """Голый английский query_en → БОГАТЫЙ арт-дирекшн-промпт (§5). Код-путь
    (без LLM): subject как есть + кинематографичный хвост. Пусто/русский → ""."""
    q = " ".join((query_en or "").split())
    if not q or _CYRILLIC.search(q):
        return ""
    return q + _ARTDIR_SUFFIX


# === §3.3 детектор-классификатор иллюстраций + сборка кандидатов =================
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
            quote = _trim_text(" ".join(w.word for w in eff.words[a:b + 1]), 120)
            # РОУТЕР (§1): источник + intent по сигналам речи (взаимоисключающий).
            src, intent = _route_source(r.get("source"), r.get("intent"),
                                        concept, quote, q)
            # Обратная совместимость: style_hint для старого рендера/плоского пути.
            style_hint = ("diagram" if src == "schematic"
                          else "icon" if src == "icon" else "photo")
            if src == "photo":
                q = " ".join(q.split())            # photo: query прошёл роутер
            else:
                q = ""                             # не-photo: без image_query
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
                "quote": quote,
                "reason": f"иллюстрация: {concept}",
                "payload": {"concept": concept, "image_query_en": q,
                            "style_hint": style_hint, "asset_kind": "none",
                            "position": "top_right",
                            "source": src, "intent": intent},
                # служебное (для стадии кандидатов; НЕ в финальный payload):
                "_route": {"source": src, "intent": intent,
                           "lo": lo, "hi": hi, "query_en": q},
            })
            n_kept += 1
    # Плотность сверх окна-лимита здесь НЕ режем — потолки планировщика
    # (<=2/мин и общие) срежут по score (§3.3: «он срежет»).
    return out


# === стадия КАНДИДАТОВ (MONTAGE_V2 §2) ==========================================
# На каждый момент-иллюстрацию собираем 2-4 кандидата в payload.candidates (юзер
# листает и выбирает; selected=0 = лучший). schematic-кандидаты — ЭАГЕРНО
# (быстро, CPU/LLM, без GPU): узкий экстрактор заполняет fields. diffusion-
# кандидат — только для photo-моментов (реальная генерация — стадия images,
# VRAM-менеджер). none — ВСЕГДА как вариант («лучше пусто, чем слоп»).
def _build_candidates(points: list, eff: _EffStream, llm, log: LogFn,
                      *, default_style: str = SCHEMATIC_STYLE_DEF,
                      run_diffusion: bool = True) -> int:
    """Заполнить ``payload.candidates``/``selected``/``source`` точкам-иллюстрациям
    (мутирует ``points``). Возврат — число точек, получивших ≥1 материальный
    (не-none) кандидат.

    Кандидаты по роутер-источнику точки (``_route``):
      schematic → узкий экстрактор fields (ЭАГЕРНО, LLM без GPU); промах
                  валидации → schematic-кандидата НЕТ (остаётся none).
      photo     → diffusion-кандидат с богатым арт-дирекшн-промптом (реальная
                  генерация N сидов — стадия images). run_diffusion=False
                  (image_source=emoji и т.п.) → diffusion-кандидата нет.
      icon      → icon-кандидат (эмодзи подберёт этап ассетов; здесь пометка).
    none добавляется КАЖДОЙ точке последним вариантом. selected=0 — первый
    материальный (или none, если материальных нет)."""
    ill = [p for p in points
           if isinstance(p, dict) and p.get("type") in (ENR_IMAGE, ENR_ANIMATION)
           and isinstance(p.get("payload"), dict)]
    style = default_style if default_style in SCHEMATIC_STYLES \
        else SCHEMATIC_STYLE_DEF
    materialized = 0
    for p in ill:
        route = p.get("_route") or {}
        src = route.get("source", p["payload"].get("source", "none"))
        intent = route.get("intent", "")
        lo, hi = route.get("lo", 0), route.get("hi", len(eff.words))
        cands: list[dict] = []
        if src == "schematic" and intent in SCHEMATIC_INTENTS:
            fields = _extract_schematic(intent, eff, lo, hi, llm,
                                        keep_alive=KEEP_ALIVE_BETWEEN, log=log)
            if fields:
                cands.append({"source": "schematic", "intent": intent,
                              "fields": fields, "style": style})
        elif src == "photo":
            prompt = _art_direction_prompt(route.get("query_en", ""))
            if prompt and run_diffusion:
                cands.append({"source": "diffusion", "prompt": prompt,
                              "seed": -1})
        elif src == "icon":
            cands.append({"source": "icon", "emoji": ""})
        # none ВСЕГДА доступен как вариант («лучше пусто, чем слоп»).
        cands.append({"source": "none"})
        clean = [c for c in (sanitize_candidate(c) for c in cands)
                 if c is not None]
        if not clean:
            clean = [{"source": "none"}]
        p["payload"]["candidates"] = clean
        p["payload"]["selected"] = 0
        p["payload"]["source"] = clean[0]["source"]
        if clean[0]["source"] == "schematic":
            p["payload"]["intent"] = clean[0].get("intent", "")
            p["payload"]["schematic_style"] = clean[0].get("style", style)
        if clean[0]["source"] != "none":
            materialized += 1
        p.pop("_route", None)                      # служебное наружу не уходит
    return materialized


# === §4 Tier 1: сопоставление папки ассетов юзера + эмодзи-фолбэк =================
# Подбор ассетов ПОСЛЕ детектора иллюстраций (§4): один LLM-вызов сопоставляет
# точки-иллюстрации (concept) с реальными файлами папки юзера; не нашли — эмодзи
# по статическому emoji_map.json (точное/частичное совпадение ключевых слов);
# совсем нет — asset_kind="none" (предложение без ассета, юзер выберет в UI).
ASSET_EXT = (".png", ".jpg", ".jpeg", ".webp")    # вайтлист индексации (§4)
ASSET_DESC_FILE = "descriptions.txt"              # «имяфайла: описание» (опц.)
ASSET_INDEX_MAX = 200                             # потолок индекса (один промпт)
# emoji_map.json кладёт P5-АССЕТЫ рядом с сабсетом Noto; загрузка ТЕРПИМА к
# отсутствию файла (в тестах мокаем через monkeypatch этой переменной/loader).
EMOJI_MAP_PATH = EMOJI_SVG_DIR.parent / "emoji_map.json"

# === SD-маршрутизация СТРОГО для фото-сцен (MONTAGE_V2 §5, c_diff §3) ============
# ВЗАИМОИСКЛЮЧАЮЩИЙ роутер (§1) уже решил источник; SD зовём ТОЛЬКО для photo.
# КРИТИЧНО (доказано кадром 05): схему/данные/числа в диффузию НЕЛЬЗЯ — кракозябры.
# Поэтому diagram→SD УДАЛЁН (был «text-free пятно без сущностей» — бесполезен, §0):
# данные идут в код-схему (schematic-кандидат), а не в SD. icon → эмодзи, не SD.
#   photo  -> SD напрямую (богатый арт-дирекшн-промпт + фото-суффикс, текст-негатив)
#   schematic/icon/прочее -> в SD НЕ уходят (False).
# Маршрутизатор лишь ПОМЕЧАЕТ photo-точку asset_kind="generate" + gen_prompt_en
# (арт-дирекшн): реальный запуск бинаря — этап images (serve.py) ПОСЛЕ unload
# Ollama (VRAM-менеджер §5). SD выключен/не настроен -> точка остаётся none.
_SD_STYLES_DIRECT = ("photo",)            # ТОЛЬКО photo идёт в диффузию (§5)

# Плоская схема (R4: optional молча выкидывается — ВСЕ поля required): на каждую
# точку модель возвращает point_idx + asset_filename ("" = нет совпадения).
_MATCH_SYSTEM = (
    "Ты — ассистент монтажёра. Тебе дают СПИСОК точек-иллюстраций (каждая — "
    "что нужно показать на экране) и СПИСОК доступных файлов-ассетов из папки "
    "пользователя (имя файла и, если есть, описание).\n\n"
    "Сопоставь каждой точке ОДИН подходящий файл из списка ассетов — тот, "
    "который по смыслу показывает то, что нужно в этой точке. Если ни один "
    "файл не подходит по смыслу — верни для этой точки пустую строку \"\" "
    "(не выдумывай файл, которого нет в списке).\n\n"
    "Для каждой точки верни:\n"
    "- point_idx — номер точки из входного списка;\n"
    "- asset_filename — ТОЧНОЕ имя файла из списка ассетов или пустая строка "
    "\"\", если подходящего нет."
)

_MATCH_USER_TMPL = (
    "Точки-иллюстрации:\n{points}\n\n"
    "Доступные ассеты:\n{assets}\n\n"
    "Верни JSON: {{\"matches\": [{{\"point_idx\": N, \"asset_filename\": "
    "\"…\"}}]}}"
)

_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point_idx": {"type": "integer"},
                    "asset_filename": {"type": "string"},
                },
                "required": ["point_idx", "asset_filename"],
            },
        }
    },
    "required": ["matches"],
}


def _safe_in_folder(folder: Path, name: str) -> Optional[Path]:
    """Файл ``name`` строго ВНУТРИ ``folder`` (path-traversal guard, паттерн
    _abs_path P2 / normcase-resolve watch-папок): отбиваем «..», абсолютные
    пути и симлинки наружу. Только имя файла (без подпапок) — индексируем
    плоско. Возврат — resolved Path или None."""
    name = (name or "").strip().strip("\"'")
    if not name or name in (".", ".."):
        return None
    # только базовое имя: режем любые разделители/компоненты пути.
    if "/" in name or "\\" in name or Path(name).name != name:
        return None
    try:
        root = folder.resolve()
        cand = (root / name).resolve()
    except OSError:
        return None
    if not cand.is_relative_to(root):    # вышли за пределы папки -> отказ
        return None
    if cand.suffix.lower() not in ASSET_EXT or not cand.is_file():
        return None
    return cand


def _read_descriptions(folder: Path) -> dict:
    """Опц. ``descriptions.txt`` («имяфайла: описание» по строке) -> словарь
    {имя_в_нижнем_регистре: описание}. Нет файла/битый — пустой словарь."""
    desc: dict = {}
    f = folder / ASSET_DESC_FILE
    try:
        text = f.read_text(encoding="utf-8")
    except OSError:
        return desc
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, _, d = line.partition(":")
        name, d = name.strip(), d.strip()
        if name and d:
            desc[name.lower()] = _trim_text(d, 120)
    return desc


def _index_assets(folder: Path) -> list[dict]:
    """Плоский индекс png/jpg/jpeg/webp папки (+ опц. описания). Скрытые файлы
    и не-файлы пропускаем; потолок ASSET_INDEX_MAX (один промпт)."""
    desc = _read_descriptions(folder)
    out: list[dict] = []
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return out
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            if not e.is_file() or e.suffix.lower() not in ASSET_EXT:
                continue
        except OSError:
            continue
        out.append({"filename": e.name, "desc": desc.get(e.name.lower(), "")})
        if len(out) >= ASSET_INDEX_MAX:
            break
    return out


def _load_emoji_map(path: Optional[Path] = None) -> dict:
    """Статический словарь {ключевое_слово: noto-имя} из emoji_map.json (его
    кладёт P5-АССЕТЫ). ТЕРПИМ к отсутствию/битому файлу — пустой словарь (тогда
    эмодзи-фолбэк просто не срабатывает, точка остаётся none). Ключи — в нижнем
    регистре; значения-непустые-строки.

    Принимает ДВА формата: (1) вшитый конверт ``{_version, _comment, map: {...}}``
    — концепты лежат под ключом ``map``; (2) плоский ``{концепт: noto}``
    (формат unit-тестов). Если есть словарь под ``map`` — разворачиваем его."""
    p = path if path is not None else EMOJI_MAP_PATH
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — нет файла / битый JSON / не словарь
        return {}
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("map"), dict):       # конверт {_version,_comment,map}
        data = data["map"]
    return {str(k).strip().lower(): str(v).strip()
            for k, v in data.items()
            if isinstance(k, str) and isinstance(v, str) and v.strip()}


def _emoji_for_concept(concept: str, emoji_map: dict) -> str:
    """Эмодзи (noto-имя) для concept по emoji_map: ТОЧНОЕ совпадение слова
    приоритетнее ЧАСТИЧНОГО (подстрока ключа в concept). Нет совпадения -> ""."""
    if not emoji_map:
        return ""
    c = concept.lower()
    words = set(re.findall(r"\w+", c, re.UNICODE))
    # ТОЛЬКО точное совпадение по ЦЕЛОМУ слову. Подстрочный матч убран намеренно:
    # на русской морфологии он даёт абсурд («ключ» ⊂ «отКЛЮЧение» → 🔑 для
    # «отключения моделей»). Нет точного слова — "" (точка уйдёт в SD или none,
    # лучше без картинки, чем нелепый эмодзи).
    for kw, name in emoji_map.items():
        if kw in words:
            return name
    return ""


def _route_generate(pl: dict) -> bool:
    """Решить, уходит ли точка в SD-генерацию (СТРОГО photo, §5).

    Мутирует payload: для ``style_hint=="photo"`` с валидным английским query
    ставит ``asset_kind="generate"`` + ``gen_prompt_en`` (БОГАТЫЙ арт-дирекшн-
    промпт, не голый query — c_diff §2б; фото-суффикс/негатив добавит imagegen).
    schematic/icon/прочее и пустой/русский query в SD НЕ уходят -> False (данные
    рисует код-схема, значок — эмодзи; «лучше пусто, чем слоп»). diagram→SD
    УДАЛЁН (кадр 05: кракозябры). Возврат — True, если точка помечена на SD."""
    if pl.get("style_hint") not in _SD_STYLES_DIRECT:
        return False                      # ТОЛЬКО photo (icon/schematic — не SD)
    prompt = _art_direction_prompt(pl.get("image_query_en"))
    if not prompt:
        return False                      # без английского предмета SD не зовём
    pl["asset_kind"] = "generate"
    pl["gen_prompt_en"] = prompt
    return True


def _legacy_sync_from_candidate(pl: dict) -> None:
    """Синхронизировать плоские поля payload (asset_kind/gen_prompt_en/asset_path)
    с ВЫБРАННЫМ кандидатом (обратная совместимость со стадией images и рендером).

    candidates[selected].source: diffusion → asset_kind="generate" + gen_prompt_en
    (его подберёт enrich_image_batch); stock → asset_kind="user" + asset_path;
    icon → asset_kind="emoji" (+emoji подберёт _attach_emoji); schematic/none →
    asset_kind="none" (schematic-PNG кладёт стадия images в кандидат, не в плоский
    asset_path — рендер берёт его через resolved_asset())."""
    cands = pl.get("candidates") or []
    sel = pl.get("selected", 0)
    if not isinstance(cands, list) or not (0 <= sel < len(cands)):
        return
    c = cands[sel]
    src = c.get("source")
    if src == "diffusion":
        pl["asset_kind"] = "generate"
        pl["gen_prompt_en"] = c.get("prompt", "")
        pl["gen_seed"] = c.get("seed", -1)
    elif src == "stock":
        pl["asset_kind"] = "user"
        pl["asset_path"] = c.get("asset_path", "")
    elif src == "icon":
        pl["asset_kind"] = "emoji"
    else:                                        # schematic / none
        pl["asset_kind"] = "none"


def _attach_emoji(pl: dict, emoji_map: dict) -> bool:
    """Подобрать эмодзи ТОЛЬКО для явного icon-кандидата (§1: эмодзи больше НЕ
    фолбэк иллюстрации). Ставит поле ``emoji`` icon-кандидату и плоскому payload.
    Возврат — True, если эмодзи нашёлся."""
    cands = pl.get("candidates") or []
    sel = pl.get("selected", 0)
    if not isinstance(cands, list) or not (0 <= sel < len(cands)):
        return False
    c = cands[sel]
    if c.get("source") != "icon":
        return False
    name = _emoji_for_concept(_trim_text(pl.get("concept"), 80), emoji_map)
    c["emoji"] = name
    pl["emoji"] = name
    return bool(name)


def match_user_assets(points: list, folder, llm, log: LogFn = _noop,
                      *, emoji_map_path: Optional[Path] = None,
                      generate: bool = False) -> int:
    """§4 (V2) — сток-кандидат из папки юзера + синк плоских полей (мутирует points).

    V2-логика (взаимоисключающий роутер уже выбрал источник, кандидаты собраны
    ``_build_candidates``): здесь только ДОБАВЛЯЕМ ``stock``-кандидат, когда файл
    из папки юзера смыслово матчит точку (ОДИН LLM-вызов, path-traversal отбит),
    и ставим его ЛУЧШИМ (``selected``=stock). Затем для icon-кандидата подбираем
    эмодзи (узко, только явный значок — §1: эмодзи больше НЕ фолбэк иллюстрации)
    и синхронизируем плоские поля payload с выбранным кандидатом (обратная
    совместимость рендера/стадии images). diffusion-кандидат метится на SD через
    плоский ``asset_kind="generate"`` (его генерит стадия images).

    ``generate`` — оставлен для совместимости сигнатуры; SD-маршрут теперь живёт
    в diffusion-кандидате (его наличие = разрешение на SD). Безопасен к сбоям:
    нет папки / пустой индекс / упавший LLM → без stock-кандидата (исключение
    наружу НЕ летит). Возврат — число точек с материальным (не-none) выбором."""
    emoji_map = _load_emoji_map(emoji_map_path)
    ill = [p for p in points
           if isinstance(p, dict) and p.get("type") in (ENR_IMAGE, ENR_ANIMATION)
           and isinstance(p.get("payload"), dict)]
    if not ill:
        return 0

    folder_p: Optional[Path] = None
    if folder:
        fp = Path(str(folder)).expanduser()
        if fp.is_dir():
            folder_p = fp
        else:
            log(f"  enrich: папка ассетов не найдена ({folder})")

    index = _index_assets(folder_p) if folder_p is not None else []
    by_idx: dict = {}                # point_idx -> filename (валидный, в папке)
    if index and folder_p is not None:
        points_txt = "\n".join(
            f"{i}: {_trim_text(p['payload'].get('concept'), 80)}"
            for i, p in enumerate(ill))
        assets_txt = "\n".join(
            (f"{a['filename']}: {a['desc']}" if a["desc"] else a["filename"])
            for a in index)
        user = _MATCH_USER_TMPL.format(points=points_txt, assets=assets_txt)
        try:
            data = llm.chat_json(_MATCH_SYSTEM, user, _MATCH_SCHEMA,
                                 keep_alive=0)
        except Exception as e:  # noqa: BLE001 — сбойный матчинг не валит пасс
            log(f"  enrich: матчинг ассетов пропущен ({e})")
            data = None
        raw = data.get("matches") if isinstance(data, dict) else None
        for r in (raw if isinstance(raw, list) else []):
            if not isinstance(r, dict):
                continue
            pi = r.get("point_idx")
            if not _valid_int(pi) or not (0 <= pi < len(ill)):
                continue
            cand = _safe_in_folder(folder_p, r.get("asset_filename") or "")
            if cand is not None:
                by_idx[pi] = str(cand)

    matched = 0
    for i, p in enumerate(ill):
        pl = p["payload"]
        cands = pl.get("candidates")
        if not isinstance(cands, list) or not cands:
            cands = [{"source": "none"}]
            pl["candidates"] = cands
        path = by_idx.get(i)
        if path:                                 # папка юзера → stock ЛУЧШИМ
            sc = sanitize_candidate({"source": "stock", "asset_path": path})
            if sc is not None:
                cands.insert(0, sc)
                pl["candidates"] = cands[:CANDIDATES_MAX]
                pl["selected"] = 0
                pl["source"] = "stock"
        _attach_emoji(pl, emoji_map)             # только для icon-кандидата
        _legacy_sync_from_candidate(pl)          # плоские поля ← выбранный кандидат
        sel = pl.get("selected", 0)
        cs = pl["candidates"]
        if 0 <= sel < len(cs) and cs[sel].get("source") != "none":
            matched += 1
    return matched


# === сборка ======================================================================
def detect_all(transcript: Optional[Transcript], cutlist: Optional[CutList],
               params: Optional[dict], llm, log: LogFn = _noop,
               on_progress=None, *, user_folder: Optional[str] = None,
               codegfx_style: str = SCHEMATIC_STYLE_DEF) -> list[EnrichItem]:
    """Полный LLM-пасс обогащения (§3): списки -> CTA -> иллюстрации -> ассеты.

    - выключенный в ``params["types"]`` тип НЕ вызывается вообще (§3.4);
    - keep_alive=300 между вызовами эпика, 0 на ПОСЛЕДНЕМ (паттерн clips);
    - per-window и per-детектор try/except: сбой не валит пасс (warnings в log);
    - прогресс по детекторам: lists 45 / cta 15 / illustrations 30 / assets 10;
    - этап assets (§4 Tier 0/1 + SD ТРЕК-2 §2): цепочка user(папка) ->
      generate(SD) -> emoji -> none. Если ``image_source`` ∈ {user_folder, auto}
      и ``user_folder`` задан — один LLM-вызов сопоставляет точки с папкой юзера;
      не нашли и ``image_source`` ∈ {generate, auto} — точка помечается на
      SD-генерацию (asset_kind="generate" + gen_prompt_en по style, реальный
      запуск бинаря — этап images в задаче ПОСЛЕ unload Ollama); иначе эмодзи-
      фолбэк по emoji_map.json (без сети). Не нашли — none. SD сам разрешён/нет
      решает serve.py (cfg.render.imagegen) — detect_all лишь МЕТИТ точки;
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
    # Этап ассетов запускаем только если есть иллюстрации (только им ставим
    # ассеты) и источник картинок их допускает (§4: auto/user_folder; emoji —
    # сразу фолбэк-эмодзи без LLM; для image_source=user_folder/auto папка
    # юзера индексируется LLM-вызовом).
    image_source = (params or {}).get("image_source", "auto")
    run_assets = run_ill                  # ассеты ставим только иллюстрациям

    eff = _build_eff(transcript, cutlist)
    if not eff.words:
        prog(1.0)
        return []

    # VRAM (§3.4): модель держим ТЁПЛОЙ ВЕСЬ пасс (детекторы → CTA-окна →
    # schematic-экстракторы _build_candidates → матч папки). Их точное число
    # заранее НЕизвестно (экстракторы зависят от вывода детектора иллюстраций),
    # поэтому НЕ угадываем «последний вызов» (старый n_calls промахивался:
    # экстракторы не считались → модель выгружалась посреди пасса и грузилась
    # заново). Вместо этого — ОДИН явный llm.unload() в конце пасса (ниже):
    # робастно и работает даже когда SD-этап пропущен.
    def ka_next() -> int:
        return KEEP_ALIVE_BETWEEN

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
    # Стадия КАНДИДАТОВ (§2): на каждый момент собираем 2-4 кандидата в
    # payload.candidates (schematic-кандидат ЭАГЕРНО узким экстрактором; diffusion
    # — для photo; none всегда). selected=0 = лучший. Реальный рендер кандидатов
    # (codegfx PNG / SD N=4) — стадия images задачи (serve.py) ПОСЛЕ unload Ollama.
    # diffusion-кандидат собираем только если image_source допускает SD
    # ({auto,generate}); при emoji/user_folder фото уходит в none (или сток ниже).
    if run_ill:
        try:
            _build_candidates(dicts, eff, llm, log,
                              default_style=codegfx_style,
                              run_diffusion=image_source in ("auto", "generate"))
        except Exception as e:  # noqa: BLE001 — сбойная стадия не валит пасс
            log(f"enrich: стадия кандидатов упала ({e}) — точки без кандидатов")
    # Этап assets (§4 V2, вес 10%): добавляем сток-кандидат из папки юзера (если
    # image_source ∈ {auto,user_folder}), узко подбираем эмодзи icon-кандидату,
    # синхронизируем плоские поля payload с выбранным кандидатом (обратная
    # совместимость рендера/стадии images). SD сам разрешён/нет решает serve.py.
    if run_assets:
        try:
            folder = user_folder if image_source in ("auto", "user_folder") \
                else None
            match_user_assets(dicts, folder, llm, log,
                              generate=image_source in ("auto", "generate"))
        except Exception as e:  # noqa: BLE001 — сбойный этап ассетов не валит пасс
            log(f"enrich: этап ассетов упал ({e}) — предложения без ассета")
    prog(1.0)

    # VRAM освобождаем ЯВНО в конце пасса (см. ka_next): единственная точка
    # выгрузки qwen3 — работает и когда SD-этап пропущен. best-effort: у боевого
    # OllamaClient unload() никогда не бросает; тестовые моки без unload()
    # просто пропускаем (getattr-гард).
    _unload = getattr(llm, "unload", None)
    if callable(_unload):
        try:
            _unload()
        except Exception as e:  # noqa: BLE001 — выгрузка best-effort
            log(f"enrich: не удалось выгрузить модель в конце пасса ({e})")

    items: list[EnrichItem] = []
    for d in dicts:
        it = item_from_dict(d, log)
        if it is not None:
            items.append(it)
    return items
