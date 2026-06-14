# -*- coding: utf-8 -*-
"""Валидатор/нормализатор data-схемы интента код-графики (MONTAGE_V2_PLAN §2/§8).

Backend (``enrich_llm``) кладёт ПЛОСКИЕ ``fields`` по интенту (форма зафиксирована
в ``vpipe/enrich.py`` «КОНТРАКТ КОД-ГРАФИКИ»). Этот модуль:

* проверяет наличие минимально-нужных полей по интенту (stat → numbers+labels;
  compare → rows + 2 колонки; tree → root + nodes; list/process → items/steps; …);
* применяет анти-слоп лимиты (≤6 строк/пунктов/шагов, ≤2 колонки compare и т.п.) —
  «лучше пусто, чем слоп» (§8). Слишком мало данных → ``None`` (кадр дропается);
* СТРОИТ из плоских fields вложенный ``PAYLOAD``, который ест ``engine.html``
  (движок собирает вложенность сам — см. контракт). Все строки чистятся
  (``_txt``: обрезка управляющих символов и лимит длины); финальное HTML-
  экранирование делает движок (esc()/textContent — XSS-safe), здесь — гигиена.

Контракт: ``build_payload(intent, fields, style) -> Optional[dict]``. Любая
негодность (неизвестный интент / нет обязательных полей / пустой результат) →
``None``. НИКОГДА не бросает на чужих данных.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# --- лимиты (анти-слоп, числа в КОДЕ — §8) ----------------------------------
MAX_ROWS = 6          # compare rows / list items / process steps / timeline / tree nodes / map points
MAX_STATS = 4         # stat: не больше 4 крупных чисел в ряд (вёрстка)
MAX_COLS = 2          # compare: ровно 2 колонки (Linux vs Windows)
TEXT_MAX = 200        # потолок любой строки (зеркало enrich.CAND_TEXT_MAX)
CODE_MAX = 1200       # код длиннее — обрезаем (терминал-карточка не безразмерна)

# Интенты таксономии (зеркало enrich.SCHEMATIC_INTENTS).
INTENTS = ("stat", "tree", "list", "timeline", "compare",
           "process", "quote", "code", "map", "callout")
STYLES = ("minimal", "neon", "business", "whiteboard")
STYLE_DEFAULT = "minimal"

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _txt(v: Any, limit: int = TEXT_MAX) -> str:
    """Скаляр → чистая строка: убрать управляющие символы, схлопнуть пробелы,
    ограничить длину. None/пусто → "". HTML-экранирование делает движок."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return ""                          # bool как текст бессмыслен
    if isinstance(v, float):
        # «96.0» → «96» (числа из речи целые), но «1.5» сохранить
        s = ("%g" % v)
    else:
        s = str(v)
    s = _CTRL.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > limit:
        s = s[:limit].rstrip()
    return s


def _items(v: Any) -> list:
    """Список или одиночка → список (не-списки оборачиваем; None → [])."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _as_dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def normalize_style(style: Optional[str]) -> str:
    s = (style or "").strip().lower()
    return s if s in STYLES else STYLE_DEFAULT


def _bar_pct(v: Any) -> Optional[int]:
    """bar → int 0..100 или None. Принимает число или строку «96» / «96%»."""
    if v is None:
        return None
    try:
        n = float(re.sub(r"[^\d.\-]", "", str(v)) or "nan")
    except ValueError:
        return None
    if n != n:                             # NaN
        return None
    return max(0, min(100, int(round(n))))


# --- построители payload по интенту -----------------------------------------
# Каждый: dict fields → dict PAYLOAD-фрагмент (без type/theme) ИЛИ None, если
# обязательных данных не хватает (анти-слоп).

def _b_stat(f: dict) -> Optional[dict]:
    raw = _items(f.get("stats"))
    stats = []
    for s in raw[:MAX_STATS]:
        s = _as_dict(s)
        val = _txt(s.get("value"))
        lab = _txt(s.get("label"))
        if not val:                        # число обязательно
            continue
        st = {"value": val, "label": lab}
        unit = _txt(s.get("unit"), 8)
        if unit:
            st["unit"] = unit
        stats.append(st)
    if not stats:                          # нет ни одного валидного числа → дроп
        return None
    out = {"stats": stats}
    bar = _bar_pct(f.get("bar"))
    if bar is not None:
        out["bar"] = bar
    return out


def _b_compare(f: dict) -> Optional[dict]:
    cols_raw = _items(f.get("columns"))
    if not cols_raw:                       # допускаем плоскую форму col_a/col_b
        a, b = _txt(f.get("col_a")), _txt(f.get("col_b"))
        cols_raw = [c for c in (a, b) if c]
    cols = [_txt(c, 40) for c in cols_raw if _txt(c, 40)][:MAX_COLS]
    if len(cols) < 2:                      # сравнение требует 2 колонки
        return None
    rows = []
    for r in _items(f.get("rows"))[:MAX_ROWS]:
        r = _as_dict(r)
        feat = _txt(r.get("feature"))
        if not feat:
            continue
        vals_raw = r.get("values")
        winner = _txt(r.get("winner"), 40).lower()
        if vals_raw is None:               # плоская форма: a/b (+winner)
            va, vb = _txt(r.get("a")), _txt(r.get("b"))
            if not (va or vb):
                continue
            cell_a = {"text": va}
            cell_b = {"text": vb}
            if winner in ("a", cols[0].lower()):
                cell_a["win"] = True; cell_b["lose"] = True
            elif winner in ("b", cols[1].lower()):
                cell_b["win"] = True; cell_a["lose"] = True
            vals = [cell_a, cell_b]
        else:                              # вложенная форма values:[{text,win,lose}]
            vals = []
            for v in _items(vals_raw)[:MAX_COLS]:
                v = _as_dict(v)
                cell = {"text": _txt(v.get("text") if "text" in v else v)}
                if v.get("win"):
                    cell["win"] = True
                if v.get("lose"):
                    cell["lose"] = True
                vals.append(cell)
            while len(vals) < 2:
                vals.append({"text": ""})
        rows.append({"feature": feat, "values": vals[:MAX_COLS]})
    if not rows:
        return None
    return {"columns": cols, "rows": rows, "tall": True}


def _b_tree(f: dict) -> Optional[dict]:
    root_label = _txt(f.get("root_label") or f.get("root"))
    nodes_raw = _items(f.get("nodes") or f.get("children"))
    # допускаем уже-вложенный root: {label, children:[...]}
    if isinstance(f.get("root"), dict):
        rd = f["root"]
        root_label = _txt(rd.get("label")) or root_label
        if not nodes_raw:
            nodes_raw = _items(rd.get("children"))
    if not root_label:
        return None
    children = []
    for n in nodes_raw[:MAX_ROWS]:
        n = _as_dict(n) if isinstance(n, dict) else {"label": n}
        lab = _txt(n.get("label") if isinstance(n, dict) else n)
        if not lab:
            continue
        node = {"label": lab}
        desc = _txt(n.get("desc"))
        if desc:
            node["desc"] = desc
        children.append(node)
    if not children:                       # дерево без веток — нечего рисовать
        return None
    return {"root": {"label": root_label, "children": children}, "tall": True}


def _b_list(f: dict) -> Optional[dict]:
    items = []
    for it in _items(f.get("items"))[:MAX_ROWS]:
        if isinstance(it, dict):
            text = _txt(it.get("text"))
            note = _txt(it.get("note"))
        else:
            text, note = _txt(it), ""
        if not text:
            continue
        d = {"text": text}
        if note:
            d["note"] = note
        items.append(d)
    if not items:
        return None
    return {"items": items, "ordered": bool(f.get("ordered"))}


def _b_process(f: dict) -> Optional[dict]:
    steps = []
    for s in _items(f.get("steps"))[:MAX_ROWS]:
        if isinstance(s, dict):
            title = _txt(s.get("title"))
            desc = _txt(s.get("desc"))
        else:
            title, desc = _txt(s), ""
        if not title:
            continue
        d = {"title": title}
        if desc:
            d["desc"] = desc
        steps.append(d)
    if len(steps) < 2:                     # «процесс» из 1 шага — не процесс
        return None
    return {"steps": steps}


def _b_timeline(f: dict) -> Optional[dict]:
    events = []
    for e in _items(f.get("events"))[:MAX_ROWS]:
        e = _as_dict(e)
        when = _txt(e.get("when"), 40)
        what = _txt(e.get("what"))
        if not (when or what):
            continue
        ev = {"when": when, "what": what}
        mark = _txt(e.get("mark"), 4)
        if mark:
            ev["mark"] = mark
        events.append(ev)
    if len(events) < 2:
        return None
    return {"events": events}


def _b_quote(f: dict) -> Optional[dict]:
    text = _txt(f.get("text"), TEXT_MAX)
    if not text:
        return None
    out = {"text": text}
    by = _txt(f.get("by"), 60)
    if by:
        out["by"] = by
    return out


def _b_code(f: dict) -> Optional[dict]:
    code = f.get("code")
    code = code if isinstance(code, str) else _txt(code, CODE_MAX)
    code = _CTRL.sub("", code.replace("\t", "    "))
    code = code[:CODE_MAX].rstrip()
    if not code.strip():
        return None
    return {"code": code}                  # движок esc() — код-текст как есть


def _b_callout(f: dict) -> Optional[dict]:
    term = _txt(f.get("term"), 80)
    definition = _txt(f.get("definition"), TEXT_MAX)
    if not term or not definition:
        return None
    return {"term": term, "definition": definition}


def _b_map(f: dict) -> Optional[dict]:
    pts = []
    for pt in _items(f.get("points"))[:MAX_ROWS]:
        if isinstance(pt, dict):
            label = _txt(pt.get("label"))
            note = _txt(pt.get("note"))
        else:
            label, note = _txt(pt), ""
        if not label:
            continue
        d = {"label": label}
        if note:
            d["note"] = note
        pts.append(d)
    if not pts:
        return None
    out = {"points": pts}
    region = _txt(f.get("region"), 60)
    if region:
        out["region"] = region
    return out


_BUILDERS = {
    "stat": _b_stat, "compare": _b_compare, "tree": _b_tree, "list": _b_list,
    "process": _b_process, "timeline": _b_timeline, "quote": _b_quote,
    "code": _b_code, "map": _b_map, "callout": _b_callout,
}


def build_payload(intent: str, fields: dict, style: str) -> Optional[dict]:
    """Плоские ``fields`` интента → вложенный PAYLOAD для ``engine.html``
    (или ``None``, если данных не хватает / интент неизвестен). Добавляет общий
    head (eyebrow/title/subtitle/source) и тему. НЕ бросает."""
    intent = (intent or "").strip().lower()
    builder = _BUILDERS.get(intent)
    if builder is None:
        return None
    if not isinstance(fields, dict):
        return None
    try:
        body = builder(fields)
    except Exception:                      # любая кривизна чужих данных → дроп
        return None
    if not body:
        return None

    payload: dict = {"type": intent, "theme": normalize_style(style)}
    payload.update(body)
    # общий head — необязательный, чистится теми же правилами
    for src_key, dst_key, lim in (("eyebrow", "eyebrow", 80),
                                  ("title", "title", 120),
                                  ("subtitle", "subtitle", TEXT_MAX),
                                  ("source", "source", 120)):
        val = _txt(fields.get(src_key), lim)
        if val:
            payload[dst_key] = val
    return payload
