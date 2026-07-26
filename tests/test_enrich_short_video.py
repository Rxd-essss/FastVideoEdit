# -*- coding: utf-8 -*-
"""Wave 4 (#71): очень короткие ролики (<~50 c) — масштабируемые чистые зоны и
пол бюджета плотности, чтобы план не отвергал АБСОЛЮТНО все оверлеи. Гард:
dur>=200 c сохраняет прежние фиксированные 30/20 c зоны байт-в-байт.

Timeline синтетический (паттерн tests/test_enrich_plan.py).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe import enrich                                              # noqa: E402
from vpipe.enrich import (EnrichItem, EnrichPlan, ImagePayload,       # noqa: E402
                          plan_render)
from vpipe.timeline import Timeline                                   # noqa: E402

W, H = 1920, 1080


def img(iid, t, score=70, dur=3.0, path="", pos="top_right", wf=0.32):
    return EnrichItem(
        id=iid, type=enrich.ENR_IMAGE, score=score, enabled=True,
        t_start=t, t_end=t + dur,
        payload=ImagePayload(asset_kind="user", asset_path=path, emoji="",
                             position=pos, width_frac=wf, kenburns=False))


def run(items, tl):
    plan = EnrichPlan(items=items)
    return plan, plan_render(plan, tl, None, None, W, H)


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "asset.png"
    p.write_bytes(b"\x89PNG fake")
    return str(p)


def test_short_video_scaled_clean_zones(png):
    # FAILS before the fix: 45-с ролик, fixed 30/20 c зоны перекрываются ->
    # ЛЮБОЙ оверлей off_limits. Масштаб (15%/10% = 6.75/4.5 c) оставляет окно.
    tl = Timeline([], duration=45)
    plan, re = run([img("a", 22.0, path=png)], tl)     # f0=22, f1=25 — в середине
    assert plan.items[0].status == enrich.ST_OK
    assert len(re.stills) == 1


def test_budget_floor_on_tiny_clip(png):
    # FAILS before the fix: 20-с ролик — clean зоны И budget int(2.5*20/60)=0
    # оба режут ВСЁ. Пол max(1,…) + масштаб зон пропускают один оверлей.
    tl = Timeline([], duration=20)
    plan, re = run([img("a", 9.0, path=png)], tl)      # f0=9, f1=12 — в [3, 18]
    assert plan.items[0].status == enrich.ST_OK
    assert len(re.stills) == 1


def test_long_video_keeps_fixed_clean_zones(png):
    # Гард-регрессия: dur>=200 c — зоны остаются 30/20 c (min(30,.15*dur)=30).
    # Оверлей на 25-й секунде по-прежнему в чистой голове -> off_limits.
    tl = Timeline([], duration=300)
    plan, re = run([img("a", 25.0, path=png)], tl)
    assert plan.items[0].status == enrich.ST_OFF_LIMITS
    assert "чистая зона" in plan.items[0].status_note
    assert re.stills == []
