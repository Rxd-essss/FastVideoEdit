"""C2 — мультиформат-рефрейм (/api/render formats): один клик — несколько
форматов вывода. CapCut делает рефрейм в облаке за Pro; у нас — локальный цикл
по форматам поверх той же vertical-механики с face-crop.

Покрывает:
 1. aspect_target — чистая геометрия: размеры цели от размеров источника
    (1920x1080 -> 9:16 = 608x1080; 1080x1920 -> 1:1 = 1080x1080; нечётные
    размеры клампятся к чётным; совпавший аспект -> None);
 2. _parse_formats — валидация поля formats + обратная совместимость:
    vertical=true -> ["9x16"], нет поля -> ["source"], мусор -> 400;
 3. _render_formats: 3 формата = 3 последовательных вызова render() с верными
    crop/target-фильтрами и именами <stem>.mp4 / <stem>_9x16.mp4 / <stem>_1x1.mp4;
 4. прогресс-агрегация percent = (i + frac)/N и stage «Формат i/N: …»
    (N=1 — легаси без префикса);
 5. совпавший с исходником аспект пропускается с пометкой «совпадает с
    исходным», render() не вызывается — и для 16:9 из 16:9 (aspect_target),
    и для 9:16 из вертикального 9:16-источника (явная сверка);
 6. сайдкары один раз: .srt/.vtt (имя без формат-суффикса) и chapters — только
    на первом удачном прогоне;
 7. устойчивость: упавший формат не валит остальные; все упали — задача падает;
    cancel между форматами сохраняет частичные результаты.
Без ffmpeg: vpipe.render.render замокан рекордером (FakeFF-граф).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import serve
import vpipe.facecrop as fc
from vpipe.config import ProfanityLists, load_config
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE, CutList, CutSegment,
                          Segment, Transcript, Word)

_SILENT = lambda *a, **k: None  # noqa: E731
_BASE_OPTS = {"subtitles": False, "chapters": False, "metadata": False}


# --- fixtures (паттерн test_serve_render) --------------------------------------
def _mk_session(tmp_path, *, duration: float = 20.0, width: int = 1920,
                height: int = 1080, cuts=()):
    n = int(duration)
    words = [Word(f"сл{i:02d}", i + 0.1, i + 0.9) for i in range(n)]
    tr = Transcript(language="ru", duration=duration, model="t", audio_hash="h",
                    segments=[Segment(0.0, duration,
                                      " ".join(w.word for w in words), words)])
    cl = CutList(source="fake.mp4", duration=duration, segments=list(cuts))
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        cfg=load_config("config.yaml"),
        inp=Path("fake.mp4"),
        media=SimpleNamespace(path="fake.mp4", duration=duration,
                              width=width, height=height, fps=30.0),
        ff=None, work_dir=work, out_dir=tmp_path / "out",
        matcher=ProfanityMatcher(ProfanityLists(roots=[], allow=[])),
        llm=None, transcript=tr, cutlist=cl)


def _patch_render(monkeypatch, *, progress=(0.0, 1.0), fail_on=()):
    """vpipe.render.render -> рекордер; шлёт on_progress, умеет «падать»."""
    calls: list[dict] = []

    def fake_render(ff, media, cl, cfg, out, work_dir, *, on_progress=None,
                    log=None, scale_h=None, fps=None, ass_path=None,
                    crop_filter=None, edge_fade=0.0):
        idx = len(calls)
        calls.append({"cl": cl, "out": Path(out), "ass_path": ass_path,
                      "crop_filter": crop_filter, "scale_h": scale_h,
                      "fps": fps, "subs": cfg.subtitles.enabled,
                      "chapters": cfg.chapters.enabled})
        if idx in fail_on:
            raise RuntimeError(f"render boom #{idx}")
        if on_progress:
            for p in progress:
                on_progress(p)
        return {"out": str(out), "encoder": "fake"}

    monkeypatch.setattr(serve.render_mod, "render", fake_render)
    return calls


def _center_half(monkeypatch):
    """Детерминированный face-crop: центр 0.5 без cv2/файла."""
    monkeypatch.setattr(serve.facecrop_mod, "detect_center",
                        lambda *a, **k: 0.5)


# --- 1. aspect_target: чистая геометрия ----------------------------------------
def test_aspect_target_spec_sizes():
    # 1920x1080 -> 9:16: высота сохраняется, ширина 1080*9/16=607.5 -> 608 (чёт.)
    assert fc.aspect_target(1920, 1080, (9, 16)) == (608, 1080)
    # 1080x1920 -> 1:1: ширина сохраняется, высота = ширине
    assert fc.aspect_target(1080, 1920, (1, 1)) == (1080, 1080)
    # 1920x1080 -> 1:1: высота сохраняется
    assert fc.aspect_target(1920, 1080, (1, 1)) == (1080, 1080)
    # 1080x1920 -> 16:9-кроп: ширина сохраняется, высота 1080*9/16 -> 608
    assert fc.aspect_target(1080, 1920, (16, 9)) == (1080, 608)
    # 4K
    assert fc.aspect_target(3840, 2160, (9, 16)) == (1216, 2160)


def test_aspect_target_noop_when_aspect_matches():
    assert fc.aspect_target(1920, 1080, (16, 9)) is None      # 16:9 из 16:9
    assert fc.aspect_target(1080, 1080, (1, 1)) is None       # квадрат из квадрата
    assert fc.aspect_target(1080, 1920, (9, 16)) is None      # 9:16 из 9:16


def test_aspect_target_odd_dims_clamped_even():
    # Нечётный источник: сохранённая сторона флорится к чётной, вычисленная —
    # к ближайшей чётной; ничего не вылезает за размеры источника.
    tw, th = fc.aspect_target(1919, 1079, (9, 16))
    assert (tw % 2, th % 2) == (0, 0)
    assert tw <= 1918 and th <= 1078
    assert (tw, th) == (606, 1078)
    tw, th = fc.aspect_target(1919, 1080, (16, 9))            # почти-16:9, нечёт.
    assert (tw, th) == (1918, 1078)
    assert tw % 2 == 0 and th % 2 == 0


def test_aspect_target_invalid():
    assert fc.aspect_target(0, 0, (1, 1)) is None
    assert fc.aspect_target(1920, 1080, (0, 1)) is None
    assert fc.aspect_target(1920, -5, (1, 1)) is None


# --- 2. _parse_formats: валидация + легаси-маппинг ------------------------------
def test_parse_formats_legacy_vertical_mapping():
    assert serve._parse_formats({}) == ["source"]
    assert serve._parse_formats({"vertical": False}) == ["source"]
    assert serve._parse_formats({"vertical": True}) == ["9x16"]
    # явные formats главнее легаси-флага
    assert serve._parse_formats({"vertical": True,
                                 "formats": ["source"]}) == ["source"]


def test_parse_formats_validation_and_dedupe():
    assert serve._parse_formats(
        {"formats": ["source", "9x16", "1x1", "16x9"]}) == \
        ["source", "9x16", "1x1", "16x9"]
    assert serve._parse_formats({"formats": ["9x16", "9x16"]}) == ["9x16"]
    for bad in ({"formats": []}, {"formats": "9x16"}, {"formats": ["4x3"]},
                {"formats": [None]}):
        with pytest.raises(HTTPException) as e:
            serve._parse_formats(bad)
        assert e.value.status_code == 400


# --- 3. три формата = три вызова render с верными crop/target и именами ---------
def test_three_formats_three_renders_crops_and_names(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)                       # 1920x1080 (16:9)
    calls = _patch_render(monkeypatch)
    _center_half(monkeypatch)
    res = serve._render_formats(s, dict(_BASE_OPTS),
                                ["source", "9x16", "1x1"], _SILENT, _SILENT)
    assert len(calls) == 3
    # source: без кропа, имя без суффикса
    assert calls[0]["crop_filter"] is None
    assert calls[0]["out"].name == "fake.mp4"
    # 9x16: прежний vertical-путь, каноничная цель Shorts 1080x1920, face-crop
    f916 = calls[1]["crop_filter"]
    assert f916 == fc.vertical_filter(1920, 1080, 0.5, (1080, 1920))
    assert "scale=1080:1920" in f916 and "ih*1080/1920" in f916
    assert calls[1]["out"].name == "fake_9x16.mp4"
    # 1x1: цель от источника (1080x1080), горизонтальный face-crop
    f11 = calls[2]["crop_filter"]
    assert f11 == fc.vertical_filter(1920, 1080, 0.5, (1080, 1080))
    assert "scale=1080:1080" in f11
    assert calls[2]["out"].name == "fake_1x1.mp4"
    # массив результатов по форматам + merged-верх (первый удачный прогон)
    fmts = res["formats"]
    assert [e["format"] for e in fmts] == ["source", "9x16", "1x1"]
    assert all(e["ok"] and not e["skipped"] for e in fmts)
    assert fmts[1]["mp4"].endswith("fake_9x16.mp4")
    assert res["mp4"].endswith("fake.mp4")          # верхний уровень = source


def test_portrait_source_1x1_and_16x9_cover_crop(monkeypatch, tmp_path):
    # Портретный источник 1080x1920: 1:1 и 16:9 — вертикальный центр-кроп
    # (cover+crop, без горизонтального face-окна — некуда двигать).
    s = _mk_session(tmp_path, width=1080, height=1920)
    calls = _patch_render(monkeypatch)
    _center_half(monkeypatch)
    serve._render_formats(s, dict(_BASE_OPTS), ["1x1", "16x9"],
                          _SILENT, _SILENT)
    assert len(calls) == 2
    assert "scale=1080:1080:force_original_aspect_ratio=increase" in \
        calls[0]["crop_filter"]
    assert "crop=1080:1080" in calls[0]["crop_filter"]
    assert calls[0]["out"].name == "fake_1x1.mp4"
    assert "scale=1080:608:force_original_aspect_ratio=increase" in \
        calls[1]["crop_filter"]
    assert "crop=1080:608" in calls[1]["crop_filter"]
    assert calls[1]["out"].name == "fake_16x9.mp4"


def test_custom_filename_gets_format_suffix(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch)
    _center_half(monkeypatch)
    serve._render_formats(s, {**_BASE_OPTS, "filename": "custom"},
                          ["9x16"], _SILENT, _SILENT)
    assert calls[0]["out"].name == "custom_9x16.mp4"


# --- 4. прогресс (i+frac)/N и stage «Формат i/N: …» -----------------------------
def test_progress_aggregation_and_stage_prefix(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    _patch_render(monkeypatch, progress=(0.0, 0.5, 1.0))
    _center_half(monkeypatch)
    percents: list[float] = []
    stages: list[str] = []
    serve._render_formats(s, dict(_BASE_OPTS), ["source", "9x16"],
                          percents.append, stages.append)
    # формат 0: 0, .25, .5; формат 1: .5, .75, 1.0 — монотонно, точно (i+f)/N
    assert percents == [0.0, 0.25, 0.5, 0.5, 0.75, 1.0]
    assert any(m.startswith("Формат 1/2: Исходный — ") for m in stages)
    assert any(m.startswith("Формат 2/2: 9:16 — ") for m in stages)


def test_single_format_keeps_legacy_plain_stages(monkeypatch, tmp_path):
    # N=1 — легаси бит-в-бит: без префикса «Формат 1/1», percent = frac.
    s = _mk_session(tmp_path)
    _patch_render(monkeypatch, progress=(0.5, 1.0))
    percents: list[float] = []
    stages: list[str] = []
    res = serve._render_formats(s, dict(_BASE_OPTS), ["source"],
                                percents.append, stages.append)
    assert percents == [0.5, 1.0]
    assert all(not m.startswith("Формат") for m in stages)
    assert res["mp4"].endswith("fake.mp4")
    assert [e["format"] for e in res["formats"]] == ["source"]


# --- 5. совпавший аспект пропускается («совпадает с исходным», без дубля) -------
def test_16x9_from_16x9_source_skipped_with_note(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)                       # 1920x1080 — уже 16:9
    calls = _patch_render(monkeypatch)
    _center_half(monkeypatch)
    res = serve._render_formats(s, dict(_BASE_OPTS), ["source", "16x9"],
                                _SILENT, _SILENT)
    assert len(calls) == 1                          # дубль НЕ рендерился
    assert calls[0]["out"].name == "fake.mp4"
    skip = res["formats"][1]
    assert skip["format"] == "16x9" and skip["skipped"] is True
    assert "совпадает с исходным" in skip["note"]
    assert "mp4" not in skip


def test_9x16_from_9x16_source_skipped_with_note(monkeypatch, tmp_path):
    # K3: 9:16 из вертикального 9:16-источника — такой же дубль, как 16:9 из
    # 16:9 — пропускается с той же пометкой (подпись UI обещает пропуск).
    s = _mk_session(tmp_path, width=1080, height=1920)
    calls = _patch_render(monkeypatch)
    _center_half(monkeypatch)
    res = serve._render_formats(s, dict(_BASE_OPTS), ["source", "9x16"],
                                _SILENT, _SILENT)
    assert len(calls) == 1                          # дубль НЕ рендерился
    assert calls[0]["out"].name == "fake.mp4"
    skip = res["formats"][1]
    assert skip["format"] == "9x16" and skip["skipped"] is True
    assert "совпадает с исходным" in skip["note"]
    assert "mp4" not in skip


def test_9x16_near_but_not_exact_aspect_still_renders(monkeypatch, tmp_path):
    # Правило integer-точное (как в aspect_target): 1080x1918 — НЕ 9:16,
    # рендерим; неизвестные размеры (0x0) совпадением тоже не считаются.
    s = _mk_session(tmp_path, width=1080, height=1918)
    calls = _patch_render(monkeypatch)
    _center_half(monkeypatch)
    serve._render_formats(s, dict(_BASE_OPTS), ["9x16"], _SILENT, _SILENT)
    assert len(calls) == 1
    assert calls[0]["out"].name == "fake_9x16.mp4"

    s0 = _mk_session(tmp_path, width=0, height=0)
    calls0 = _patch_render(monkeypatch)
    serve._render_formats(s0, dict(_BASE_OPTS), ["9x16"], _SILENT, _SILENT)
    assert len(calls0) == 1                         # 0x0 — рендерим, как раньше


def test_all_formats_skipped_returns_formats_only(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch)
    res = serve._render_formats(s, dict(_BASE_OPTS), ["16x9"],
                                _SILENT, _SILENT)
    assert calls == []
    assert res["formats"][0]["skipped"] is True
    assert "mp4" not in res                         # рендера не было — честно


# --- 6. сайдкары один раз: сабы без формат-суффикса, главы на первом прогоне ----
def test_sidecars_generated_once_with_plain_stem(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch)
    _center_half(monkeypatch)
    subs_calls: list[Path] = []

    def fake_subs(tr, removed, scfg, mask, matcher, base, log=None):
        subs_calls.append(Path(base))
        return {"srt": str(base) + ".srt", "vtt": str(base) + ".vtt", "cues": 3}

    chap_calls: list[Path] = []

    def fake_chapters(tr, removed, ccfg, out_path, **kw):
        chap_calls.append(Path(out_path))
        return {"path": str(out_path), "chapters": 2}

    monkeypatch.setattr(serve.subs_mod, "generate", fake_subs)
    monkeypatch.setattr(serve.chapters_mod, "generate", fake_chapters)
    res = serve._render_formats(
        s, {"subtitles": True, "chapters": True, "metadata": False},
        ["9x16", "1x1"], _SILENT, _SILENT)
    # сабы/главы — ровно один раз, на ПЕРВОМ формате
    assert len(subs_calls) == 1 and len(chap_calls) == 1
    assert subs_calls[0].name == "fake"             # без _9x16: кроп сабы не меняет
    assert calls[0]["subs"] is True and calls[1]["subs"] is False
    assert calls[1]["chapters"] is False
    # merged-верх несёт сайдкары первого прогона
    assert res["srt"].endswith("fake.srt")
    assert res["n_chapters"] == 2


# --- 7. устойчивость: падения и cancel ------------------------------------------
def test_failed_format_does_not_kill_the_rest(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch, fail_on={0})
    _center_half(monkeypatch)
    res = serve._render_formats(s, dict(_BASE_OPTS), ["source", "9x16"],
                                _SILENT, _SILENT)
    assert len(calls) == 2                          # второй формат отрендерился
    e0, e1 = res["formats"]
    assert e0["ok"] is False and "boom" in e0["error"]
    assert e1["ok"] is True
    # merged-верх — от первого УДАЧНОГО (9x16), сайдкары не потеряны
    assert res["mp4"].endswith("fake_9x16.mp4")


def test_all_formats_failed_raises(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    _patch_render(monkeypatch, fail_on={0, 1})
    _center_half(monkeypatch)
    with pytest.raises(RuntimeError, match="ни один формат"):
        serve._render_formats(s, dict(_BASE_OPTS), ["source", "9x16"],
                              _SILENT, _SILENT)


def test_single_format_failure_propagates_legacy(monkeypatch, tmp_path):
    # N=1 — прежний контракт: ошибка рендера = ошибка задачи (без обёртки).
    s = _mk_session(tmp_path)
    _patch_render(monkeypatch, fail_on={0})
    with pytest.raises(RuntimeError, match="boom"):
        serve._render_formats(s, dict(_BASE_OPTS), ["source"],
                              _SILENT, _SILENT)


def test_cancel_between_formats_keeps_partial_results(monkeypatch, tmp_path):
    s = _mk_session(tmp_path)
    calls = _patch_render(monkeypatch)
    _center_half(monkeypatch)
    cancelled = {"v": False}

    def fake_render_then_cancel(*a, **k):
        cancelled["v"] = True                       # «cancel» после 1-го формата
        calls.append({"out": Path(a[4])})
        return {"out": str(a[4]), "encoder": "fake"}

    monkeypatch.setattr(serve.render_mod, "render", fake_render_then_cancel)
    res = serve._render_formats(s, dict(_BASE_OPTS),
                                ["source", "9x16", "1x1"], _SILENT, _SILENT,
                                is_cancelled=lambda: cancelled["v"])
    assert len(calls) == 1                          # цикл остановлен между форматами
    assert [e["format"] for e in res["formats"]] == ["source"]
    assert res["mp4"].endswith("fake.mp4")          # частичный результат сохранён
