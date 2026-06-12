"""C1 — пресеты стилей вшитых караоке-субтитров (CAPTION_PRESETS в serve.py).

Закрывает контракт фичи без ffmpeg/рендера:
  * ровно 4 пресета, уникальные ключи, человеческие label/hint;
  * каждый ``style`` — ПОЛНЫЙ набор полей AssStyleCfg (кроме ``enabled``):
    точное равенство множества ключей (extra="ignore" у pydantic молча съел бы
    опечатку) + парсинг моделью без «починки» значений валидатором;
  * write_ass не падает ни на одном пресете и пишет ожидаемое ключевое поле
    стиля (по одному на пресет: Alignment/PrimaryColour/OutlineColour/Font);
  * _resolve_render_opts применяет пресет ЦЕЛИКОМ — включая новые
    outline/shadow/margin_v, которых нет среди сырых полей UI (+ клампы и
    толерантность к мусору);
  * GET /api/state отдаёт caption_presets (источник истины для фронта).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                                  # noqa: E402
from vpipe.config import (AssStyleCfg, MaskingCfg, ProfanityLists,  # noqa: E402
                          load_config)
from vpipe.detect.profanity import ProfanityMatcher           # noqa: E402
from vpipe.models import Word                                 # noqa: E402
from vpipe.subtitles import Cue, write_ass                    # noqa: E402

# Полный стилевой набор = все поля модели, кроме ортогонального тумблера burn.
STYLE_FIELDS = set(AssStyleCfg.model_fields) - {"enabled"}


# --- форма списка -------------------------------------------------------------
def test_four_presets_unique_keys_and_labels():
    assert len(serve.CAPTION_PRESETS) == 4
    keys = [p["key"] for p in serve.CAPTION_PRESETS]
    assert len(set(keys)) == 4
    for p in serve.CAPTION_PRESETS:
        assert p["label"].strip() and p["hint"].strip()


def test_preset_styles_are_complete_assstylecfg_sets():
    for p in serve.CAPTION_PRESETS:
        style = p["style"]
        # точное множество ключей — единственная защита от опечаток при
        # extra="ignore"; «полный набор» — контракт фронта (шлёт style целиком)
        assert set(style) == STYLE_FIELDS, p["key"]
        cfg = AssStyleCfg(**style)
        for k, v in style.items():
            # значения проходят валидацию КАК ЕСТЬ (без коэрции в другой тип)
            assert getattr(cfg, k) == v, (p["key"], k)


# --- write_ass на каждом пресете + 1 ключевое поле ------------------------------
def _style_line(txt: str) -> list[str]:
    line = next(l for l in txt.splitlines() if l.startswith("Style: Default,"))
    return line.split(",")
    # f[1]=Font f[2]=Size f[3]=Primary f[4]=Secondary f[5]=OutlineColour …
    # хвост шаблона: …,{align},40,40,{margin_v},1 -> f[-5]=Align f[-2]=MarginV


@pytest.mark.parametrize("key,expect", [
    # классика: снизу, «спетое» слово жёлтым (Primary = karaoke_color)
    ("classic", {"align": "2", "primary": "&H0000D4FF"}),
    # неон: бирюзовая подсветка, приподнят над низом (MarginV из пресета)
    ("neon",    {"align": "2", "primary": "&H00FFE500", "margin_v": "160"}),
    # минимал: полупрозрачная обводка-плашка
    ("minimal", {"align": "2", "outline_c": "&H78000000"}),
    # крупный: по центру кадра, Impact
    ("bold",    {"align": "5", "font": "Impact", "size": "78"}),
])
def test_write_ass_per_preset(tmp_path, key, expect):
    p = next(x for x in serve.CAPTION_PRESETS if x["key"] == key)
    style = AssStyleCfg(**p["style"])
    out = tmp_path / f"{key}.ass"
    write_ass([Cue(0.0, 2.0, "пример субтитра")], out, style,
              karaoke=style.karaoke,
              words=[Word("пример", 0.0, 1.0), Word("субтитра", 1.0, 2.0)],
              matcher=ProfanityMatcher(ProfanityLists()), mask=MaskingCfg(),
              play_res=(1080, 1920))
    txt = out.read_text(encoding="utf-8-sig")
    f = _style_line(txt)
    if "align" in expect:
        assert f[-5] == expect["align"]
    if "primary" in expect:
        assert f[3] == expect["primary"]
    if "outline_c" in expect:
        assert f[5] == expect["outline_c"]
    if "font" in expect:
        assert f[1] == expect["font"]
    if "size" in expect:
        assert f[2] == expect["size"]
    if "margin_v" in expect:
        assert f[-2] == expect["margin_v"]
    # караоке-пресеты реально производят \k-строки (а не голый текст)
    assert "Dialogue:" in txt and "\\k" in txt


# --- _resolve_render_opts: пресет применяется целиком ---------------------------
def _mk_session(tmp_path):
    return SimpleNamespace(
        cfg=load_config("config.yaml"),
        inp=Path("fake.mp4"),
        media=SimpleNamespace(duration=20.0, width=1920, height=1080, fps=30.0),
        out_dir=tmp_path / "out")


@pytest.mark.parametrize("p", serve.CAPTION_PRESETS, ids=lambda p: p["key"])
def test_resolve_render_opts_applies_full_preset(tmp_path, p):
    s = _mk_session(tmp_path)
    cfg, *_ = serve._resolve_render_opts(
        s, {"burn_subtitles": True, "burn_style": dict(p["style"]),
            "subtitles": False, "chapters": False, "metadata": False})
    b = cfg.subtitles.burn
    assert b.enabled is True
    for k, v in p["style"].items():
        assert getattr(b, k) == v, (p["key"], k)


def test_resolve_render_opts_clamps_and_ignores_junk(tmp_path):
    s = _mk_session(tmp_path)
    before = s.cfg.subtitles.burn
    cfg, *_ = serve._resolve_render_opts(
        s, {"burn_subtitles": True,
            "burn_style": {"outline": 99, "shadow": -5, "margin_v": "мусор"},
            "subtitles": False, "chapters": False, "metadata": False})
    b = cfg.subtitles.burn
    assert b.outline == 10.0                      # кламп сверху
    assert b.shadow == 0.0                        # кламп снизу
    assert b.margin_v == before.margin_v          # мусор -> значение из конфига


# --- /api/state отдаёт пресеты (источник истины фронта) -------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    cfg = load_config("config.yaml")
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.out_dir = str(tmp_path / "out")
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", cfg.paths.out_dir)
    media = SimpleNamespace(duration=10.0, fps=25.0, width=1920, height=1080)
    sess = SimpleNamespace(
        inp=Path("clip.mp4"), media=media, audio_hash="a" * 12,
        transcript=None, cutlist=None, llm=None, cfg=cfg,
        out_dir=Path(cfg.paths.out_dir),
        task={"name": None, "running": False})
    Path(cfg.paths.out_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(serve, "SESSION", sess)
    return TestClient(serve.app)


def test_state_exposes_caption_presets(client):
    j = client.get("/api/state").json()
    assert j["caption_presets"] == serve.CAPTION_PRESETS
    # форма каждой записи — ровно то, что ждёт фронт (key/label/hint/style)
    for p in j["caption_presets"]:
        assert set(p) == {"key", "label", "hint", "style"}
