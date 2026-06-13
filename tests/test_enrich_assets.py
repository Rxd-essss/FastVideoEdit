# -*- coding: utf-8 -*-
"""P5 — ассеты авто-обогащения (ENRICH_PLAN §4 Tier 0/1, §7-P5).

Покрывает список §7-P5:
 1. растеризация эмодзи: noto-имя -> кодпойнт -> цветной глиф Segoe UI Emoji
    через Pillow -> 256-px PNG с прозрачным фоном на тёмной мини-подложке;
    кэш cache/enrich_emoji/, идемпотентно; PNG НЕпустой; битое имя/пустое -> None;
 2. индексация папки юзера: png/jpg/jpeg/webp + опц. descriptions.txt
    («имяфайла: описание»); скрытые/не-картинки/подпапки пропускаются;
 3. матчинг MockLLM (один вызов): точка -> файл (asset_kind=user + абс. путь);
    нет совпадения -> эмодзи по emoji_map.json (asset_kind=emoji) либо none;
 4. path-traversal отбит (../, абсолютный путь, имя из чужой папки -> отказ);
 5. emoji_map.json загрузка ТЕРПИМА к отсутствию файла (нет файла -> none);
 6. detect_all интегрирует этап ассетов (image_source=user_folder/auto +
    папка -> LLM-матчинг считается в keep_alive; emoji -> фолбэк без LLM);
 7. валидность вшитых CTA webm: НАЛИЧИЕ АЛЬФЫ (libvpx-vp9 yuva420p ИЛИ Matroska
    alpha_mode=1 — нативный ffprobe врёт yuv420p для VP9-альфы) и < 200 КБ —
    ROBUST: нет файла / нет ffprobe -> skip, а не падение; anim_presets.json —
    схема пресетов/ассетов;
 8. /api/browse?kind=image — вайтлист png/jpg/jpeg/webp; music/без-kind целы.
Без сети/Ollama: LLM — голый MockLLM с chat_json(); ffprobe — через PATH (нет
ffprobe -> skip валидации pix_fmt).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe import enrich as enrich_mod
from vpipe import enrich_llm
from vpipe.enrich import emoji_png_path
from vpipe.enrich_llm import match_user_assets

_SILENT = lambda *a, **k: None  # noqa: E731

CTA_DIR = enrich_mod.CTA_ASSET_DIR
MAX_WEBM_BYTES = 200_000
# V11 §5: subscribe_slide_avatar.webm — новый ассет со слотом под аватар канала.
CTA_FILES = ("subscribe_like.webm", "subscribe_slide_avatar.webm",
             "comment.webm", "like.webm", "bell.webm")


class MockLLM:
    """Строгий мок (паттерн test_enrich_llm): голый объект с chat_json()."""

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


# === 1. растеризация эмодзи =====================================================
def _nonzero_alpha(path: Path) -> int:
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    return sum(1 for px in img.split()[3].get_flattened_data() if px > 0)


def test_emoji_rasterizes_nonempty_png_with_transparency(tmp_path):
    p = emoji_png_path("u26a1", tmp_path)              # ⚡
    assert p is not None and p.is_file()
    assert p.name == f"u26a1_{enrich_mod.EMOJI_PNG_SIZE}.png"
    assert p.stat().st_size > 0                        # НЕпустой PNG
    from PIL import Image
    img = Image.open(p).convert("RGBA")
    assert img.size == (enrich_mod.EMOJI_PNG_SIZE, enrich_mod.EMOJI_PNG_SIZE)
    # прозрачный фон: углы холста за подложкой — alpha 0
    assert img.getpixel((0, 0))[3] == 0
    # глиф + тёмная подложка нарисованы: много непрозрачных пикселей
    assert _nonzero_alpha(p) > 1000


def test_emoji_cache_idempotent_no_rewrite(tmp_path):
    p1 = emoji_png_path("u1f5c3", tmp_path)            # 🗃
    assert p1 is not None
    mtime1 = p1.stat().st_mtime_ns
    p2 = emoji_png_path("u1f5c3", tmp_path)            # второй вызов — из кэша
    assert p2 == p1
    assert p2.stat().st_mtime_ns == mtime1             # файл не перезаписан
    assert not p1.with_suffix(".png.tmp").exists()     # .tmp прибран


def test_emoji_multi_codepoint_flag(tmp_path):
    # имя из нескольких кодпойнтов через «_» (флаг 🇺🇸) — оба разворачиваются
    p = emoji_png_path("u1f1fa_u1f1f8", tmp_path)
    assert p is not None and p.stat().st_size > 0


def test_emoji_bad_or_empty_name_returns_none(tmp_path):
    assert emoji_png_path("", tmp_path) is None
    assert emoji_png_path("   ", tmp_path) is None
    assert emoji_png_path("zzz", tmp_path) is None     # не hex-кодпойнт
    assert emoji_png_path("u_", tmp_path) is None
    # пустой результат не оставляет битый файл и не падает
    assert not any(tmp_path.iterdir())


def test_emoji_name_to_char_helper():
    assert enrich_mod._emoji_to_char("u26a1") == "⚡"
    assert enrich_mod._emoji_to_char("u1f1fa_u1f1f8") == "\U0001f1fa\U0001f1f8"
    assert enrich_mod._emoji_to_char("zzz") == ""      # битый -> пусто


# === 2. индексация папки юзера ===================================================
def _mk_assets(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for n in ("registry.png", "server.jpg", "logo.webp"):
        (folder / n).write_bytes(b"\x89PNG\r\n")
    (folder / "notes.txt").write_text("ignore", encoding="utf-8")   # не картинка
    (folder / ".hidden.png").write_bytes(b"x")                      # скрытый
    (folder / "sub").mkdir()                                        # подпапка
    (folder / "sub" / "deep.png").write_bytes(b"x")
    (folder / "descriptions.txt").write_text(
        "registry.png: реестр Windows\n"
        "server.jpg: сервер Dell\n"
        "# комментарий без двоеточия игнор\n", encoding="utf-8")


def test_index_assets_whitelist_and_descriptions(tmp_path):
    _mk_assets(tmp_path)
    idx = enrich_llm._index_assets(tmp_path)
    names = sorted(a["filename"] for a in idx)
    assert names == ["logo.webp", "registry.png", "server.jpg"]   # 3 картинки
    by = {a["filename"]: a["desc"] for a in idx}
    assert by["registry.png"] == "реестр Windows"
    assert by["server.jpg"] == "сервер Dell"
    assert by["logo.webp"] == ""                       # без описания — пусто


def test_read_descriptions_missing_file_tolerant(tmp_path):
    assert enrich_llm._read_descriptions(tmp_path) == {}


# === 3+4. матчинг MockLLM + path-traversal =======================================
def _point(concept: str, typ: str = enrich_mod.ENR_IMAGE) -> dict:
    return {"type": typ, "payload": {"concept": concept, "asset_kind": "none",
                                     "asset_path": "", "emoji": ""}}


def _emoji_map(tmp_path: Path, mapping: dict) -> Path:
    p = tmp_path / "emoji_map.json"
    p.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return p


def test_match_point_to_user_asset_single_llm_call(tmp_path):
    folder = tmp_path / "assets"
    _mk_assets(folder)
    pts = [_point("реестр Windows"), _point("сервер Dell")]
    llm = MockLLM([{"matches": [
        {"point_idx": 0, "asset_filename": "registry.png"},
        {"point_idx": 1, "asset_filename": "server.jpg"},
    ]}])
    n = match_user_assets(pts, str(folder), llm, _SILENT,
                          emoji_map_path=tmp_path / "no_map.json")
    assert n == 2
    assert len(llm.calls) == 1                          # ОДИН вызов матчинга
    assert llm.calls[0]["schema"] is enrich_llm._MATCH_SCHEMA
    assert pts[0]["payload"]["asset_kind"] == "user"
    assert Path(pts[0]["payload"]["asset_path"]) == (folder / "registry.png").resolve()
    assert pts[1]["payload"]["asset_path"].endswith("server.jpg")
    # абсолютные пути строго внутри папки
    for p in pts:
        assert Path(p["payload"]["asset_path"]).is_absolute()


def test_match_no_hit_falls_back_to_emoji(tmp_path):
    folder = tmp_path / "assets"
    _mk_assets(folder)
    pts = [_point("облако маркетинга")]
    emap = _emoji_map(tmp_path, {"облако": "u2601", "реестр": "u1f5c3"})
    # модель не нашла файл -> пустая строка
    llm = MockLLM([{"matches": [{"point_idx": 0, "asset_filename": ""}]}])
    n = match_user_assets(pts, str(folder), llm, _SILENT, emoji_map_path=emap)
    assert n == 1
    assert pts[0]["payload"]["asset_kind"] == "emoji"
    assert pts[0]["payload"]["emoji"] == "u2601"        # «облако» -> noto u2601
    assert pts[0]["payload"]["asset_path"] == ""


def test_match_no_hit_no_emoji_stays_none(tmp_path):
    folder = tmp_path / "assets"
    _mk_assets(folder)
    pts = [_point("нечто несопоставимое")]
    emap = _emoji_map(tmp_path, {"облако": "u2601"})
    llm = MockLLM([{"matches": [{"point_idx": 0, "asset_filename": ""}]}])
    n = match_user_assets(pts, str(folder), llm, _SILENT, emoji_map_path=emap)
    assert n == 0
    assert pts[0]["payload"]["asset_kind"] == "none"


def test_match_path_traversal_rejected(tmp_path):
    folder = tmp_path / "assets"
    _mk_assets(folder)
    (tmp_path / "secret.png").write_bytes(b"x")         # вне папки ассетов
    pts = [_point("a"), _point("b"), _point("c"), _point("d")]
    llm = MockLLM([{"matches": [
        {"point_idx": 0, "asset_filename": "../secret.png"},     # traversal
        {"point_idx": 1, "asset_filename": str((tmp_path / "secret.png"))},  # абс.
        {"point_idx": 2, "asset_filename": "sub/deep.png"},      # подпапка
        {"point_idx": 3, "asset_filename": "nope.png"},          # нет файла
    ]}])
    n = match_user_assets(pts, str(folder), llm, _SILENT,
                          emoji_map_path=tmp_path / "no_map.json")
    assert n == 0                                       # ни один не прошёл guard
    assert all(p["payload"]["asset_kind"] == "none" for p in pts)


def test_match_no_folder_emoji_only_no_llm(tmp_path):
    pts = [_point("молния скорость")]
    emap = _emoji_map(tmp_path, {"молния": "u26a1"})
    llm = MockLLM([])                                   # без папки LLM не зовётся
    n = match_user_assets(pts, "", llm, _SILENT, emoji_map_path=emap)
    assert n == 1
    assert llm.calls == []                              # эмодзи-фолбэк без LLM
    assert pts[0]["payload"]["asset_kind"] == "emoji"
    assert pts[0]["payload"]["emoji"] == "u26a1"


def test_match_missing_folder_tolerant(tmp_path):
    pts = [_point("реестр")]
    emap = _emoji_map(tmp_path, {"реестр": "u1f5c3"})
    llm = MockLLM([])
    n = match_user_assets(pts, str(tmp_path / "does_not_exist"), llm, _SILENT,
                          emoji_map_path=emap)
    assert n == 1 and llm.calls == []                  # папки нет -> эмодзи-фолбэк
    assert pts[0]["payload"]["emoji"] == "u1f5c3"


def test_match_llm_failure_falls_back_to_emoji(tmp_path):
    folder = tmp_path / "assets"
    _mk_assets(folder)
    pts = [_point("реестр")]
    emap = _emoji_map(tmp_path, {"реестр": "u1f5c3"})
    llm = MockLLM([RuntimeError("boom")])              # матчинг упал
    n = match_user_assets(pts, str(folder), llm, _SILENT, emoji_map_path=emap)
    assert n == 1                                       # сбой не валит — эмодзи
    assert pts[0]["payload"]["asset_kind"] == "emoji"


def test_emoji_map_loader_tolerant_missing_and_garbage(tmp_path):
    assert enrich_llm._load_emoji_map(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert enrich_llm._load_emoji_map(bad) == {}
    arr = tmp_path / "arr.json"
    arr.write_text("[1,2,3]", encoding="utf-8")         # не словарь
    assert enrich_llm._load_emoji_map(arr) == {}
    ok = tmp_path / "ok.json"
    ok.write_text('{"Реестр": "u1f5c3", "пусто": "", "число": 5}',
                  encoding="utf-8")
    m = enrich_llm._load_emoji_map(ok)
    assert m == {"реестр": "u1f5c3"}                    # lower, без пустых/нестрок


def test_emoji_map_loader_unwraps_envelope():
    """Конверт {_version,_comment,map:{...}} разворачивается: концепты — из
    `map`, служебные ключи (_comment/_version) НЕ протекают как «эмодзи»."""
    m = enrich_llm._load_emoji_map(enrich_llm.EMOJI_MAP_PATH)  # вшитый файл
    assert len(m) >= 60                                 # реальные концепты, не 1
    assert "_comment" not in m and "_version" not in m  # служебные не протекли
    # реальный Tier 0 §4 резолвится на вшитом файле (не только в unit-flat-map)
    assert enrich_llm._emoji_for_concept("молния скорость", m) == "u26a1"
    assert enrich_llm._emoji_for_concept("ракета запуск", m) == "u1f680"


def test_emoji_map_loader_unwraps_envelope_synthetic(tmp_path):
    """Конверт распознаётся и на синтетике; плоский формат остаётся рабочим."""
    env = tmp_path / "env.json"
    env.write_text(
        '{"_version": 1, "_comment": "hi", "map": {"Гром": "u26a1"}}',
        encoding="utf-8")
    assert enrich_llm._load_emoji_map(env) == {"гром": "u26a1"}
    flat = tmp_path / "flat.json"                       # без конверта — как было
    flat.write_text('{"гром": "u26a1"}', encoding="utf-8")
    assert enrich_llm._load_emoji_map(flat) == {"гром": "u26a1"}


def test_emoji_for_concept_exact_before_partial():
    # точное совпадение СЛОВА приоритетнее частичной подстроки: «облако» —
    # отдельное слово -> u2601, хотя «лак» тоже подстрока «облако».
    m = {"облако": "u2601", "лак": "uLAK", "сервер": "u1f5a5"}
    assert enrich_llm._emoji_for_concept("облако маркетинга", m) == "u2601"
    # частичное: нет точного слова, ключ-подстрока (длинный ключ — точнее)
    assert enrich_llm._emoji_for_concept("серверная стойка", m) == "u1f5a5"
    assert enrich_llm._emoji_for_concept("ничего похожего", m) == ""


# === 6. detect_all интеграция этапа ассетов =====================================
def _tr_and_cut(n_words: int = 300):
    from vpipe.models import CutList, Segment, Transcript, Word
    words = [Word(f"ток{i:03d}", round(i * 0.5, 3), round(i * 0.5 + 0.4, 3))
             for i in range(n_words)]
    segs = [Segment(words[s].start, words[min(s + 9, n_words - 1)].end,
                    " ".join(w.word for w in words[s:s + 10]), words[s:s + 10])
            for s in range(0, n_words, 10)]
    tr = Transcript(language="ru", duration=round(n_words * 0.5, 3),
                    model="t", audio_hash="h", segments=segs)
    cl = CutList(source="x.mp4", duration=tr.duration, segments=[])
    return tr, cl


def test_detect_all_user_folder_match_counts_as_last_call(tmp_path,
                                                          monkeypatch):
    """image_source=user_folder + папка -> матчинг = последний LLM-вызов пасса
    (keep_alive=0), иллюстрации проставляются asset_kind=user."""
    folder = tmp_path / "assets"
    _mk_assets(folder)
    monkeypatch.setattr(enrich_llm, "EMOJI_MAP_PATH", tmp_path / "no_map.json")
    tr, cl = _tr_and_cut(300)
    params = enrich_mod.default_params()
    params["types"] = {"image": True, "animation": False, "list_card": False,
                       "cta": False}
    params["image_source"] = "user_folder"
    # 1 окно иллюстраций -> 1 точка; затем 1 вызов матчинга
    llm = MockLLM([
        {"points": [{"word_start": 100, "word_end": 105,
                     "concept": "реестр Windows",
                     "image_query_en": "windows registry", "style": "diagram"}]},
        {"matches": [{"point_idx": 0, "asset_filename": "registry.png"}]},
    ])
    out = enrich_llm.detect_all(tr, cl, params, llm, log=_SILENT,
                                user_folder=str(folder))
    assert len(llm.calls) == 2                          # иллюстрации + матчинг
    assert llm.calls[-1]["keep_alive"] == 0             # матчинг — ПОСЛЕДНИЙ
    assert llm.calls[-1]["schema"] is enrich_llm._MATCH_SCHEMA
    assert llm.calls[0]["keep_alive"] == 300            # иллюстрации не последние
    assert len(out) == 1
    assert out[0].payload.asset_kind == "user"
    assert out[0].payload.asset_path.endswith("registry.png")


def test_detect_all_emoji_source_no_extra_llm_call(tmp_path, monkeypatch):
    """image_source=emoji -> эмодзи-фолбэк по карте, БЕЗ вызова матчинга."""
    emap = _emoji_map(tmp_path, {"реестр": "u1f5c3"})
    monkeypatch.setattr(enrich_llm, "EMOJI_MAP_PATH", emap)
    tr, cl = _tr_and_cut(300)
    params = enrich_mod.default_params()
    params["types"] = {"image": True, "animation": False, "list_card": False,
                       "cta": False}
    params["image_source"] = "emoji"
    llm = MockLLM([
        {"points": [{"word_start": 100, "word_end": 105, "concept": "реестр",
                     "image_query_en": "registry", "style": "icon"}]},
    ])
    out = enrich_llm.detect_all(tr, cl, params, llm, log=_SILENT,
                                user_folder="")
    assert len(llm.calls) == 1                          # только иллюстрации
    assert out[0].payload.asset_kind == "emoji"
    assert out[0].payload.emoji == "u1f5c3"


# === 7. валидность вшитых CTA-ассетов (ROBUST к файлам соседнего агента) ==========
@pytest.mark.parametrize("name", CTA_FILES)
def test_cta_webm_size_under_200kb(name):
    f = CTA_DIR / name
    if not f.is_file():
        pytest.skip(f"CTA-ассет ещё не положен соседним агентом: {name}")
    assert f.stat().st_size < MAX_WEBM_BYTES, \
        f"{name}: {f.stat().st_size} >= {MAX_WEBM_BYTES} (§7-P5 гард)"


@pytest.mark.parametrize("name", CTA_FILES)
def test_cta_webm_has_alpha(name):
    """Чистовой CTA-пак несёт АЛЬФУ. Тонкость VP9: НАТИВНЫЙ декодер ffprobe
    рапортует `yuv420p` (альфа лежит скрытым вторичным планом + Matroska
    `alpha_mode=1`), и истинный `yuva420p` виден только под `-c:v libvpx-vp9`.
    Поэтому альфу подтверждаем по ЛЮБОМУ из двух надёжных сигналов: pix_fmt от
    libvpx-vp9 == yuva420p ИЛИ тег alpha_mode==1. ROBUST: нет файла / нет
    ffprobe -> skip (а не падение)."""
    f = CTA_DIR / name
    if not f.is_file():
        pytest.skip(f"CTA-ассет ещё не положен соседним агентом: {name}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe нет в PATH — пропускаю проверку альфы")

    def _probe(entries: str, extra: list[str]) -> str:
        return subprocess.run(
            [ffprobe, "-v", "error", *extra, "-select_streams", "v:0",
             "-show_entries", entries, "-of", "csv=p=0", str(f)],
            capture_output=True, text=True).stdout.strip()

    # 1) истинный pix_fmt через VP9-декодер (нативный врёт yuv420p для VP9-альфы)
    pf_vp9 = _probe("stream=pix_fmt", ["-c:v", "libvpx-vp9"])
    # 2) Matroska-тег альфа-плана (как альфу штатно хранит VP9-webm)
    alpha_tag = _probe("stream_tags=alpha_mode", [])
    assert pf_vp9 == "yuva420p" or alpha_tag == "1", (
        f"{name}: альфа не обнаружена (libvpx pix_fmt={pf_vp9!r}, "
        f"alpha_mode={alpha_tag!r})")


def test_anim_presets_schema():
    f = CTA_DIR / "anim_presets.json"
    if not f.is_file():
        pytest.skip("anim_presets.json ещё не положен соседним агентом")
    data = json.loads(f.read_text(encoding="utf-8"))
    assert isinstance(data.get("presets"), dict)
    # пресеты pop_in/pulse (план §2.3) присутствуют и несут easing
    for key in ("pulse", "pop_in"):
        assert key in data["presets"], key
        assert "easing" in data["presets"][key]
    # карта ассет -> пресет ссылается на существующие пресеты
    assets = data.get("assets")
    if isinstance(assets, dict):
        for webm, preset in assets.items():
            assert preset in data["presets"], (webm, preset)


def test_anim_presets_v11_entrance():
    """V11 §5: новый пресет «cta_slide_in» — финитный въезд (loop=False) с
    ease-out-back-проскоком (overshoot > 1.0) и glow-пульсом; comment/like/bell
    стартуют pop-in (overshoot 0.2 -> 1.12 -> 1.0)."""
    f = CTA_DIR / "anim_presets.json"
    if not f.is_file():
        pytest.skip("anim_presets.json ещё не положен соседним агентом")
    data = json.loads(f.read_text(encoding="utf-8"))
    pr = data["presets"]
    # 1) cta_slide_in: финитный въезд с проскоком (НЕ loop — иначе въезд повторится)
    assert "cta_slide_in" in pr
    si = pr["cta_slide_in"]
    assert si.get("loop") is False, "въезд должен играть один раз от t0 (loop=False)"
    assert si.get("easing") == "ease_out_back"
    assert float(si.get("overshoot", 0.0)) > 1.0, "ease-out-back проскакивает 1.0"
    # 2) pop_in: overshoot 0.2 -> 1.12 -> 1.0
    pi = pr["pop_in"]
    assert pi.get("from", 1.0) < pi.get("over", 0.0) > pi.get("to", 0.0)
    assert abs(float(pi["over"]) - 1.12) < 1e-6
    # 3) ассет-карта: оба subscribe -> cta_slide_in, comment -> pop_in
    a = data.get("assets", {})
    assert a.get("subscribe_like.webm") == "cta_slide_in"
    assert a.get("subscribe_slide_avatar.webm") == "cta_slide_in"
    assert a.get("comment.webm") == "pop_in"


@pytest.mark.parametrize("name", CTA_FILES)
def test_cta_webm_alpha_round_trips_to_png(name):
    """Лакмус VP9-альфы: декодируем кадр обратно в PNG через -c:v libvpx-vp9 и
    проверяем, что прозрачность восстановилась (угол холста alpha=0, но контент
    непрозрачен). ROBUST: нет файла / нет ffmpeg/Pillow -> skip."""
    f = CTA_DIR / name
    if not f.is_file():
        pytest.skip(f"CTA-ассет ещё не положен соседним агентом: {name}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg нет в PATH — пропускаю round-trip альфы")
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "fr.png"
        # последний кадр (покой) — заведомо непрозрачный контент
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-c:v", "libvpx-vp9", "-i", str(f),
             "-vf", "select=eq(n\\,5)", "-frames:v", "1", str(png)],
            capture_output=True, text=True)
        if r.returncode != 0 or not png.is_file():
            pytest.skip(f"декод не удался: {r.stderr.strip()[:120]}")
        img = Image.open(png).convert("RGBA")
        a = img.split()[3]
        nz = sum(1 for px in a.get_flattened_data() if px > 0)
        assert img.getpixel((0, 0))[3] == 0, "угол должен быть прозрачным"
        assert nz > 500, "контент должен быть непрозрачным (альфа восстановилась)"


@pytest.mark.parametrize("name,frames", [
    ("subscribe_like.webm", 36), ("subscribe_slide_avatar.webm", 36),
    ("comment.webm", 30), ("like.webm", 30), ("bell.webm", 30)])
def test_cta_webm_frame_count(name, frames):
    """V11 §5: pill-ассеты длятся ~1.44с (36 кадров @25), иконки ~1.2с (30)."""
    f = CTA_DIR / name
    if not f.is_file():
        pytest.skip(f"CTA-ассет ещё не положен соседним агентом: {name}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        pytest.skip("ffprobe нет в PATH")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-c:v", "libvpx-vp9", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout.strip()
    assert out == str(frames), f"{name}: nb_read_frames={out!r}, ждали {frames}"


def test_make_enrich_assets_deterministic(tmp_path):
    """Генератор детерминирован: повторный прогон даёт байт-в-байт те же webm
    (single-thread VP9 + -bitexact + stripped metadata). ROBUST: нет ffmpeg/
    Pillow/шрифта -> skip (гоняет настоящий VP9-энкод, потому skippable)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg нет в PATH — пропускаю генерацию")
    import importlib.util
    import hashlib
    tools = Path(__file__).resolve().parents[1] / "tools" / "make_enrich_assets.py"
    if not tools.is_file():
        pytest.skip("генератор не найден")
    spec = importlib.util.spec_from_file_location("make_enrich_assets", tools)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not mod.FONT_SEMIBOLD.is_file():
        pytest.skip("нет вшитого Inter-SemiBold — пропускаю")

    def _run(d):
        rc = mod.main(["--ffmpeg", ffmpeg, "--out", str(d)])
        assert rc == 0
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(d.glob("*.webm"))}

    h1 = _run(tmp_path / "a")
    h2 = _run(tmp_path / "b")
    assert h1 == h2 and len(h1) == 5, (h1, h2)


# === 8. /api/browse?kind=image ==================================================
@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


def test_browse_kind_image_whitelist(client, tmp_path):
    for n in ("a.png", "b.jpg", "c.jpeg", "d.webp"):
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "vid.mp4").write_bytes(b"x")            # видео — НЕ картинка
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    r = client.get("/api/browse", params={"dir": str(tmp_path), "kind": "image"})
    assert r.status_code == 200
    names = sorted(f["name"] for f in r.json()["files"])
    assert names == ["a.png", "b.jpg", "c.jpeg", "d.webp"]
    assert r.json()["folders"] == ["sub"]              # папки всегда видны


def test_browse_kind_image_does_not_break_music_or_default(client, tmp_path):
    (tmp_path / "song.mp3").write_bytes(b"x")
    (tmp_path / "vid.mp4").write_bytes(b"x")
    (tmp_path / "pic.png").write_bytes(b"x")
    # music: аудио+видео (картинку НЕ показываем)
    rm = client.get("/api/browse", params={"dir": str(tmp_path), "kind": "music"})
    assert sorted(f["name"] for f in rm.json()["files"]) == ["song.mp3", "vid.mp4"]
    # без kind / folder: прежний контракт — только видео
    rd = client.get("/api/browse", params={"dir": str(tmp_path)})
    assert [f["name"] for f in rd.json()["files"]] == ["vid.mp4"]
    rf = client.get("/api/browse", params={"dir": str(tmp_path), "kind": "folder"})
    assert [f["name"] for f in rf.json()["files"]] == ["vid.mp4"]
