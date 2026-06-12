# -*- coding: utf-8 -*-
"""C3 — фоновая музыка + локальный авто-дакинг (render.music).

CapCut делает Auto-Duck облачными кейфреймами за Pro; здесь — один ffmpeg
``sidechaincompress`` локально. Слои, покрытые этим файлом:

  * ``MusicCfg`` — дефолты (выключено; gain -18 дБ, duck -12 дБ, threshold
    0.02, ratio 8, attack 20 мс, release 400 мс).
  * ``build_music_mix`` — точная строка графа: лупленый вход -> atrim до
    финальной длительности -> volume -> sidechaincompress (речь = key,
    duck_db -> mix) -> amix (normalize=0, duration=first).
  * ``render()`` — музыка входит ПОСЛЕДНИМ входом (-stream_loop -1), микс
    собирается ДО loudnorm (мастеринг меряет опубликованный звук); music off /
    битый путь / video-only -> графы байт-в-байт прежние (existing tests +
    явные проверки здесь).
  * 2-pass loudnorm — измерительный граф включает ТОТ ЖЕ музыкальный блок,
    что и кодирующий (симметрия measured-значений).
  * ``serve._resolve_render_opts`` — валидация path/расширения (400 на
    запросе), клампы -40..0 / -30..0, контракт «нет opts.music -> выключено».
  * /api/clips/render — клипы Shorts НИКОГДА не получают подложку (сервер
    шлёт music=None в _resolve_render_opts) — функциональный тест эндпоинта.
  * /api/browse?kind=music — файл-пикер музыки видит аудио+видео.
  * UI-маркеры в web/index.html / web/app.js.

Без сети, GPU и реального ffmpeg — всё на фейках (паттерн test_mastering /
test_loudnorm_2pass / test_api_clips).
"""
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.config import Config, MusicCfg, load_config                # noqa: E402
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE,                  # noqa: E402
                          CutList, CutSegment)
from vpipe.probe import MediaInfo                                     # noqa: E402
from vpipe.render import build_music_mix, render                      # noqa: E402

import serve                                                          # noqa: E402

LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"
MEASURE = LOUDNORM + ":print_format=json"
LINEAR = (LOUDNORM + ":measured_I=-27.61:measured_TP=-9.11:measured_LRA=18.06"
          ":measured_thresh=-39.20:offset=0.47:linear=true")
ARESAMPLE = "aresample=48000"
DUCK = ("sidechaincompress=threshold=0.02:ratio=8:attack=20:release=400"
        ":mix=0.749")
AMIX = "amix=inputs=2:duration=first:dropout_transition=0:normalize=0"

# Реалистичный stderr измерительного пасса (как в test_loudnorm_2pass).
SAMPLE_STDERR = """\
[Parsed_loudnorm_2 @ 000001f2a3b4c5d6]
{
\t"input_i" : "-27.61",
\t"input_tp" : "-9.11",
\t"input_lra" : "18.06",
\t"input_thresh" : "-39.20",
\t"output_i" : "-14.47",
\t"output_tp" : "-1.50",
\t"output_lra" : "16.70",
\t"output_thresh" : "-25.67",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.47"
}
"""


# --- MusicCfg defaults ---------------------------------------------------------
def test_music_defaults_off_and_sane():
    m = MusicCfg()
    assert m.enabled is False and m.path is None
    assert m.gain_db == -18.0          # музыка тише речи
    assert m.duck_db == -12.0          # на сколько давить при речи
    assert m.threshold == 0.02 and m.ratio == 8.0
    assert m.attack == 20.0 and m.release == 400.0


def test_config_has_music_field_off():
    assert Config().render.music.enabled is False


def test_repo_config_yaml_music_off():
    # Блок music в config.yaml закомментирован -> pydantic-дефолт (выключено).
    assert load_config("config.yaml").render.music.enabled is False


# --- build_music_mix -----------------------------------------------------------
def _mcfg(**over) -> MusicCfg:
    return MusicCfg(enabled=True, path="bgm.mp3", **over)


def test_build_music_mix_exact_string():
    g = build_music_mix(_mcfg(), 1, "[0:a]", "[outa]", 10.0)
    assert g == (
        "[1:a]atrim=start=0:end=10.000,asetpts=PTS-STARTPTS,volume=-18dB[bgm];"
        "[0:a]asplit=2[spd][spk];"
        f"[bgm][spk]{DUCK}[duck];"
        f"[spd][duck]{AMIX}[outa]")


def test_build_music_mix_duck_to_mix_mapping():
    # duck_db -> mix = 1 - 10^(duck/20): 0 дБ = без дакинга, -30 дБ ~ 0.968.
    assert "mix=0.000" in build_music_mix(_mcfg(duck_db=0.0), 1, "[0:a]",
                                          "[outa]", 5.0)
    assert "mix=0.968" in build_music_mix(_mcfg(duck_db=-30.0), 1, "[0:a]",
                                          "[outa]", 5.0)
    # Положительный duck_db (мусор в конфиге) клампится к 0 — без «усиления».
    assert "mix=0.000" in build_music_mix(_mcfg(duck_db=6.0), 1, "[0:a]",
                                          "[outa]", 5.0)


def test_build_music_mix_gain_and_idx():
    g = build_music_mix(_mcfg(gain_db=-25.0), 2, "[outa_raw]", "[mix]", 7.5)
    assert g.startswith("[2:a]atrim=start=0:end=7.500,")
    assert "volume=-25dB[bgm]" in g
    assert "[outa_raw]asplit=2[spd][spk]" in g
    assert g.endswith("[mix]")


# --- render() integration (FakeFF — паттерн test_mastering) ---------------------
class FakeFF:
    def __init__(self, measure_stderr=SAMPLE_STDERR):
        self.runs: list[list[str]] = []
        self.measure_stderr = measure_stderr

    def has_filter(self, name):
        return True

    def has_encoder(self, name):          # форсируем x264 (детерминизм)
        return False

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.runs.append(list(args))
        if args and args[-1] == "-":      # измерительный пасс `-f null -`
            return self.measure_stderr
        try:
            Path(args[-1]).write_bytes(b"")
        except OSError:
            pass
        return ""


def _media(has_audio=True):
    return MediaInfo(path="in.mp4", duration=10.0, fps=30.0, width=1920,
                     height=1080, vcodec="h264", acodec="aac",
                     has_audio=has_audio, sample_rate=48000)


def _cutlist(*, cut=False):
    segs = []
    if cut:
        segs.append(CutSegment(id="c", start=2.0, end=3.0, type=TYPE_PAUSE,
                               action=ACTION_REMOVE, enabled=True))
    return CutList(source="in.mp4", duration=10.0, segments=segs)


def _graph(args: list[str]) -> str:
    if "-filter_complex" in args:
        return args[args.index("-filter_complex") + 1]
    return ""


def _music_cfg(tmp_path, *, exists=True, **denoise) -> Config:
    cfg = Config()
    bgm = tmp_path / "bgm.mp3"
    if exists:
        bgm.write_bytes(b"\x00")
    cfg.render.music.enabled = True
    cfg.render.music.path = str(bgm)
    for k, v in denoise.items():
        setattr(cfg.render.denoise, k, v)
    return cfg


def _run(cfg, cl, tmp_path, *, has_audio=True, ff=None, log=None):
    ff = ff or FakeFF()
    out = str(tmp_path / "out.mp4")
    info = render(ff, _media(has_audio=has_audio), cl, cfg, out,
                  str(tmp_path), log=(log if log is not None
                                      else (lambda *a, **k: None)))
    return ff, info


def test_render_music_no_cuts_graph_and_inputs(tmp_path):
    cfg = _music_cfg(tmp_path)
    ff, info = _run(cfg, _cutlist(), tmp_path)
    args = ff.runs[-1]
    # Музыка — последний вход, лупится -stream_loop -1.
    i = args.index("-stream_loop")
    assert args[i:i + 4] == ["-stream_loop", "-1", "-i",
                             str(tmp_path / "bgm.mp3")]
    # Граф = ровно build_music_mix (без apost tail музыка сразу даёт [outa]).
    assert _graph(args) == build_music_mix(cfg.render.music, 1, "[0:a]",
                                           "[outa]", 10.0)
    # Видео по-прежнему копируется (музыка не форсит перекодирование видео).
    assert args[args.index("-c:v") + 1] == "copy"
    assert info["music"] is True and info["encoder"] == "copy"


def test_render_music_mix_before_loudnorm(tmp_path):
    # КРИТИЧНО: sidechaincompress/amix строго ДО loudnorm — мастеринг -14 LUFS
    # меряет итоговый микс, а не голую речь; де-эссер (речевой) — до микса.
    cfg = _music_cfg(tmp_path, deess=True, loudnorm=True)
    ff, _ = _run(cfg, _cutlist(), tmp_path)
    g = _graph(ff.runs[-1])
    assert g.startswith("[0:a]deesser=i=0.4[sp];")          # речь чистится до микса
    assert "[sp]asplit=2[spd][spk]" in g
    assert g.index("sidechaincompress") < g.index("loudnorm")
    assert g.endswith(f"[mix]{LOUDNORM},{ARESAMPLE}[outa]")


def test_render_music_with_cuts_uses_retimed_speech_as_key(tmp_path):
    cfg = _music_cfg(tmp_path)
    ff, info = _run(cfg, _cutlist(cut=True), tmp_path)
    g = _graph(ff.runs[-1])
    # Ключ дакинга — речь ПОСЛЕ вырезов (concat-выход), не исходная дорожка.
    assert "concat=n=2:v=1:a=1[outv][outa_raw]" in g
    assert "[outa_raw]asplit=2[spd][spk]" in g
    # Музыка обрезана до ФИНАЛЬНОЙ длительности (10 - 1 c выреза = 9).
    assert "atrim=start=0:end=9.000" in g
    assert info["music"] is True


def test_render_music_off_keeps_graphs_byte_for_byte(tmp_path):
    # Дефолтный конфиг (music выключена): чистый copy-фастпас без графа…
    ff, info = _run(Config(), _cutlist(), tmp_path)
    args = ff.runs[-1]
    assert "-filter_complex" not in args and "-stream_loop" not in args
    assert info["music"] is False
    # …а с loudnorm — прежний граф буква в букву (без музыкальных фильтров).
    cfg = Config()
    cfg.render.denoise.loudnorm = True
    ff2, _ = _run(cfg, _cutlist(), tmp_path)
    assert _graph(ff2.runs[-1]) == f"[0:a]{LOUDNORM},{ARESAMPLE}[outa]"
    assert "-stream_loop" not in ff2.runs[-1]


def test_render_music_missing_file_falls_back_honestly(tmp_path):
    logs: list[str] = []
    cfg = _music_cfg(tmp_path, exists=False)
    ff, info = _run(cfg, _cutlist(), tmp_path, log=logs.append)
    args = ff.runs[-1]
    assert "-stream_loop" not in args
    assert "sidechaincompress" not in _graph(args)
    assert info["music"] is False
    assert any("не найден" in m and "без музыки" in m for m in logs)


def test_render_music_video_only_skipped(tmp_path):
    logs: list[str] = []
    cfg = _music_cfg(tmp_path)
    ff, info = _run(cfg, _cutlist(cut=True), tmp_path, has_audio=False,
                    log=logs.append)
    assert "-stream_loop" not in ff.runs[-1]
    assert info["music"] is False
    assert any("нет звуковой дорожки" in m for m in logs)


# --- 2-pass loudnorm: измерение слышит ТО ЖЕ, что войдёт в loudnorm -------------
def test_render_2pass_measure_includes_music_mix(tmp_path):
    cfg = _music_cfg(tmp_path, loudnorm=True, loudnorm_mode="2pass")
    ff, _ = _run(cfg, _cutlist(), tmp_path)
    assert len(ff.runs) == 2
    m, f = ff.runs[0], ff.runs[1]
    # Пасс 1: audio-only, музыкальный вход уже подключён, граф кончается
    # измерительным loudnorm ПОСЛЕ микса.
    assert m[-3:] == ["-f", "null", "-"]
    assert "-stream_loop" in m
    mg = _graph(m)
    assert mg.endswith(f";[mmix]{MEASURE}[mout]")
    assert "volume=-18dB" in mg and DUCK in mg
    # Пасс 2: linear-loudnorm с измеренными значениями — ПОСЛЕ amix.
    fg = _graph(f)
    assert fg.endswith(f"[mix]{LINEAR},{ARESAMPLE}[outa]")
    # Симметрия: музыкальный блок (gain/duck-параметры) в обоих графах идентичен.
    block = re.compile(r"sidechaincompress=[^\[]+")
    assert block.search(mg).group(0) == block.search(fg).group(0)


def test_render_2pass_music_with_cuts_measures_concat(tmp_path):
    cfg = _music_cfg(tmp_path, loudnorm=True, loudnorm_mode="2pass")
    ff, _ = _run(cfg, _cutlist(cut=True), tmp_path)
    mg = _graph(ff.runs[0])
    # Измерение реплицирует финальную речь (тримы+concat) и тот же микс.
    assert "concat=n=2:v=0:a=1[mraw]" in mg
    assert "[mraw]asplit=2[spd][spk]" in mg
    assert "atrim=start=0:end=9.000" in mg          # музыка — до финальной длины
    assert mg.endswith(f";[mmix]{MEASURE}[mout]")


# --- serve._resolve_render_opts --------------------------------------------------
def _fake_session(tmp_path):
    cfg = load_config("config.yaml")
    media = SimpleNamespace(height=1080, fps=30.0)
    return SimpleNamespace(cfg=cfg, media=media, out_dir=tmp_path / "out",
                           inp=SimpleNamespace(stem="clip"))


def _bgm(tmp_path, name="bgm.mp3") -> str:
    p = tmp_path / name
    p.write_bytes(b"\x00")
    return str(p)


def test_resolve_music_absent_means_off_even_if_session_cfg_on(tmp_path):
    # Контракт «без opts.music — выключено» = серверная гарантия клипам без
    # подложки (clips_render/autopack шлют music=None).
    s = _fake_session(tmp_path)
    s.cfg.render.music.enabled = True
    s.cfg.render.music.path = _bgm(tmp_path)
    for opts in ({}, {"music": None}, {"music": "yes"},
                 {"music": {"enabled": False, "path": "anything"}}):
        cfg, *_ = serve._resolve_render_opts(s, opts)
        assert cfg.render.music.enabled is False


def test_resolve_music_valid_and_clamped(tmp_path):
    p = _bgm(tmp_path)
    cfg, *_ = serve._resolve_render_opts(
        _fake_session(tmp_path),
        {"music": {"enabled": True, "path": p,
                   "gain_db": -100, "duck_db": 5}})
    mc = cfg.render.music
    assert mc.enabled is True
    assert mc.path == str(Path(p).resolve())
    assert mc.gain_db == -40.0          # кламп -40..0
    assert mc.duck_db == 0.0            # кламп -30..0


def test_resolve_music_junk_levels_keep_defaults(tmp_path):
    cfg, *_ = serve._resolve_render_opts(
        _fake_session(tmp_path),
        {"music": {"enabled": True, "path": _bgm(tmp_path),
                   "gain_db": "loud", "duck_db": None}})
    assert cfg.render.music.gain_db == -18.0
    assert cfg.render.music.duck_db == -12.0


def test_resolve_music_missing_file_400(tmp_path):
    with pytest.raises(HTTPException) as e:
        serve._resolve_render_opts(
            _fake_session(tmp_path),
            {"music": {"enabled": True,
                       "path": str(tmp_path / "missing.mp3")}})
    assert e.value.status_code == 400


def test_resolve_music_bad_extension_400(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(HTTPException) as e:
        serve._resolve_render_opts(
            _fake_session(tmp_path),
            {"music": {"enabled": True, "path": str(bad)}})
    assert e.value.status_code == 400


def test_resolve_music_enabled_without_path_400(tmp_path):
    for path in ("", None, "   "):
        with pytest.raises(HTTPException) as e:
            serve._resolve_render_opts(
                _fake_session(tmp_path),
                {"music": {"enabled": True, "path": path}})
        assert e.value.status_code == 400


def test_resolve_music_video_file_allowed(tmp_path):
    # Видео в белом списке: ffmpeg возьмёт из него звуковую дорожку.
    cfg, *_ = serve._resolve_render_opts(
        _fake_session(tmp_path),
        {"music": {"enabled": True, "path": _bgm(tmp_path, "bed.mp4")}})
    assert cfg.render.music.enabled is True


# --- /api/browse?kind=music -------------------------------------------------------
@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


def test_browse_kind_music_lists_audio_and_video(client, tmp_path):
    (tmp_path / "song.mp3").write_bytes(b"x")
    (tmp_path / "vid.mp4").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    r = client.get("/api/browse", params={"dir": str(tmp_path),
                                          "kind": "music"})
    assert r.status_code == 200
    names = sorted(f["name"] for f in r.json()["files"])
    assert names == ["song.mp3", "vid.mp4"]
    # Без kind — прежний контракт: только видео.
    r2 = client.get("/api/browse", params={"dir": str(tmp_path)})
    assert [f["name"] for f in r2.json()["files"]] == ["vid.mp4"]


# --- /api/clips/render: клипы Shorts ВСЕГДА без подложки ---------------------------
class _ClipSession:
    """Минимальная сессия с настоящей task-механикой (паттерн test_api_clips)."""

    start_task = serve.Session.start_task
    set_progress = serve.Session.set_progress
    stage = serve.Session.stage

    def __init__(self, tmp_path):
        self.cfg = load_config("config.yaml")
        self.inp = Path("fake.mp4")
        self.media = SimpleNamespace(path="fake.mp4", duration=40.0,
                                     width=1920, height=1080, fps=30.0)
        self.ff = None
        self.work_dir = tmp_path / "work"
        self.out_dir = tmp_path / "out"
        for d in (self.work_dir, self.out_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.last_out_dir = str(self.out_dir.resolve())
        self.audio_hash = "a8" * 20
        self.llm = None
        self.transcript = SimpleNamespace()     # только проверка на not-None
        self.cutlist = CutList(source="fake.mp4", duration=40.0, segments=[])
        self.task = {"name": None, "running": False, "percent": 0.0,
                     "stage": "", "error": None, "done": False, "results": None}


def _wait_done(sess, timeout=5.0):
    t0 = time.time()
    while sess.task["running"]:
        if time.time() - t0 > timeout:
            raise AssertionError(f"задача не завершилась: {sess.task}")
        time.sleep(0.01)


def test_clips_render_strips_music(client, monkeypatch, tmp_path):
    sess = _ClipSession(tmp_path)
    # Музыка «включена» и в конфиге сессии, и в render_opts клиента — клип
    # всё равно обязан рендериться без подложки (серверная гарантия C3).
    bgm = _bgm(tmp_path)
    sess.cfg.render.music.enabled = True
    sess.cfg.render.music.path = bgm
    monkeypatch.setattr(serve, "SESSION", sess)
    calls: list[dict] = []

    def fake_pipeline(s, cfg, scale_h, fps, out_dir, base, on_progress,
                      on_stage, cutlist_override=None, edge_fade=0.0, **kw):
        calls.append({"cfg": cfg})
        return {"mp4": str(base) + ".mp4", "encoder": "fake"}

    monkeypatch.setattr(serve, "_run_render_pipeline", fake_pipeline)
    r = client.post("/api/clips/render", json={
        "clips": [{"start": 5.0, "end": 25.0}],
        "render_opts": {"music": {"enabled": True, "path": bgm,
                                  "gain_db": -18, "duck_db": -12}}})
    assert r.status_code == 200
    _wait_done(sess)
    assert sess.task["error"] is None
    assert calls and calls[0]["cfg"].render.music.enabled is False


def test_clips_and_autopack_strip_music_in_source():
    # Дешёвый сторож: обе точки входа клипов шлют music=None (clips_render и
    # clip_opts Авто-пака) — функционально clips покрыт тестом выше.
    src = Path(serve.__file__).read_text(encoding="utf-8")
    assert src.count('"music": None') >= 2


# --- UI wiring ---------------------------------------------------------------------
WEB = Path(serve.__file__).resolve().parent / "web"


def test_render_modal_has_music_section():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for marker in ('id="rMusic"', 'id="rMusicOpts"', 'id="rMusicPath"',
                   'id="rMusicBrowse"', 'id="rMusicGain"', 'id="rMusicDuck"'):
        assert marker in html, marker
    # Обещанная подсказка — дословно про «стелется и приглушается».
    assert "Музыка стелется под речь и автоматически приглушается" in html


def test_app_js_sends_music_opts_and_uses_music_picker():
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "music: {" in js                  # collectRenderOpts шлёт music{}
    assert "kind=music" in js                # файл-пикер ходит в /api/browse
    assert "pickFileCb" in js                # режим выбора файла в #files
