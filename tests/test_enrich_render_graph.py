# -*- coding: utf-8 -*-
"""P1.3 — enrich-оверлеи в графе render() (ENRICH_PLAN §2.1, §7-P1).

FakeFF-сетка (паттерн test_music_duck): ни ffmpeg, ни GPU — только точные
строки args/filter_complex. Слои:

  * ГЛАВНЫЙ регрессионный: enrich=None / пустой RenderEnrich -> все ветки
    (copy fast-path, no-cuts re-encode, cuts, cuts+ass, музыка C3, video-only,
    2-pass measure) дают args БАЙТ-В-БАЙТ как без параметра.
  * Входы: ПОСЛЕ музыки, динамические индексы; PNG строго «-loop 1 -t t1+0.5»
    (R2 §1); WebM строго «[-stream_loop -1] -c:v libvpx-vp9» ДО «-i»
    (R2 капкан №1 — нативный vp9-декодер молча выбрасывает альфу).
  * Граф: vpre (crop->scale->fps) -> overlay-узлы -> enrich.ass(fontsdir)
    ПЕРВЫМ -> burn.ass ПОСЛЕДНИМ (§2.2: скрим карточки не темнит караоке);
    enable= в ФИНАЛЬНЫХ координатах с ТОЧКОЙ в дробях (локаль не влияет);
    shortest=1 ТОЛЬКО у лупленых WebM (R2 капкан №2), у конечных — нет
    (shortest=1 у конечного оборвал бы весь ролик).
  * Ветки: без вырезов (copy fast-path гаснет), video-only, музыка C3 жива
    (аудио-граф не тронут), measure-пасс 2-pass loudnorm не ломается лишними
    видео-входами (живая проверка ffmpeg 8.1.1 — см. коммент в render.py).
  * Лимиты движка: >6 PNG / >3 WebM — warning и трим (страховка планировщика).
"""
import locale
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.config import Config                                       # noqa: E402
from vpipe.enrich import (MAX_ANIMS, MAX_STILLS, AnimOverlay,          # noqa: E402
                          RenderEnrich, StillOverlay, ZoomWindow)
from vpipe.render import (_enrich_video_chain, _punch_dims,            # noqa: E402
                          _punch_filter, _smoothstep_z_expr)
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE,                   # noqa: E402
                          CutList, CutSegment)
from vpipe.probe import MediaInfo                                      # noqa: E402
from vpipe.render import build_music_mix, render                       # noqa: E402

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
    return MediaInfo(path="in.mp4", duration=120.0, fps=30.0, width=1920,
                     height=1080, vcodec="h264", acodec="aac",
                     has_audio=has_audio, sample_rate=48000)


def _cutlist(*, cut=False):
    segs = []
    if cut:
        segs.append(CutSegment(id="c", start=20.0, end=21.0, type=TYPE_PAUSE,
                               action=ACTION_REMOVE, enabled=True))
    return CutList(source="in.mp4", duration=120.0, segments=segs)


def _graph(args: list[str]) -> str:
    if "-filter_complex" in args:
        return args[args.index("-filter_complex") + 1]
    return ""


def _still(**over) -> StillOverlay:
    d = dict(path="pic1.png", x_expr="W-w-48", y_expr="48", scale_w=614,
             t0=40.0, t1=43.0, fade_s=0.22, kenburns=False)
    d.update(over)
    return StillOverlay(**d)


def _anim(**over) -> AnimOverlay:
    d = dict(path="cta.webm", x_expr="48", y_expr="H-h-160", scale_w=220,
             t0=70.0, t1=74.0, loop=True)
    d.update(over)
    return AnimOverlay(**d)


def _enrich(stills=(), anims=(), cards_ass=None,
            fonts_dir="fonts", punches=(), card_windows=()) -> RenderEnrich:
    return RenderEnrich(stills=list(stills), anims=list(anims),
                        cards_ass=cards_ass, fonts_dir=fonts_dir,
                        punches=list(punches), card_windows=list(card_windows))


def _run(cfg, cl, tmp_path, *, has_audio=True, ff=None, log=None, **kw):
    ff = ff or FakeFF()
    out = str(tmp_path / "out.mp4")
    info = render(ff, _media(has_audio=has_audio), cl, cfg, out,
                  str(tmp_path), log=(log if log is not None
                                      else (lambda *a, **k: None)), **kw)
    return ff, info


def _music_cfg(tmp_path, **denoise) -> Config:
    cfg = Config()
    bgm = tmp_path / "bgm.mp3"
    bgm.write_bytes(b"\x00")
    cfg.render.music.enabled = True
    cfg.render.music.path = str(bgm)
    for k, v in denoise.items():
        setattr(cfg.render.denoise, k, v)
    return cfg


_FULL = _enrich(stills=[_still()], anims=[_anim()], cards_ass="enrich_in.ass")


# === ГЛАВНЫЙ регрессионный тест: enrich=None -> графы байт-в-байт прежние ======
def _legacy_matrix(tmp_path):
    """(имя, cfg-фабрика, cutlist, kwargs) — все ветки render()."""
    def plain():
        return Config()

    def loudnorm():
        c = Config()
        c.render.denoise.loudnorm = True
        return c

    def twopass():
        c = Config()
        c.render.denoise.loudnorm = True
        c.render.denoise.loudnorm_mode = "2pass"
        return c

    return [
        ("copy fast-path", plain, _cutlist(), {}),
        ("no cuts + apost", loudnorm, _cutlist(), {}),
        ("no cuts + scale", plain, _cutlist(), {"scale_h": 720}),
        ("cuts plain", plain, _cutlist(cut=True), {}),
        ("cuts + ass", plain, _cutlist(cut=True), {"ass_path": "burn.ass"}),
        ("cuts + music", lambda: _music_cfg(tmp_path), _cutlist(cut=True), {}),
        ("cuts video-only", plain, _cutlist(cut=True), {"has_audio": False}),
        ("2pass measure", twopass, _cutlist(cut=True), {}),
    ]


def test_enrich_none_keeps_every_graph_byte_for_byte(tmp_path):
    for name, mkcfg, cl, kw in _legacy_matrix(tmp_path):
        has_audio = kw.pop("has_audio", True)
        base, _ = _run(mkcfg(), cl, tmp_path, has_audio=has_audio, **kw)
        none_, _ = _run(mkcfg(), cl, tmp_path, has_audio=has_audio,
                        enrich=None, **kw)
        empty, _ = _run(mkcfg(), cl, tmp_path, has_audio=has_audio,
                        enrich=RenderEnrich(), **kw)
        assert none_.runs == base.runs, f"enrich=None изменил args: {name}"
        assert empty.runs == base.runs, f"пустой RenderEnrich изменил args: {name}"
        for args in base.runs:           # и никаких следов enrich в legacy args
            joined = " ".join(args)
            assert "libvpx-vp9" not in joined and "fontsdir" not in joined
            assert "overlay=" not in _graph(args), name


def test_enrich_none_copy_fast_path_still_copies(tmp_path):
    ff, info = _run(Config(), _cutlist(), tmp_path, enrich=None)
    args = ff.runs[-1]
    assert info["encoder"] == "copy"
    assert args[args.index("-c:v") + 1] == "copy"
    assert "-filter_complex" not in args


# === входы: после музыки, точные паттерны PNG/WebM ==============================
def test_inputs_appended_after_music_with_dynamic_indices(tmp_path):
    cfg = _music_cfg(tmp_path)
    ff, _ = _run(cfg, _cutlist(cut=True), tmp_path, enrich=_FULL,
                 ass_path="burn.ass")
    args = ff.runs[-1]
    i_media = args.index("in.mp4")
    i_music = args.index(str(tmp_path / "bgm.mp3"))
    i_png = args.index("pic1.png")
    i_webm = args.index("cta.webm")
    assert i_media < i_music < i_png < i_webm   # media -> музыка -> PNG -> WebM
    g = _graph(args)
    # музыка = вход 1, оверлеи = 2 и 3 (паттерн music_idx, без хардкода).
    assert "[1:a]atrim=" in g
    assert "[2:v]format=rgba" in g and "[3:v]scale=220:-1" in g


def test_png_input_loop_1_t_window_end_plus_half(tmp_path):
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                 enrich=_enrich(stills=[_still(t0=40.0, t1=43.25)]))
    args = ff.runs[-1]
    i = args.index("pic1.png")
    # PNG обязателен «-loop 1 -t {t1+0.5}» (R2 §1: иначе один кадр без fade).
    assert args[i - 5:i + 1] == ["-loop", "1", "-t", "43.750", "-i", "pic1.png"]


def test_webm_input_codec_strictly_before_i(tmp_path):
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                 enrich=_enrich(anims=[_anim(loop=True)]))
    args = ff.runs[-1]
    i = args.index("cta.webm")
    # КАПКАН R2 №1: -stream_loop -1 -c:v libvpx-vp9 строго ДО -i.
    assert args[i - 5:i + 1] == ["-stream_loop", "-1", "-c:v", "libvpx-vp9",
                                 "-i", "cta.webm"]


def test_webm_input_no_loop_no_stream_loop(tmp_path):
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                 enrich=_enrich(anims=[_anim(loop=False)]))
    args = ff.runs[-1]
    i = args.index("cta.webm")
    assert args[i - 3:i + 1] == ["-c:v", "libvpx-vp9", "-i", "cta.webm"]
    assert "-stream_loop" not in args


# === граф: порядок vpre -> overlay -> enrich.ass(fontsdir) -> burn.ass ==========
def test_graph_full_order_with_cuts(tmp_path):
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path, enrich=_FULL,
                 ass_path="burn.ass", scale_h=720, fps=25)
    g = _graph(ff.runs[-1])
    # vpre после concat-метки [vc], оверлеи между vpre и субтитрами.
    assert "concat=n=2:v=1:a=1[vc][outa];[vc]scale=-2:720,fps=25[vb0];" in g
    assert ("[1:v]format=rgba,scale=614:-1,"
            "fade=t=in:st=40.000:d=0.220:alpha=1,"
            "fade=t=out:st=42.780:d=0.220:alpha=1[ov0];"
            "[vb0][ov0]overlay=W-w-48:48"
            ":enable='between(t,40.000,43.000)'[vb1];") in g
    assert ("[2:v]scale=220:-1,setpts=PTS+70.000/TB[an0];"
            "[vb1][an0]overlay=48:H-h-160"
            ":enable='between(t,70.000,74.000)':shortest=1[vb2];") in g
    # enrich.ass с fontsdir ПЕРВЫМ subtitles-фильтром, burn.ass — ПОСЛЕДНИМ.
    assert g.endswith("[vb2]subtitles='enrich_in.ass':fontsdir='fonts',"
                      "subtitles='burn.ass'[outv]")
    order = [g.index("scale=-2:720"), g.index("[ov0]overlay"),
             g.index("[an0]overlay"), g.index("subtitles='enrich_in.ass'"),
             g.index("subtitles='burn.ass'")]
    assert order == sorted(order)


def test_graph_cards_ass_only_comma_chain_fontsdir_first_burn_last(tmp_path):
    ff, info = _run(Config(), _cutlist(cut=True), tmp_path,
                    enrich=_enrich(cards_ass="enrich_in.ass"),
                    ass_path="burn.ass")
    g = _graph(ff.runs[-1])
    assert g.endswith("[vc]subtitles='enrich_in.ass':fontsdir='fonts',"
                      "subtitles='burn.ass'[outv]")
    assert info["encoder"] == "x264"


def test_graph_overlays_without_any_ass_end_in_outv(tmp_path):
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                 enrich=_enrich(stills=[_still()], anims=[_anim()]))
    g = _graph(ff.runs[-1])
    assert "subtitles" not in g
    assert g.endswith(":enable='between(t,70.000,74.000)':shortest=1[outv]")


def test_enable_times_use_dot_decimals_even_under_ru_locale(tmp_path):
    # ЛОКАЛЬ-капкан: дробные секунды в enable= обязаны быть с ТОЧКОЙ.
    old = locale.setlocale(locale.LC_NUMERIC)
    try:
        for loc in ("ru_RU.UTF-8", "Russian_Russia.1251", "ru_RU"):
            try:
                locale.setlocale(locale.LC_NUMERIC, loc)
                break
            except locale.Error:
                continue
        ff, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                     enrich=_enrich(stills=[_still(t0=41.125, t1=44.5)]))
        g = _graph(ff.runs[-1])
        m = re.search(r"enable='between\(t,([^)]*)\)'", g)
        assert m and m.group(1) == "41.125,44.500"
        assert ("-t" in ff.runs[-1]
                and ff.runs[-1][ff.runs[-1].index("-t") + 1] == "45.000")
    finally:
        locale.setlocale(locale.LC_NUMERIC, old)


def test_shortest_only_on_looped_webm(tmp_path):
    enr = _enrich(anims=[_anim(loop=True, t0=40.0, t1=44.0),
                         _anim(path="pop.webm", loop=False,
                               t0=70.0, t1=72.0)])
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path, enrich=enr)
    g = _graph(ff.runs[-1])
    # лупленый — строго с shortest=1 (КАПКАН R2 №2: иначе вечный рендер)…
    assert ":enable='between(t,40.000,44.000)':shortest=1" in g
    # …конечный — строго БЕЗ (shortest=1 оборвал бы ролик на конце анимации).
    assert ":enable='between(t,70.000,72.000)'[" in g
    assert g.count("shortest=1") == 1


# === ветки: без вырезов / video-only / музыка ===================================
def test_no_cuts_branch_overlays_force_reencode(tmp_path):
    ff, info = _run(Config(), _cutlist(), tmp_path,
                    enrich=_enrich(stills=[_still()]))
    args = ff.runs[-1]
    g = _graph(args)
    # copy fast-path погашен; цепочка строится от [0:v] (вырезов нет).
    assert info["encoder"] == "x264"
    assert "copy" not in args
    assert g.startswith("[1:v]format=rgba,scale=614:-1,")
    assert "[0:v][ov0]overlay=W-w-48:48" in g and "[vc]" not in g
    assert args[args.index("-map") + 1] == "[outv]"


def test_no_cuts_cards_only_forces_reencode(tmp_path):
    ff, info = _run(Config(), _cutlist(), tmp_path,
                    enrich=_enrich(cards_ass="enrich_in.ass"))
    assert info["encoder"] == "x264"
    assert _graph(ff.runs[-1]) == ("[0:v]subtitles='enrich_in.ass'"
                                   ":fontsdir='fonts'[outv]")


def test_video_only_branch_with_cuts(tmp_path):
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path, has_audio=False,
                 enrich=_FULL, ass_path="burn.ass")
    args = ff.runs[-1]
    g = _graph(args)
    assert "concat=n=2:v=1:a=0[vc];" in g
    # video-only: оверлеи на входах 1 и 2 (нет ни audio_src, ни музыки).
    assert "[1:v]format=rgba" in g and "[2:v]scale=220:-1" in g
    assert g.endswith("subtitles='enrich_in.ass':fontsdir='fonts',"
                      "subtitles='burn.ass'[outv]")
    assert "-an" in args


def test_music_c3_alive_with_enrich(tmp_path):
    # Аудио-граф (дакинг C3) не тронут оверлеями: тот же блок, те же метки.
    cfg = _music_cfg(tmp_path)
    base, _ = _run(_music_cfg(tmp_path), _cutlist(cut=True), tmp_path)
    ff, info = _run(cfg, _cutlist(cut=True), tmp_path, enrich=_FULL,
                    ass_path="burn.ass")
    g, g0 = _graph(ff.runs[-1]), _graph(base.runs[-1])
    expected = build_music_mix(cfg.render.music, 1, "[outa_raw]",
                               "[outa]", 119.0)
    assert expected in g and expected in g0
    assert "sidechaincompress" in g and "amix=inputs=2" in g
    assert "concat=n=2:v=1:a=1[vc][outa_raw];" in g
    assert info["music"] is True


# === 2-pass measure: лишние видео-входы не ломают измерение =====================
def test_measure_pass_gets_enrich_inputs_but_audio_only_graph(tmp_path):
    cfg = Config()
    cfg.render.denoise.loudnorm = True
    cfg.render.denoise.loudnorm_mode = "2pass"
    ff, _ = _run(cfg, _cutlist(cut=True), tmp_path, enrich=_FULL,
                 ass_path="burn.ass")
    assert len(ff.runs) == 2
    m, enc = ff.runs[0], ff.runs[1]
    # Измерение видит ТЕ ЖЕ входы (включая enrich)…
    assert m[-3:] == ["-f", "null", "-"]
    assert "pic1.png" in m and "cta.webm" in m
    assert m[:m.index("-filter_complex")] == enc[:enc.index("-filter_complex")]
    # …но его граф — чисто аудио (видео-входы не декодируются, R1 §1.6).
    mg = _graph(m)
    assert "overlay" not in mg and "subtitles" not in mg
    assert mg.endswith("[mout]")
    # Кодирующий пасс получил linear-loudnorm из измерения.
    assert "linear=true" in _graph(enc)


# === лимиты движка (страховка планировщика, §2.1 п.5) ===========================
def test_engine_caps_trim_extra_overlays_with_warning(tmp_path):
    logs: list[str] = []
    stills = [_still(path=f"p{i}.png", t0=30.0 + 5 * i, t1=33.0 + 5 * i)
              for i in range(MAX_STILLS + 1)]
    anims = [_anim(path=f"a{i}.webm", t0=70.0 + 5 * i, t1=73.0 + 5 * i)
             for i in range(MAX_ANIMS + 1)]
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path, log=logs.append,
                 enrich=_enrich(stills=stills, anims=anims))
    args = ff.runs[-1]
    assert args.count("-loop") == MAX_STILLS
    assert args.count("libvpx-vp9") == MAX_ANIMS
    assert f"p{MAX_STILLS}.png" not in args     # лишние отброшены с конца
    assert f"a{MAX_ANIMS}.webm" not in args
    assert sum("лимит движка" in m or "ВНИМАНИЕ" in m for m in logs) >= 2


# === V11 §3 — blur-backplate (R2 trimdscale: split/даунскейл/trim/eof_action) ===
def _zw(**over) -> ZoomWindow:
    d = dict(t0=40.0, t1=43.0, z_max=1.08, cx=0.5, cy=0.5)
    d.update(over)
    return ZoomWindow(**d)


def test_blur_backplate_only_forces_reencode_and_chain(tmp_path):
    # карточек-окно без stills/anims/cards_ass: чистый эффект цепочки, без входов.
    ff, info = _run(Config(), _cutlist(cut=True), tmp_path,
                    enrich=_enrich(card_windows=[(100.0, 106.0)]))
    g = _graph(ff.runs[-1])
    assert info["encoder"] == "x264"
    assert "libvpx-vp9" not in " ".join(ff.runs[-1])     # blur не добавляет входов
    # split -> trim-В-ОКНО -> даунскейл-blur -> плоский bilinear upscale -> overlay.
    assert "split=2[bbs0][bbk0]" in g
    # blur-ветка целиком (КРИТ-ПОРЯДОК: trim ПЕРВЫМ, плоский scale=W:H, НЕ scale2ref):
    assert ("[bbk0]trim=100.000:106.000,setpts=PTS-STARTPTS+100.000/TB,"
            "scale=480:270,boxblur=4:2,eq=brightness=-0.16:saturation=0.85,"
            "scale=1920:1080:flags=bilinear[bbb0]") in g
    assert ("overlay=0:0:enable='between(t,100.000,106.000)'"
            ":eof_action=pass[outv]") in g


def test_blur_backplate_trim_gates_boxblur_before_compute(tmp_path):
    # ПЕРФ-ИНВАРИАНТ (LAW §3/§4, R2 ловушка #1): дорогой boxblur ДОЛЖЕН считаться
    # ТОЛЬКО в окне карточки. Это достигается ТОЛЬКО если trim стоит ДО boxblur на
    # blur-ветке. scale2ref-форма тянула boxblur на ВЕСЬ клип (+537%) — запрещена.
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                 enrich=_enrich(card_windows=[(100.0, 106.0)]))
    g = _graph(ff.runs[-1])
    # на blur-ветке trim предшествует boxblur (иначе boxblur на полном клипе).
    branch = g[g.index("[bbk0]"):g.index("[bbb0]")]
    assert branch.index("trim=") < branch.index("boxblur="), (
        "trim ДОЛЖЕН стоять ДО boxblur, иначе blur считается на весь клип")
    # scale2ref запрещён: он лок-степит blur-ветку к полнодлинной reference-ветке,
    # из-за чего boxblur вычисляется на каждый кадр клипа независимо от trim-после.
    assert "scale2ref" not in g
    # upscale = плоский scale в финальные дименшены кадра с bilinear-флагом.
    assert "scale=1920:1080:flags=bilinear" in g


def test_blur_backplate_runs_before_subtitles_after_concat(tmp_path):
    # blur-backplate стоит ПОСЛЕ concat([vc]) и ДО enrich.ass/burn.ass.
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                 ass_path="burn.ass",
                 enrich=_enrich(cards_ass="enrich_in.ass",
                                card_windows=[(100.0, 106.0)]))
    g = _graph(ff.runs[-1])
    assert "concat=n=2:v=1:a=1[vc]" in g
    order = [g.index("split=2[bbs0]"), g.index("eof_action=pass"),
             g.index("subtitles='enrich_in.ass'"),
             g.index("subtitles='burn.ass'")]
    assert order == sorted(order)
    # инвариант: enrich.ass ПЕРВЫМ, burn.ass ПОСЛЕДНИМ
    assert g.endswith("subtitles='enrich_in.ass':fontsdir='fonts',"
                      "subtitles='burn.ass'[outv]")


def test_multiple_card_windows_chain_sequentially(tmp_path):
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                 enrich=_enrich(card_windows=[(100.0, 104.0), (200.0, 205.0)]))
    g = _graph(ff.runs[-1])
    assert "split=2[bbs0][bbk0]" in g and "split=2[bbs1][bbk1]" in g
    assert "trim=100.000:104.000" in g and "trim=200.000:205.000" in g
    assert g.count("eof_action=pass") == 2


# === V11 §4a — punch-zoom (crop+scale smoothstep, НЕ zoompan) ===================
def test_punch_zoom_crop_scale_not_zoompan(tmp_path):
    ff, info = _run(Config(), _cutlist(cut=True), tmp_path,
                    enrich=_enrich(punches=[_zw(t0=40.0, t1=43.0, z_max=1.08)]))
    g = _graph(ff.runs[-1])
    assert info["encoder"] == "x264"
    # crop=w='iw/Z':h='ih/Z':x:y центрированный + scale обратно в W:H.
    assert "crop=w='iw/(" in g and ":h='ih/(" in g
    assert ":x='(iw-iw/(" in g and ":y='(ih-ih/(" in g
    assert "scale=1920:1080" in g
    # НЕ zoompan (доказано дёрганей — R3).
    assert "zoompan" not in g
    # smoothstep p*p*(3-2p) в выражении Z(t).
    assert "*(3-2*" in g
    # окно/потолок зума: between(t,t0,t1) и пик 1.080 (z_max=1.08, ≤ потолка)
    assert "between(t,40.000,43.000)" in g
    assert "1.080" in g


def test_punch_zoom_z_expr_smoothstep_envelope():
    z = _smoothstep_z_expr(_zw(t0=10.0, t1=15.0, z_max=1.08))
    # вне окна Z=1; рамп-ин/hold/рамп-аут (lt/gt по краям рамп)
    assert z.startswith("if(between(t,10.000,15.000)")
    assert z.endswith(",1)")                            # else 1.0 вне окна
    assert "lt(t,10.700)" in z and "gt(t,14.300)" in z  # рампы 0.7 c
    assert "1.080" in z                                 # пик


def test_punch_zoom_runs_before_overlays_and_subs(tmp_path):
    # порядок: vpre -> punch -> overlay(PiP) -> subtitles.
    ff, _ = _run(Config(), _cutlist(cut=True), tmp_path, scale_h=720,
                 ass_path="burn.ass",
                 enrich=_enrich(stills=[_still(t0=80.0, t1=83.0)],
                                punches=[_zw(t0=40.0, t1=43.0)]))
    g = _graph(ff.runs[-1])
    order = [g.index("scale=-2:720"), g.index("crop=w='iw/("),
             g.index("[ov0]overlay"), g.index("subtitles='burn.ass'")]
    assert order == sorted(order)


def test_punch_dims_from_scale_h_and_crop_filter():
    assert _punch_dims(_media(), 720, None) == (1280, 720)        # пропорция
    assert _punch_dims(_media(), None, None) == (1920, 1080)      # исходная
    # vertical: цель запечена в crop_filter (...,scale=W:H)
    assert _punch_dims(_media(), None,
                       "crop=600:1067:660:0,scale=1080:1920") == (1080, 1920)


def test_punch_filter_empty_without_windows():
    assert _punch_filter([], 1920, 1080) == ""
    assert _punch_filter([_zw()], 0, 0) == ""           # без дименшенов — no-op


def test_dynamics_keep_legacy_byte_for_byte_when_empty(tmp_path):
    # пустые punches/card_windows -> граф БАЙТ-В-БАЙТ как без них.
    base, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                   enrich=_enrich(stills=[_still()]))
    dyn, _ = _run(Config(), _cutlist(cut=True), tmp_path,
                  enrich=_enrich(stills=[_still()], punches=[],
                                 card_windows=[]))
    assert _graph(base.runs[-1]) == _graph(dyn.runs[-1])
