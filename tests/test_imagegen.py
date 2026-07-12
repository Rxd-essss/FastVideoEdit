# -*- coding: utf-8 -*-
"""ТРЕК-2 §2 — локальная SD-генерация (vpipe/imagegen.py).

subprocess ЗАМОКАН: модель в pytest НЕ гоняем (тяжёлый GPU). Покрывает:
 1. резолв путей бинаря/модели (абсолют / repo-root / cwd / PATH; нет → None);
 2. детерминированный сид (-1 → стабильный хэш query; >=0 как есть);
 3. кэш-ключ: разные промпт/суффикс/негатив/сид/размер/шаги/модель → разные
    ключи; идемпотентность (готовый PNG отдаётся без запуска бинаря);
 4. сбой → None: нет бинаря, нет модели, exit!=0, пустой PNG, таймаут, OSError;
 5. enrich_image_batch: маршрутизация generate→user(успех)/emoji/none(сбой),
    работает и на dict-кандидатах, и на EnrichItem; прогресс.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from vpipe import imagegen
from vpipe.enrich import ENR_IMAGE, EnrichItem, ImagePayload

_SILENT = lambda *a, **k: None  # noqa: E731


def _cfg(tmp_path, **over):
    """ImageGenCfg-подобный объект (duck-typed; реальный pydantic не нужен)."""
    base = dict(imagegen_enabled=True,
                imagegen_bin=str(tmp_path / "sd-cli.exe"),
                imagegen_model=str(tmp_path / "m.gguf"),
                imagegen_size=768, imagegen_steps=4,
                imagegen_cfg=1.5, imagegen_candidates=4, imagegen_vae="",
                imagegen_vae_on_cpu=False)
    base.update(over)
    return SimpleNamespace(**base)


def _touch(p, data=b"x"):
    p.write_bytes(data)
    return p


# === 1. резолв путей =============================================================
def test_resolve_sd_bin_absolute(tmp_path):
    exe = _touch(tmp_path / "sd-cli.exe")
    assert imagegen._resolve_sd_bin(str(exe)) == str(exe)
    assert imagegen._resolve_sd_bin(str(tmp_path / "nope.exe")) is None
    assert imagegen._resolve_sd_bin("") is None
    assert imagegen._resolve_sd_bin(None) is None


def test_resolve_sd_bin_repo_root_relative(monkeypatch, tmp_path):
    monkeypatch.setattr(imagegen, "_REPO_ROOT", tmp_path)
    (tmp_path / "tools").mkdir()
    exe = _touch(tmp_path / "tools" / "sd-cli.exe")
    assert imagegen._resolve_sd_bin("tools/sd-cli.exe") == str(exe)


def test_resolve_sd_bin_falls_back_to_path(monkeypatch, tmp_path):
    monkeypatch.setattr(imagegen, "_REPO_ROOT", tmp_path)   # без repo-root-копии
    # имя, которого нет ни в repo-root, ни в cwd → доходим до PATH (shutil.which).
    monkeypatch.setattr(imagegen.shutil, "which",
                        lambda name: "/usr/bin/sd" if name in
                        ("nx-sd.exe", "nx-sd") else None)
    assert imagegen._resolve_sd_bin("nx-sd.exe") == "/usr/bin/sd"


def test_resolve_model(monkeypatch, tmp_path):
    monkeypatch.setattr(imagegen, "_REPO_ROOT", tmp_path)
    m = _touch(tmp_path / "m.gguf")
    assert imagegen._resolve_model(str(m)) == str(m)        # абсолют
    assert imagegen._resolve_model("m.gguf") == str(m)      # repo-root-rel
    assert imagegen._resolve_model("") is None
    assert imagegen._resolve_model(str(tmp_path / "no.gguf")) is None


def test_resolve_model_repo_root_relative(monkeypatch, tmp_path):
    """#89: модель живёт в repo-local models/ (gitignored) — резолвится через
    _REPO_ROOT, иммунна к чистке D:/tmp (зеркало resolve_sd_bin для models/)."""
    monkeypatch.setattr(imagegen, "_REPO_ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    m = _touch(tmp_path / "models" / "sdxl-turbo-Q4_0.gguf")
    assert imagegen._resolve_model("models/sdxl-turbo-Q4_0.gguf") == str(m)


# === 2. детерминированный сид ===================================================
def test_seed_for_deterministic_hash_and_passthrough():
    assert imagegen._seed_for("ubuntu linux desktop", 42) == 42      # >=0 как есть
    assert imagegen._seed_for("ubuntu", 0) == 0
    s1 = imagegen._seed_for("ubuntu linux desktop", -1)
    s2 = imagegen._seed_for("ubuntu linux desktop", -1)
    s3 = imagegen._seed_for("windows desktop", -1)
    assert s1 == s2 and s1 != s3                                     # стабилен и зависит от query
    assert 0 <= s1 <= 0x7FFFFFFF


# === 3. кэш-ключ ================================================================
def test_cache_key_changes_with_every_factor():
    base = ("p", "sfx", "neg", 1, 768, 768, 4, "sig")
    k0 = imagegen._cache_key(*base)
    variants = [
        ("P2", "sfx", "neg", 1, 768, 768, 4, "sig"),    # prompt
        ("p", "SFX2", "neg", 1, 768, 768, 4, "sig"),    # suffix
        ("p", "sfx", "NEG2", 1, 768, 768, 4, "sig"),    # negative
        ("p", "sfx", "neg", 2, 768, 768, 4, "sig"),     # seed
        ("p", "sfx", "neg", 1, 512, 768, 4, "sig"),     # W
        ("p", "sfx", "neg", 1, 768, 512, 4, "sig"),     # H
        ("p", "sfx", "neg", 1, 768, 768, 8, "sig"),     # steps
        ("p", "sfx", "neg", 1, 768, 768, 4, "sig2"),    # model
    ]
    keys = {imagegen._cache_key(*v) for v in variants}
    assert k0 not in keys
    assert len(keys) == len(variants)               # каждый фактор уникален


# === 4. генерация: мок subprocess ==============================================
def _ok_run(out_arg_writer):
    """Фабрика fake-subprocess.run, который «пишет» PNG в -o путь и возвращает 0."""
    def fake_run(cmd, **kw):
        out = cmd[cmd.index("-o") + 1]
        out_arg_writer(out)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return fake_run


def test_generate_happy_writes_png_and_caches(monkeypatch, tmp_path):
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf", b"M" * 100)
    cfg = _cfg(tmp_path)
    cache = tmp_path / "cache"
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        # флаги §5 (c_diff §2): cfg-scale из cfg (1.5), euler_a, diffusion-fa
        assert "--diffusion-fa" in cmd
        assert cmd[cmd.index("--steps") + 1] == "4"     # cfg.imagegen_steps
        assert cmd[cmd.index("--cfg-scale") + 1] == "1.5"
        assert cmd[cmd.index("--sampling-method") + 1] == "euler_a"
        assert cmd[cmd.index("-W") + 1] == "768"
        out = cmd[cmd.index("-o") + 1]
        from pathlib import Path
        Path(out).write_bytes(b"\x89PNG fake")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    p1 = imagegen.generate_image("ubuntu linux desktop", imagegen.STYLE_SUFFIX,
                                 -1, 768, 768, cfg=cfg, cache_dir=cache,
                                 log=_SILENT)
    assert p1 is not None and p1.endswith(".png")
    from pathlib import Path
    assert Path(p1).is_file() and Path(p1).stat().st_size > 0
    # атомарно: временный .tmp.png не остался (sd-cli требует .png-расширение)
    assert not Path(p1).with_name(Path(p1).stem + ".tmp.png").exists()
    # второй вызов с тем же запросом — кэш-хит, бинарь НЕ запускается снова
    p2 = imagegen.generate_image("ubuntu linux desktop", imagegen.STYLE_SUFFIX,
                                 -1, 768, 768, cfg=cfg, cache_dir=cache,
                                 log=_SILENT)
    assert p2 == p1 and calls["n"] == 1


def test_generate_no_bin_or_model_returns_none(monkeypatch, tmp_path):
    # бинаря нет
    cfg = _cfg(tmp_path)
    assert imagegen.generate_image("q", imagegen.STYLE_SUFFIX, 1, 768, 768,
                                   cfg=cfg, cache_dir=tmp_path / "c",
                                   log=_SILENT) is None
    # бинарь есть, модели нет
    _touch(tmp_path / "sd-cli.exe")
    cfg2 = _cfg(tmp_path, imagegen_model="")
    assert imagegen.generate_image("q", imagegen.STYLE_SUFFIX, 1, 768, 768,
                                   cfg=cfg2, cache_dir=tmp_path / "c",
                                   log=_SILENT) is None


def test_generate_empty_query_returns_none(tmp_path):
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf")
    assert imagegen.generate_image("   ", imagegen.STYLE_SUFFIX, 1, 768, 768,
                                   cfg=_cfg(tmp_path), cache_dir=tmp_path / "c",
                                   log=_SILENT) is None


@pytest.mark.parametrize("kind", ["nonzero", "empty_png", "timeout", "oserror"])
def test_generate_failure_modes_return_none_and_clean_tmp(monkeypatch, tmp_path,
                                                          kind):
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf")
    cfg = _cfg(tmp_path)
    cache = tmp_path / "cache"
    from pathlib import Path

    def fake_run(cmd, **kw):
        out = cmd[cmd.index("-o") + 1]
        if kind == "nonzero":
            Path(out).write_bytes(b"partial")
            return SimpleNamespace(returncode=1, stdout="", stderr="CUDA oom")
        if kind == "empty_png":
            Path(out).write_bytes(b"")               # пустой выход
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if kind == "timeout":
            raise subprocess.TimeoutExpired(cmd, imagegen.SD_TIMEOUT_S)
        raise OSError("exec format error")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = imagegen.generate_image("q english", imagegen.STYLE_SUFFIX, 1,
                                  768, 768, cfg=cfg, cache_dir=cache,
                                  log=_SILENT)
    assert out is None
    # ни итоговый .png, ни временный .tmp.png не остались висеть
    if cache.is_dir():
        assert not list(cache.glob("*.png"))         # включая *.tmp.png


def test_generate_vae_on_cpu_flag(monkeypatch, tmp_path):
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        from pathlib import Path
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    imagegen.generate_image("q english", imagegen.STYLE_SUFFIX, 1, 768, 768,
                            cfg=_cfg(tmp_path, imagegen_vae_on_cpu=True),
                            cache_dir=tmp_path / "c", log=_SILENT)
    assert "--vae-on-cpu" in seen["cmd"]


# === 5. enrich_image_batch =====================================================
def _gen_item(iid, **pl):
    base = dict(asset_kind="generate", gen_prompt_en="dell server",
                image_query_en="dell server", gen_seed=-1, emoji="")
    base.update(pl)
    return EnrichItem(id=iid, type=ENR_IMAGE, t_start=10.0, t_end=13.0,
                      payload=ImagePayload(**base))


def test_batch_success_marks_user_path(monkeypatch, tmp_path):
    monkeypatch.setattr(imagegen, "generate_image",
                        lambda *a, **k: str(tmp_path / "gen.png"))
    it = _gen_item("enr_1")
    other = EnrichItem(id="enr_2", type=ENR_IMAGE, t_start=20.0, t_end=23.0,
                       payload=ImagePayload(asset_kind="emoji", emoji="u26a1"))
    n = imagegen.enrich_image_batch([it, other], _cfg(tmp_path), log=_SILENT)
    assert n == 1
    assert it.payload.asset_kind == "user"
    assert it.payload.asset_path == str(tmp_path / "gen.png")
    assert other.payload.asset_kind == "emoji"      # не-generate не тронут


def test_batch_failure_falls_back_to_emoji_then_none(monkeypatch, tmp_path):
    monkeypatch.setattr(imagegen, "generate_image", lambda *a, **k: None)
    with_emoji = _gen_item("enr_1", emoji="u1f4c1")
    no_emoji = _gen_item("enr_2", emoji="")
    n = imagegen.enrich_image_batch([with_emoji, no_emoji], _cfg(tmp_path),
                                    log=_SILENT)
    assert n == 0
    assert with_emoji.payload.asset_kind == "emoji"
    assert no_emoji.payload.asset_kind == "none"


def test_batch_handles_dict_candidates(monkeypatch, tmp_path):
    """enrich_image_batch работает и на сырых dict-кандидатах (до сборки в
    EnrichItem) — payload как dict."""
    monkeypatch.setattr(imagegen, "generate_image",
                        lambda *a, **k: str(tmp_path / "g.png"))
    pt = {"type": "image", "payload": {"asset_kind": "generate",
                                       "gen_prompt_en": "linux desktop",
                                       "gen_seed": -1, "emoji": ""}}
    n = imagegen.enrich_image_batch([pt], _cfg(tmp_path), log=_SILENT)
    assert n == 1
    assert pt["payload"]["asset_kind"] == "user"
    assert pt["payload"]["asset_path"] == str(tmp_path / "g.png")


def test_batch_no_generate_points_returns_zero(tmp_path):
    it = EnrichItem(id="x", type=ENR_IMAGE, t_start=1.0, t_end=4.0,
                    payload=ImagePayload(asset_kind="emoji", emoji="u26a1"))
    assert imagegen.enrich_image_batch([it], _cfg(tmp_path), log=_SILENT) == 0


def test_batch_progress_reaches_full(monkeypatch, tmp_path):
    monkeypatch.setattr(imagegen, "generate_image",
                        lambda *a, **k: str(tmp_path / "g.png"))
    seen = []
    imagegen.enrich_image_batch([_gen_item("a"), _gen_item("b")],
                                _cfg(tmp_path), log=_SILENT,
                                on_progress=seen.append)
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_batch_success_writes_into_diffusion_candidate(monkeypatch, tmp_path):
    """V2: успешная генерация выбранного diffusion-кандидата пишет asset_path/
    preview в кандидат (рендер берёт через resolved_asset())."""
    monkeypatch.setattr(imagegen, "generate_image",
                        lambda *a, **k: str(tmp_path / "gen.png"))
    pl = ImagePayload(asset_kind="generate", gen_prompt_en="dell server",
                      source="diffusion", selected=0,
                      candidates=[{"source": "diffusion", "prompt": "dell server",
                                   "seed": -1}])
    it = EnrichItem(id="d1", type=ENR_IMAGE, t_start=1.0, t_end=4.0, payload=pl)
    n = imagegen.enrich_image_batch([it], _cfg(tmp_path), log=_SILENT)
    assert n == 1
    assert pl.candidates[0]["asset_path"] == str(tmp_path / "gen.png")
    assert pl.candidates[0]["preview"] == str(tmp_path / "gen.png")


# === 6. диффузия §5: арт-промпт, фото-суффикс, N=4, негатив, vae ================
def test_style_suffix_is_photographic_not_illustration():
    """Старый суффикс «clean illustration/orange accent/minimal» УДАЛЁН (c_diff
    §1: тянул в мультик); новый — фотографический (§5.3)."""
    s = imagegen.STYLE_SUFFIX
    assert "photorealistic" in s and "depth of field" in s
    assert "clean illustration" not in s
    assert "orange accent" not in s and "minimal" not in s


def test_negative_prompt_strong_and_people_variant():
    neg = imagegen.NEGATIVE_PROMPT
    for w in ("text", "letters", "cartoon", "plastic", "deformed"):
        assert w in neg                               # сильный негатив (§5.4)
    assert "extra fingers" in imagegen.NEGATIVE_PEOPLE


def test_diffusion_defaults_8_steps_cfg_1_5_n4():
    assert imagegen.DIFFUSION_STEPS == 8              # §5.4 (было 4)
    assert imagegen.CFG_SCALE == 1.5                  # §5.4 (было 1.0)
    assert imagegen.DIFFUSION_CANDIDATES == 4         # §5.5 (-b 4)


def test_generate_image_people_prompt_adds_hand_negative(monkeypatch, tmp_path):
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        from pathlib import Path
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    imagegen.generate_image("a developer coding at night", imagegen.STYLE_SUFFIX,
                            1, 768, 768, cfg=_cfg(tmp_path),
                            cache_dir=tmp_path / "c", log=_SILENT)
    neg = seen["cmd"][seen["cmd"].index("-n") + 1]
    assert "extra fingers" in neg                     # люди → +негатив рук


def test_generate_image_vae_flag_when_configured(monkeypatch, tmp_path):
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf")
    vae = _touch(tmp_path / "vae.safetensors")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        from pathlib import Path
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    imagegen.generate_image("data center", imagegen.STYLE_SUFFIX, 1, 768, 768,
                            cfg=_cfg(tmp_path, imagegen_vae=str(vae)),
                            cache_dir=tmp_path / "c", log=_SILENT)
    assert "--vae" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--vae") + 1] == str(vae)


def test_generate_max_vram_flag_default(monkeypatch, tmp_path):
    """VRAM-гард (#40): sd-cli зовётся с --max-vram -1.0 по умолчанию —
    авто-детект свободной VRAM (фолбэк, обещанный llm.unload/_wait_ollama)."""
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        from pathlib import Path
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    imagegen.generate_image("q english", imagegen.STYLE_SUFFIX, 1, 768, 768,
                            cfg=_cfg(tmp_path), cache_dir=tmp_path / "c",
                            log=_SILENT)
    assert "--max-vram" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--max-vram") + 1] == "-1.0"


def test_generate_max_vram_disabled_when_zero(monkeypatch, tmp_path):
    """cfg=0 — эскейп-хэтч: флаг не добавляется (неограниченное поведение)."""
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        from pathlib import Path
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    imagegen.generate_image("q english", imagegen.STYLE_SUFFIX, 1, 768, 768,
                            cfg=_cfg(tmp_path, imagegen_max_vram=0.0),
                            cache_dir=tmp_path / "c", log=_SILENT)
    assert "--max-vram" not in seen["cmd"]


def test_generate_candidates_batch_single_process(monkeypatch, tmp_path):
    """#82: N кандидатов = ОДИН sd-cli-процесс (`-b N -s base`), не N загрузок
    ~4 ГБ модели. Мокаем subprocess: ровно один вызов; флаги -b N / -s base /
    -o %03d / --max-vram; пишем N фейковых PNG по %03d-шаблону → N путей назад."""
    from pathlib import Path
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf", b"M" * 100)
    cfg = _cfg(tmp_path)
    cache = tmp_path / "cache"
    calls = {"n": 0}
    base = imagegen._seed_for("data center aisle", -1)

    def fake_run(cmd, **kw):
        calls["n"] += 1
        assert cmd[cmd.index("-b") + 1] == "4"             # батч на 4 сида
        assert cmd[cmd.index("-s") + 1] == str(base)       # детерминированный base
        assert "--max-vram" in cmd                         # VRAM-гард не потерян
        outp = cmd[cmd.index("-o") + 1]
        assert "%03d" in outp                              # image-sequence шаблон
        for i in range(4):
            Path(outp % i).write_bytes(b"\x89PNG fake")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    paths = imagegen.generate_candidates("data center aisle", cfg,
                                         cache_dir=cache, log=_SILENT)
    assert calls["n"] == 1                                 # ОДИН процесс, не 4
    assert len(paths) == 4 and len(set(paths)) == 4
    for p in paths:
        assert Path(p).is_file() and p.endswith(".png")
    # временные b_%03d.tmp.png переименованы в кэш-пути, не висят
    assert not list(Path(cache).resolve().glob("b_*.tmp.png"))


def test_generate_candidates_all_cached_no_subprocess(monkeypatch, tmp_path):
    """Идемпотентность: если все N кандидатов уже в кэше — 0 subprocess (#82)."""
    from pathlib import Path
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf", b"M" * 100)
    cfg = _cfg(tmp_path)
    cache = tmp_path / "cache"

    def writing_run(cmd, **kw):
        outp = cmd[cmd.index("-o") + 1]
        for i in range(4):
            Path(outp % i).write_bytes(b"\x89PNG fake")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", writing_run)
    first = imagegen.generate_candidates("linux server room", cfg,
                                         cache_dir=cache, log=_SILENT)
    assert len(first) == 4

    def boom(cmd, **kw):
        raise AssertionError("subprocess must NOT run when all cached")

    monkeypatch.setattr(subprocess, "run", boom)
    again = imagegen.generate_candidates("linux server room", cfg,
                                         cache_dir=cache, log=_SILENT)
    assert again == first                                 # fast-path, тот же список


def test_generate_candidates_batch_failure_falls_back(monkeypatch, tmp_path):
    """Батч отвалился (или его -b/%03d-семантика иная) → посидовый generate_image
    фолбэк (корректность > 1 загрузка). Мок: -b → non-zero, одиночный → пишет PNG."""
    from pathlib import Path
    _touch(tmp_path / "sd-cli.exe")
    _touch(tmp_path / "m.gguf", b"M" * 100)
    cfg = _cfg(tmp_path)
    cache = tmp_path / "cache"
    seen = {"batch": 0, "single": 0}

    def fake_run(cmd, **kw):
        if "-b" in cmd:                                   # батч → падаем
            seen["batch"] += 1
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        seen["single"] += 1                               # посидовый → успех
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    paths = imagegen.generate_candidates("edge network node", cfg,
                                         cache_dir=cache, log=_SILENT)
    assert seen["batch"] == 1                             # попробовали батч один раз
    assert seen["single"] == 4                            # откатились на 4 одиночных
    assert len(paths) == 4 and len(set(paths)) == 4


def test_generate_candidates_empty_prompt_returns_empty(tmp_path):
    assert imagegen.generate_candidates("", _cfg(tmp_path), log=_SILENT) == []


def test_generate_candidates_failures_skipped(monkeypatch, tmp_path):
    """Упавший сид (None) просто не попадает в список — остальные живут."""
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return None if calls["n"] % 2 else str(tmp_path / f"ok{calls['n']}.png")

    monkeypatch.setattr(imagegen, "generate_image", flaky)
    paths = imagegen.generate_candidates("x scene", _cfg(tmp_path), log=_SILENT)
    assert 0 < len(paths) < 4                          # часть отсеялась
