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
        # флаги R1 спайка присутствуют
        assert "--diffusion-fa" in cmd
        assert cmd[cmd.index("--steps") + 1] == "4"
        assert cmd[cmd.index("--cfg-scale") + 1] == "1.0"
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
