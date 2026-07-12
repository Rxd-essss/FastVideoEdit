"""Локальная SD-генерация контекстных картинок (PLAN_V11 §2, ТРЕК-2).

Бэкенд — внешний бинарь stable-diffusion.cpp CUDA (паттерн
``render._resolve_deepfilter_bin`` / ``enhance_audio``: ``sd-cli.exe`` + DLL
рядом), модель SDXL-Turbo Q4_0 GGUF (~3.94 ГБ) скачивает ПОЛЬЗОВАТЕЛЬ — в репо
не кладём (паттерн DeepFilterNet). Полностью оффлайн (zero-upload), torch не
нужен.

Контракт graceful-degrade как у нейроденойза: ЛЮБОЙ сбой (нет бинаря/модели,
ненулевой exit, пустой/битый PNG, таймаут) -> ``None`` (или 0 для батча) и
честная строка в лог; задача обогащения НЕ падает — маршрутизатор откатывается
на эмодзи-фолбэк (см. ``vpipe/enrich_llm.py``).

КРИТИЧНО про текст в кадре (§2): SDXL-Turbo рисует кракозябры вместо букв.
Поэтому маршрутизация по ``style`` живёт в ``enrich_llm.match_user_assets``:
photo -> SD напрямую; diagram/chart -> переписанный text-free промпт; icon ->
эмодзи (НЕ SD). Здесь только сам запуск бинаря по уже готовому промпту.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

LogFn = Callable[..., None]


def _noop(*_a, **_k) -> None:
    pass


# Repo root (родитель vpipe/) — `tools/sd-cli.exe` лежит относительно него.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# --- константы генерации (MONTAGE_V2 §5, c_diff §2; A/B 07a vs 07b) -----------
# СТРОГО фото-сцены (роутер enrich_llm уже отсёк данные/схемы → код-графика).
# Слоп лечит ПРОМПТ, не модель (доказано кадром): фото-суффикс + сильный негатив.
# Старый суффикс «clean illustration, orange accent, minimal» УДАЛЁН (тянул в
# мультик + красил всё оранжевым — c_diff §1). Новый — фотографический (§5.3).
STYLE_SUFFIX = (", cinematic, photorealistic, professional photography, "
                "moody dark scene, teal and cyan ambient light, "
                "shallow depth of field, ultra detailed")
# Сильный негатив (c_diff §2б, критичен): бьёт текст/буквы/цифры + мультяшность/
# пластик/cgi/деформации. Для людей добавляется NEGATIVE_PEOPLE (руки/пальцы).
NEGATIVE_PROMPT = (
    "text, words, letters, numbers, labels, watermark, logo, signature, "
    "cartoon, illustration, drawing, painting, blurry, low quality, deformed, "
    "distorted, oversaturated, plastic, cgi, render, ugly, jpeg artifacts")
NEGATIVE_PEOPLE = ", extra fingers, mutated hands"   # +для сцен с людьми (§5.4)
_PEOPLE_RE = re.compile(
    r"\b(person|people|man|woman|men|women|coder|developer|programmer|hand|"
    r"hands|face|portrait|human|guy|girl|boy)\b", re.IGNORECASE)
DIFFUSION_STEPS = 8               # SDXL-Turbo: 8 шагов чище 4 (+3 c, c_diff §2а)
CFG_SCALE = 1.5                  # Turbo на 1.5 крепче держит композицию (§5.4)
SAMPLING_METHOD = "euler_a"
DIFFUSION_CANDIDATES = 4         # N сидов на момент (-b 4 -s -1) → выбор (§5.5)
SD_TIMEOUT_S = 300               # один кадр на RTX 3080 ~ секунды; гард на зависание
CACHE_DIR = Path("cache") / "enrich_img"


def _resolve_sd_bin(configured: str) -> Optional[str]:
    """Резолв ``sd-cli`` к запускаемому пути (или ``None``). НИКОГДА не бросает —
    отсутствие SD-бинаря должно деградировать на эмодзи, а не валить задачу
    (зеркало ``render._resolve_deepfilter_bin``).

    Порядок: абсолютный путь -> repo-root-relative (вендоренный
    ``tools/sd-cli.exe``) -> cwd-relative -> PATH (полное имя, затем голый
    stem, чтобы ``sd-cli`` в PATH сработал даже когда конфиг говорит
    ``tools/sd-cli.exe``)."""
    configured = str(configured or "").strip()
    if not configured:
        return None
    p = Path(configured)
    if p.is_absolute():
        return str(p) if p.exists() else None
    cand = _REPO_ROOT / configured
    if cand.exists():
        return str(cand)
    if p.exists():
        return str(p)
    return shutil.which(configured) or shutil.which(p.name) or shutil.which(p.stem)


def _resolve_model(configured: str) -> Optional[str]:
    """Путь к .gguf-модели: абсолют -> repo-root-relative -> cwd-relative.
    Пусто/не существует -> ``None`` (SD не настроен — эмодзи-фолбэк). НЕ бросает."""
    configured = str(configured or "").strip()
    if not configured:
        return None
    p = Path(configured)
    if p.is_absolute():
        return str(p) if p.exists() else None
    cand = _REPO_ROOT / configured
    if cand.exists():
        return str(cand)
    return str(p) if p.exists() else None


def _model_hash(model_path: str) -> str:
    """Дешёвая сигнатура модели для кэш-ключа: имя + размер + mtime (НЕ хэш
    всего .gguf — 4 ГБ читать на каждый кадр нельзя). Меняется при подмене
    файла модели -> кэш честно инвалидируется."""
    try:
        st = os.stat(model_path)
        return f"{Path(model_path).name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return Path(model_path).name


def _cache_key(prompt: str, suffix: str, negative: str, seed: int,
               W: int, H: int, steps: int, model_sig: str) -> str:
    """SHA1 кэш-ключ кадра: всё, что меняет картинку (промпт+суффикс+негатив+
    сид+размеры+шаги+сигнатура модели). Сид ВХОДИТ в ключ — -1 (хэш query) и
    фиксированный сид дают разные файлы (§2: детерминизм)."""
    blob = f"{prompt}\x00{suffix}\x00{negative}\x00{seed}\x00{W}x{H}" \
           f"\x00{steps}\x00{model_sig}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _seed_for(query_en: str, seed: int) -> int:
    """Детерминированный сид (§2): seed>=0 — как есть; -1 — стабильный хэш
    query_en (один и тот же запрос -> одна и та же картинка между прогонами,
    не случайная). Держим в положительном int32, sd-cli берёт >=0."""
    if seed is not None and seed >= 0:
        return int(seed)
    h = hashlib.sha1(query_en.encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def generate_image(query_en: str, style_suffix: str, seed: int, W: int, H: int,
                   *, cfg, cache_dir: Optional[Path] = None,
                   log: LogFn = _noop) -> Optional[str]:
    """Сгенерировать ОДИН кадр через ``sd-cli`` -> путь к PNG или ``None``.

    ``query_en`` — БОГАТЫЙ арт-дирекшн-промпт photo-сцены (роутер enrich_llm уже
    отсёк данные/схемы в код-графику — c_diff §3). ``style_suffix`` — фото-хвост
    (дефолт ``STYLE_SUFFIX``); негатив — ``NEGATIVE_PROMPT`` (+``NEGATIVE_PEOPLE``
    при людях в промпте — §5.4). Кэш по ``_cache_key`` в
    ``cache/enrich_img/<sha1>.png`` (идемпотентно: есть файл — отдаём; пишем
    атомарно .tmp->replace). ЛЮБОЙ сбой (нет бинаря/модели, exit!=0, пустой PNG,
    таймаут) -> ``None`` (фолбэк выше по стеку).

    Флаги (§5.4, c_diff §2): ``--steps`` (cfg.imagegen_steps), ``--cfg-scale``
    (cfg.imagegen_cfg, дефолт 1.5), ``--sampling-method euler_a --diffusion-fa
    -W/-H``; ``--vae`` (cfg.imagegen_vae, если задан — sdxl_vae_fp16fix);
    ``imagegen_vae_on_cpu`` -> ``--vae-on-cpu`` (аварийный VRAM-путь). Работаем
    строго по cfg."""
    query_en = " ".join((query_en or "").split())
    if not query_en:
        return None
    binp = _resolve_sd_bin(getattr(cfg, "imagegen_bin", "tools/sd-cli.exe"))
    if not binp:
        log("  SD: бинарь sd-cli не найден — фолбэк.")
        return None
    model = _resolve_model(getattr(cfg, "imagegen_model", ""))
    if not model:
        log("  SD: модель .gguf не настроена (imagegen_model) — фолбэк.")
        return None

    suffix = style_suffix if style_suffix is not None else STYLE_SUFFIX
    steps = max(1, int(getattr(cfg, "imagegen_steps", DIFFUSION_STEPS)))
    cfg_scale = float(getattr(cfg, "imagegen_cfg", CFG_SCALE) or CFG_SCALE)
    real_seed = _seed_for(query_en, seed)
    prompt = query_en + suffix
    # Сильнее негатив для сцен с людьми (руки/пальцы — c_diff §2б, кадр 03).
    negative = NEGATIVE_PROMPT
    if _PEOPLE_RE.search(prompt):
        negative += NEGATIVE_PEOPLE
    vae = _resolve_model(getattr(cfg, "imagegen_vae", "") or "")

    # АБСОЛЮТНЫЙ путь обязателен: enrich.ImagePayload._abs_path (P2 path-traversal
    # guard) обнуляет любой ОТНОСИТЕЛЬНЫЙ asset_path при перезагрузке/санитайзе
    # плана — иначе сгенерированная картинка «отвязывается» от карточки (asset_kind
    # =user, но asset_path пустой → blank-превью, рендер дропает). resolve() здесь.
    cache = (Path(cache_dir) if cache_dir is not None else CACHE_DIR).resolve()
    # cfg-scale/vae входят в ключ (меняют картинку): добавлены в model_sig-слот
    # как часть сигнатуры (старые кэши инвалидируются честно — новые настройки §5).
    key = _cache_key(prompt, suffix, negative, real_seed, W, H,
                     steps, f"{_model_hash(model)}:cfg{cfg_scale}:vae{bool(vae)}")
    out_png = cache / f"{key}.png"
    if out_png.is_file() and out_png.stat().st_size > 0:
        return str(out_png)             # идемпотентно: уже сгенерировано

    cache.mkdir(parents=True, exist_ok=True)
    # ВАЖНО: sd-cli выбирает формат по расширению ``-o`` и НЕ понимает ``.tmp``
    # (дописывает ``.png`` -> файл ``<key>.png.tmp.png``). Поэтому временный путь
    # обязан оканчиваться на ``.png``; затем атомарно переименовываем в out_png.
    tmp = out_png.with_name(out_png.stem + ".tmp.png")
    cmd = [
        binp, "-M", "img_gen",
        "-m", model,
        "-p", prompt,
        "-n", negative,
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--sampling-method", SAMPLING_METHOD,
        "--diffusion-fa",
        "-W", str(int(W)), "-H", str(int(H)),
        "-s", str(real_seed),
        "-o", str(tmp),
    ]
    if vae:
        cmd += ["--vae", vae]           # sdxl_vae_fp16fix: стабильнее цвет (§5.6)
    if bool(getattr(cfg, "imagegen_vae_on_cpu", False)):
        cmd.append("--vae-on-cpu")
    # VRAM-гард (#40): sd-cli режет граф под доступную VRAM, когда qwen3 не
    # успела выгрузиться (8 ГБ карта) — обещанный llm.unload/_wait_ollama_unloaded
    # фолбэк. <0 = авто-детект свободной VRAM; 0 = выкл (неограниченно).
    try:
        max_vram = float(getattr(cfg, "imagegen_max_vram", -1.0))
    except (TypeError, ValueError):
        max_vram = -1.0
    if max_vram != 0.0:
        cmd += ["--max-vram", str(max_vram)]
    log(f"  SD: генерация «{query_en}» ({W}x{H}, {steps} шагов, "
        f"cfg {cfg_scale}, seed {real_seed})…")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=SD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log(f"  SD: таймаут {SD_TIMEOUT_S} c — эмодзи-фолбэк.")
        _unlink(tmp)
        return None
    except OSError as e:
        log(f"  SD: запуск не удался ({e}) — эмодзи-фолбэк.")
        _unlink(tmp)
        return None
    if r.returncode != 0:
        tail = " | ".join((r.stderr or r.stdout or "").strip().splitlines()[-3:])
        log(f"  SD: exe завершился с ошибкой (exit {r.returncode})"
            + (f": {tail}" if tail else "") + " — эмодзи-фолбэк.")
        _unlink(tmp)
        return None
    if not (tmp.is_file() and tmp.stat().st_size > 0):
        log("  SD: вышел пустой/отсутствующий PNG — эмодзи-фолбэк.")
        _unlink(tmp)
        return None
    try:
        os.replace(tmp, out_png)
    except OSError as e:
        log(f"  SD: не удалось сохранить кадр ({e}) — эмодзи-фолбэк.")
        _unlink(tmp)
        return None
    return str(out_png)


def generate_candidates(prompt: str, cfg, *, n: Optional[int] = None,
                        cache_dir: Optional[Path] = None,
                        log: LogFn = _noop) -> list[str]:
    """N кандидатов одного photo-момента (§5.5, c_diff §2г): N разных сидов одного
    арт-дирекшн-промпта → список путей к PNG (юзер выберет лучший в «Монтаже»).

    N = cfg.imagegen_candidates (дефолт ``DIFFUSION_CANDIDATES``=4). Сиды
    детерминированы (хэш «prompt#i») — повторный прогон даёт те же 4 кадра (кэш
    переиспользуется). Каждый кадр через ``generate_image`` (graceful-degrade:
    упавший сид просто не попадает в список). Пустой/без-бинаря → []. Реальный
    батч ``-b N`` амортизирует загрузку модели — здесь последовательно (бинарь
    кэширует модель в рамках процесса при общем прогоне; простота > 1 загрузка)."""
    prompt = " ".join((prompt or "").split())
    if not prompt:
        return []
    size = max(64, int(getattr(cfg, "imagegen_size", 768)))
    count = n if (isinstance(n, int) and n > 0) else \
        max(1, int(getattr(cfg, "imagegen_candidates", DIFFUSION_CANDIDATES)))
    out: list[str] = []
    seen: set[str] = set()
    for i in range(count):
        # детерминированный РАЗНЫЙ сид на кандидата: хэш «prompt#i» → стабильно
        # между прогонами, но 4 разных кадра (не один и тот же сид × 4).
        seed = _seed_for(f"{prompt}#{i}", -1)
        try:
            path = generate_image(prompt, STYLE_SUFFIX, seed, size, size,
                                  cfg=cfg, cache_dir=cache_dir, log=log)
        except Exception as e:  # noqa: BLE001 — один сид не валит остальные
            log(f"  SD: кандидат {i} упал ({e}).")
            path = None
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def _payload_of(p):
    """Достать payload-объект точки: ``EnrichItem`` (с дата-классом payload, как
    после ``detect_all``) ИЛИ сырой dict-кандидат (``payload`` — dict, как из
    детектора до сборки). Возврат — объект с атрибутами/ключами asset_kind и др.,
    или ``None`` если это не точка-иллюстрация."""
    payload = getattr(p, "payload", None)
    if payload is None and isinstance(p, dict):
        payload = p.get("payload")
    return payload


def _pl_get(pl, key, default=None):
    if isinstance(pl, dict):
        return pl.get(key, default)
    return getattr(pl, key, default)


def _pl_set(pl, key, value) -> None:
    if isinstance(pl, dict):
        pl[key] = value
    else:
        setattr(pl, key, value)


def enrich_image_batch(points: list, cfg, log: LogFn = _noop,
                       on_progress=None) -> int:
    """Сгенерировать SD-картинки точкам с ``asset_kind=="generate"`` (мутирует
    ``points``). Принимает и ``EnrichItem`` (после detect_all), и сырые
    dict-кандидаты. Возврат — число успешно сгенерированных.

    Маршрутизатор enrich_llm уже проставил таким точкам ``asset_kind="generate"``,
    ``gen_prompt_en`` (фактический промпт после diagram->text-free) и эмодзи-
    фолбэк в поле ``emoji``; ``gen_seed`` (-1 = хэш query). Успех ->
    ``asset_kind="user"`` + абсолютный ``asset_path`` (рендер ест
    сгенерированный единым user-путём); сбой -> откат на эмодзи (поле ``emoji``
    непусто), иначе ``asset_kind="none"``. Безопасен к сбоям: одна упавшая точка
    не валит остальные, исключение наружу не летит."""
    gen: list = []
    for p in points:
        pl = _payload_of(p)
        if pl is not None and _pl_get(pl, "asset_kind") == "generate":
            gen.append(pl)
    if not gen:
        return 0
    size = max(64, int(getattr(cfg, "imagegen_size", 768)))
    prog = on_progress if on_progress is not None else (lambda _f: None)
    n = len(gen)
    made = 0
    for i, pl in enumerate(gen):
        prog(i / n)
        query = _pl_get(pl, "gen_prompt_en") or _pl_get(pl, "image_query_en") or ""
        seed = _pl_get(pl, "gen_seed")
        seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else -1
        try:
            path = generate_image(query, STYLE_SUFFIX, seed, size, size,
                                  cfg=cfg, log=log)
        except Exception as e:  # noqa: BLE001 — одна точка не валит батч
            log(f"  SD: точка «{query}» упала ({e}) — фолбэк.")
            path = None
        if path:
            _pl_set(pl, "asset_kind", "user")  # рендер ест как user-ассет
            _pl_set(pl, "asset_path", path)
            _write_diffusion_candidate(pl, path)  # V2: и в выбранный кандидат
            made += 1
        elif _pl_get(pl, "emoji"):
            _pl_set(pl, "asset_kind", "emoji")  # маршрутизатор подобрал эмодзи
        else:
            _pl_set(pl, "asset_kind", "none")   # ни SD, ни эмодзи — без ассета
    prog(1.0)
    return made


def _write_diffusion_candidate(pl, path: str) -> None:
    """Записать сгенерированный PNG в ВЫБРАННЫЙ diffusion-кандидат (V2): рендер
    берёт ассет через ``ImagePayload.resolved_asset()`` (= candidate.asset_path).
    Без кандидатов (старый плоский план) — no-op (рендер возьмёт плоский
    asset_path по шиму обратной совместимости)."""
    cands = _pl_get(pl, "candidates")
    sel = _pl_get(pl, "selected", 0)
    if not isinstance(cands, list) or not (isinstance(sel, int) and
                                           0 <= sel < len(cands)):
        return
    c = cands[sel]
    if isinstance(c, dict) and c.get("source") == "diffusion":
        c["asset_path"] = path
        c["preview"] = path
