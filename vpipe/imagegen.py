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
from types import SimpleNamespace
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


def _sd_cmd_core(binp: str, model: str, prompt: str, negative: str, seed: int,
                 W: int, H: int, steps: int, cfg_scale: float,
                 vae: Optional[str], cfg) -> list[str]:
    """Общий костяк ``sd-cli``-команды: один кадр (``generate_image``) и батч
    (``_run_sd_batch``) делят ИДЕНТИЧНЫЕ флаги (§5.4, #40) — включая ``--vae`` и
    ``--max-vram``, чтобы батч-путь НЕ терял VRAM-гард. Вызывающий дописывает
    ``-o`` (и ``-b N`` для батча)."""
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
        "-s", str(int(seed)),
    ]
    if vae:
        cmd += ["--vae", vae]           # sdxl_vae_fp16fix: стабильнее цвет (§5.6)
    if bool(getattr(cfg, "imagegen_vae_on_cpu", False)):
        cmd.append("--vae-on-cpu")
    # VRAM-гард (#40): <0 = авто-детект свободной VRAM; 0 = выкл (неограниченно).
    try:
        max_vram = float(getattr(cfg, "imagegen_max_vram", -1.0))
    except (TypeError, ValueError):
        max_vram = -1.0
    if max_vram != 0.0:
        cmd += ["--max-vram", str(max_vram)]
    return cmd


def _plan_frame(query_en: str, style_suffix, seed: int, W: int, H: int, cfg,
                cache_dir, model: str) -> SimpleNamespace:
    """Детерминированный план ОДНОГО кадра: итоговый кэш-путь + разрешённые
    параметры команды. ``generate_image`` и ``generate_candidates`` (батч) зовут
    ЕГО, чтобы посидовый кэш-ключ совпадал байт-в-байт — тогда повторный прогон
    батча попадает в all-cached fast-path (0 subprocess). НЕ бросает при валидной
    модели."""
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
    # плана — иначе сгенерированная картинка «отвязывается» от карточки. resolve().
    cache = (Path(cache_dir) if cache_dir is not None else CACHE_DIR).resolve()
    # cfg-scale/vae входят в ключ (меняют картинку) через слот model_sig.
    key = _cache_key(prompt, suffix, negative, real_seed, W, H, steps,
                     f"{_model_hash(model)}:cfg{cfg_scale}:vae{bool(vae)}")
    return SimpleNamespace(out_png=cache / f"{key}.png", cache=cache,
                           prompt=prompt, negative=negative, seed=real_seed,
                           steps=steps, cfg_scale=cfg_scale, vae=vae)


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

    # §5-план кадра (кэш-ключ + разрешённые параметры) — общий с батч-путём, чтобы
    # посидовый кэш совпадал байт-в-байт (см. _plan_frame / generate_candidates).
    fr = _plan_frame(query_en, style_suffix, seed, W, H, cfg, cache_dir, model)
    out_png = fr.out_png
    if out_png.is_file() and out_png.stat().st_size > 0:
        return str(out_png)             # идемпотентно: уже сгенерировано

    fr.cache.mkdir(parents=True, exist_ok=True)
    # ВАЖНО: sd-cli выбирает формат по расширению ``-o`` и НЕ понимает ``.tmp``
    # (дописывает ``.png`` -> файл ``<key>.png.tmp.png``). Поэтому временный путь
    # обязан оканчиваться на ``.png``; затем атомарно переименовываем в out_png.
    tmp = out_png.with_name(out_png.stem + ".tmp.png")
    cmd = _sd_cmd_core(binp, model, fr.prompt, fr.negative, fr.seed, W, H,
                       fr.steps, fr.cfg_scale, fr.vae, cfg) + ["-o", str(tmp)]
    log(f"  SD: генерация «{query_en}» ({W}x{H}, {fr.steps} шагов, "
        f"cfg {fr.cfg_scale}, seed {fr.seed})…")
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


def _run_sd_batch(prompt: str, base: int, count: int, W: int, H: int, cfg,
                  cache_dir, log: LogFn) -> Optional[list[Path]]:
    """Один процесс sd-cli на N кандидатов (``-b N -s base``): грузит ~4 ГБ GGUF
    ОДИН раз вместо N (#82). Возвращает упорядоченный список ``count`` временных
    PNG (сиды base..base+count-1, шаблон ``b_%03d.tmp.png``) или ``None`` при
    любом сбое/несовпадении числа выходов — вызывающий откатывается на посидовый
    цикл (корректность > 1 загрузка; -b/%03d-семантика бинаря может отличаться)."""
    binp = _resolve_sd_bin(getattr(cfg, "imagegen_bin", "tools/sd-cli.exe"))
    model = _resolve_model(getattr(cfg, "imagegen_model", ""))
    if not binp or not model:
        return None
    fr = _plan_frame(prompt, STYLE_SUFFIX, base, W, H, cfg, cache_dir, model)
    fr.cache.mkdir(parents=True, exist_ok=True)
    tmpl = fr.cache / "b_%03d.tmp.png"
    cmd = _sd_cmd_core(binp, model, fr.prompt, fr.negative, base, W, H,
                       fr.steps, fr.cfg_scale, fr.vae, cfg) \
        + ["-b", str(count), "-o", str(tmpl)]
    log(f"  SD: батч {count} кандидатов «{prompt}» "
        f"({W}x{H}, {fr.steps} шагов, seed base {base})…")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=SD_TIMEOUT_S * max(1, count))
    except (subprocess.TimeoutExpired, OSError) as e:
        log(f"  SD: батч упал ({e}) — посидовый фолбэк.")
        return None
    if r.returncode != 0:
        tail = " | ".join((r.stderr or r.stdout or "").strip().splitlines()[-3:])
        log(f"  SD: батч завершился с ошибкой (exit {r.returncode})"
            + (f": {tail}" if tail else "") + " — посидовый фолбэк.")
        return None
    produced: list[Path] = []
    for i in range(count):
        f = fr.cache / f"b_{i:03d}.tmp.png"
        if f.is_file() and f.stat().st_size > 0:
            produced.append(f)
    if len(produced) != count:          # наименование/начало-индекс не совпали
        for f in produced:
            _unlink(f)
        return None
    return produced


def generate_candidates(prompt: str, cfg, *, n: Optional[int] = None,
                        cache_dir: Optional[Path] = None,
                        log: LogFn = _noop) -> list[str]:
    """N кандидатов одного photo-момента (§5.5): N ПОСЛЕДОВАТЕЛЬНЫХ детерминиро-
    ванных сидов (base..base+N-1, base=хэш промпта) → список путей к PNG (юзер
    выберет лучший в «Монтаже»).

    N = cfg.imagegen_candidates (дефолт ``DIFFUSION_CANDIDATES``=4). #82: ОДИН
    батч-процесс sd-cli (``-b N``) грузит ~4 ГБ GGUF один раз вместо N. Идемпо-
    тентно: если все N уже в кэше — 0 subprocess (fast-path). Батч недоступен/
    сбой/не то число выходов → откат на посидовый ``generate_image`` (корректность
    > 1 загрузка; упавший сид просто не попадает в список). Пустой промпт → []."""
    prompt = " ".join((prompt or "").split())
    if not prompt:
        return []
    size = max(64, int(getattr(cfg, "imagegen_size", 768)))
    count = n if (isinstance(n, int) and n > 0) else \
        max(1, int(getattr(cfg, "imagegen_candidates", DIFFUSION_CANDIDATES)))
    binp = _resolve_sd_bin(getattr(cfg, "imagegen_bin", "tools/sd-cli.exe"))
    model = _resolve_model(getattr(cfg, "imagegen_model", ""))
    base = _seed_for(prompt, -1)
    seeds = [(base + i) & 0x7FFFFFFF for i in range(count)]
    # Плановый итоговый кэш-путь на КАЖДЫЙ сид (ТОТ ЖЕ ключ, что у generate_image).
    planned = ([(_plan_frame(prompt, STYLE_SUFFIX, s, size, size, cfg,
                             cache_dir, model).out_png, s) for s in seeds]
               if model else [])
    if planned:
        cached = [str(p) for (p, _s) in planned
                  if p.is_file() and p.stat().st_size > 0]
        if len(cached) == count:                 # FAST-PATH: 0 subprocess
            return list(dict.fromkeys(cached))
    # Один батч-процесс амортизирует загрузку модели на все N сидов (#82).
    if binp and model and planned:
        produced = _run_sd_batch(prompt, base, count, size, size, cfg,
                                 cache_dir, log)
        if produced is not None and len(produced) == count:
            out: list[str] = []
            for (final_png, _s), src in zip(planned, produced):
                try:
                    os.replace(src, final_png)   # b_%03d.tmp.png → <key>.png
                    out.append(str(final_png))
                except OSError:
                    _unlink(src)
            if out:
                return list(dict.fromkeys(out))
    # ФОЛБЭК: посидовый цикл (батч не поддержан/сбой/не то число выходов).
    out2: list[str] = []
    seen: set[str] = set()
    for s in seeds:
        try:
            p = generate_image(prompt, STYLE_SUFFIX, s, size, size,
                               cfg=cfg, cache_dir=cache_dir, log=log)
        except Exception as e:  # noqa: BLE001 — один сид не валит остальные
            log(f"  SD: кандидат {s} упал ({e}).")
            p = None
        if p and p not in seen:
            seen.add(p)
            out2.append(p)
    return out2


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
