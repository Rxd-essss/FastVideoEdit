# FastVideoEdit — детальные патчи по аудиту (2026-07-12)

> **Полный apply-ready список всех исправлений** по 103 находкам аудита (96 патчей после мержей).
> Патчи ещё **НЕ применены** — это спецификация для реализации. Каждый патч: текущий код → изменение,
> обоснование, риск/безопасность, тест. Порядок: волна → серьёзность. Сначала прочитайте раздел
> «Три сквозные темы» — многие патчи ссылаются на общие хелперы, их нельзя реализовывать вразнобой.

| Волна | Патчей | | Серьёзность | Патчей |
|---|---|---|---|---|
| 1. Стабильность (краши / зависания / обрыв вывода) | 19 | | 🔴 CRITICAL | 6 |
| 2. Субтитры | 12 | | 🟠 HIGH | 17 |
| 3. «Монтаж» + настройки рендера | 21 | | 🟡 MEDIUM | 37 |
| 4. Устойчивость UI + краевые случаи | 28 | | ⚪ LOW | 36 |
| 5. Производительность + долги тестов | 16 | |  |  |

## Три сквозные темы (реализовать как общий код, не дублировать)

**R1 — единый «эффективный cutlist».** Субтитры/.srt/главы/enrich строятся по полному `Timeline`, а `render()` выкидывает короткие сегменты. Ввести один хелпер, считающий sliver-фильтрованный cutlist ОДИН раз, и использовать его И для энкода, И для всех генераторов субтитров/оверлеев/глав. Патчи `subtitles.py:356/166`, `render.py:876`, сборка субтитров в `serve.py` ссылаются на него, а не изобретают свой трим.

**R2 — не гнать гигантский граф инлайном + валидировать вывод.** Большой `-filter_complex` писать во временный файл и передавать `-filter_complex_script` (render/censor/measure). После рендера — `ffprobe` длительности и сравнение с ожидаемой. `render.py:1173`, `ffmpeg_utils.py:149`, `serve.py:1338`, `probe.py:37`, `censor.py:88` — одно скоординированное изменение.

**R3 — единый контракт UI↔бэкенд.** Персист настроек модалки + настройки очереди/watch + схема кандидатов «Монтаж» сходятся на реальном payload бэкенда (`candidates`/`selected`, `/api/enrich/select`).

Заметки по связям внутри кластеров (порядок применения, общие хелперы, мержи) — в конце документа.

---


# Волна 1 — Стабильность (краши / зависания / обрыв вывода)

## 1. plan_render drops every schematic (code-graphics) point: it switches on flat pl2.asset_kind (always 'none' for schematic) instead of resolved_asset()
🔴 CRITICAL · симптом 4 · effort ? · `schematic-render-via-resolved-asset`

- **Файл / якорь:** `vpipe/enrich.py` — plan_render() step 5, image/animation asset-selection branch (~lines 1329-1362)
- **Закрывает находки:** vpipe/enrich.py:1348, vpipe/enrich.py:1330

**Проблема.** The flagship Montage V2 code-graphics PNG never reaches the rendered mp4. plan_render step 5 builds StillOverlay/AnimOverlay off the flat pl2.asset_kind, hitting the `else: continue` for any schematic-selected point (asset_kind is DELIBERATELY 'none' per _legacy_sync_from_candidate). The rendered PNG lives only in candidates[selected].asset_path/preview, resolvable ONLY via ImagePayload.resolved_asset() (which .kadr export at serve.py:2567 already uses). Diffusion survives only because _generate_point_candidates coincidentally syncs flat fields; schematic is silently omitted with no status_note.

**Почему так.** resolved_asset() is the documented single-source resolver (docstring: «Единая точка выбранный кандидат → что и откуда рисовать»); .kadr export and _legacy_sync_from_candidate already treat it as canonical. Routing the render through it makes schematic (and stock, and re-selected diffusion) correct in ONE place instead of relying on fragile flat-field syncs. AnimationPayload has no chosen_candidate/resolved_asset, so animations fall to the untouched flat path. The full-frame branch is because codegfx PNGs are 1920x1080 transparent cut-outs, not corner thumbnails — a 32% PiP would render the infographic as a broken thumbnail.

**Текущий код:**
```
        # image / animation (PiP в верхнем углу)
        pl2 = it.payload
        path: Optional[str] = None
        if pl2.asset_kind == "user":
            if pl2.asset_path and Path(pl2.asset_path).is_file():
                path = pl2.asset_path
            else:
                it.status_note = f"ассет не найден: {pl2.asset_path or '(пусто)'}"
                lg(f"  enrich: {it.status_note} ({it.id})")
                continue                # дроп из рендера, item остаётся в плане
        elif pl2.asset_kind == "emoji":
            p = emoji_png_path(pl2.emoji, EMOJI_CACHE_DIR)
            if p is None:
                it.status_note = (f"эмодзи-ассет {pl2.emoji or '(пусто)'} "
                                  "не растеризовался (битый кодпойнт / нет "
                                  "шрифта эмодзи)")
                lg(f"  enrich: {it.status_note} ({it.id})")
                continue
            path = str(p)
        else:                           # asset_kind=none: предложение без ассета
            continue
        x, y = _pip_xy(pl2.position, H)
        scale_w = max(1, round(W * pl2.width_frac))
        if it.type == ENR_ANIMATION and path.lower().endswith(".webm"):
            anims.append((it.score, AnimOverlay(
                path=path, x_expr=x, y_expr=y, scale_w=scale_w,
                t0=c.f0, t1=c.f1, loop=(pl2.preset == "pulse")), it))
        else:
            # animation с png/emoji деградирует в still c fade — webm-пресеты
            # поверх эмодзи генерит P5; kenburns — только для type=image.
            stills.append((it.score, StillOverlay(
                path=path, x_expr=x, y_expr=y, scale_w=scale_w,
                t0=c.f0, t1=c.f1, fade_s=pl2.fade_ms / 1000.0,
                kenburns=bool(getattr(pl2, "kenburns", False))), it))
```

**Изменение:**
```
Replace the branch with resolved_asset()-first routing (kept 100% backward-compatible for payloads WITHOUT candidates):

        # image / animation (PiP в верхнем углу)
        pl2 = it.payload
        path: Optional[str] = None
        is_schematic = False
        # «Монтаж V2»: ассет ВЫБРАННОГО кандидата резолвит ЕДИНАЯ точка
        # ImagePayload.resolved_asset() (schematic-PNG код-графики / диффузия /
        # сток — по candidates[selected]), а НЕ плоский asset_kind: для schematic
        # он НАМЕРЕННО "none" (_legacy_sync_from_candidate), рендер обязан брать
        # PNG через resolved_asset (иначе флагманская код-графика молча выпадает
        # из ролика). Старый плоский план / AnimationPayload (нет candidates) —
        # по-прежнему по asset_kind.
        _chosen = getattr(pl2, "chosen_candidate", None)
        _cand = _chosen() if callable(_chosen) else None
        if _cand is not None:
            ap, src = pl2.resolved_asset()
            if src in ("none", "kinetic"):
                continue                # нет материального ассета — тихо
            if src == "icon":
                p = emoji_png_path(pl2.emoji or _cand.get("emoji", ""),
                                   EMOJI_CACHE_DIR)
                if p is None:
                    it.status_note = (f"эмодзи-ассет {pl2.emoji or '(пусто)'} "
                                      "не растеризовался (битый кодпойнт / нет "
                                      "шрифта эмодзи)")
                    lg(f"  enrich: {it.status_note} ({it.id})")
                    continue
                path = str(p)
            elif ap and Path(ap).is_file():
                path = ap
                is_schematic = (src == "schematic")
            else:
                it.status_note = f"визуал не готов ({src}): {ap or '(нет файла)'}"
                lg(f"  enrich: {it.status_note} ({it.id})")
                continue                # дроп из рендера, item остаётся в плане
        elif pl2.asset_kind == "user":
            if pl2.asset_path and Path(pl2.asset_path).is_file():
                path = pl2.asset_path
            else:
                it.status_note = f"ассет не найден: {pl2.asset_path or '(пусто)'}"
                lg(f"  enrich: {it.status_note} ({it.id})")
                continue
        elif pl2.asset_kind == "emoji":
            p = emoji_png_path(pl2.emoji, EMOJI_CACHE_DIR)
            if p is None:
                it.status_note = (f"эмодзи-ассет {pl2.emoji or '(пусто)'} "
                                  "не растеризовался (битый кодпойнт / нет "
                                  "шрифта эмодзи)")
                lg(f"  enrich: {it.status_note} ({it.id})")
                continue
            path = str(p)
        else:                           # asset_kind=none: предложение без ассета
            continue
        # schematic код-графики — полнокадровый cut-out с альфой (contract
        # §КОД-ГРАФИКА: PNG 1920x1080, прозрачный фон, «cut-out поверх talking-
        # head»), а не угловой PiP; kenburns на карточке с текстом не нужен.
        if is_schematic:
            x, y, scale_w = "0", "0", W
        else:
            x, y = _pip_xy(pl2.position, H)
            scale_w = max(1, round(W * pl2.width_frac))
        if it.type == ENR_ANIMATION and path.lower().endswith(".webm"):
            anims.append((it.score, AnimOverlay(
                path=path, x_expr=x, y_expr=y, scale_w=scale_w,
                t0=c.f0, t1=c.f1, loop=(pl2.preset == "pulse")), it))
        else:
            stills.append((it.score, StillOverlay(
                path=path, x_expr=x, y_expr=y, scale_w=scale_w,
                t0=c.f0, t1=c.f1, fade_s=pl2.fade_ms / 1000.0,
                kenburns=(False if is_schematic
                          else bool(getattr(pl2, "kenburns", False)))), it))
```

**Риск / безопасность.** MODERATE. Blast radius contained by the `_cand is not None` gate: every existing plan test (all build ImagePayload with empty candidates) keeps the identical legacy flat path — verify test_asset_kind_none_skipped_silently and the user/emoji/CTA overlay tests still pass byte-for-byte. Preserve the emoji rasterization and the exact status_note strings so UI regex/tests don't break. Two things to VERIFY against the real codegfx build before merge: (a) the schematic PNG is a full-frame 16:9 alpha cut-out (if it's actually a small card the full-frame scale_w=W is wrong — fall back to PiP); (b) at RENDER time the cache/codegfx/<sha1>.png still exists (deterministic cache persists from the suggest images stage; if evicted, this patch now honestly drops with a status_note instead of silent omission). Do NOT let full-frame schematic path through kenburns (jitters text).

**Тест (зафиксировать поведение).** Add to tests/test_enrich_plan.py: (1) test_schematic_candidate_renders_fullframe_still — build ImagePayload(source='schematic', selected=0, candidates=[{'source':'schematic','intent':'stat','fields':{'stats':[{'value':'96','unit':'%'}]},'style':'minimal','preview': png}]) on an EnrichItem(type=ENR_IMAGE) at t=100 in Timeline([],300); assert len(re.stills)==1, st.path==png, (st.x_expr,st.y_expr,st.scale_w)==('0','0',W), st.kenburns is False, item.status==ST_OK. (2) test_schematic_candidate_without_png_dropped_with_note — same but candidate has no preview/asset_path; assert re.stills==[] and 'не готов' in item.status_note. (3) regression: a diffusion candidate with asset_path=png selected renders as a corner PiP (scale_w==round(W*width_frac), not full-frame). Use the existing `png` fixture (absolute path — required by resolved_asset's is_absolute() guard).

---
## 2. Spill large -filter_complex graphs to a script file (kills WinError 206 on long/many-cut/censor-dense renders)
🔴 CRITICAL · симптом 2 · effort M · `r2-filter-complex-script`

- **Файл / якорь:** `vpipe/ffmpeg_utils.py` — FFmpeg.run(), cmd build + Popen, ~line 147-151
- **Закрывает находки:** vpipe/ffmpeg_utils.py:147, vpipe/ffmpeg_utils.py:149, vpipe/render.py:1173, vpipe/render.py:1205, vpipe/render.py:1211, vpipe/render.py:1225, vpipe/render.py:437, vpipe/censor.py:88, vpipe/render.py:474

**Проблема.** Every ffmpeg pass passes the entire filtergraph as one argv element to subprocess.Popen (ffmpeg_utils.py:147-149). The cut graph grows ~196 chars/kept-segment (render.py:1173-1177) and the censor asplit graph ~150+ chars/piece (censor.py:88-99); at ~165 kept segments (or ~123-194 censors depending on method) the Windows CreateProcess 32,767-char command-line limit is exceeded and Popen raises OSError [WinError 206] before ffmpeg starts. Renders of long talking-head videos with hundreds of filler/pause/hesitation cuts fail deterministically. No -filter_complex_script exists anywhere in the repo.

**Почему так.** Centralizing in run() is the single coordinated R2 change: it fixes findings #3/#9/#12/#14 (all four WinError-206 variants) at once, cannot be forgotten by a future new ffmpeg call site, and keeps the caller-facing contract (args with inline graph) intact so no existing test or the error-reporter breaks.

**Текущий код:**
```
        cmd = [self.ffmpeg, "-hide_banner", "-nostdin", "-y",
               "-progress", "pipe:1", "-nostats", *args]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        _register_proc(proc)
```

**Изменение:**
```
Add `import tempfile` at the top of ffmpeg_utils.py. Add a module-level constant and helper (near the process-registry block):

    # A many-cut/censor-dense -filter_complex overflows the Windows 32767-char
    # CreateProcess limit (~196 chars/segment). Anything longer than this is
    # written to a temp file and passed via -filter_complex_script so the graph
    # never touches the command line. Small graphs pass through unchanged so
    # short renders keep their exact argv (and every FakeFF test still sees an
    # inline -filter_complex).
    _FILTERGRAPH_INLINE_MAX = 8000

    def _spill_filtergraph(cmd: list[str]) -> tuple[list[str], Optional[str]]:
        try:
            i = cmd.index("-filter_complex")
        except ValueError:
            return cmd, None
        graph = cmd[i + 1] if i + 1 < len(cmd) else ""
        if len(graph) <= _FILTERGRAPH_INLINE_MAX:
            return cmd, None
        fd, path = tempfile.mkstemp(suffix=".ffscript", prefix="fve_fc_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(graph)
        out = list(cmd)
        out[i:i + 2] = ["-filter_complex_script", path]
        return out, path

Then in run(), rewrite the cmd build and wrap the process body so the temp file is always cleaned up. IMPORTANT: only rewrite the local `cmd`; leave the caller's `args` untouched (the FFmpegError block at ~198 and the FakeFF tests read the inline graph from `args`).

        cmd = [self.ffmpeg, "-hide_banner", "-nostdin", "-y",
               "-progress", "pipe:1", "-nostats", *args]
        cmd, _fc_script = _spill_filtergraph(cmd)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            _register_proc(proc)
            ...  # existing drain/progress/returncode body UNCHANGED
        finally:
            if _fc_script:
                try:
                    os.remove(_fc_script)
                except OSError:
                    pass

The existing `try/finally` that does `_unregister_proc(proc)` stays as an inner block; the new outer finally only removes the script file. Both returns (raise FFmpegError / success return) then pass through the outer finally.
```

**Риск / безопасность.** Low. -filter_complex_script is semantically identical to inline -filter_complex (ffmpeg 8.1.1 supports it). Fast-path preserved two ways: (1) graphs <=8000 chars are byte-for-byte unchanged, so all short-render golden tests and the copy/remux fast-paths are untouched; (2) the swap happens only in the local `cmd`, never in the caller's `args`, so FakeFF-based tests (test_edge_fade.py:122, test_serve_render.py:236, test_loudnorm_2pass, test_music_duck, etc.) that inspect `args[args.index('-filter_complex')+1]` keep passing. The censor pass, measure pass, and both cut branches all inherit the fix for free because they all call ff.run. Do NOT lower the 8000 threshold below the largest single-statement graph used by the enrich/music fast paths, or you would start spilling those too (harmless but changes their argv). Temp file uses the OS temp dir (fine) — cleaned in finally even on raise/cancel.

**Тест (зафиксировать поведение).** Add tests/test_ffmpeg_runner.py::test_spill_filtergraph. (1) Build cmd=['ffmpeg','-i','x','-filter_complex', 'a'*9000, 'out.mp4']; assert _spill_filtergraph returns a cmd containing '-filter_complex_script' (and NOT '-filter_complex'), the returned path exists, and its contents == 'a'*9000. (2) Small graph ('scale=1280:-2') passes through unchanged, path is None. (3) Add tests/test_serve_render.py::test_render_200_cuts_builds_script — build a CutList producing 200 kept segments, run render() with the real FFmpeg wrapper against tests/_media (or a real-ffmpeg-gated marker) and assert it does NOT raise OSError. Also assert via FakeFF that render() still emits an inline '-filter_complex' in args (contract for other tests).

---
## 3. Karaoke burn-in silently drops the last word of every budget/max_cps/overlap-trimmed cue in continuous speech
🔴 CRITICAL · симптом 1 · effort S · `karaoke-lastword-drop`

- **Файл / якорь:** `vpipe/subtitles.py` — _karaoke_text() in_cue selection, ~line 356-357
- **Закрывает находки:** vpipe/subtitles.py:356, vpipe/subtitles.py:357

**Проблема.** _karaoke_text re-selects each cue's words from raw word timings by FULL CONTAINMENT (w.start >= cue.start-eps AND w.end <= cue.end+eps). But build_cues trims cue ends after grouping (overlap pass line 169-171 sets cue.end = next.start - min_gap; max_cps pass line 166 can shrink too). In continuous Russian speech Whisper words are back-to-back (gap ~0, and the pause-cut pipeline snaps inter-word gaps to exactly 0 at every seam), so the trimmed cue.end lands BELOW the last word's end, that word fails w.end <= cue.end+eps, and its TEXT vanishes from the burned karaoke line — while the .srt (built from cue.text) keeps it. Single-word cues trimmed this way make in_cue empty and fall back to a solid-yellow plain-text flash line. This is a primary root cause of «субтитры корявые».

**Почему так.** Because every non-last cue is guaranteed end <= next.start - min_gap < next.start (max_cps + overlap passes both cap at next.start - min_gap), and the next cue's first word starts at next.start, overlap selection can NEVER bleed a neighbour word in: prev-cue last word has w.end <= cue.start (fails w.end>cue.start+eps); next-cue first word has w.start >= next.start > cue.end (fails w.start<cue.end-eps). It only RESCUES the trimmed tail word. Spans still clamp start to max(prev_end,cue.start); if the rescued word's end exceeds cue.end its \k is trimmed by the existing drift line (min 1cs) so text is never lost even before P2/P6 fix the timing.

**Текущий код:**
```
    in_cue = [w for w in words
              if w.start >= cue.start - eps and w.end <= cue.end + eps]
```

**Изменение:**
```
    # Overlap (not full-containment) selection: build_cues trims cue ends
    # below the last word's end for back-to-back speech, so containment silently
    # dropped the tail word. A word belongs to the cue iff its span OVERLAPS the
    # cue window. The eps keeps the previous cue's last word (ends at cue.start)
    # and the next cue's first word (starts at cue.end) out.
    in_cue = [w for w in words
              if w.start < cue.end - eps and w.end > cue.start + eps]
```

**Риск / безопасность.** Low. Verified against all existing write_ass/subtitles karaoke tests (test_write_ass.py:51-99, test_subtitles.py:98-151): in every one the cue exactly bounds its words, so overlap selection yields the identical word set and \k sums (200/250/278) are unchanged. Keep the fast path: do NOT touch total_cs/spans/drift. The eps (0.02) must stay strict on BOTH sides to avoid boundary double-count.

**Тест (зафиксировать поведение).** Add tests/test_write_ass.py::test_karaoke_keeps_last_word_when_cue_end_trimmed: words=[Word('раз',0.0,1.0),Word('два',1.0,2.0)], cue=Cue(0.0,1.90,'раз два') (cue.end 1.90 < last word end 2.00, exactly the trim build_cues produces). Assert 'два' in _karaoke_text(cue,words,ProfanityMatcher(ProfanityLists()),MaskingCfg()) and txt.count('\\k')==2. Pre-fix this fails (1 \k, 'два' missing).

---
## 4. SSE watchdog: detect a dead/restarted backend during a task instead of looping the spinner forever
🟠 HIGH · симптом 2 · effort M · `sse-dead-server-watchdog`

- **Файл / якорь:** `web/app.js` — followTask() — es.onerror / es.onmessage, ~line 2101-2103
- **Закрывает находки:** web/app.js:2102, web/app.js:250 (spinner-forever half)

**Проблема.** followTask()'s `es.onerror` is a no-op, so if serve.py dies or restarts mid-task the EventSource reconnect-loops (or the first reconnect hits /api/events → 409 and the stream permanently fails), the progress bar freezes at the last percent, st.task stays set, and every action button stays disabled with no recovery but a manual reload. This is the same root cause as the line-250 'spinner forever after restart' finding.

**Почему так.** Turns the two worst 'backend crash looks like a frontend hang' symptoms (frozen spinner, all-buttons-disabled) into a bounded ~9-15s recovery with an actionable toast, while correctly distinguishing a transient reconnect from a genuine death via a cheap state probe.

**Текущий код:**
```
  if (es) es.close()
  es = new EventSource('/api/events')
  es.onerror = () => { /* let EventSource auto-retry; final state arrives via message */ }
  es.onmessage = (ev) => {
    let t; try { t = JSON.parse(ev.data) } catch { return }
    const pct = Math.max(0, Math.min(100, t.percent || 0))
```

**Изменение:**
```
Reset the error counter on every good message and add a counted watchdog + /api/state probe. Replace the onerror line and add a helper:

  if (es) es.close()
  st._sseErrs = 0
  es = new EventSource('/api/events')
  es.onerror = () => {
    // Transient blips: let EventSource auto-retry. But a dead/restarted backend
    // makes reconnects fail forever (or the first reconnect 409s and the stream
    // permanently closes). After a few consecutive errors, probe /api/state to
    // decide: same task still running (keep waiting) vs task gone (recover UI).
    st._sseErrs = (st._sseErrs || 0) + 1
    if (st._sseErrs < 3) return
    st._sseErrs = 0
    fetch('/api/state').then((r) => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))).then((s) => {
      const t = s && s.task
      if (s.no_session || !t || t.name !== name) { taskLost() }              // session/task is gone
      else if (t.running) { /* server alive, same task runs — ES keeps retrying */ }
      else { if (es) { es.close(); es = null }; followTask(name) }            // finished during the blip → re-open, /api/events flushes the final snapshot
    }).catch(() => taskLost())                                              // /api/state itself unreachable → server down
  }
  es.onmessage = (ev) => {
    st._sseErrs = 0
    let t; try { t = JSON.parse(ev.data) } catch { return }
    const pct = Math.max(0, Math.min(100, t.percent || 0))

Add next to followTask():

  function taskLost() {
    if (es) { es.close(); es = null }
    st._sseErrs = 0
    $('#progress').classList.add('hidden'); $('#progress').classList.remove('indeterminate')
    setRunning(null)   // re-enables buttons, hides Cancel
    toast('Связь с задачей потеряна — возможно, сервер перезапустился. Обновите страницу.', 'error', { sticky: true })
  }
```

**Риск / безопасность.** Blast radius is one function. The `else { followTask(name) }` branch re-opens the stream only when the SAME task exists but is no longer running (a real SSE hiccup on a live server); /api/events immediately flushes the terminal snapshot so the normal results/error path in onmessage still runs (no duplicate rendering because onmessage closes es on !running). The probe GET /api/state is unguarded server-side (no 409). Do NOT lower the threshold below ~3: a single dropped frame fires onerror once even on a healthy stream and must not trip recovery. Preserve the existing onmessage terminal-state handling untouched — the watchdog only recovers the UI, it does not reimplement result handling.

**Тест (зафиксировать поведение).** Backend contract (lockable now, tests/test_events_dead.py): with serve.SESSION=None, TestClient GET /api/events → 409 and GET /api/state → {no_session:true}; with a SimpleNamespace session whose task={'running':False,'name':'render','done':True,...}, /api/events flushes a final data frame with running:false. Frontend (needs jsdom+vitest harness under web/tests/): stub EventSource; fire 3 onerror events with fetch('/api/state') mocked to {no_session:true} → assert setRunning(null) called, #progress hidden, sticky error toast; separately mock to {task:{name,running:false,done:true}} → assert followTask re-invoked.

---
## 5. Clamp media.duration to the SHORTEST real stream + verify rendered output duration, so a short track can't silently truncate/desync the render
🟠 HIGH · симптом 2 · effort M · `probe-perstream-duration`

- **Файл / якорь:** `vpipe/probe.py` — probe_media(), line 37 (duration =) + companion post-render verify in vpipe/render.py after _run_atomic (~line 1229)
- **Закрывает находки:** vpipe/probe.py:37, vpipe/render.py:1173, vpipe/render.py:1229, vpipe/timeline.py:92

**Проблема.** probe_media takes duration only from container format.duration (the MAX of all stream end-times), so a track that ends early (OBS audio outliving video by 0.5-2s, damaged recordings, cover-art clips) is invisible. Timeline.kept_segments() extends the last kept segment to that max and render.py turns kept segments into trim/atrim verbatim; a kept segment past the shorter stream's EOF yields a short/empty trim, concat emits mismatched-length streams, ffmpeg exits 0, _run_atomic promotes .part to final mp4, and serve reports a COMPUTED (never measured) new_duration with render:True. Net: silently truncated/desynced tail reported as success.

**Почему так.** Fixes the root (probe is the single source of duration truth, so clamping here propagates to Timeline/subtitles/chapters/enrich per R1) AND adds the R2 measured-output check so any residual mismatch is surfaced instead of reported as success. Two independent layers means even if the clamp is later relaxed, the verify still catches truncation.

**Текущий код:**
```
    duration = float(fmt.get("duration", 0.0) or 0.0)
    if duration <= 0 and v and v.get("duration"):
        duration = float(v["duration"])

    return MediaInfo(
        path=str(path),
        duration=duration,
```

**Изменение:**
```
Insert a per-stream clamp between the duration computation and the MediaInfo return in probe_media:

    duration = float(fmt.get("duration", 0.0) or 0.0)
    if duration <= 0 and v and v.get("duration"):
        duration = float(v["duration"])
    # Container format.duration is the MAX of all streams, so a track that ends
    # early (OBS audio outliving video, damaged/cover-art recordings) is invisible
    # and a kept segment past its EOF makes concat silently truncate/desync. Clamp
    # to the SHORTEST real stream so every downstream trim/atrim stays in-bounds.
    # Only when BOTH streams are present AND BOTH report a positive duration, and
    # only when the gap exceeds ~1 frame (ignore frame-quantization noise).
    v_dur = float(v.get("duration", 0.0) or 0.0) if v else 0.0
    a_dur = float(a.get("duration", 0.0) or 0.0) if a else 0.0
    if v is not None and a is not None and v_dur > 0.0 and a_dur > 0.0:
        shortest = min(v_dur, a_dur)
        if duration - shortest > 0.05:
            duration = shortest

COMPANION (R2 safety net) in vpipe/render.py — add a module helper and call it right after the CUT-path _run_atomic (currently line 1229-1230, `total=new_dur`):

    def _verify_output_duration(ff, out_path, expected, log, tol=0.5):
        try:
            info = ff.probe(out_path)
            actual = float(info.get("format", {}).get("duration", 0.0) or 0.0)
        except Exception:
            return
        if expected > 0 and actual > 0 and (expected - actual) > max(tol, 0.02 * expected):
            log(f"  ВНИМАНИЕ: рендер короче ожидаемого на {expected - actual:.2f}с "
                f"({actual:.1f}с вместо {expected:.1f}с) — вероятно усечён из-за "
                f"короткой дорожки (проверьте A/V-синхронизацию к концу).")

    ... after `_run_atomic(ff, args, out_path, total=new_dur, ...)`:
        _verify_output_duration(ff, out_path, new_dur, log)
```

**Риск / безопасность.** The min-clamp trims content of the LONGER stream: a talking-head with a silent video outro (end card held with no audio) would have media.duration shortened to the audio length, chopping the outro. Mitigated by (a) the 0.05s deadband (ignores frame noise), and (b) this being uncommon for the target talking-head footage where the tail is silence anyway. If the outro-trim risk is unacceptable, the more-correct-but-invasive alternative is to apad/tpad the shorter stream to the longer one in render.py instead of clamping — note this in the PR. The post-render _verify_output_duration WARNS (never raises) so it cannot break existing real-render tests on legit small deltas; it only converts silent truncation into a visible log line. Blast radius of the clamp is wide (every duration consumer) but strictly toward reality — old_duration/new_duration become accurate. Preserve: do NOT clamp when either stream duration is missing/0 (some containers omit stream duration) — the guard already handles this.

**Тест (зафиксировать поведение).** Add tests/test_probe.py::test_duration_clamped_to_shortest_stream — build a fake ff whose .probe returns {'format':{'duration':'10.0'},'streams':[{'codec_type':'video','width':1920,'height':1080,'duration':'10.0','avg_frame_rate':'30/1'},{'codec_type':'audio','codec_name':'aac','duration':'8.0','sample_rate':'48000'}]}; assert probe_media(ff,'x').duration == 8.0. Add ::test_duration_not_clamped_when_stream_duration_absent (audio stream has no 'duration' key) → stays 10.0. Add ::test_duration_frame_deadband (v=10.00, a=9.98) → stays 10.0. Plus one @pytest.mark.slow real-ffmpeg integration test that muxes a 5s video + 7s audio, renders with no cuts near the tail, and asserts the produced mp4's ffprobe duration is within tol of the reported new_duration (locks that _verify_output_duration fires and the clamp keeps them equal).

---
## 6. ffprobe the produced .part and reject a grossly-truncated output before promoting it (R2 validate-output half)
🟠 HIGH · симптом 2 · effort S · `r2-validate-output-duration`

- **Файл / якорь:** `vpipe/render.py` — _run_atomic(), after ff.run success, before os.replace, ~line 124-131
- **Закрывает находки:** vpipe/render.py:131, vpipe/ffmpeg_utils.py:123

**Проблема.** _run_atomic promotes the encoded .part to the final path with no check that ffmpeg actually produced a full-length file. A pass that exits 0 but writes a truncated stream (e.g. a starved input, a partial concat, an EOF mid-encode) is reported to the user as a successful render — the 'silently truncated but reported done' symptom. R2 mandates ffprobe-ing the produced file and comparing its duration to the expected duration.

**Почему так.** This is the R2 'compare duration to expected' half; placing it in _run_atomic covers every encode/remux path in one spot (cut, no-cut, video-only) and guarantees a rejected render leaves neither a half file nor a false success.

**Текущий код:**
```
    try:
        ff.run(run_args, total=total, on_progress=on_progress, desc=desc)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, out_path)
```

**Изменение:**
```
Between the successful ff.run and os.replace, when a positive `total` (expected duration) was passed, ffprobe the temp file and reject a gross shortfall (keeps container/keyframe slack so it never flaps on near-exact renders):

    try:
        ff.run(run_args, total=total, on_progress=on_progress, desc=desc)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    if total and total > 0:
        try:
            got = float(ff.probe(tmp).get("format", {}).get("duration", 0.0) or 0.0)
        except Exception:      # noqa: BLE001 — probe failure must not fail a good render
            got = 0.0
        # Only a GROSS shortfall (missing >10% AND >0.5s) is treated as truncation;
        # small keyframe/container rounding is expected and must never trip this.
        if got > 0.0 and got < total - max(0.5, 0.10 * total):
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise FFmpegError(
                f"{desc}: output truncated — got {got:.1f}s of an expected "
                f"{total:.1f}s. The render was rejected (no file written).")
    os.replace(tmp, out_path)

Import FFmpegError into render.py if not already (it imports FFmpeg from .ffmpeg_utils — extend to `from .ffmpeg_utils import FFmpeg, FFmpegError`).
```

**Риск / безопасность.** Medium-low. The 10%+0.5s tolerance is deliberately generous so legitimate outputs (which land within a frame or two of expected) never trip it; only a real truncation crosses it. A probe that itself fails is swallowed (got=0.0) so a broken ffprobe never fails a good render. Note the expected `total` differs per call site: media.duration for remux/copy, new_dur for the cut pass — both are already the true expected output length, so no new plumbing. Keep tolerance conservative; do NOT tighten it, or CFR/keyframe padding on short clips could cause false rejects.

**Тест (зафиксировать поведение).** tests/test_serve_render.py::test_run_atomic_rejects_truncated_output — monkeypatch ff.run to write a short valid mp4 (or fake ff.probe to return duration=5.0) while total=60.0; assert _run_atomic raises FFmpegError and out_path was NOT created and the .part was removed. And test_run_atomic_accepts_within_tolerance — probe returns 59.7 for total=60.0 → os.replace happens, no raise.

---
## 7. Session.__init__: validate the on-disk cutlist belongs to THIS video, and stop running full detection (LLM+VAD) synchronously inside the request
🟠 HIGH · симптом 4 · effort ? · `cutlist-source-guard-and-defer-detect`

- **Файл / якорь:** `serve.py` — Session.__init__ disk-load block (~L255-259) + Session._detect (~L281) + open_session (~L388-391) + put_cutlist (~L2231-2236); models.py CutList (~L137-162)
- **Закрывает находки:** serve.py:255, serve.py:256, serve.py:257, serve.py:258, serve.py:259, serve.py:388, vpipe/models.py:139

**Проблема.** Two merged high-sev backend bugs on the same ctor load block. (#7) `if self.cutlist_path.exists(): self.cutlist = CutList.load_json(...)` trusts a cutlist keyed by stem only in a SHARED out_dir — opening a different (or re-recorded) video with the same stem applies another video's cut timestamps; `_detect` then even merges the stale TYPE_MANUAL cuts. (#8) `elif self.transcript is not None: self.cutlist = self._detect()` runs run_detection (qwen3 bad-takes + Silero-VAD) synchronously inside the /api/open HTTP request — minutes of frozen UI, no progress/cancel, and a second /api/open races two detections onto the same file.

**Почему так.** The hash is the only correct invalidator (source path is identical for a re-recorded file; stem collides cross-folder). Storing it in the cutlist mirrors clips.json/enrich.json which already hash-validate. Deferring detection to a real task preserves the auto-detect-on-open UX (the frontend reloads after /api/open and followTask()s any running task — app.js:250, and the SSE 'detect' handler reloadCutlist()s on done, app.js:2149) while giving progress/cancel and killing the request-thread hang. The queue path is unaffected: it constructs Session() directly (not open_session) and already detects explicitly at serve.py:1565-1567 when _ctor_fresh_detect is False (now always False) — one detection, now WITH stage reporting.

**Текущий код:**
```
        # Load anything already on disk.
        cache_file = self.cache_dir / f"{self.audio_hash}.transcript.json"
        if cfg.transcribe.cache and cache_file.exists():
            self.transcript = Transcript.load(cache_file)
        if self.cutlist_path.exists():
            self.cutlist = CutList.load_json(self.cutlist_path)
        elif self.transcript is not None:
            self.cutlist = self._detect()
            self._ctor_fresh_detect = True
```

**Изменение:**
```
Rewrite the load block so it (1) validates the cutlist and (2) NEVER detects in the ctor:

        if cfg.transcribe.cache and cache_file.exists():
            self.transcript = Transcript.load(cache_file)
        if self.cutlist_path.exists():
            loaded = CutList.load_json(self.cutlist_path)
            if self._cutlist_matches(loaded):
                self.cutlist = loaded
            # else: a same-stem cutlist from a DIFFERENT / re-recorded video —
            # ignore it. Crucially do NOT keep it as self.cutlist, or a fresh
            # _detect() would merge its stale manual cuts (serve.py:278-279).
        # NOTE: detection is NO LONGER run here. A cached transcript with no
        # valid cutlist leaves self.cutlist=None; the editor launches detect as
        # a background TASK (open_session), the queue detects explicitly.

Add the guard method next to _detect:

    def _cutlist_matches(self, cl: CutList) -> bool:
        """The cutlist file is keyed by stem in a shared out_dir; make sure it
        belongs to THIS input before trusting it (cross-video / re-record)."""
        h = getattr(cl, "audio_hash", "")
        if h:                                  # new cutlists carry the hash
            return h == self.audio_hash
        # legacy cutlist (no hash): fall back to exact source path + duration.
        src_ok = (not cl.source) or cl.source == str(self.inp)
        dur_ok = (not cl.duration) or abs(cl.duration - self.media.duration) <= 0.5
        return src_ok and dur_ok

In Session._detect, stamp the hash before saving (just before `cl.save_json(self.cutlist_path)`):

        cl.audio_hash = self.audio_hash

In put_cutlist (after `cl.source = str(s.inp)`), stamp it too so user-edited cutlists carry the hash:

        cl.audio_hash = s.audio_hash

In open_session, launch detection as a background task (editor path only):

    def open_session(path: str) -> Session:
        global SESSION
        SESSION = Session(path, APP["cfg"], APP["out_dir"], APP["use_llm"])
        s = SESSION
        if s.cutlist is None and s.transcript is not None:
            try:
                s.start_task("detect", lambda: s._detect())   # progress + cancel
            except HTTPException:
                pass    # queue busy — user can detect once it frees up
        return SESSION

Companion models.py edit — add an optional field to CutList so the hash round-trips:
  * dataclass: add `audio_hash: str = ""` after `version: int = 1`
  * to_dict: add `"audio_hash": self.audio_hash`
  * from_dict: add `audio_hash=d.get("audio_hash", "")`
```

**Риск / безопасность.** Blast radius: Session ctor + open_session + put_cutlist + CutList schema. (1) Legacy cutlists (no audio_hash) still load via the source-path+duration fallback, so existing single-folder edits are preserved byte-for-byte on reopen. (2) Because the ctor no longer detects, opening a video whose transcript is cached but whose cutlist is missing now shows an empty timeline UNTIL the deferred detect task finishes — verify the frontend follows it (it does via reload+followTask; if a regression appears, the fallback is to keep the empty state and expose a 'detect' button). (3) start_task raises 409 if _queue_running — the try/except swallows it so a busy queue can't fail /api/open. (4) models.py field is additive (default ""), safe for all existing readers. Keep the vestigial `_ctor_fresh_detect=False` and the queue's conditional as-is (harmless). Do NOT hash-key the cutlist FILENAME — that would orphan every existing out/<stem>.cutlist.json and silently wipe users' saved manual edits on next open.

**Тест (зафиксировать поведение).** Add tests/test_cutlist_guard.py. (a) test_stale_cutlist_from_other_video_ignored: write out/<stem>.cutlist.json with audio_hash='OTHER' and a manual cut; monkeypatch serve.run_detection to a recorder and probe_media to a stub; construct Session on a fixture whose audio_hash!='OTHER'; assert s.cutlist is None after ctor (not the stale list) and that when _detect runs it does NOT contain the stale manual cut. (b) test_matching_hash_cutlist_loaded: same file with audio_hash==session hash → s.cutlist loaded, run_detection NOT called. (c) test_legacy_cutlist_source_mismatch_ignored: no audio_hash, source='D:/other/x.mp4' → ignored. (d) test_open_session_defers_detect: monkeypatch Session._detect to a recorder and start_task to run inline; call open_session on a fixture with cached transcript + no cutlist; assert a 'detect' task was started (not run inside __init__). (e) test_queue_still_detects_once: existing test_queue coverage — assert _detect runs exactly once for a job with a cached transcript.

---
## 8. Double extension-stripping collapses dotted stems: every Clip Maker clip and same-tail source overwrites one file
🟠 HIGH · симптом 3 · effort ? · `output-filename-double-stem`

- **Файл / якорь:** `serve.py` — _resolve_render_opts stem/base (~L1108-1111); _run_render_pipeline mp4 path (~L1279); _render_formats stem (~L1428); companion vpipe/subtitles.py:228,232
- **Закрывает находки:** serve.py:1110, serve.py:1279, serve.py:1428

**Проблема.** `stem = Path((opts.get('filename') or s.inp.stem).strip() or s.inp.stem).stem` applies `.stem` to an ALREADY extension-less name (UI strips ext at app.js:2563; s.inp.stem is pathlib), truncating everything after the last remaining dot. `лекция.часть1_clip01` → `лекция` for EVERY clip. Worse, even after stem is right, `base.with_suffix('.mp4')` (L1279) and subtitles' `base.with_suffix('.srt/.vtt')` re-truncate at the last dot — so the mp4 ALSO collapses. Result: N Clip Maker clips render to one `лекция.mp4`; `проект.v1.mp4` and `проект.v2.mp4` overwrite each other silently.

**Почему так.** The guard strips ONLY real media suffixes, so `.часть1`/`_clip01`/`.v2` are preserved while a stray `.mp4` a user typed is still cleaned. _with_ext / string-append avoid Path.with_suffix's last-dot truncation on the mp4 and sidecars. Fixing 1110 alone is insufficient — the corrected finding proves the mp4 re-collapses at 1279; both layers must change together, which is why this is one patch across serve.py + subtitles.py.

**Текущий код:**
```
    out_dir = Path(opts.get("out_dir") or s.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path((opts.get("filename") or s.inp.stem).strip() or s.inp.stem).stem
    base = out_dir / stem
    return cfg, scale_h, fps, out_dir, base

# ... _run_render_pipeline (~L1279):
    rr = render_mod.render(
        s.ff, s.media, cl, cfg, base.with_suffix(".mp4"), s.work_dir,

# ... _render_formats (~L1428):
    stem = Path((opts.get("filename") or s.inp.stem).strip() or s.inp.stem).stem

# ... vpipe/subtitles.py generate (~L227-232):
    base = Path(out_base)
    srt = str(base.with_suffix(".srt"))
    ...
        vtt = str(base.with_suffix(".vtt"))
```

**Изменение:**
```
Add two module-level helpers in serve.py (near _resolve_render_opts):

    def _clean_stem(raw: str, fallback: str) -> str:
        """Output stem from a UI/queue filename OR the input stem. Both are
        already extension-less, so a blind Path(...).stem truncates a DOTTED
        name at its last dot (`лекция.часть1` -> `лекция`), collapsing every
        `stem_clipNN` clip onto one file. Strip a trailing suffix ONLY when it
        is a known media extension."""
        raw = (str(raw) or "").strip() or fallback
        p = Path(raw)
        return (p.stem if p.suffix.lower() in VIDEO_EXT else p.name) or fallback

    def _with_ext(base: Path, ext: str) -> Path:
        """Append ext WITHOUT Path.with_suffix (which truncates a dotted stem
        at its last dot). Identical to with_suffix for non-dotted stems."""
        return base.parent / (base.name + ext)

Replace the stem line at ~L1110:
        stem = _clean_stem(opts.get("filename") or s.inp.stem, s.inp.stem)
Replace the duplicate at ~L1428 identically.
Replace the mp4 path at ~L1279:
        s.ff, s.media, cl, cfg, _with_ext(base, ".mp4"), s.work_dir,
Companion (vpipe/subtitles.py:228,232) — same append fix so dotted-stem sidecars aren't re-truncated:
        srt = str(Path(str(base) + ".srt"))
        ...
        vtt = str(Path(str(base) + ".vtt"))
```

**Риск / безопасность.** For every non-dotted stem (the overwhelming common case) output is byte-for-byte identical: _clean_stem returns the same value, and str(base)+ext == with_suffix(ext). Only dotted stems change (they were broken). Verify no other caller relies on with_suffix truncating a user-supplied '.mp4' typed into #rFilename — _clean_stem handles that by stripping known exts. Low blast radius. Keep _render_formats' sidecar_base=out_dir/stem consistent (it now carries the un-truncated stem, so _with_ext-based sidecars line up).

**Тест (зафиксировать поведение).** Add to tests/test_serve_render.py. test_dotted_filename_not_truncated: set s.inp=Path('лекция.часть1.mp4'); cfg,_,_,_,base = _resolve_render_opts(s, {'filename':'лекция.часть1_clip01'}); assert base.name=='лекция.часть1_clip01' and str(_with_ext(base,'.mp4')).endswith('лекция.часть1_clip01.mp4'). test_two_clips_distinct_outputs: resolve for 'x.a_clip01' and 'x.a_clip02' → different mp4 paths. test_plain_stem_unchanged: filename 'output' → base.name=='output', _with_ext(base,'.mp4') == base.with_suffix('.mp4'). In tests/test_subtitles.py add test_dotted_base_srt_keeps_tail: generate(...,out_base=tmp/'лекция.часть1') → srt endswith 'лекция.часть1.srt'.

---
## 9. ffprobe the rendered mp4 and fail on a short/truncated file instead of reporting a computed duration as done
🟠 HIGH · симптом 2 · effort ? · `render-output-validate`

- **Файл / якорь:** `serve.py` — _run_render_pipeline, immediately after the render_mod.render(...) call (~L1278-1284) and the results dict new_duration line (~L1340)
- **Закрывает находки:** serve.py:1338, serve.py:1340
- **Зависит от:** `cutlist-source-guard-and-defer-detect`

**Проблема.** After the encode nothing probes the produced mp4. The results dict reports `new_duration` COMPUTED from the cutlist timeline, not MEASURED. Any path where ffmpeg exits 0 but writes less than expected (interrupted/crash source recordings, VFR containers whose metadata overstates the decodable stream — probe.py:37 trusts format.duration) is invisible: the file is atomically renamed into place (render.py:131) and the UI toasts «Рендер завершён» with the full expected duration.

**Почему так.** A single ffprobe (already available via s.ff.probe, ffmpeg_utils.py:123) turns a silent truncation into a clean task error the UI already renders. One-sided (only fails when SHORTER) so a slightly-longer container (padding) never false-alarms. The 0.5s+1frame tolerance covers muxer rounding; exactness comes once R1 makes render() report the sliver-filtered duration.

**Текущий код:**
```
    rr = render_mod.render(
        s.ff, s.media, cl, cfg, base.with_suffix(".mp4"), s.work_dir,
        on_progress=on_progress,
        log=lambda m="": on_stage(str(m).strip() or "Рендер видео…"),
        scale_h=scale_h, fps=fps, ass_path=ass_path, crop_filter=crop_filter,
        edge_fade=edge_fade, **enrich_kw)

    sr: dict = {}
...
    tl = Timeline(removed, s.media.duration)
    return {
        "mp4": rr.get("out"), "encoder": rr.get("encoder"),
        "new_duration": round(tl.new_duration(), 1),
```

**Изменение:**
```
Insert a best-effort validation right after the render() call, before the subtitles block:

    # The mp4 is the irreplaceable artifact — but ffmpeg can exit 0 and still
    # write a SHORT file (crash-interrupted source, VFR metadata overstating
    # the decodable stream). Report a MEASURED duration and fail loudly on a
    # gross shortfall instead of toasting a truncated render as done. A probe
    # failure must NOT lose the mp4 (best-effort, mirrors the burn/enrich try).
    expected = float(rr.get("new_duration") or 0.0)   # R1: effective (post-sliver) duration
    probed = None
    out_mp4 = rr.get("out")
    if s.ff is not None and out_mp4 and expected > 0:
        try:
            fmt = s.ff.probe(out_mp4).get("format", {})
            probed = float(fmt.get("duration", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            probed = None
    if probed is not None:
        frame = (1.0 / s.media.fps) if s.media.fps else 0.04
        if probed + 0.5 + frame < expected:
            raise RuntimeError(
                f"Рендер оборвался: выходной файл {probed:.1f} с короче "
                f"ожидаемых {expected:.1f} с (ffmpeg завершился с кодом 0, но "
                f"записал меньше). Файл оставлен для проверки: {out_mp4}. / "
                f"Render truncated: output {probed:.1f}s vs expected {expected:.1f}s.")

Then surface the measured value in the results dict (add one key next to new_duration):
        "new_duration": round(tl.new_duration(), 1),
        "new_duration_probed": round(probed, 1) if probed is not None else None,
```

**Риск / безопасность.** Could newly FAIL renders that currently 'succeed'. Kept one-sided + tolerant to avoid that on healthy files. The real exposure is R1: render() currently returns tl.new_duration() PRE sliver-drop (render.py:868/1234) so `expected` overstates by dropped slivers (each <0.04s or <1 frame); a video with ~12+ dropped slivers could exceed 0.5s and false-alarm. Mitigation: depends_on the R1 render.py:876 fix (return effective duration) — then tolerance is exact. Best-effort probe (guarded by s.ff is not None) keeps all monkeypatched unit tests green (fixture ff=None → probed None → no-op). In the multi-format loop a raise on one format is caught and doesn't lose the others (serve.py:1486); for a single render it fails the task, which is the desired 'truncation = error'. Do not change the existing `new_duration` key (consumers/tests depend on it) — only ADD new_duration_probed.

**Тест (зафиксировать поведение).** Extend tests/test_serve_render.py. test_truncated_output_raises: give the fixture session a stub ff with `probe` returning {'format': {'duration': 5.0}}; monkeypatch render to return {'out': <path>, 'encoder': 'fake', 'new_duration': 20.0}; assert serve._run_render_pipeline(...) raises RuntimeError mentioning both 5.0 and 20.0. test_matching_output_ok: probe returns 20.0 → no raise, result['new_duration_probed']==20.0. test_probe_failure_keeps_mp4: probe raises → no exception, result returned, new_duration_probed is None. Add a real-ffmpeg smoke (marked slow): render tests/_media/test.mp4 with a 2-cut list, then truncate the .part-analog and assert the guard trips — locks the class the audit flagged as untested.

---
## 10. Batch-queue worker (_queue_process_one/_queue_worker) has zero tests: error transition, GPU-flag release, cancel, ctor-failure and persistence ordering all unverified
🟠 HIGH · симптом 2 · effort M · `queue-worker-tests`

- **Файл / якорь:** `tests/test_queue_worker.py` — serve._queue_process_one (serve.py:1516) + serve._queue_worker (serve.py:1582-1610)
- **Закрывает находки:** serve.py:1516, serve.py:1582, serve.py:1601, serve.py:1608, tests/test_queue.py:9

**Проблема.** tests/test_queue.py:9 explicitly admits the worker is not exercised ('drives real ffmpeg/whisper'). But it is mostly orchestration that mocks cleanly. Untested: job.status->'error'+job.error on exception (serve.py:1601-1603), _queue_running release under TASK_LOCK on loop exit (1608-1610), _chk() cancel at stage boundaries (1529-1533), the _ctor_fresh_detect skip (1565-1567), pending->running->done _save_queue ordering (1592-1606), and Session() ctor failure becoming a job error (not a dead worker thread). A regression = 'render silently stops partway' with the queue wedged and _queue_running never cleared (editor 409s until restart).

**Текущий код:**
```
# tests/test_queue.py:9 (module docstring)
"The worker itself (_queue_process_one) drives real ffmpeg/whisper, so it is not
exercised here; instead we assert the queue plumbing around it."
# tests/test_queue.py:129-136 shows the monkeypatch(serve,'QUEUE',[...]) pattern already used for jobs.
```

**Изменение:**
````
New file tests/test_queue_worker.py. Reuse the monkeypatch(serve,'QUEUE',...) / monkeypatch(serve,'_queue_running',...) pattern from test_queue.py. Fake the heavy stages so no ffmpeg/whisper runs:

```python
import threading, time
from types import SimpleNamespace
import pytest
import serve

def _fake_session_factory(monkeypatch, *, transcript=True, fresh=True, ctor_exc=None):
    calls = {}
    def _ctor(path, cfg, out_dir, use_llm):
        if ctor_exc: raise ctor_exc
        ls = SimpleNamespace(
            path=path, out_dir=out_dir, cfg=cfg,
            media=SimpleNamespace(has_audio=True, duration=10.0, height=1080, fps=30.0),
            ff=None, inp=SimpleNamespace(stem='clip'),
            transcript=(object() if transcript else None),
            audio_hash='h', cache_dir='c', work_dir='w',
            _ctor_fresh_detect=fresh)
        ls._detect = lambda: calls.setdefault('detect', 0) or calls.__setitem__('detect', calls.get('detect',0)+1)
        return ls
    monkeypatch.setattr(serve, 'Session', _ctor)
    return calls

def _patch_pipeline(monkeypatch, *, raise_in_render=None):
    monkeypatch.setattr(serve, '_resolve_render_opts',
                        lambda s, opts: (s.cfg, None, None, __import__('pathlib').Path('o'), __import__('pathlib').Path('o/clip')))
    def _rrp(*a, **k):
        if raise_in_render: raise raise_in_render
        return {'mp4': 'out.mp4'}
    monkeypatch.setattr(serve, '_run_render_pipeline', _rrp)

# 1) HAPPY PATH: cached transcript + fresh detect -> done, result set, no _detect re-run

def test_process_one_success(monkeypatch):
    _fake_session_factory(monkeypatch, transcript=True, fresh=True)
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(serve, '_queue_cancel', __import__('threading').Event())
    job = serve.QueueJob(id='j', path='clip.mp4', out_dir='/o')
    serve._queue_process_one(job)
    assert job.status == 'done' and job.result == {'mp4': 'out.mp4'} and job.percent == 100.0

# 2) MID-JOB EXCEPTION via the WORKER LOOP -> error + _queue_running released; next job still runs

def test_worker_marks_error_and_releases_flag_and_continues(monkeypatch):
    _fake_session_factory(monkeypatch)
    _patch_pipeline(monkeypatch, raise_in_render=RuntimeError('boom'))
    saves = []
    monkeypatch.setattr(serve, '_save_queue', lambda: saves.append([(j.id, j.status) for j in serve.QUEUE]))
    j1 = serve.QueueJob(id='a', path='clip.mp4', out_dir='/o', status='pending')
    j2 = serve.QueueJob(id='b', path='clip.mp4', out_dir='/o', status='pending')
    monkeypatch.setattr(serve, 'QUEUE', [j1, j2])
    monkeypatch.setattr(serve, '_queue_running', True)
    monkeypatch.setattr(serve, '_queue_cancel', __import__('threading').Event())
    serve._queue_worker()   # runs synchronously to completion (2 jobs)
    assert j1.status == 'error' and j1.error == 'boom'
    assert j2.status == 'error'          # second job also processed, not skipped
    assert serve._queue_running is False # released under TASK_LOCK on loop exit
    # persistence ordering: each job goes pending->running->error, snapshotted
    assert ('a','running') in [tuple(s[0]) for s in saves] or any(('a','running') in snap for snap in saves)

# 3) CTOR FAILURE -> job error, worker survives

def test_ctor_failure_becomes_job_error(monkeypatch):
    _fake_session_factory(monkeypatch, ctor_exc=ValueError('corrupt'))
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(serve, '_save_queue', lambda: None)
    j = serve.QueueJob(id='a', path='bad.mp4', out_dir='/o', status='pending')
    monkeypatch.setattr(serve, 'QUEUE', [j])
    monkeypatch.setattr(serve, '_queue_running', True)
    monkeypatch.setattr(serve, '_queue_cancel', __import__('threading').Event())
    serve._queue_worker()
    assert j.status == 'error' and 'corrupt' in j.error and serve._queue_running is False

# 4) CANCEL AT STAGE BOUNDARY -> _chk() raises 'Очередь остановлена', mapped to that error

def test_cancel_at_stage_boundary(monkeypatch):
    _fake_session_factory(monkeypatch)
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(serve, '_save_queue', lambda: None)
    ev = __import__('threading').Event(); ev.set()
    monkeypatch.setattr(serve, '_queue_cancel', ev)
    j = serve.QueueJob(id='a', path='clip.mp4', out_dir='/o', status='pending')
    monkeypatch.setattr(serve, 'QUEUE', [j])
    monkeypatch.setattr(serve, '_queue_running', True)
    serve._queue_worker()
    assert j.status == 'error' and j.error == 'Очередь остановлена'
```
Note: monkeypatch `serve._queue_cancel` with a fresh Event per test (it is a module global) so a set() from one test never leaks. Because _queue_worker() runs the loop to exhaustion synchronously with fakes, no thread/join is needed.
````

**Риск / безопасность.** Low blast radius: net-new file, no source edits. Watchpoint: _queue_worker() reads several module globals (QUEUE, _queue_running, _queue_cancel, TASK_LOCK) and calls _save_queue — each MUST be monkeypatched/isolated or the test will mutate real ./cache/queue.json (the exact phantom-job bug test_queue.py:29 already guards). Set _queue_cancel to a fresh Event and _save_queue to a no-op/recorder. Do not call _start_queue_worker (spawns a daemon thread); call _queue_worker() directly for determinism.

**Тест (зафиксировать поведение).** test_worker_marks_error_and_releases_flag_and_continues: patch Session+pipeline so _run_render_pipeline raises; assert job.status=='error', job.error=='boom', the NEXT pending job still processes, and serve._queue_running is False after the loop.

---
## 11. Make CutList.save_json atomic (tmp + os.replace) — it is the most-frequently-written state file and the only non-atomic persistence path
🟡 MEDIUM · симптом 4 · effort S · `cutlist-atomic-save`

- **Файл / якорь:** `vpipe/models.py` — CutList.save_json(), line 164-166 (+ add `import os` at top; Transcript.save at line 98 shares the flaw)
- **Закрывает находки:** vpipe/models.py:164

**Проблема.** save_json does a direct Path(path).write_text (open 'w' truncates first), so a crash/power-loss/disk-full mid-write leaves truncated JSON under the LIVE cutlist name. It is the hottest write in the app: the frontend PUTs the whole cutlist on a 700ms debounce after every toggle/edit, and _detect rewrites it after every detection. Every other persistence path in the codebase already uses the tmp+os.replace pattern (serve.py:416/445/553/656/760/2213/2399/3196/3346, enrich.py:835, render.py:131). NOTE: the finding claims Transcript.save is atomic — it is NOT (models.py:98 is also a bare write_text); fix both while here.

**Текущий код:**
```
    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")
```

**Изменение:**
```
Add `import os` after `import json` (line 4). Then replace save_json with the atomic pattern used everywhere else:

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, p)

Apply the IDENTICAL tmp+os.replace transform to Transcript.save (models.py:98-100) since it shares the flaw (written on every transcript edit).
```

**Риск / безопасность.** Very low. os.replace is atomic on the same filesystem (out/ is always local, same volume as the .tmp). The only behavioral change is a transient .cutlist.json.tmp file that is renamed away; ensure any directory-listing/watch logic ignores .tmp (it already does — watch.json glob targets media, not .tmp). Preserve identical JSON bytes (same ensure_ascii/indent) so no test comparing serialized output changes. p.with_suffix appends .tmp correctly because the path ends in .json (single suffix replaced) — verify: 'x.cutlist.json' → with_suffix('.json.tmp')? with_suffix replaces the LAST suffix only, giving 'x.cutlist.json.tmp' via `p.suffix + '.tmp'` = '.json.tmp' → correct. (Do NOT use with_suffix('.tmp') alone — that would drop '.json'.)

**Тест (зафиксировать поведение).** Add tests/test_models_atomic.py::test_save_json_atomic_leaves_no_partial — monkeypatch os.replace to raise mid-call, call save_json, assert the ORIGINAL file (pre-existing valid JSON) is still intact and parseable (the failed write hit only the .tmp). And ::test_save_json_roundtrip_bytes_unchanged — assert the produced file bytes equal the old direct-write bytes for a sample CutList (locks serialization is byte-identical).

---
## 12. Guard the ctor transcript/cutlist loads so a corrupt on-disk file re-detects instead of turning /api/open into a cryptic 400 (or exit 2)
🟡 MEDIUM · симптом 4 · effort M · `session-ctor-corrupt-guard`

- **Файл / якорь:** `serve.py` — Session.__init__, lines 253-259 (cache_file / cutlist_path loads)
- **Закрывает находки:** vpipe/models.py:164, serve.py:255
- **Зависит от:** `cutlist-atomic-save`

**Проблема.** Session.__init__ calls Transcript.load (line 254) and CutList.load_json (line 256) with no error handling. A truncated/corrupt file (the failure #2 prevents going forward, but existing corrupt files remain) raises json.JSONDecodeError (a ValueError) out of the ctor; /api/open turns it into a 400 whose body is the raw parser error ('Expecting value: line 1 column 5...'), and the clip is unopenable until the user manually finds and deletes out/<stem>.cutlist.json. With --video at startup, main() prints the parse error and exits code 2.

**Текущий код:**
```
        if cfg.transcribe.cache and cache_file.exists():
            self.transcript = Transcript.load(cache_file)
        if self.cutlist_path.exists():
            self.cutlist = CutList.load_json(self.cutlist_path)
        elif self.transcript is not None:
            self.cutlist = self._detect()
            self._ctor_fresh_detect = True
```

**Изменение:**
```
Add a module-level helper near the other file helpers:

    def _quarantine_corrupt(path: Path, log=print) -> None:
        """Rename a corrupt state file aside so the ctor can re-detect instead of
        failing the open. Best-effort — never raises."""
        try:
            bak = path.with_suffix(path.suffix + f".corrupt-{int(time.time())}")
            os.replace(path, bak)
            log(f"  повреждённый файл {path.name} отложен как {bak.name}; пересчитываю.")
        except OSError:
            pass

Then replace the load block (note: `elif` becomes an independent `if` so a quarantined cutlist still triggers re-detect):

        if cfg.transcribe.cache and cache_file.exists():
            try:
                self.transcript = Transcript.load(cache_file)
            except (ValueError, OSError):
                _quarantine_corrupt(cache_file, log)
        if self.cutlist_path.exists():
            try:
                self.cutlist = CutList.load_json(self.cutlist_path)
            except (ValueError, OSError):
                _quarantine_corrupt(self.cutlist_path, log)
        if self.cutlist is None and self.transcript is not None:
            self.cutlist = self._detect()
            self._ctor_fresh_detect = True

(Confirm `time` and `os` are imported in serve.py — both are; `log` is available in __init__ scope, else pass print.)
```

**Риск / безопасность.** Low, but two things to preserve: (1) the original `elif` meant 'only detect when NO cutlist file exists' — the rewrite to `if self.cutlist is None` preserves that for the healthy path (file present & valid → cutlist set → no detect) AND correctly adds re-detect after quarantining a corrupt cutlist. (2) json.JSONDecodeError IS a ValueError subclass so `except (ValueError, OSError)` catches truncation; do not narrow to JSONDecodeError only (a zero-byte or encoding-broken file can raise other ValueErrors/UnicodeDecodeError — UnicodeDecodeError is a ValueError subclass, good). Don't swallow bare Exception (would hide programming errors). Renaming via os.replace to a sibling name is atomic and can't lose data.

**Тест (зафиксировать поведение).** Add tests/test_serve_open_corrupt.py::test_open_with_truncated_cutlist_redetects — write out/<stem>.cutlist.json with half-bytes of valid JSON, POST /api/open (with a cached transcript present) and assert 200 + a fresh cutlist returned + the corrupt file renamed to *.corrupt-* (not deleted). ::test_open_with_corrupt_transcript_cache_recovers — corrupt the transcript cache, assert open still succeeds (transcript re-derived or gracefully absent) rather than 400.

---
## 13. Thread a should_cancel callback into render() and honor it at each pass boundary (cancel during measure/DFN/inter-pass no longer swallowed)
🟡 MEDIUM · симптом 2 · effort M · `cancel-flag-pass-boundaries`

- **Файл / якорь:** `vpipe/render.py` — render() signature ~822-829; measure_loudness except ~441-444; enhance_audio except ~571-574; pass boundaries
- **Закрывает находки:** vpipe/render.py:441, vpipe/render.py:571, serve.py:2667, serve.py:1278
- **Зависит от:** `dfn-popen-register-heartbeat`

**Проблема.** POST /api/cancel sets task['cancelled'] and calls cancel_all(), which only kills currently-live registered ffmpeg. The flag is never checked inside render(). So: a cancel during the 2-pass loudnorm measurement kills that ffmpeg but its error is swallowed by `except Exception` (render.py:441-444) as a best-effort fallback and the FULL encode then launches; same for the DFN wav-extraction swallow (render.py:571-574); a cancel landing between passes kills nothing and the next multi-hour encode still starts. The user presses Отмена, the toast says 'Отменяю…', yet the render burns GPU/CPU to completion.

**Почему так.** Makes the first Отмена press actually stop the pipeline at the next boundary instead of only working when it happens to land while a registered ffmpeg is mid-encode; pairs with P5 which makes the DFN stage itself killable.

**Текущий код:**
```
def render(ff: FFmpeg, media: MediaInfo, cl: CutList, cfg: Config,
           out_path: str | Path, work_dir: str | Path,
           on_progress=None, log=print,
           scale_h: Optional[int] = None, fps: Optional[float] = None,
           ass_path: Optional[str] = None,
           crop_filter: Optional[str] = None,
           edge_fade: float = 0.0,
           enrich: Optional[RenderEnrich] = None) -> dict:
...
    except Exception as e:  # noqa: BLE001 — measurement is strictly best-effort
        log(f"  loudnorm 2pass: измерение не удалось ({e}) — "
            "включаю обычный однопроходный режим.")
        return None
```

**Изменение:**
```
1) Add a module-level sentinel: `class RenderCancelled(RuntimeError): pass`.

2) Add `should_cancel: Optional[Callable[[], bool]] = None` to the signatures of render(), measure_loudness(), enhance_audio(), and censor_audio() (import Callable). Add a tiny local helper in render(): `def _ck():\n    if should_cancel is not None and should_cancel():\n        raise RenderCancelled('cancelled')`.

3) Call `_ck()` at every pass boundary in render(): immediately before `censored = censor_audio(...)` (~909), before the `if has_audio and _wants_loudnorm_2pass(cfg)` measure block (~1059), before the DFN `enhance_audio(...)` call (~927), and before EACH `_run_atomic(...)` encode/remux invocation.

4) In the measure_loudness except (render.py:441) and the enhance_audio excepts (render.py:571-574, 583-586) re-raise instead of swallowing when cancelled:

    except Exception as e:  # noqa: BLE001 — measurement is strictly best-effort
        if should_cancel is not None and should_cancel():
            raise RenderCancelled("cancelled")
        log(f"  loudnorm 2pass: измерение не удалось ({e}) — "
            "включаю обычный однопроходный режим.")
        return None

   Pass should_cancel from render() into measure_loudness/enhance_audio/censor_audio call sites.

5) serve.py:1278 render call — add `should_cancel=lambda: bool(s.task.get('cancelled'))`.

The worker's existing except already reports «cancelled» from task['cancelled'], so RenderCancelled propagates as a clean cancel, not a raw error dump.
```

**Риск / безопасность.** Medium. should_cancel defaults to None so every non-serve caller (clips, queue via their own paths, tests) is byte-for-byte unchanged and never raises. The re-raise is gated on should_cancel() being truthy, so a genuine measurement failure (no cancel) still falls back exactly as before — the loudnorm best-effort contract is preserved. Confirm the serve worker treats a raised exception with task['cancelled']==True as a cancel and not an error toast (it already does per the /api/cancel comment at serve.py:2670). Do not check the flag inside tight loops — only at pass boundaries — to avoid perf noise.

**Тест (зафиксировать поведение).** tests/test_serve_render.py::test_cancel_during_measure_aborts — set a should_cancel that returns True after the first ff.run; drive render() with loudnorm_mode='2pass' and assert it raises RenderCancelled and NEVER reaches the encode _run_atomic (spy/monkeypatch the encode call). test_cancel_between_passes — should_cancel flips True after censor; assert encode is not launched.

---
## 14. Run deep-filter.exe via a registered Popen with a heartbeat so the DFN stage is cancellable and the bar shows motion
🟡 MEDIUM · симптом 2 · effort M · `dfn-popen-register-heartbeat`

- **Файл / якорь:** `vpipe/render.py` — enhance_audio(), deep-filter.exe subprocess.run, ~line 576-601
- **Закрывает находки:** vpipe/render.py:581, vpipe/render.py:576

**Проблема.** With denoise.engine=deepfilter the render runs deep-filter.exe via blocking `subprocess.run(cmd, capture_output=True)` (render.py:581): no progress ticks, no timeout, and — because it is never added to _running_procs — cancel_all() cannot touch it at all. On a 60-min video the worker blocks for tens of minutes with the bar frozen at ~25-35% and Отмена is a no-op, which reads exactly as 'the render silently stopped partway'. The 48kHz mono wav intermediates also churn ~700MB/hour in work_dir during this window.

**Почему так.** Removes the single longest uncancellable, motionless window in the pipeline; also supplies the registered-process mechanism finding #4 needs for its 'register the DFN process' clause.

**Текущий код:**
```
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except OSError as e:
        log(f"  DeepFilterNet: запуск не удался ({e}) — использую afftdn.")
        _cleanup_dfn_temp(wd)
        return None
    if r.returncode != 0:
        tail = " | ".join((r.stderr or r.stdout or "").strip().splitlines()[-3:])
```

**Изменение:**
```
Expose the registry helpers from ffmpeg_utils (they already exist: _register_proc/_unregister_proc, cancel_all) — import them into render.py: `from .ffmpeg_utils import FFmpeg, FFmpegError, _register_proc, _unregister_proc` (or add thin public wrappers register_proc/unregister_proc in ffmpeg_utils and use those). Replace the blocking run with a registered Popen plus an elapsed-time heartbeat:

    import time
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace")
    except OSError as e:
        log(f"  DeepFilterNet: запуск не удался ({e}) — использую afftdn.")
        _cleanup_dfn_temp(wd)
        return None
    _register_proc(proc)
    start = time.monotonic()
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=5.0)
                break
            except subprocess.TimeoutExpired:
                el = time.monotonic() - start
                if on_progress is not None:
                    # DFN gives no fraction; nudge the bar within its slice so the
                    # UI shows motion (cap below 1.0 so it never claims 'done').
                    on_progress(min(0.95, el / max(total or el, el + 1.0)))
                if should_cancel is not None and should_cancel():
                    proc.terminate()
                    out, err = proc.communicate()
                    _cleanup_dfn_temp(wd)
                    raise RenderCancelled("cancelled")
    finally:
        _unregister_proc(proc)
    rc = proc.returncode
    if rc != 0:
        tail = " | ".join((err or out or "").strip().splitlines()[-3:])
        ...  # existing non-zero handling, using rc instead of r.returncode

Keep the rest (out_wav existence check) unchanged. Requires should_cancel plumbed by P4.
```

**Риск / безопасность.** Medium. Registering the DFN process lets cancel_all() terminate it like any ffmpeg — but deep-filter.exe on SIGTERM leaves a partial dfn_out wav, so the terminate path calls _cleanup_dfn_temp (already the failure convention here, audit C-1). The heartbeat only advances a bounded 0..0.95 within the DFN slice and never emits 1.0, so it cannot make the bar prematurely complete. communicate(timeout=) avoids the pipe-deadlock risk of reading capture_output manually. If should_cancel is None (non-serve callers), behavior is a blocking wait identical in effect to today minus the heartbeat. Do not change the DFN degrade-to-afftdn contract: any non-zero exit / missing output still returns None.

**Тест (зафиксировать поведение).** tests/test_deepfilter.py::test_dfn_cancellable — monkeypatch subprocess.Popen to a fake long-running process; set should_cancel True after first heartbeat; assert enhance_audio terminates it, cleans temps, and raises RenderCancelled. test_dfn_heartbeat_ticks — assert on_progress is called with strictly-increasing values <1.0 while the fake runs.

---
## 15. Add a no-progress watchdog so a stalled ffmpeg is killed instead of hanging the task (and every endpoint) forever
🟡 MEDIUM · симптом 2 · effort M · `ffmpeg-stall-watchdog`

- **Файл / якорь:** `vpipe/ffmpeg_utils.py` — FFmpeg.run(), progress-read loop + proc.wait(), ~line 171-186
- **Закрывает находки:** vpipe/ffmpeg_utils.py:183, vpipe/ffmpeg_utils.py:171

**Проблема.** run() does `for line in proc.stdout: ...` then `proc.wait()` with no timeout and no stall detection (ffmpeg_utils.py:171-184). If ffmpeg stalls without exiting (NVENC/TDR hang, dropped network-drive input, starved filtergraph input) the worker thread blocks forever, task['running'] stays True, and _guard_no_task makes every mutating endpoint 409 «Идёт фоновая задача». The progress bar freezes at the last percent with no error, indefinitely; the queue worker has the same property (one stuck job blocks all pending jobs).

**Почему так.** Turns an unrecoverable silent hang (app looks dead, all endpoints 409) into a bounded, self-clearing task error the worker reports normally, unblocking the editor and the queue.

**Текущий код:**
```
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_us=") and total and on_progress:
                    try:
                        us = int(line.split("=", 1)[1])
                        on_progress(min(1.0, max(0.0, (us / 1e6) / total)))
                    except (ValueError, ZeroDivisionError):
                        pass
                elif line == "progress=end" and on_progress:
                    on_progress(1.0)
            proc.wait()
            t.join(timeout=1.0)
        finally:
            _unregister_proc(proc)
```

**Изменение:**
```
Track the monotonic timestamp of the last progress line and run a daemon watchdog that terminates the process if no progress arrives for a configurable window. Add `import time` (already may be present — verify). Read the window from config; add `stall_timeout: float = 180.0` to FfmpegCfg (config.py:22) and store it on the FFmpeg instance in __init__ (`self.stall_timeout = float(getattr(cfg, 'stall_timeout', 180.0) or 0.0)`).

    last_progress = [time.monotonic()]
    stalled = [False]
    win = getattr(self, "stall_timeout", 0.0)

    def _watchdog():
        while proc.poll() is None:
            if win and (time.monotonic() - last_progress[0]) > win:
                stalled[0] = True
                try:
                    proc.terminate()
                except Exception:
                    pass
                return
            time.sleep(2.0)

    wd = None
    if win and total:
        wd = threading.Thread(target=_watchdog, daemon=True)
        wd.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_us=") and total and on_progress:
                last_progress[0] = time.monotonic()
                try:
                    us = int(line.split("=", 1)[1])
                    on_progress(min(1.0, max(0.0, (us / 1e6) / total)))
                except (ValueError, ZeroDivisionError):
                    pass
            elif line == "progress=end":
                last_progress[0] = time.monotonic()
                if on_progress:
                    on_progress(1.0)
        proc.wait()
        t.join(timeout=1.0)
    finally:
        _unregister_proc(proc)

After the returncode check, if stalled[0] is True, raise FFmpegError with a clear message instead of the raw exit-1 dump:
    if stalled[0]:
        raise FFmpegError(f"{desc}: нет прогресса дольше {int(win)} c — процесс остановлен (возможно, зависание кодировщика или потерян вход).")
```

**Риск / безопасность.** Medium. The watchdog only arms when both a stall window and `total` are set, so capability probes (`-encoders`/`-filters`) and any zero-total call are never watched. Default 180s is comfortably longer than any real inter-progress gap for NVENC/x264 on this hardware (progress ticks sub-second during a healthy encode), but a very slow CPU x264 'veryslow' preset on a huge frame could theoretically gap — keep the default generous and make it config-tunable; a user hitting false kills raises ffmpeg.stall_timeout. Watchdog is daemon so it never blocks shutdown. Must interact cleanly with cancel_all(): a manual cancel also terminates proc, poll() becomes non-None, watchdog exits — no double-raise because returncode!=0 path already fires.

**Тест (зафиксировать поведение).** tests/test_ffmpeg_runner.py::test_stall_watchdog — run a fake ffmpeg (a small python script via sys.executable) that prints two out_time_us lines then sleeps 10s without exiting, with stall_timeout=2 and total=100; assert run() raises FFmpegError whose message contains 'нет прогресса' within ~3s, and that a healthy fake that keeps printing progress is NOT killed.

---
## 16. Kill in-flight ffmpeg on server exit (atexit + FastAPI shutdown) so a parent-crash/kill can't orphan ffmpeg.exe holding the GPU + locking the .part file
🟡 MEDIUM · симптом 2 · effort S · `serve-shutdown-cancel-all`

- **Файл / якорь:** `serve.py` — top imports (~line 12-25); module app scope (~325); main() before uvicorn.run (~4460)
- **Закрывает находки:** serve.py:4461, serve.py:4420-4430, serve.py:321

**Проблема.** Renders run on daemon threads (serve.py:321) and nothing in serve.py calls ffmpeg_utils.cancel_all() on process exit — no atexit, no FastAPI shutdown/lifespan, no signal handler. ffmpeg is a plain Popen with no Windows Job Object (ffmpeg_utils.py:149), so a parent kill/crash leaves ffmpeg.exe encoding for hours: it holds NVENC/GPU while the restarted server renders, and holds out.part open (no FILE_SHARE_DELETE) so the startup *.part sweep (serve.py:4420-4430) fails unlink under `except OSError: pass` and the multi-GB scrap survives.

**Почему так.** Owners of process lifetime (interpreter exit + ASGI shutdown) now both invoke the kill primitive the /api/cancel path already trusts, closing the orphan window for the common exit routes.

**Текущий код:**
```
# --- imports (serve.py:12-25) ---
import argparse
import copy
import hashlib
import json
import logging
import math
import os
...
import threading
import time
import uuid
import webbrowser

# --- module scope (serve.py:324-325) ---
SESSION: Optional[Session] = None
app = FastAPI(title="FastVideoEdit")

# --- main(), just before serving (serve.py:4458-4461) ---
    _install_connection_reset_filter()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
```

**Изменение:**
```
1) Add `import atexit` to the stdlib import block (alphabetical, right after `import argparse`).

2) Register a FastAPI shutdown hook at module scope, immediately after `app = FastAPI(title="FastVideoEdit")` (line 325):

@app.on_event("shutdown")
def _cancel_ffmpeg_on_shutdown() -> None:
    """uvicorn graceful shutdown (Ctrl+C / SIGTERM) -> terminate any tracked
    ffmpeg so a daemon render thread cannot keep an orphan encoding after the
    server has quit. Idempotent; safe when nothing is running."""
    try:
        ffmpeg_utils.cancel_all()
    except Exception:  # noqa: BLE001 — best-effort teardown, never raise on exit
        pass

3) In main(), register the same teardown for the plain-interpreter-exit path (covers normal `return`/SystemExit that does not fire the ASGI shutdown), just before the uvicorn.run call (line ~4460):

    _install_connection_reset_filter()
    atexit.register(ffmpeg_utils.cancel_all)   # parent exits -> don't orphan ffmpeg.exe
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0

Register atexit INSIDE main() (not at import) so import-only test processes don't add a global hook.
```

**Риск / безопасность.** Low blast radius: adds callers of an existing idempotent primitive; changes no render path. `@app.on_event("shutdown")` is deprecated-but-supported in current FastAPI and fires under both real uvicorn and TestClient-as-context-manager; if a lifespan handler is added later, fold this into it. atexit.register in main() only runs when actually serving. NOTE this does NOT cover a hard native crash (segfault/CUDA fault) — atexit/shutdown don't run then; the only complete fix for that is a Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE around the Popen in ffmpeg_utils.py:149 (ctypes, Windows-only, higher risk — recommend as a separate hardening patch, out of scope here). Do NOT weaken the existing *.part sweep; once cancel_all runs on exit the child releases the lock and the next sweep succeeds.

**Тест (зафиксировать поведение).** Add tests/test_shutdown.py::test_shutdown_calls_cancel_all — monkeypatch ffmpeg_utils.cancel_all to append to a list, then `with TestClient(serve.app):  pass` (context-manager entry/exit fires startup+shutdown events); assert the list is non-empty. Plus test_atexit_registers_cancel_all — assert `ffmpeg_utils.cancel_all` is the callable passed to atexit.register by patching atexit.register to capture args and invoking the relevant main() branch (or assert the registration line exists via a thin refactor). Cross-platform (no real ffmpeg).

---
## 17. max_cps enforcement can SHRINK a cue's end below its last word instead of only extending it (trims last cue early)
🟡 MEDIUM · симптом 1 · effort S · `maxcps-no-shrink-below-last-word`

- **Файл / якорь:** `vpipe/subtitles.py` — build_cues() max_cps reading-speed pass, ~line 166
- **Закрывает находки:** vpipe/subtitles.py:166

**Проблема.** The reading-speed pass is documented extend-only ('A cue that is too short for its text is extended on its END') but applies limit=cue_next_start[i]-min_gap as a hard OUTER min(), so a too-fast cue whose original end exceeds limit gets REDUCED. Net-new damage is the LAST cue (the overlap pass at 169-171 skips it): when speech runs to the timeline end (Timeline.remap_clamped snaps the final word end to new_duration), limit=total-min_gap lands below the last word's end, so cues[-1].end is pulled ~50ms early. This trims the .srt/.vtt sidecar early AND (compounding P1) makes the burned last word's \k window and Dialogue End finish before the word is spoken.

**Почему так.** When limit < c.end (no room / would shrink) max(c.end,limit)=c.end, inner min collapses to c.end, outer max leaves c.end unchanged. When limit > c.end the formula extends up to min(c.start+required, limit, total) exactly as before. So the only behaviour change is: the shrink case is neutralised. Fixes both the sidecar early-trim and one of the two triggers of the P1 word-drop, keeping .srt and burn consistent.

**Текущий код:**
```
            if (c.end - c.start) < required:
                limit = cue_next_start[i] - subs.min_gap
                c.end = min(max(c.end, c.start + required), max(c.start, limit), total)
```

**Изменение:**
```
            if (c.end - c.start) < required:
                limit = cue_next_start[i] - subs.min_gap
                # Extend-only: never pull the end BELOW the original (that would
                # drop the cue's own last word). Wrapping the whole target in an
                # outer max(c.end, ...) makes the limit cap the EXTENSION only.
                c.end = max(c.end, min(c.start + required, max(c.end, limit), total))
```

**Риск / безопасность.** Very low, one-line swap. No existing test exercises a dense-enough cue to hit this branch, so nothing regresses. Preserve the extension semantics exactly (verified: identical output for the has-room and no-room cases).

**Тест (зафиксировать поведение).** Add tests/test_subtitles.py::test_max_cps_never_shrinks_cue_end_below_last_word: build_cues over one dense last cue whose text length L satisfies L/17 > cue duration and whose last word ends within min_gap of total — e.g. words spanning 3.0..5.98 with ~55 chars, total=6.0 (required≈3.2 > dur 2.98; limit=5.95 < last word end 5.98). Assert cues[-1].end == pytest.approx(5.98) (pre-fix it is 5.95).

---
## 18. FFmpeg.run progress/failure parsing, cancel_all, and _run_atomic's truncation guard have no direct tests — the exact code deciding done-vs-crashed-vs-truncated
🟡 MEDIUM · симптом 2 · effort M · `ffmpeg-runner-atomic-tests`

- **Файл / якорь:** `tests/test_ffmpeg_runner.py` — vpipe.ffmpeg_utils.FFmpeg.run (ffmpeg_utils.py:133-211) + cancel_all (39-55) + vpipe.render._run_atomic (render.py:98-131)
- **Закрывает находки:** vpipe/ffmpeg_utils.py:133, vpipe/ffmpeg_utils.py:175, vpipe/ffmpeg_utils.py:188, vpipe/ffmpeg_utils.py:39, vpipe/render.py:98

**Проблема.** FFmpeg.run is only tested for stderr head/tail windowing (test_loudnorm_2pass.py:398-410). Untested: out_time_us progress parsing+clamping and on_progress(1.0) on 'progress=end' (ffmpeg_utils.py:175-182), nonzero-exit FFmpegError assembly incl. the filter_complex dump (188-204), and cancel_all() actually terminating registered procs (test_csrf.py replaces cancel_all with a lambda). _run_atomic (render.py:98-131) — the ONLY guard that a crashed render leaves no truncated playable mp4 — has no test that (a) an ff.run exception removes the .part and leaves NO out.mp4, (b) the -f fmt injection handles the trailing output arg, (c) os.replace only on success. All FakeFF tests run only the success path.

**Текущий код:**
```
# tests/test_loudnorm_2pass.py:375-395 — the reusable fake Popen + FFmpeg.__new__ pattern
class _FakeProc:
    def __init__(self, stderr_text):
        self.stdout = io.StringIO('')          # <-- today only stderr is fed
        self.stderr = io.StringIO(stderr_text)
        self.returncode = 0
    def wait(self): return 0
    def poll(self): return self.returncode
def _fake_ffmpeg(monkeypatch, stderr_text):
    ff = ffmpeg_utils.FFmpeg.__new__(ffmpeg_utils.FFmpeg)
    ff.ffmpeg, ff.ffprobe, ff._caps = 'ffmpeg', 'ffprobe', {}
    monkeypatch.setattr(ffmpeg_utils.subprocess, 'Popen', lambda *a, **k: _FakeProc(stderr_text))
    return ff
```

**Изменение:**
````
New file tests/test_ffmpeg_runner.py. Extend the _FakeProc to also emit stdout progress lines and a settable returncode.

```python
import io, os
from pathlib import Path
import pytest
from vpipe import ffmpeg_utils
from vpipe.ffmpeg_utils import FFmpeg, FFmpegError, cancel_all
from vpipe.render import _run_atomic

class _Proc:
    def __init__(self, stdout_lines, stderr_text='', rc=0):
        self.stdout = io.StringIO(''.join(stdout_lines))
        self.stderr = io.StringIO(stderr_text)
        self.returncode = rc
    def wait(self): return self.returncode
    def poll(self): return self.returncode
    def terminate(self): self.returncode = -15

def _ff(monkeypatch, proc):
    ff = FFmpeg.__new__(FFmpeg); ff.ffmpeg=ff.ffprobe='ffmpeg'; ff._caps={}
    monkeypatch.setattr(ffmpeg_utils.subprocess, 'Popen', lambda *a, **k: proc)
    return ff

# 1) progress: out_time_us -> clamped 0..1 fractions, progress=end -> 1.0

def test_run_progress_sequence(monkeypatch):
    lines = ['out_time_us=0\n','out_time_us=5000000\n','out_time_us=20000000\n','progress=end\n']
    seen=[]
    ff=_ff(monkeypatch,_Proc(lines))
    ff.run(['-i','x','out.mp4'], total=10.0, on_progress=seen.append)
    assert seen[0]==0.0 and abs(seen[1]-0.5)<1e-9
    assert seen[2]==1.0            # 20s/10s clamped to 1.0
    assert seen[-1]==1.0           # progress=end

# 2) failure: nonzero exit -> FFmpegError with stderr tail AND filter_complex dump

def test_run_failure_includes_graph(monkeypatch):
    ff=_ff(monkeypatch,_Proc([], stderr_text='X\nboom: invalid argument\n', rc=1))
    with pytest.raises(FFmpegError) as ei:
        ff.run(['-filter_complex','[0:v]bad[outv]','-i','x','out.mp4'], desc='render')
    msg=str(ei.value)
    assert 'exit 1' in msg and 'boom' in msg and '[0:v]bad[outv]' in msg

# 3) cancel_all terminates a live registered proc

def test_cancel_all_terminates(monkeypatch):
    p=_Proc([]); p.returncode=None      # poll()->None means running
    ffmpeg_utils._register_proc(p)
    try:
        assert cancel_all()==1 and p.returncode==-15
    finally:
        ffmpeg_utils._unregister_proc(p)

# 4) _run_atomic: crash removes .part, leaves NO final file

def test_run_atomic_failure_cleans_part(tmp_path):
    out=str(tmp_path/'out.mp4')
    class Boom:
        def run(self, args, total=None, on_progress=None, desc='ffmpeg'):
            Path(args[-1]).write_bytes(b'\x00'*10)   # write the .part like a real crash mid-encode
            raise FFmpegError('died')
    with pytest.raises(FFmpegError):
        _run_atomic(Boom(), ['-i','x',out], out, total=1.0)
    assert not Path(out).exists() and not Path(out+'.part').exists()

# 5) _run_atomic: success replaces atomically + injects -f <fmt> before the temp arg

def test_run_atomic_success_injects_format_and_replaces(tmp_path):
    out=str(tmp_path/'out.mp4'); seen={}
    class Ok:
        def run(self, args, total=None, on_progress=None, desc='ffmpeg'):
            seen['args']=list(args); Path(args[-1]).write_bytes(b'ok')
    _run_atomic(Ok(), ['-i','x',out], out, total=1.0)
    a=seen['args']
    assert a[-3:]==['-f','mp4', out+'.part']      # fmt injected, temp path substituted
    assert Path(out).read_bytes()==b'ok' and not Path(out+'.part').exists()
```
````

**Риск / безопасность.** Net-new file. cancel_all touches the real module-global _running_procs set — always _unregister in a finally so a stray proc can't leak into another test's cancel_all count. _run_atomic uses os.replace which needs the tmp on the same filesystem — tmp_path guarantees that. Do not assert exact stderr formatting beyond substrings (head/... /tail joins vary with buffer sizes).

**Тест (зафиксировать поведение).** test_run_atomic_failure_cleans_part: a fake ff.run that writes the .part then raises FFmpegError; assert Path(out) absent AND Path(out+'.part') absent — locks the truncation guard that prevents a playable-looking half-render.

---
## 19. POST /api/queue/stop only kills ffmpeg when the queue is actually running — else a stale Stop button (or out-of-band call) murders the editor's in-flight render with a raw exit-1 dump
⚪ LOW · симптом 2 · effort S · `queue-stop-guard`

- **Файл / якорь:** `serve.py` — queue_stop() (~line 2925-2931)
- **Закрывает находки:** serve.py:2930

**Проблема.** queue_stop() calls the process-wide ffmpeg_utils.cancel_all() unconditionally, without checking _queue_running. cancel_all terminates EVERY registered ffmpeg (editor, queue, previews). Fired while the queue is idle and the editor is mid-render, it kills the editor's ffmpeg; because task["cancelled"] is never set on this path the task surfaces the ugly "render failed (exit 1). ffmpeg said: …" dump instead of a clean «Задача отменена». This violates the exact invariant /api/cancel documents (serve.py:2660-2662).

**Почему так.** The endpoint now enforces the documented invariant instead of trusting a poll-derived UI flag; the guard is one lock-read with no effect on the legitimate running-queue path.

**Текущий код:**
```
@app.post("/api/queue/stop")
def queue_stop():
    """Abort the running queue: flag it and kill any in-flight ffmpeg. The
    current job ends in 'error: Очередь остановлена'; pending jobs stay pending."""
    _queue_cancel.set()
    ffmpeg_utils.cancel_all()
    return {"ok": True}
```

**Изменение:**
```
@app.post("/api/queue/stop")
def queue_stop():
    """Abort the running queue: flag it and kill any in-flight ffmpeg. The
    current job ends in 'error: Очередь остановлена'; pending jobs stay pending."""
    _queue_cancel.set()
    # cancel_all() is PROCESS-WIDE (kills every registered ffmpeg). Only fire it
    # when the queue is really running — otherwise a stale Stop button (queue
    # finished, editor render started) or an out-of-band call would terminate the
    # editor's ffmpeg with a raw exit-1 dump instead of a clean cancel. This is
    # the same editor<->queue exclusion invariant /api/cancel relies on.
    with TASK_LOCK:
        running = _queue_running
    if running:
        ffmpeg_utils.cancel_all()
    return {"ok": True}
```

**Риск / безопасность.** Minimal. When the queue IS running, behavior is byte-identical. When idle, we still set _queue_cancel (harmless; cleared by _start_queue_worker at line 1624 on next start) but skip the kill. _queue_running is read under TASK_LOCK, matching how the flag is written (serve.py:298-306, 1617-1624). No change to the editor's own /api/cancel path.

**Тест (зафиксировать поведение).** tests/test_api_queue.py::test_queue_stop_idle_does_not_cancel_ffmpeg — monkeypatch serve._queue_running=False and ffmpeg_utils.cancel_all to record calls; POST /api/queue/stop (with a valid local Origin); assert 200 and cancel_all NOT called. Companion test_queue_stop_running_cancels — monkeypatch _queue_running=True; assert cancel_all IS called. Mirrors the cancel_all-capture pattern already in tests/test_csrf.py:74.

---

# Волна 2 — Субтитры

## 20. One shared effective-cutlist helper so burned/sidecar subs, chapters, metadata and enrich are built from the SAME sliver-filtered timeline render() executes (fixes cumulative subtitle desync)
🔴 CRITICAL · симптом 1 · effort M · `r1-effective-cutlist-helper`

- **Файл / якорь:** `vpipe/render.py` — render() sliver-drop block ~866-881; new module-level helper effective_cut()
- **Закрывает находки:** vpipe/render.py:873, serve.py:1162, serve.py:1209, serve.py:1251, serve.py:1290, serve.py:1302, serve.py:1322, vpipe/subtitles.py:223

**Проблема.** render() drops kept slivers shorter than min_seg = max(1.2/fps, cfg.render.min_segment) (render.py:873-876), effectively enlarging the removed regions — it even logs 'dropped N tiny kept sliver(s)'. But every final-coordinate artifact is built from the UNfiltered removed: burn.ass Timeline (serve.py:1209), sidecar .srt/.vtt (subtitles.generate → serve.py:1290), chapters (serve.py:1302), metadata (serve.py:1322) and enrich windows (serve.py:1251). Each dropped sliver makes the rendered video shorter than the subtitle Timeline, so every subtitle after that point is LATE and the error accumulates monotonically — tens of slivers × ~40-50ms = 0.5-2s+ drift by the end of a long talky video, worse if the user raises render.min_segment. Words inside a dropped sliver also survive in the subs though their audio/video is gone.

**Почему так.** This is the canonical R1 fix: a single helper computes the sliver-filtered effective cutlist ONCE and both the encoder and every overlay/subtitle/chapter generator consume it, eliminating the monotonic desync at its source instead of patching each generator separately.

**Текущий код:**
```
    removed, censors = resolve(cl)
    tl = Timeline(removed, media.duration)
    new_dur = tl.new_duration()
    frame = (1.2 / media.fps) if media.fps > 0 else 0.04
    min_seg = max(frame, float(getattr(cfg.render, "min_segment", 0.0) or 0.0))
    all_kept = tl.kept_segments()
    kept = [(a, b) for (a, b) in all_kept if (b - a) >= min_seg]
    if len(kept) < len(all_kept):
        log(f"  dropped {len(all_kept) - len(kept)} tiny kept sliver(s) "
            f"(< {min_seg*1000:.0f} ms) — merged near-adjacent cuts.")
```

**Изменение:**
```
1) Extract the sliver logic into ONE module-level helper in render.py (import merge_intervals: `from .timeline import Timeline, merge_intervals`):

    def sliver_min_seg(media: MediaInfo, cfg: Config) -> float:
        frame = (1.2 / media.fps) if media.fps > 0 else 0.04
        return max(frame, float(getattr(cfg.render, "min_segment", 0.0) or 0.0))

    def effective_cut(cl: CutList, media: MediaInfo, cfg: Config
                      ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """THE cutlist render() actually executes. Returns (kept, removed_eff):
        kept = sliver-filtered kept segments; removed_eff = merged removed whose
        complement over [0,duration] is exactly `kept`. Every subtitle / chapter /
        metadata / enrich Timeline MUST be built from removed_eff so its timestamps
        match the encoded video."""
        removed, _ = resolve(cl)
        tl = Timeline(removed, media.duration)
        min_seg = sliver_min_seg(media, cfg)
        all_kept = tl.kept_segments()
        kept = [(a, b) for (a, b) in all_kept if (b - a) >= min_seg]
        dropped = [(a, b) for (a, b) in all_kept if (b - a) < min_seg]
        removed_eff = merge_intervals(list(tl.removed) + dropped)
        return kept, removed_eff

2) In render() replace lines 866-879 to use it (preserving the log and censors):

    removed, censors = resolve(cl)
    kept, _removed_eff = effective_cut(cl, media, cfg)
    tl = Timeline(_removed_eff, media.duration)
    new_dur = tl.new_duration()
    all_kept = Timeline(removed, media.duration).kept_segments()
    if len(kept) < len(all_kept):
        log(f"  dropped {len(all_kept) - len(kept)} tiny kept sliver(s) — merged near-adjacent cuts.")

   (Downstream uses of `removed`/`tl`/`new_dur`/`kept` stay valid: tl is now built on removed_eff, tl.total_removed/new_duration are consistent with kept, and `out_adur = sum(b-a for a,b in kept)` is unchanged.)

3) In serve.py compute the effective removed ONCE and feed it everywhere in the main render path. Replace serve.py:1162 `removed, _ = resolve(cl)` with:

    removed, _ = resolve(cl)                       # raw — kept for facecrop range only
    _, removed_eff = render_mod.effective_cut(cl, s.media, cfg)

   Then change the Timeline/removed consumers to removed_eff: burn.ass (serve.py:1209 `Timeline(removed_eff, ...)` and remap_words on it), enrich (serve.py:1251 `Timeline(removed_eff, ...)`), subtitles.generate (serve.py:1290 pass removed_eff), chapters.generate (serve.py:1302 pass removed_eff), metadata.generate (serve.py:1322 pass removed_eff), and the return-summary Timeline (serve.py:1337 `Timeline(removed_eff, ...)`). Leave the facecrop detect-range block (serve.py:1185) on raw `removed` (it only needs the outer span).
```

**Риск / безопасность.** Medium. render()'s ENCODE output is byte-for-byte unchanged: `kept` is computed by the identical formula, so the exact same trim/atrim graph is emitted; only the subtitle-side Timelines shift to match it. subtitles.generate / chapters.generate / metadata.generate keep their signatures (they already take a `removed` list) — the caller just passes the filtered list, so no cross-module signature churn. merge_intervals(list(tl.removed)+dropped) is safe: each dropped sliver touches its neighboring removed intervals (its endpoints ARE removed-interval endpoints) so they merge with gap=0, and the complement equals `kept` exactly. Scope is deliberately the MAIN render path; clip-maker (cutlist_override) and NLE-export consumers (serve.py:2502/2535/3049/3334) are out of scope — clips don't get enrich and their sub timing is a separate finding. Watch: tr.duration vs media.duration — subtitles/chapters build Timeline on transcript.duration internally; pass the same removed_eff (interval math is duration-independent for the removed set).

**Тест (зафиксировать поведение).** tests/test_cut_smoothing.py::test_effective_cut_matches_render_kept — build a cutlist with two cuts 30ms apart (remove [10,12] and [12.03,15]); assert effective_cut() drops the 30ms sliver, removed_eff merges [10,15], and Timeline(removed_eff).new_duration() == sum(b-a for kept). tests/test_subtitles.py::test_subs_use_effective_cutlist — stack N such near-adjacent pairs, generate subs from removed_eff and assert a cue after the last cut lands at the render-consistent time (no N×30ms drift) vs the old raw-removed path. Add an end-to-end serve test asserting rendered new_duration == sr sidecar last-cue-consistent duration.

---
## 21. Honor rotation metadata (phone portrait): read side_data_list/displaymatrix and swap width/height so PlayRes, punch-zoom, backplate and aspect logic see DISPLAY dims
🟠 HIGH · симптом 1 · effort M · `probe-rotation-swap`

- **Файл / якорь:** `vpipe/probe.py` — probe_media() return, lines 45-46 (width/height)
- **Закрывает находки:** vpipe/probe.py:45

**Проблема.** probe_media reads coded width/height raw; nothing in vpipe inspects side_data_list/displaymatrix. ffmpeg autorotates at decode, so a phone clip tagged rotation=90 probes as 1920x1080 while every frame in the filtergraph is 1080x1920. Consequences: (a) _final_render_dims computes PlayResX/Y=1920x1080 for a portrait render → burned ASS subtitles get wrong font scale/margins/wrapping (the «субтитры корявые» phone case); (b) _punch_dims and the blur-backplate scale={out_w}:{out_h} hard-scale portrait frames to landscape dims → enrich punch-zoom/cards squish the whole video; (c) _render_formats duplicate-skip misjudges aspect (a 16x9 reframe of a rotated portrait clip is skipped as 'matches source'); (d) the enrich planner receives swapped out_w/out_h.

**Текущий код:**
```
        fps=parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0") if v else 0.0,
        width=int(v.get("width", 0)) if v else 0,
        height=int(v.get("height", 0)) if v else 0,
```

**Изменение:**
```
Add a helper above probe_media:

    def _stream_rotation(v: dict) -> int:
        """Net display rotation in degrees (0/90/180/270). Reads the Display
        Matrix side-data (ffprobe reports a signed 'rotation', e.g. -90) and the
        legacy tags.rotate. Returns a normalized non-negative multiple of 90."""
        rot = 0.0
        for sd in (v.get("side_data_list") or []):
            if "rotation" in sd:
                try:
                    rot = float(sd.get("rotation") or 0.0)
                except (TypeError, ValueError):
                    rot = 0.0
                break
        else:
            try:
                rot = float((v.get("tags", {}) or {}).get("rotate", 0) or 0)
            except (TypeError, ValueError):
                rot = 0.0
        return int(round(rot)) % 360

Then compute display dims in probe_media before the MediaInfo return:

    width = int(v.get("width", 0)) if v else 0
    height = int(v.get("height", 0)) if v else 0
    if v is not None and _stream_rotation(v) in (90, 270):
        width, height = height, width

and use these `width`/`height` locals in the MediaInfo(...) call (replace the two `int(v.get(...))` expressions with `width=width, height=height`).
```

**Риск / безопасность.** Changes width/height ONLY for rotated inputs (rot 90/270); non-rotated files (rot 0/180) are byte-for-byte unchanged, so no existing landscape test moves. Rotated phone footage was already broken, so every downstream (PlayRes, punch dims, aspect dup-skip, enrich planner) strictly improves. Watch: the displaymatrix sign convention — ffprobe emits rotation=-90 for a clockwise-90 display; we only care about the axis swap, so mod-360 mapping 90 and 270 both swap and 180 does not — correct for W/H swap regardless of sign. side_data_list is present in `ffprobe -show_streams` (ffmpeg_utils.py:125 already passes it) for streams with a Display Matrix; the tags.rotate fallback covers older muxes. Do not also swap the raw coded dims used for hashing (hash_input reads bytes, unaffected).

**Тест (зафиксировать поведение).** Add tests/test_probe.py::test_rotation_90_swaps_dims — fake ff.probe returns a video stream width=1920,height=1080 with side_data_list=[{'side_data_type':'Display Matrix','rotation':-90}]; assert probe_media(ff,'x').width==1080 and .height==1920. ::test_rotation_180_no_swap (rotation:180) → dims unchanged. ::test_legacy_tags_rotate (no side_data, tags={'rotate':'90'}) → swapped. ::test_no_rotation_landscape_unchanged → 1920x1080 stays.

---
## 22. Snap every kept-segment boundary to the source frame grid so concat inserts no silence padding and Timeline arithmetic matches the true output timeline
🟠 HIGH · симптом 1 · effort M · `frame-snap-cut-boundaries`

- **Файл / якорь:** `vpipe/render.py` — effective_cut() helper (from P6) + render.py:1173/1211 trim build
- **Закрывает находки:** vpipe/render.py:1174, vpipe/render.py:1211
- **Зависит от:** `r1-effective-cutlist-helper`

**Проблема.** Kept segments are cut with frame-quantized `trim=start=a:end=b` for video (segment duration = whole frames, off from b−a by up to +1 frame) but sample-exact `atrim` for audio. At each seam where the video segment overshoots the audio, `concat=n=N:v=1:a=1` pads the shorter audio with silence, so actual output positions are cumsum(max(V_i, A_i)) while burned-karaoke / enrich-overlay / chapter times are computed as cumsum(b−a) via Timeline arithmetic. The excess is one-sided, so drift grows LINEARLY (~1/(6·fps) ≈ 5-6ms/cut); over 300-500 cuts burned subtitles and overlays end up ~1.7-2.8s EARLY near the end of a long file.

**Почему так.** Eliminates the one-sided concat-padding drift at the source by making the video and audio cut lengths identical; combined with P6 it guarantees subs/overlays/chapters ride the exact rendered timeline for the whole file, killing the 'субтитры съезжают на длинных видео' symptom.

**Текущий код:**
```
    def effective_cut(cl, media, cfg):
        removed, _ = resolve(cl)
        tl = Timeline(removed, media.duration)
        min_seg = sliver_min_seg(media, cfg)
        all_kept = tl.kept_segments()
        kept = [(a, b) for (a, b) in all_kept if (b - a) >= min_seg]
        dropped = [(a, b) for (a, b) in all_kept if (b - a) < min_seg]
        removed_eff = merge_intervals(list(tl.removed) + dropped)
        return kept, removed_eff

    # render.py:1173-1174
    vparts.append(f"[0:v]trim=start={_f(a)}:end={_f(b)},setpts=PTS-STARTPTS[v{i}]")
```

**Изменение:**
```
Extend the P6 helper to snap kept-boundaries to the frame grid BEFORE filtering slivers, gated by a config flag so byte-for-byte output is opt-out. Add `frame_snap: bool = True` to RenderCfg (config.py:353, next to min_segment) with a comment.

    def effective_cut(cl, media, cfg):
        removed, _ = resolve(cl)
        tl = Timeline(removed, media.duration)
        min_seg = sliver_min_seg(media, cfg)
        all_kept = tl.kept_segments()
        fps = media.fps if (getattr(cfg.render, "frame_snap", True) and media.fps > 0) else 0.0
        snapped = []
        for (a, b) in all_kept:
            if fps:
                a = round(a * fps) / fps
                b = round(b * fps) / fps
            if b - a >= min_seg:
                snapped.append((a, b))
        kept = snapped
        # removed_eff = complement of the SNAPPED kept over [0, duration]
        removed_eff, cursor = [], 0.0
        for (a, b) in kept:
            if a > cursor:
                removed_eff.append((cursor, a))
            cursor = b
        if cursor < media.duration:
            removed_eff.append((cursor, media.duration))
        removed_eff = merge_intervals(removed_eff)
        return kept, removed_eff

Because render() emits trim=start/end from THIS `kept` (P6 wired it) and all subtitle/enrich/chapter Timelines are built from removed_eff (also P6), audio, video, subs now share one frame-quantized timeline: V_i == A_i so concat inserts no padding, and Timeline arithmetic == true output positions.
```

**Риск / безопасность.** High-ish blast radius on ENCODE bytes: when frame_snap=True the trim boundaries shift by up to half a frame, so rendered output is NOT byte-for-byte identical to today and any golden-hash render test will change. Two safety valves: (1) the flag defaults True (correct behavior) but can be set False to restore exact legacy cuts; (2) snapping keys off media.fps>0 so audio-only or unknown-fps inputs are unaffected. Update/relax any exact-duration or golden test to the snapped expectation. Frame_snap must run BEFORE the sliver filter (a snap can shrink a segment below min_seg). Keep _f() 4-decimal formatting — round(a*fps)/fps still prints cleanly. Apply P6 first; this patch only edits the one helper, so render()/serve.py wiring is already done.

**Тест (зафиксировать поведение).** tests/test_cut_smoothing.py::test_frame_snap_aligns_av — with fps=25, cuts producing 200 kept segments at non-grid word times, assert every kept boundary * fps is within 1e-6 of an integer, and that Timeline(removed_eff).new_duration() equals the sum of frame-quantized segment lengths (i.e. == the length concat will actually produce). test_frame_snap_off_is_legacy — frame_snap=False reproduces the pre-patch kept list exactly. Add a real-ffmpeg-gated test rendering ~200 cuts and asserting output duration == expected within <1 frame.

---
## 23. ASS style size/margins are absolute px while PlayRes tracks output resolution — same preset renders at wildly different relative size per format
🟠 HIGH · симптом 1 · effort M · `ass-size-scale-by-playres`

- **Файл / якорь:** `vpipe/subtitles.py` — write_ass() Style: Default line, ~line 418 + 438-446
- **Закрывает находки:** vpipe/subtitles.py:438, vpipe/subtitles.py:441

**Проблема.** write_ass sets PlayResX/Y to the FINAL output dims (serve.py:1216→1226) but emits Fontsize/MarginV/MarginL,R/Outline/Shadow as raw px. ASS sizes are in PlayResY units, so preset 'classic' size=52 is 4.8% of frame height at 1080p but 2.4% on a 4K 'as source' render, 10.8% at 480p, 2.7% on a 1080x1920 Shorts crop. A single multiformat job burns visibly different relative caption sizes per format from ONE preset. enrich_cards already scales everything by H/1080 (_px) — burn.ass is the odd one out.

**Почему так.** k = pr_y/1080 is exactly the enrich_cards convention, so burn and enrich overlays finally scale identically. At pr_y=1080 k=1.0 and int(round(v*1.0))==int(v) / _num(x*1.0)==_num(x), so every 1080-baseline output (the common case and all 1080 golden tests) is unchanged byte-for-byte. Non-1080 outputs now hold a constant cap-height fraction.

**Текущий код:**
```
    pr_x, pr_y = int(play_res[0] or 1920), int(play_res[1] or 1080)
    align = _ASS_ALIGN.get(style.position, 2)
...
        ("Style: Default,{font},{size},{primary},{secondary},{outline_c},"
         "&H64000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},{align},"
         "40,40,{margin_v},1").format(
            font=style.font, size=int(style.size),
            primary=(style.karaoke_color if karaoke else style.primary_color),
            secondary=style.primary_color,
            outline_c=style.outline_color,
            outline=_num(style.outline), shadow=_num(style.shadow),
            align=align, margin_v=int(style.margin_v)),
```

**Изменение:**
```
Add after the pr_x/pr_y line (~418):
    # Presets are defined @1080 (serve.CAPTION_PRESETS). PlayResY == output height,
    # so scale metrics by pr_y/1080 to keep a constant fraction of frame height
    # across resolutions/formats (mirrors enrich_cards._px). k==1.0 at 1080 → the
    # 1080p/1920x1080 render stays byte-identical.
    k = pr_y / 1080.0
Then rewrite the Style line's scaled fields:
         "...,1,{outline},{shadow},{align},"
         "{mlr},{mlr},{margin_v},1").format(
            font=style.font, size=max(1, int(round(style.size * k))),
            primary=(style.karaoke_color if karaoke else style.primary_color),
            secondary=style.primary_color,
            outline_c=style.outline_color,
            outline=_num(style.outline * k), shadow=_num(style.shadow * k),
            align=align, mlr=int(round(40 * k)),
            margin_v=int(round(style.margin_v * k))),
(i.e. replace the literal '40,40' with '{mlr},{mlr}' and add the mlr kwarg.)
```

**Риск / безопасность.** One existing test PINS THE BUG and must flip: test_caption_presets.py::test_write_ass_per_preset renders at play_res=(1080,1920) (k=1920/1080=1.7778) and asserts bold size f[2]=='78' and neon margin_v f[-2]=='160'. After the fix those become '139' (round(78*1.7778)) and '284' (round(160*1.7778)). Update those two expects. Do NOT introduce a force_style anywhere (render.py:1002 passes none — no double-scaling). Residual (out of this cluster): app.js:2714 preview hardcodes size*0.36 assuming 1080, so the modal preview still under-shows on non-1080 formats — flag for the frontend/R3 work, not this patch.

**Тест (зафиксировать поведение).** (1) Update the two pinned expects in test_caption_presets.py::test_write_ass_per_preset as above. (2) Add tests/test_write_ass.py::test_ass_size_scales_with_playres: write same AssStyleCfg(size=52,margin_v=40) at (1920,1080)→Fontsize field=='52'; at (1920,2160)→'104'; at (1280,720)→'35' (round(52*0.6667)). Assert MarginV scales likewise (40→80 at 2160).

---
## 24. Untested cross-module contract: render() drops sub-min_segment kept slivers AFTER subtitles/ASS were timed on the un-dropped Timeline — cumulative A/V-vs-subs desync + overstated duration have no test
🟠 HIGH · симптом 1 · effort M · `render-sliver-desync-contract`

- **Файл / якорь:** `tests/test_render_sliver.py` — vpipe.render.render sliver drop (render.py:869-879, 1234) vs serve ASS/generate on full Timeline (serve.py:1209-1213, 1290, 1337-1340)
- **Закрывает находки:** vpipe/render.py:876, vpipe/render.py:1234, serve.py:1209, serve.py:1290, serve.py:1340

**Проблема.** render() encodes only kept segments >= min_seg=max(1.2/fps, cfg.render.min_segment) (render.py:876) but returns new_duration=tl.new_duration() computed on the FULL timeline (868->1234), and serve builds burn-ASS cues (serve.py:1209-1213) and sidecar .srt (1290) and the reported new_duration (1340) all from the same un-dropped `removed`. So every dropped sliver makes the real video shorter than the subtitle timeline; all cues after the sliver drift late by the sliver length, and the reported duration overstates the file. Grep 'min_segment'/'sliver' in tests/: zero. This is R1: the fix is ONE shared effective-cutlist helper used by encode + every subtitle/overlay/chapter generator.

**Текущий код:**
```
# tests/test_edge_fade.py:79-133 — the FakeFF + _media + _run helpers to reuse
class FakeFF: ...            # records runs, writes empty output file
def _media(has_audio=True):  # MediaInfo(duration=10.0, fps=30.0, 1920x1080, ...)
def _run(cfg, cl, tmp_path, ...): return render(ff, _media(...), cl, cfg, out, work, ...)
def _graph(ff): return args[args.index('-filter_complex')+1]
# No test anywhere constructs a sub-min_seg kept sliver.
```

**Изменение:**
````
New file tests/test_render_sliver.py, importing FakeFF/_media/_graph/_run helpers from test_edge_fade (or copy them). Scenario: 10s source at fps=30 => frame=1.2/30=0.04s, cfg.render.min_segment default 0.04 => min_seg=0.04. Cuts [5.0,6.0] and [6.02,7.0] leave kept complement (0,5),(6.0,6.02),(7,10); the 0.02s sliver is dropped => kept=[(0,5),(7,10)], concat n=2. tl.new_duration()=10-1.98=8.02, but encoded sum=8.00.

```python
import re
import pytest
from vpipe.render import render
from vpipe.timeline import Timeline, remap_words
from vpipe import subtitles as subs
from vpipe.config import SubsCfg, MaskingCfg, ProfanityLists
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import CutList, CutSegment, Word, Segment, Transcript, TYPE_MANUAL, ACTION_REMOVE
# reuse FakeFF/_media/_graph from test_edge_fade
from test_edge_fade import FakeFF, _media, _graph, _cfg

def _cl():
    segs=[CutSegment(id='r0',start=5.0,end=6.0,type=TYPE_MANUAL,action=ACTION_REMOVE,enabled=True),
          CutSegment(id='r1',start=6.02,end=7.0,type=TYPE_MANUAL,action=ACTION_REMOVE,enabled=True)]
    return CutList(source='in.mp4', duration=10.0, segments=segs)

def _trims(g):  # -> [(start,end)] parsed from trim=start=..:end=..
    return [(float(s),float(e)) for s,e in re.findall(r'trim=start=([0-9.]+):end=([0-9.]+)', g)]

def test_sliver_dropped_from_encode_graph(tmp_path):
    ff=FakeFF(); render(ff,_media(),_cl(),_cfg(),str(tmp_path/'o.mp4'),str(tmp_path),log=lambda*a,**k:None)
    tr=_trims(_graph(ff))
    assert tr==[(0.0,5.0),(7.0,10.0)]             # sliver (6.0,6.02) gone; concat n=2
    encoded=sum(b-a for a,b in tr)
    assert abs(encoded-8.0)<1e-6

@pytest.mark.xfail(strict=True, reason='R1: render new_duration is tl.new_duration() (8.02), not encoded kept sum (8.0); flip green when the shared effective-cutlist helper lands')
def test_reported_duration_matches_encoded(tmp_path):
    ff=FakeFF(); rr=render(ff,_media(),_cl(),_cfg(),str(tmp_path/'o.mp4'),str(tmp_path),log=lambda*a,**k:None)
    assert round(rr['new_duration'],2)==8.0

@pytest.mark.xfail(strict=True, reason='R1: ASS/generate use the full Timeline, so cues after a dropped sliver are 20ms late vs the encoded video')
def test_subs_share_effective_timeline(tmp_path):
    # a word at original t=7.0 sits at encoded 5.0, but remap on the FULL timeline puts it at 5.02
    tl_full=Timeline([(5.0,6.0),(6.02,7.0)], 10.0)
    w=remap_words([Word('после',7.0,7.5)], tl_full)[0]
    assert abs(w.start-5.0)<1e-6      # should equal the ENCODED position, currently 5.02
```
````

**Риск / безопасность.** The two xfail(strict=True) tests are the debt-lock: they MUST stay red until the R1 source fix lands, then auto-flip (strict makes an unexpected pass fail, forcing removal of the marker). Do NOT delete the fast-path/byte-for-byte render tests. Importing from test_edge_fade requires tests/ on sys.path (test_edge_fade already does sys.path.insert at line 31) — if pytest rootdir import fails, copy the ~15-line FakeFF/_media helpers instead of importing. test_sliver_dropped_from_encode_graph is green today and locks the drop behavior itself.

**Тест (зафиксировать поведение).** test_sliver_dropped_from_encode_graph (green, locks the drop) + test_reported_duration_matches_encoded and test_subs_share_effective_timeline as xfail(strict=True) that flip to pass exactly when the R1 shared effective-cutlist helper unifies encode + subtitle timelines.

---
## 25. subtitles.generate() — the only production path writing sidecar .srt/.vtt after cuts — is never called by any test; the one in-context caller stubs it out
🟠 HIGH · симптом 1 · effort M · `subtitles-generate-golden`

- **Файл / якорь:** `tests/test_subtitles_generate.py` — vpipe.subtitles.generate (subtitles.py:219-241)
- **Закрывает находки:** vpipe/subtitles.py:219, tests/test_multiformat.py:303, tests/test_preview_f2.py:47

**Проблема.** generate() chains Timeline -> remap_words -> build_cues -> write_srt/write_vtt/write_transcript and returns the paths dict serve consumes (serve.py:1290). No test imports it: test_subtitles.py tests build_cues/mask pieces; test_preview_f2.py:47-72 re-implements the chain by hand; test_multiformat.py:313 monkeypatches serve.subs_mod.generate with a fake. A regression inside generate() (wrong total to build_cues, with_suffix collisions, write_vtt/transcript plumbing) is undetectable, and no test reads back a produced .srt to check numbering/timestamps/ordering — the exact 'субтитры корявые' artifact.

**Текущий код:**
```
# tests/test_multiformat.py:303-305 — generate() is STUBBED, never run:
def fake_subs(tr, removed, scfg, mask, matcher, base, log=None):
    subs_calls.append(Path(base))
    return {'srt': str(base)+'.srt', 'vtt': str(base)+'.vtt', 'cues': 3}
monkeypatch.setattr(serve.subs_mod, 'generate', fake_subs)
# tests/test_preview_f2.py:59-64 — the chain re-implemented by hand instead of calling generate()
```

**Изменение:**
````
New file tests/test_subtitles_generate.py — a golden round-trip that PARSES the emitted .srt.

```python
from pathlib import Path
import re
from vpipe.subtitles import generate
from vpipe.config import SubsCfg, MaskingCfg, ProfanityLists
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Transcript, Segment, Word

def _parse_srt(text):
    blocks=[b for b in text.strip().split('\n\n') if b.strip()]
    out=[]
    for b in blocks:
        idx, times, *txt = b.splitlines()
        s,e = times.split(' --> ')
        out.append((int(idx), s, e, '\n'.join(txt)))
    return out

def _sec(ts):  # HH:MM:SS,mmm -> float
    h,m,rest=ts.split(':'); s,ms=rest.split(','); return int(h)*3600+int(m)*60+int(s)+int(ms)/1000

def _tr():
    words=[Word('привет',0.0,0.5),Word('это',0.6,1.0),Word('блядь',1.1,1.6),
           Word('вырезано',2.5,3.5), Word('дальше',6.0,6.6), Word('конец.',6.7,7.2)]
    return Transcript(language='ru',duration=8.0,model='t',audio_hash='h',
                      segments=[Segment(0.0,8.0,'txt',words)])

def test_generate_srt_golden(tmp_path):
    matcher=ProfanityMatcher(ProfanityLists(roots=['бля'],allow=[]))
    removed=[(2.0,4.0)]                       # drops 'вырезано', shifts later words -2s
    res=generate(_tr(), removed, SubsCfg(write_vtt=True, write_transcript=True),
                 MaskingCfg(), matcher, tmp_path/'clip')
    srt=_parse_srt(Path(res['srt']).read_text(encoding='utf-8'))
    assert res['cues']==len(srt) and len(srt)>=1
    # 1) monotonic, non-overlapping, 1-based contiguous numbering
    assert [b[0] for b in srt]==list(range(1,len(srt)+1))
    prev_end=-1.0
    for _,s,e,txt in srt:
        ss,ee=_sec(s),_sec(e)
        assert ss>=prev_end-1e-6 and ee>ss and ee<=6.0+1e-6   # within new_duration (8-2)
        prev_end=ee
    joined=' '.join(b[3] for b in srt)
    assert 'вырезано' not in joined            # word inside the cut is gone
    assert 'б***ь' in joined and 'блядь' not in joined  # profanity masked in the sidecar
    # 2) .vtt + transcript.txt actually written and flagged
    assert Path(res['vtt']).exists() and Path(res['vtt']).read_text(encoding='utf-8').startswith('WEBVTT')
    assert Path(res['transcript']).exists() and 'вырезано' not in Path(res['transcript']).read_text(encoding='utf-8')

def test_generate_no_optional_outputs(tmp_path):
    matcher=ProfanityMatcher(ProfanityLists())
    res=generate(_tr(), [], SubsCfg(write_vtt=False, write_transcript=False),
                 MaskingCfg(), matcher, tmp_path/'clip')
    assert 'vtt' not in res and 'transcript' not in res and Path(res['srt']).exists()
```
````

**Риск / безопасность.** Net-new, fully hermetic (no ffmpeg). Confirm SubsCfg field names write_vtt/write_transcript exist (they are read in generate() at subtitles.py:231/235). Keep assertions on INVARIANTS (monotonic, in-range, masked, numbering) not exact millisecond values, so legitimate cue-timing tuning doesn't churn the golden. transcript.txt is written to base.parent/'transcript.txt' (subtitles.py:236) — assert on res['transcript'] path, not a guessed name.

**Тест (зафиксировать поведение).** test_generate_srt_golden: transcript+removed -> generate() -> parse the .srt and assert cue count matches res['cues'], 1-based contiguous numbering, monotonic non-overlapping times all <= new_duration, dropped word absent, profanity masked, and .vtt/transcript.txt written when their flags are on.

---
## 26. Karaoke path discards _wrap's balanced line breaks — max_lines and Russian break rules are lost, libass re-wraps to 3-5 ragged lines
🟡 MEDIUM · симптом 1 · effort M · `karaoke-wrap-newlines`

- **Файл / якорь:** `vpipe/subtitles.py` — _karaoke_text() return, ~line 400
- **Закрывает находки:** vpipe/subtitles.py:400

**Проблема.** _karaoke_text returns ' '.join(out_parts) with NO \N, contradicting its own docstring (lines 344-346). The carefully balanced 2-line _wrap output (avoids breaks after Russian prepositions via _BAD_LINE_TAIL, balances line lengths) is used only for .srt and non-karaoke burn. For karaoke (the default) libass WrapStyle 0 re-wraps the flat string by pixel width: max_lines=2 is not enforced (an 84-char budget cue at preset size 62-78 on a 1080x1920 frame with 40px margins wraps to 3-4 lines), breaks land after «в»/«не»/«что», and burned layout never matches the .srt/preview — «кривые переносы».

**Почему так.** cue.text already carries _wrap's '\n'-separated balanced layout (built from the same group's disp words); mapping the \k tokens onto those line lengths reproduces it exactly and joins with ASS \N so libass stops re-wrapping. The sum-check is the safety valve for the case the corrected finding flagged (in_cue count != cue.text token count): it degrades to today's flat join instead of shifting words across lines.

**Текущий код:**
```
        elapsed_cs += cs
    return " ".join(out_parts)
```

**Изменение:**
```
        elapsed_cs += cs
    # Reapply cue.text's balanced 2-line layout to the \k tokens: split the
    # tokens positionally into the same per-line counts _wrap chose and join line
    # groups with the ASS hard break \N. Fall back to a flat line when the token
    # count doesn't match the wrapped text (e.g. eps dropped a word) so we never
    # misalign the karaoke fill.
    line_counts = [len(ln.split(" ")) for ln in cue.text.split("\n")]
    if len(line_counts) > 1 and sum(line_counts) == len(out_parts):
        chunks, idx = [], 0
        for c in line_counts:
            chunks.append(" ".join(out_parts[idx:idx + c]))
            idx += c
        return "\\N".join(chunks)
    return " ".join(out_parts)
```

**Риск / безопасность.** Low, minimal blast radius: single-line cues (cue.text has no '\n' → len(line_counts)==1) keep the exact ' '.join(out_parts) behaviour, so ALL existing kinetic/karaoke tests (which use single-line _cue_words) are untouched. Only multi-line cues gain \N. Depends conceptually on P1 (P1 makes token counts match so the \N branch is taken rather than the fallback).

**Тест (зафиксировать поведение).** Add tests/test_subtitles.py::test_karaoke_wraps_to_two_lines: build a cue whose wrapped cue.text has exactly one '\n' (e.g. cue.text='слово раз два\nтри четыре пять' with 6 matching words) and assert '\\N' appears exactly once in _karaoke_text output and each side has the right \k count. Add ::test_karaoke_single_line_no_newline: a short single-line cue → no '\\N' in output (regression guard for the fallback).

---
## 27. Kinetic keyword pop never restores colour — the popped word AND every later word in the cue stay accent-amber
🟡 MEDIUM · симптом 1 · effort M · `kinetic-pop-restore-colour`

- **Файл / якорь:** `vpipe/subtitles.py` — _kinetic_pop_tag() + _karaoke_text() call, ~lines 315-329, 332-335, 395, 465-466
- **Закрывает находки:** vpipe/subtitles.py:327

**Проблема.** _kinetic_pop_tag emits \t(t0,t1,\fscx120\fscy120\1c{accent}\3c{accent}) then \t(t2,t3,\fscx100\fscy100) — the second transform resets only SCALE. Per ASS semantics a \t's end values persist to the end of the Dialogue event, so \1c/\3c stay amber: the popped word's outline (normally black) turns amber, and every subsequent word in the cue sings amber instead of karaoke_color — reads as a random mis-coloured tail. Kinetic pop is hardcoded on (write_ass passes it unconditionally), so nearly every cue shows this.

**Почему так.** The second \t now animates \1c back to the karaoke fill and \3c back to the outline colour, so after the pop the word (and every following word) resumes the correct sung-colour/black-outline. Defaults on the new params match AssStyleCfg defaults so direct/legacy calls are unaffected. The 8-digit no-trailing-'&' accent token is left as-is (cosmetic per the verified finding; both libass and VSFilter parse it — and test_kinetic_pop_adds_t_scale_and_accent asserts '\\1c&H000B9EF5').

**Текущий код:**
```
def _kinetic_pop_tag(start_ms: int, accent: str) -> str:
...
    return (f"\\t({t0},{t1},\\fscx{KINETIC_SCALE}\\fscy{KINETIC_SCALE}"
            f"\\1c{accent}\\3c{accent})"
            f"\\t({t2},{t3},\\fscx100\\fscy100)")
...
def _karaoke_text(cue: Cue, words: list[Word], matcher: ProfanityMatcher,
                  mask: MaskingCfg, eps: float = 0.02, *,
                  kinetic: bool = True,
                  accent: str = _KINETIC_DEFAULT_ACCENT) -> str:
...
            pop = _kinetic_pop_tag(elapsed_cs * 10, accent)
...
            text = _karaoke_text(c, words or [], kmatcher, kmask,
                                 accent=_KINETIC_DEFAULT_ACCENT)
```

**Изменение:**
```
1) _kinetic_pop_tag gains restore colours (no test calls it directly):
def _kinetic_pop_tag(start_ms: int, accent: str,
                     restore_fill: str = "&H0000FFFF",
                     restore_outline: str = "&H00000000") -> str:
    ...
    return (f"\\t({t0},{t1},\\fscx{KINETIC_SCALE}\\fscy{KINETIC_SCALE}"
            f"\\1c{accent}\\3c{accent})"
            f"\\t({t2},{t3},\\fscx100\\fscy100"
            f"\\1c{restore_fill}\\3c{restore_outline})")
2) _karaoke_text signature gains: karaoke_color: str = "&H0000FFFF", outline_color: str = "&H00000000" (after accent).
3) Its call site (~395): pop = _kinetic_pop_tag(elapsed_cs * 10, accent, karaoke_color, outline_color)
4) write_ass call site (~465): text = _karaoke_text(c, words or [], kmatcher, kmask, accent=_KINETIC_DEFAULT_ACCENT, karaoke_color=style.karaoke_color, outline_color=style.outline_color)
```

**Риск / безопасность.** Low. Existing kinetic tests still pass: they assert the first \t (accent + \fscx120) and that '\\fscx100\\fscy100' is present — the second \t still contains it, now with colour restores appended. Co-located with P1/P4 in _karaoke_text (different lines). Keep the \k fill counts/sum untouched (test_kinetic_pop_keeps_karaoke_fill_intact).

**Тест (зафиксировать поведение).** Add tests/test_subtitles.py::test_kinetic_pop_restores_colours: call _karaoke_text(..., kinetic=True, accent='&H000B9EF5', karaoke_color='&H00AABBCC', outline_color='&H00112233') and assert '\\fscx100\\fscy100\\1c&H00AABBCC\\3c&H00112233' in output (the restoring second \t).

---
## 28. Caption preset tests are tautological on the scaling property; fixed-pixel Fontsize vs per-format PlayRes means the same preset shrinks ~2x on 9:16 vs 16:9 — and _final_render_dims has no direct test
🟡 MEDIUM · симптом 1 · effort S · `caption-cross-format-scale`

- **Файл / якорь:** `tests/test_caption_presets.py` — vpipe.subtitles.write_ass PlayRes (subtitles.py:418-446) + serve._final_render_dims (serve.py:1115-1128)
- **Закрывает находки:** tests/test_caption_presets.py:93, serve.py:1115, vpipe/subtitles.py:438

**Проблема.** write_ass emits absolute-pixel Fontsize/MarginV against PlayResX/Y = final render dims (subtitles.py:418,438). Preset size=78 is 7.2% of a 1080-tall frame but 4.1% of a 1920-tall Shorts frame. test_write_ass_per_preset renders at play_res=(1080,1920) but asserts f[2]=='78' — the input echoed back — which can NEVER catch cross-format mis-scaling. No test compares relative size across two play_res, and _final_render_dims (the source of those dims) has zero direct tests.

**Текущий код:**
```
# tests/test_caption_presets.py:74-96 — asserts input==output only:
def test_write_ass_per_preset(tmp_path, key, expect):
    ...
    write_ass([...], out, style, ..., play_res=(1080, 1920))
    f = _style_line(txt)
    if 'size' in expect: assert f[2] == expect['size']   # '78' echoed back == tautology
```

**Изменение:**
````
Append to tests/test_caption_presets.py. (A) Pin the ACTUAL (absolute-pixel) cross-format contract non-tautologically; (B) add a direct _final_render_dims test.

```python
import serve
from vpipe.subtitles import Cue, write_ass
from vpipe.config import AssStyleCfg

def _size_and_playresy(txt):
    fs=next(l for l in txt.splitlines() if l.startswith('Style: Default,')).split(',')[2]
    pry=next(l for l in txt.splitlines() if l.startswith('PlayResY:')).split(':')[1].strip()
    return int(fs), int(pry)

def test_write_ass_font_is_absolute_across_formats(tmp_path):
    style=AssStyleCfg(**next(p for p in serve.CAPTION_PRESETS if p['key']=='bold')['style'])
    a=tmp_path/'h.ass'; b=tmp_path/'v.ass'
    write_ass([Cue(0,2,'x')], a, style, karaoke=False, play_res=(1920,1080))
    write_ass([Cue(0,2,'x')], b, style, karaoke=False, play_res=(1080,1920))
    sa,pya=_size_and_playresy(a.read_text(encoding='utf-8-sig'))
    sb,pyb=_size_and_playresy(b.read_text(encoding='utf-8-sig'))
    # CONTRACT DECISION (documented): font size is ABSOLUTE px, identical across formats...
    assert sa==sb==int(style.size)
    # ...therefore its RELATIVE height differs ~1.78x between 16:9 and 9:16 (the mis-scaling risk).
    rel_16x9=sa/pya; rel_9x16=sb/pyb
    assert abs(rel_16x9/rel_9x16 - (1920/1080)) < 0.02

def test_final_render_dims_direct():
    s=SimpleNamespace(media=SimpleNamespace(width=1920,height=1080))
    assert serve._final_render_dims(s, None, None)==(1920,1080)      # source
    assert serve._final_render_dims(s, 720, None)==(1280,720)        # scaled height, proportional width
    assert serve._final_render_dims(s, None, (1080,1920))==(1080,1920)  # vertical target wins
    s2=SimpleNamespace(media=SimpleNamespace(width=None,height=None))
    assert serve._final_render_dims(s2, None, None)==(1920,1080)     # defaults when probe blank
```
SimpleNamespace is already imported at test_caption_presets.py:17.
````

**Риск / безопасность.** test_write_ass_font_is_absolute_across_formats PINS the current absolute-pixel decision; if the team later chooses to scale size by PlayResY, this test is the single place that must flip (assert sa!=sb, rel constant) — that is exactly the surfaced decision the finding asks for. No source change here. _final_render_dims returns (out_w,out_h) with out_w rounded (serve.py:1127) — use the concrete 1280x720 from 1920x1080@720 which is exact.

**Тест (зафиксировать поведение).** test_write_ass_font_is_absolute_across_formats: render the same 'bold' preset at (1920,1080) and (1080,1920), assert Fontsize is byte-identical (absolute) AND that size/PlayResY differs by the 16:9-vs-9:16 aspect ratio — a real, non-tautological cross-format assertion that documents the mis-scaling contract.

---
## 29. Line-wrapping heuristic _wrap has zero direct tests; the >2-line greedy path silently DELETES words (lines[:max_lines]) and the 2-line path can emit lines longer than max_chars
🟡 MEDIUM · симптом 1 · effort S · `wrap-line-heuristic-tests`

- **Файл / якорь:** `tests/test_subtitles.py` — vpipe.subtitles._wrap (subtitles.py:67-117)
- **Закрывает находки:** vpipe/subtitles.py:117, vpipe/subtitles.py:89

**Проблема.** _wrap decides how every cue looks. Its >2-line fallback ends `return '\n'.join(lines[:max_lines])` (subtitles.py:117) — words beyond max_lines are dropped from the subtitle. Its max_lines==2 branch accepts a best split even when both lines overflow max_chars (over is only penalized, 89-94), producing lines libass re-wraps unpredictably ('корявые'). No test imports _wrap: grep '_wrap' in tests/ = nothing; build_cues tests check only gap-splitting.

**Текущий код:**
```
# tests/test_subtitles.py:42-51 — the closest existing coverage, only gap-splitting:
def test_build_cues_splits_on_gap():
    ...
    cues = build_cues(words, m, SubsCfg(new_cue_gap=0.7), MaskingCfg(), total=7.0)
    assert len(cues) == 2
# _wrap is never imported.
```

**Изменение:**
````
Add `_wrap` to the import at tests/test_subtitles.py:6-8, then append these tests. Two GREEN today (2-line contract) and two xfail(strict=True) that pin the intended contract the source fix must satisfy.

```python
from vpipe.subtitles import _wrap   # add to existing import line

def _tokens(s):
    return [t for t in s.replace('\n',' ').split(' ') if t]

# GREEN: 2-line balance splits at the best point and both lines are produced

def test_wrap_two_line_balances():
    out=_wrap(['раз','два','три','четыре'], max_chars=8, max_lines=2)
    assert out.count('\n')==1 and _tokens(out)==['раз','два','три','четыре']

# GREEN: never break right after a short preposition/conjunction (bad tail)

def test_wrap_avoids_bad_tail():
    # 'и' must not end line 1
    out=_wrap(['я','и','ты','здесь'], max_chars=6, max_lines=2)
    first=out.split('\n')[0]
    assert not first.rstrip().endswith(' и') and not first.strip()=='я и'

# XFAIL: >2-line greedy path must NOT drop words (currently lines[:max_lines] truncates)
@pytest.mark.xfail(strict=True, reason='subtitles._wrap:117 lines[:max_lines] deletes words beyond max_lines; fix must keep every word')
def test_wrap_preserves_all_words_multiline():
    words=['один','два','три','четыре','пять','шесть','семь']
    out=_wrap(words, max_chars=8, max_lines=2)
    assert set(_tokens(out))==set(words)          # no word silently dropped

# XFAIL: 2-line split must not emit a line longer than max_chars when a wrap exists
@pytest.mark.xfail(strict=True, reason='subtitles._wrap 89-94 only penalizes overflow; a fittable split can still exceed max_chars')
def test_wrap_two_line_respects_max_chars():
    words=['аааа','бббб','вввв','гггг']           # 4x4chars, a 2/2 split fits max_chars=9
    out=_wrap(words, max_chars=9, max_lines=2)
    for line in out.split('\n'):
        assert len(line)<=9 or len(line.split(' '))==1   # overflow only if single unbreakable token
```
Note: pytest is already imported in most test files; test_subtitles.py currently does NOT import pytest — add `import pytest` at top.
````

**Риск / безопасность.** Two xfail(strict=True) encode the intended _wrap contract; they flip to green when the subtitles cluster fixes the truncation (drop lines[:max_lines] / raise a warning) and the overflow acceptance. The two GREEN tests pin today's good 2-line behavior so a refactor can't regress balancing/bad-tail. Choose word/char sizes carefully so the GREEN cases are unambiguous (verify by hand-tracing against the greedy loop before committing). Adding tests to an existing file: append only, don't reorder existing tests.

**Тест (зафиксировать поведение).** test_wrap_preserves_all_words_multiline (xfail-strict): 7 short words, max_chars=8, max_lines=2 — assert set of output tokens == set of input words; currently fails because lines[:max_lines] deletes the overflow, auto-passes once _wrap stops truncating.

---
## 30. _ass_text_escape uses escapes ASS does not define — literal backslashes double and braces render as stray characters in VSFilter
⚪ LOW · симптом 1 · effort S · `ass-text-escape-invalid-escapes`

- **Файл / якорь:** `vpipe/subtitles.py` — _ass_text_escape(), ~lines 256-260
- **Закрывает находки:** vpipe/subtitles.py:256

**Проблема.** The escaper does '\'→'\\' then '{'/'}'→'\{'/'\}'. ASS has no general backslash escape: a lone '\' renders literally, so 'C:\New' → 'C:\\New' which libass shows as 'C:\' + a hard break + 'ew'. The brace escape is a libass-only extension — the downloadable burn.ass (served via /api/output) opened in a VSFilter player shows a spurious '\' and misparses '{...}'. Only reachable via user-edited transcript text (whisper never emits these).

**Почему так.** Fullwidth braces (U+FF5B/FF5D) and a reverse-solidus lookalike (U+29F5) are common ASS practice: they display as the intended glyph in every renderer and can never start an override block or form a control pair. The \N we add for real newlines still uses a genuine backslash and is applied last, so hard breaks are preserved.

**Текущий код:**
```
    text = text.replace("\\", "\\\\")        # backslash first
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\N")
    return text
```

**Изменение:**
```
    # ASS has no general '\' escape and '\{' is libass-only. Neutralise the
    # ambiguous backslash (avoid accidental \N/\h) and use fullwidth brace
    # lookalikes so the text is safe in BOTH libass and VSFilter. Do the literal
    # substitutions BEFORE inserting our own \N for newlines.
    text = text.replace("\\", "⧵")               # ⧵ reverse-solidus glyph
    text = text.replace("{", "｛").replace("}", "｝")  # ｛ ｝
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\N")
    return text
```

**Риск / безопасность.** This flips a PINNED test: test_write_ass.py::test_ass_text_escape asserts '\{' in out, '\}' in out, and _ass_text_escape('a\\b')=='a\\\\b'. Those encode the bug and must be rewritten. Low product risk (whisper output has no braces/backslashes); the only behaviour change is for user-typed special chars. If a plain ASCII display is preferred over U+29F5, replacing '\' with '/' is an acceptable alternative (documented in the finding).

**Тест (зафиксировать поведение).** Rewrite test_write_ass.py::test_ass_text_escape: keep 'a\nb'→'a\\Nb' and 'a\r\nb'→'a\\Nb'; replace the brace/backslash asserts with: out=_ass_text_escape('x {y} z'); assert '{' not in out and '｛' in out and '｝' in out; and assert '\\\\' not in _ass_text_escape('a\\b') (no doubling).

---
## 31. Forced 50ms gap between back-to-back cues blinks the subtitle at every boundary in continuous speech
⚪ LOW · симптом 1 · effort S · `gap-blink-chain-continuous-cues`

- **Файл / якорь:** `vpipe/subtitles.py` — build_cues() no-overlap pass, ~lines 168-171
- **Закрывает находки:** vpipe/subtitles.py:170

**Проблема.** The no-overlap pass always leaves min_gap (0.05s ≈ 1-3 frames) of empty screen between consecutive cues, even in continuous speech where the pipeline already cut the pauses — so the subtitle area flashes off/on at nearly every boundary («дёрганые субтитры»). For karaoke/kinetic burned captions, chaining continuous cues end-to-start looks smoother; keep a gap only for real pauses.

**Почему так.** When cue i already ends within min_gap of cue i+1 (the continuous-speech case) and the natural gap is <0.15s, set cue.end = next.start — cues touch (no overlap, no blank frame). Because next.start >= this group's last word end, this ALSO keeps the last word fully inside the cue (reinforces P1). Real pauses (gap >= 0.15s) keep the existing min_gap clamp.

**Текущий код:**
```
    # Enforce no overlap (keep order, leave a small gap).
    for i in range(len(cues) - 1):
        if cues[i].end > cues[i + 1].start - subs.min_gap:
            cues[i].end = max(cues[i].start + 0.2, cues[i + 1].start - subs.min_gap)
    return cues
```

**Изменение:**
```
    # Enforce no overlap. For CONTINUOUS speech (next cue begins within ~0.15s)
    # chain end-to-start (no flicker); a real pause keeps the min_gap.
    for i in range(len(cues) - 1):
        nxt = cues[i + 1].start
        if cues[i].end > nxt - subs.min_gap:
            if nxt - cues[i].end < 0.15 and nxt > cues[i].start + 0.2:
                cues[i].end = nxt
            else:
                cues[i].end = max(cues[i].start + 0.2, nxt - subs.min_gap)
    return cues
```

**Риск / безопасность.** Low. test_build_cues_splits_on_gap has a big gap (cue0 ends 1.0, next starts 5.0) so the branch never fires — its 'cues[0].end <= cues[1].start' still holds (1.0<=5.0). Chaining yields cue.end == next.start (touching, not overlapping), which still satisfies <=. This is style polish, not the primary «корявые» fix (that's P1); include it low.

**Тест (зафиксировать поведение).** Add tests/test_subtitles.py::test_continuous_cues_chain_without_gap: two contiguous budget-split cues (gap 0 between last word of cue0 and first word of cue1) → cues[0].end == pytest.approx(cues[1].start). And ::test_real_pause_keeps_gap: a >=0.2s pause between them → cues[0].end <= cues[1].start - min_gap.

---

# Волна 3 — «Монтаж» + настройки рендера

## 32. Wire Montage candidate UI to the backend flat payload.candidates/selected schema and POST /api/enrich/select
🔴 CRITICAL · симптом 5 · effort L · `montage-visual-schema`

- **Файл / якорь:** `web/app.js` — enrichVisual() ~3377-3400 + satellites enrichCandTile ~3482, enrichStyleSwitch ~3556, enrichChooseCandidate ~3524, enrichSetStyle ~3650, enrichSetAsset ~3747
- **Закрывает находки:** web/app.js:3379, web/app.js:3380, web/app.js:3524, web/app.js:3650, web/app.js:3556, web/app.js:3482, web/app.js:3747, vpipe/enrich.py:416, serve.py:3795

**Проблема.** enrichVisual() reads a non-existent payload.visual.{candidates,chosen}; the backend stores candidates FLAT (ImagePayload.to_dict -> payload.candidates[] + payload.selected, keys source/style/preview/fields/asset_path). So rendered candidates are never shown (strip falls to a <=1-tile shim; schematic points show 'nothing to illustrate' despite codegfx PNGs). Tile clicks write payload.visual which the server sanitizer drops, and POST /api/enrich/select is never called, so payload.selected stays 0 and the user's choice is discarded.

**Почему так.** The backend already ships the correct schema via GET /api/enrich (item.payload.candidates[]/selected, keys source/style/preview) and the correct write endpoint POST /api/enrich/select {id,idx} (documented UI contract, re-syncs flat source server-side). This makes the UI consume exactly that: previews render, selection persists through the mutating endpoint, schematic_style is stored in a whitelisted field, and the sanitizer-dropped payload.visual path is removed.

**Текущий код:**
```
function enrichVisual(it) {
  const p = it.payload || {}
  if (p.visual && typeof p.visual === 'object') {
    const v = p.visual
    const cands = Array.isArray(v.candidates) ? v.candidates.filter((c) => c && (c.preview || c.id != null)) : []
    return {
      source: v.source || 'none', intent: v.intent || null, candidates: cands,
      chosen: Number.isInteger(v.chosen) ? Math.max(0, Math.min(v.chosen, Math.max(0, cands.length - 1))) : 0,
      schematic_style: v.schematic_style || v.style || null,
    }
  }
  let source = 'none'
  if (p.asset_kind === 'generate') source = 'diffusion'
  else if (p.asset_kind === 'user') source = 'stock'
  else if (p.asset_kind === 'emoji') source = 'icon'
  const cands = []
  if (p.asset_path && (p.asset_kind === 'generate' || p.asset_kind === 'user')) {
    cands.push({ kind: source === 'diffusion' ? 'seed' : 'asset', id: 0, preview: p.asset_path })
  }
  return { source, intent: null, candidates: cands, chosen: 0, schematic_style: null }
}
```

**Изменение:**
```
ONE coordinated fix across enrichVisual + 5 satellites. (1) REPLACE enrichVisual to read flat schema: if Array.isArray(p.candidates)&&length -> cands=p.candidates.filter(c=>c&&(c.preview||c.asset_path)); sel=clamp(p.selected,0,len-1); chosenC=cands[sel]||{}; return {source:chosenC.source||p.source||'none', intent:chosenC.intent||p.intent||null, candidates:cands, chosen:sel, schematic_style:chosenC.style||p.schematic_style||null}. else keep the flat asset_kind shim but emit candidate as {source, preview:p.asset_path} (drop kind/id) and schematic_style:p.schematic_style||null. (2) enrichCandTile ~3487: `enrichPreviewSrc(c.preview)` -> `enrichPreviewSrc(c.preview || c.asset_path)`; replace every `c.kind === 'theme'` with `c.source === 'schematic'` and every theme-name `c.id` with `c.style` (alt + tag block). (3) enrichStyleSwitch ~3556: `v.candidates[v.chosen] && v.candidates[v.chosen].id` -> `... .style`. (4) enrichChooseCandidate ~3524 becomes async and calls the endpoint instead of writing visual: optimistically set it.payload.selected=idx + renderEnrich({silent:true}); POST /api/enrich/select {id,idx}; on network error or !res.ok revert selected=prevSel + renderEnrich + toast/failToast; on ok set it.payload.selected=j.selected (and it.payload.source=j.source) then loadEnrichFromCache(). (5) enrichSetStyle ~3650: findIndex by c.source==='schematic'&&c.style===theme; if found enrichChooseCandidate; else persist WHITELISTED flat field via enrichQueueSave(id,{payload:{schematic_style:theme}}) and set it.payload.schematic_style=theme (NOT payload.visual). (6) enrichSetAsset ~3747: drop the dead `visual` key, keep flat asset_kind/asset_path, and clear candidates+selected so the user's file wins: empty -> {asset_kind:'none',asset_path:'',emoji:'',candidates:[],selected:0}; path -> {asset_kind:'user',asset_path:val,emoji:'',candidates:[],selected:0}.
```

**Риск / безопасность.** MEDIUM blast radius (whole candidate strip). Keep the flat-asset shim so plain user/emoji points still show a tile. Candidate previews depend on /api/output serving the codegfx/SD cache dir (basename mapping via enrichPreviewSrc); if not served, tiles show the broken-image placeholder, not a crash. enrichChooseCandidate becomes async but its callers are fire-and-forget onclicks. HARD DEPENDENCY: fixes persistence only; plan_render (vpipe/enrich.py:1332-1349) still reads flat asset_kind and skips schematic, so the chosen candidate won't reach the video until a companion backend patch uses ImagePayload.resolved_asset()/chosen_candidate() — ship together. No render fast-path exists (UI was fully non-functional).

**Тест (зафиксировать поведение).** tests/test_api_enrich.py: test_enrich_select_roundtrip — plan with an ENR_IMAGE item + >=2 candidates, POST /api/enrich/select {id,idx:1}, assert 200 + {selected:1,source:...} and reload has payload.selected==1. test_enrich_state_flat_schema — GET /api/enrich, assert image item's payload has list candidates + int selected and NO visual key. Frontend render half needs a new jsdom test (enrichVisual reads flat candidates) — flag no harness exists.

---
## 33. Re-running «Предложить монтаж» silently discards all user review because the merge matches by item id, but detectors mint a fresh random uuid every run
🟠 HIGH · симптом 4 · effort M · `enrich-merge-content-key`

- **Файл / якорь:** `serve.py` — _merge_enrich_user_state() (~line 3596-3611)
- **Закрывает находки:** serve.py:3596

**Проблема.** _merge_enrich_user_state re-links the old plan to the new one by `it.id` only (prev_by_id.get(it.id)). But detector dict builders (_card_from_list, _detect_cta, _detect_illustrations in enrich_llm.py) emit dicts WITHOUT an "id", so item_from_dict assigns new_item_id()=enr_<uuid> on every run (enrich.py:726). prev_by_id therefore ALWAYS misses for LLM suggestions: every user toggle (enabled=False) and every text edit is dropped on re-suggest; only source=="user" manual items survive. This is exactly the CRITICAL P2 fix the function claims to implement, and it is a no-op for detector items.

**Почему так.** Matching on stable original-word coordinates re-links a re-detected suggestion to its prior review even though its id is randomized each run, delivering the P2 'повторный запуск не теряет enabled/edited' guarantee the code only pretended to provide.

**Текущий код:**
```
    prev_by_id = {it.id: it for it in old.items}
    carried: set[str] = set()
    for it in items:
        prev = prev_by_id.get(it.id)
        if prev is None:
            continue
        carried.add(it.id)
        it.enabled = prev.enabled            # решение юзера — закон
        if prev.edited:                      # правки текста/таймингов — тоже
            it.payload = prev.payload
            it.t_start = prev.t_start
            it.t_end = prev.t_end
            it.edited = True
    items.extend(p for p in old.items
                 if p.source == "user" and p.id not in carried)
    return items
```

**Изменение:**
```
    prev_by_id = {it.id: it for it in old.items}
    # Детекторы дают КАЖДОМУ предложению свежий random id (item_from_dict ->
    # new_item_id) на каждом прогоне, поэтому матч только по id НИКОГДА не
    # переносит ревью LLM-предложений между запусками. Стабильный ключ — по
    # ОРИГИНАЛЬНЫМ word-индексам: они не смещаются при смене катлиста (ровно
    # тогда юзер и перезапускает анализ) -> (type, word_start, word_end). Ручные
    # (source:"user") в ключ НЕ кладём — они переезжают явным extend ниже.
    prev_by_key: dict[tuple, EnrichItem] = {}
    for it in old.items:
        if it.source == "user":
            continue
        prev_by_key.setdefault((it.type, it.word_start, it.word_end), it)
    carried_ids: set[str] = set()            # id СТАРЫХ items, уже перенесённых
    for it in items:
        prev = prev_by_id.get(it.id) or \
            prev_by_key.get((it.type, it.word_start, it.word_end))
        if prev is None:
            continue
        carried_ids.add(prev.id)
        it.enabled = prev.enabled            # решение юзера — закон
        if prev.edited:                      # правки текста/таймингов — тоже
            it.payload = prev.payload
            it.t_start = prev.t_start
            it.t_end = prev.t_end
            it.edited = True
    items.extend(p for p in old.items
                 if p.source == "user" and p.id not in carried_ids)
    return items

(EnrichItem is already imported via `from vpipe import enrich as enrich_mod`; if a bare type hint is undesired, annotate the dict as `dict[tuple, object]`.)
```

**Риск / безопасность.** The extend filter changes from `p.id not in carried` (new ids) to `p.id not in carried_ids` (old ids consumed). Because source=="user" items are excluded from prev_by_key, a user item's id can enter carried_ids only via an impossible random-id collision, so user items still always carry — behavior preserved. LLM old items that get content-matched land in carried_ids but are ignored by the source=="user" extend clause. Only enabled + (if edited) payload/timings carry; fresh score/quote/reason stay new (unchanged intent). Edge: two old LLM items sharing (type,word_start,word_end) — setdefault keeps the first (rare; detectors dedup). Original word indices are stable across cutlist edits (EnrichItem.word_start/word_end are ORIGINAL indices, enrich.py:671-672), so the key survives the very re-run that triggers the bug.

**Тест (зафиксировать поведение).** The existing tests/test_api_enrich.py::test_suggest_rerun_matched_id_carries_enabled_and_edits MASKS this bug — it monkeypatches _run_enrich_detectors to return items with EXPLICIT stable ids (enr_img001), which never happens in prod. Add test_suggest_rerun_content_key_carries_when_ids_change: (1) _write_plan with an image item id='old1', word_start=10, word_end=12, enabled=False, edited=True with a hand-tuned payload.concept; (2) monkeypatch serve._run_enrich_detectors to return a FRESH item with a DIFFERENT id ('fresh_'+uuid) but SAME type + word_start=10 + word_end=12 and score=90; (3) POST /api/enrich/suggest, _wait_done; (4) assert the saved item has enabled==False and payload.concept==(the edited value) and edited==True and score==90 (fresh detector field). Add a negative: a detector item at word_start=50 does NOT inherit old1's disable. IMPORTANT: construct the items with explicit word_start/word_end (the _img helper defaults them to 0/0, which would match trivially).

---
## 34. The 4,984-line frontend has zero tests; the UI->server render-opts contract exists twice with no shared schema, so a renamed key silently stops reaching ffmpeg
🟠 HIGH · симптом 5 · effort M · `render-opts-ui-contract`

- **Файл / якорь:** `tests/test_render_opts_contract.py` — web/app.js collectRenderOpts (app.js:2786-2831) <-> serve._resolve_render_opts (serve.py:892-1111)
- **Закрывает находки:** web/app.js:2786, serve.py:892, serve.py:1067

**Проблема.** collectRenderOpts assembles the render payload from ~25 DOM fields (app.js:2786-2831); _resolve_render_opts whitelists keys server-side and silently ignores junk (serve.py:892-1111). No package.json, no JS test runner, no contract test. A typo'd/renamed key (e.g. loudnorm_mode values, denoise_engine, music.gain_db) is dropped with all 773 backend tests green — 'render settings have bugs' (symptom 3/5) with nothing red.

**Текущий код:**
```
# tests/test_queue.py:180-197 — the closest thing: it feeds a HAND-WRITTEN opts dict, not a
# recorded frontend payload, and asserts a few keys apply. Nothing ties app.js field names to the server.
def test_resolve_render_opts_applies_overrides(tmp_path):
    opts = {'encoder':'x264','quality':20,'audio_bitrate':'256k','censor_method':'pitch', ...}
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(s, opts)
    assert cfg.render.encoder == 'x264'; ...
```

**Изменение:**
````
Two artifacts. (1) A recorded fixture tests/fixtures/render_opts_payload.json capturing a full collectRenderOpts() output (produce it once by pasting the object collectRenderOpts builds, or by console.log JSON.stringify(collectRenderOpts()) in the browser). Include EVERY top-level key the function can emit: encoder, quality, scale_h, fps, audio_bitrate, censor_method, subtitles, chapters, formats, denoise, denoise_deess, denoise_loudnorm, loudnorm_mode, cut_fade, music{enabled,path,gain_db,duck_db}, out_dir, filename, enrich{enabled}, denoise_strength, denoise_normalize, denoise_engine, burn_subtitles, burn_style. (2) tests/test_render_opts_contract.py:

```python
import json, re
from pathlib import Path
from types import SimpleNamespace
import serve
from vpipe.config import load_config

FIX=Path(__file__).parent/'fixtures'/'render_opts_payload.json'
# Every top-level key _resolve_render_opts / _parse_formats actually consumes.
ACCEPTED={'encoder','quality','scale_h','fps','audio_bitrate','censor_method','subtitles',
          'chapters','metadata','formats','vertical','vertical_target','vertical_center',
          'denoise','denoise_strength','denoise_highpass','denoise_normalize','denoise_engine',
          'denoise_deess','denoise_loudnorm','loudnorm_mode','cut_fade','music','enrich',
          'out_dir','filename','burn_subtitles','burn_style'}

def _sess(tmp_path):
    cfg=load_config('config.yaml')
    return SimpleNamespace(cfg=cfg, media=SimpleNamespace(height=1080,fps=30.0),
                           out_dir=tmp_path/'out', inp=SimpleNamespace(stem='clip'))

def test_every_ui_key_is_a_known_server_key():
    payload=json.loads(FIX.read_text(encoding='utf-8'))
    unknown=set(payload)-ACCEPTED
    assert not unknown, f'app.js emits keys the server ignores: {unknown}'

def test_recorded_payload_applies_without_error(tmp_path):
    payload=json.loads(FIX.read_text(encoding='utf-8'))
    payload['music']={'enabled':False}   # avoid the 400 path needing a real file
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(_sess(tmp_path), payload)
    # spot-check that representative options actually took effect (not silently dropped)
    if payload.get('encoder') in ('nvenc','x264'): assert cfg.render.encoder==payload['encoder']
    if payload.get('loudnorm_mode') in ('dynamic','2pass'): assert cfg.render.denoise.loudnorm_mode==payload['loudnorm_mode']
    if payload.get('denoise'): assert cfg.render.denoise.enabled is True

def test_app_js_field_ids_have_no_orphans():
    # lightweight static guard: the ids collectRenderOpts reads should exist in index.html/app render modal
    js=Path('web/app.js').read_text(encoding='utf-8')
    body=js[js.index('function collectRenderOpts'):js.index('function submitRender')]
    ids=set(re.findall(r"\$\('#(\w+)'\)", body))
    html=Path('web/index.html').read_text(encoding='utf-8')
    missing=[i for i in ids if f'id="{i}"' not in html and f"id='{i}'" not in html]
    assert not missing, f'collectRenderOpts reads DOM ids absent from index.html: {missing}'
```
Ideal follow-up (not required for MVP, note in file docstring): add package.json + a jsdom/vitest unit test that mounts the modal and asserts collectRenderOpts() output equals the fixture, plus a Playwright smoke intercepting the /api/render body.
````

**Риск / безопасность.** Medium: the static app.js/index.html scan (test 3) is the brittle part — confirm the actual template filename (web/index.html) and the $('#id') access pattern before committing; if ids are built dynamically, drop test 3 and keep tests 1-2 (the real contract lock). The fixture must be regenerated whenever collectRenderOpts legitimately gains a field — that regeneration IS the review signal. ACCEPTED set must be kept in sync with _resolve_render_opts; put a comment pointing at serve.py:892 so a reviewer updates both.

**Тест (зафиксировать поведение).** test_every_ui_key_is_a_known_server_key: load the recorded collectRenderOpts payload fixture and assert set(payload_keys) - ACCEPTED == empty — renaming a key in app.js (or the server whitelist) without updating the other side turns this red instead of silently dropping the option before ffmpeg.

---
## 35. Restore Главы/Мета on reload; intercept rich-HTML paste; guard hand-edited metadata against silent loss
🟡 MEDIUM · симптом 5 · effort M · `chapters-metadata-restore-client`

- **Файл / якорь:** `web/app.js` — init() bootstrap (~app.js:248-249, beside loadClipsFromCache/loadEnrichFromCache); meta field wiring (~app.js:4217-4219); beforeunload guard (app.js:179)
- **Закрывает находки:** web/app.js:984, web/app.js:949
- **Зависит от:** `chapters-metadata-restore-server`

**Проблема.** renderChapters()/renderMetadata() run only from live task results in followTask, so after F5 both tabs are empty and the user must re-run minutes of LLM generation. The Мета contenteditable fields invite manual polishing but no code ever persists them and beforeunload (app.js:179) only guards the cutlist — a reload silently destroys hand-edited YouTube copy. The fields also accept rich-HTML paste (no plaintext handler), garbling the field.

**Почему так.** Closes the reload-loss and paste-garble defects with a frontend-only change riding on data the server already persists, and converts a silent data-loss into an explicit leave warning.

**Текущий код:**
```
  loadClipsFromCache()   // F4: панель клипов из out/<stem>.clips.json — без LLM
  loadEnrichFromCache()  // P4: вкладка «Монтаж» из out/<stem>.enrich.json — без LLM
  if (s.task && s.task.running) followTask(s.task.name)

... (bindUI, ~4217-4219)
  for (const el of document.querySelectorAll('.metaCopyBtn')) { el.onclick = () => copyField(el.dataset.field) }
  $('#btnMetaCopyAll').onclick = copyAllMeta
  $('#metaTitle').addEventListener('input', () => { const l = $('#metaTitleLen'); const t = ($('#metaTitle').textContent || '').trim(); if (l) l.textContent = t ? `(${t.length}/100)` : '' })

... (beforeunload, app.js:179)
window.addEventListener('beforeunload', (e) => {
  if (st.dirty || st.saving) { e.preventDefault(); e.returnValue = ''; return '' }
})
```

**Изменение:**
```
1) Bootstrap restore in init() (only when transcript exists, so empty sessions stay empty):
  loadClipsFromCache()
  loadEnrichFromCache()
  if (s.has_transcript) { loadChaptersFromCache(); loadMetadataFromCache() }
  if (s.task && s.task.running) followTask(s.task.name)

Add the two cache loaders (mirror loadClipsFromCache at app.js:1139):
  async function loadChaptersFromCache() {
    try { const r = await fetch('/api/chapters'); if (!r.ok) return; const j = await r.json()
      if (j && Array.isArray(j.chapters) && j.chapters.length) renderChapters(j.chapters) } catch {}
  }
  async function loadMetadataFromCache() {
    try { const r = await fetch('/api/metadata'); if (!r.ok) return; const j = await r.json()
      if (j && j.metadata) renderMetadata(j.metadata) } catch {}
  }

2) Paste-to-plaintext + dirty flag on the four meta fields (in bindUI, after line 4219):
  for (const sel of ['#metaTitle', '#metaDesc', '#metaTags', '#metaHook']) {
    const el = $(sel); if (!el) continue
    el.addEventListener('paste', (e) => {
      e.preventDefault()
      const txt = (e.clipboardData || window.clipboardData).getData('text/plain')
      document.execCommand('insertText', false, txt)
    })
    el.addEventListener('input', () => { st.metaDirty = true })
  }

3) Include hand-edited meta in the leave guard (app.js:179):
window.addEventListener('beforeunload', (e) => {
  if (st.dirty || st.saving || st.metaDirty) { e.preventDefault(); e.returnValue = ''; return '' }
})

(Clear st.metaDirty=false inside renderMetadata() after `set(...)` calls, so a fresh generation isn't treated as unsaved.)
```

**Риск / безопасность.** renderChapters/renderMetadata are idempotent DOM fills and already run in followTask, so calling them from init is safe. Restore is gated on s.has_transcript to keep empty sessions clean. Paste interception uses execCommand('insertText') (deprecated but universally supported and the only synchronous plaintext-into-contenteditable primitive; keep it). This patch stops SILENT loss (adds the leave warning) but does not yet SERVER-persist manual edits — full persistence needs a PUT + debounce-save endpoint and is a larger follow-up; call it out so reviewers don't expect autosave.

**Тест (зафиксировать поведение).** Needs jsdom+vitest harness (web/tests/): (a) mock GET /api/chapters/metadata → assert renderChapters/renderMetadata called and tabs populated; (b) dispatch a paste event carrying HTML → assert only text/plain lands in the field; (c) input on #metaDesc sets st.metaDirty and beforeunload returns a truthy value. Server side is locked by chapters-metadata-restore-server's test.

---
## 36. Add GET /api/chapters and GET /api/metadata that read the already-persisted generated files
🟡 MEDIUM · симптом 5 · effort S · `chapters-metadata-restore-server`

- **Файл / якорь:** `serve.py` — next to @app.get("/api/clips") (serve.py:3246); reads existing work_dir/preview_chapters.txt (3102) and preview_metadata.json (3140)
- **Закрывает находки:** web/app.js:984 (server half of the reload-loss finding)

**Проблема.** The route table has only POST /api/preview/chapters and POST /api/preview/metadata (task starters). The generated data is already written to disk (work_dir/preview_chapters.txt at serve.py:3102, work_dir/preview_metadata.json at serve.py:3140) but there is no read endpoint, so the Главы/Мета tabs cannot be restored after reload the way clips are (GET /api/clips exists at 3246).

**Почему так.** Supplies the missing read side of a POST-only contract using data the server ALREADY persists, unblocking the client restore with no new write path.

**Текущий код:**
```
@app.get("/api/clips")
def get_clips():
    ...   # (existing cache-read endpoint for the Клипы tab — the pattern to mirror)
```

**Изменение:**
```
Add two GET endpoints mirroring /api/clips (read-only, no _guard_no_task, return empty payload when the file is absent):

@app.get("/api/chapters")
def get_chapters():
    s = S()
    p = s.work_dir / "preview_chapters.txt"
    if not p.exists():
        return {"chapters": []}
    try:
        return {"chapters": _parse_chapters_txt(str(p))}
    except Exception:
        return {"chapters": []}

@app.get("/api/metadata")
def get_metadata():
    s = S()
    p = s.work_dir / "preview_metadata.json"
    if not p.exists():
        return {"metadata": None}
    try:
        return {"metadata": json.loads(p.read_text(encoding="utf-8"))}
    except Exception:
        return {"metadata": None}

(_parse_chapters_txt already exists — it is used at serve.py:3117.)
```

**Риск / безопасность.** Purely additive, read-only. The restored data can be STALE if the user edited cuts after generating (chapters/metadata files carry no cutlist hash, unlike clips.json). Acceptable: restoring the last-generated copy is strictly better than a blank tab, and matches the existing 'clips are cached' UX. Optional hardening (note, not required for this patch): stamp the cutlist/transcript hash into the files and return a `stale` flag so the client can badge it. Coordinate route names with the R3/serve.py owner so they aren't duplicated.

**Тест (зафиксировать поведение).** tests/test_chapters.py (or new tests/test_preview_restore.py): with a SimpleNamespace session whose work_dir contains a preview_chapters.txt and preview_metadata.json fixture, TestClient GET /api/chapters returns the parsed list and GET /api/metadata returns the dict; with the files absent, both return the empty shape (200, not 404).

---
## 37. Queue jobs must carry the full render settings, not a 7-key subset
🟡 MEDIUM · симптом 3 · effort S · `queue-render-opts`

- **Файл / якорь:** `web/app.js` — currentRenderOpts() ~4531-4541
- **Закрывает находки:** web/app.js:4531, serve.py:1818
- **Зависит от:** `subs-chapters-defaults`, `quality-per-encoder`

**Проблема.** currentRenderOpts() sends only encoder/quality/audio_bitrate/censor_method/subtitles/chapters/out_dir. Because _resolve_render_opts uses 'absent key = force OFF' for burn/vertical/denoise/deess/loudnorm/music/enrich (and unconditionally overwrites deess/loudnorm), a file queued via the queue renders with all of those OFF even when enabled in the modal or config.yaml, contradicting the docstring 'a queued render behaves exactly like the manual one'.

**Почему так.** collectRenderOpts already assembles the complete, server-validated opts dict; forwarding it verbatim minus filename is the smallest change achieving queue/manual parity, leaving _resolve_render_opts semantics untouched so clips/autopack (which send music=null etc.) are unaffected. seedRenderModal is the proven seeding path already used by autopack.

**Текущий код:**
```
function currentRenderOpts() {
  const d = st.rdefaults || {}
  return {
    encoder: d.encoder || 'nvenc',
    quality: (d.quality != null) ? d.quality : null,
    audio_bitrate: d.audio_bitrate || '320k',
    censor_method: d.censor_method || 'partial',
    subtitles: true, chapters: true,
    out_dir: st.outDir || '',
  }
}
```

**Изменение:**
```
Reuse collectRenderOpts() so a queued render equals a manual one; seed modal fields first if never opened this session (openAutopackModal pattern); drop only per-file filename (server defaults each job's stem to its own input basename, _resolve_render_opts:1110):
function currentRenderOpts() {
  if (!st._rSeeded) seedRenderModal()
  const o = collectRenderOpts()
  delete o.filename
  return o
}
```

**Риск / безопасность.** LOW. seedRenderModal is idempotent and already called this way by openAutopackModal. Note: (a) enrich for OTHER queued files sends enrich:{enabled:true} from the current session's suggestions — server enrich stage no-ops without an enrich.json for that file. (b) COMPANION SERVER FIX: watch folder at serve.py:~1818 uses render_opts={} (drops everything) — fix separately by persisting last-used opts server-side or key-presence gating for the empty case; do NOT blanket-flip _resolve_render_opts to key-presence gating (would leak config vertical/music into clips/autopack).

**Тест (зафиксировать поведение).** tests/test_queue.py: test_queue_job_applies_full_render_opts — POST /api/queue/add with render_opts incl denoise:true+denoise_loudnorm:true, assert stored QueueJob.render_opts round-trips them and _resolve_render_opts yields cfg.render.denoise.enabled and .loudnorm True (vs subset -> False). Client half needs a jsdom test (currentRenderOpts==collectRenderOpts minus filename).

---
## 38. Render modal must stop re-seeding every open so user choices survive a reopen
🟡 MEDIUM · симптом 3 · effort M · `render-modal-persist`

- **Файл / якорь:** `web/app.js` — openRenderModal() ~2567-2571
- **Закрывает находки:** web/app.js:2567, web/app.js:2557
- **Зависит от:** `subs-chapters-defaults`, `quality-per-encoder`

**Проблема.** openRenderModal unconditionally calls seedRenderModal on every open, overwriting all controls from static config defaults; the enrich box even force-re-checks itself (rEnrich.checked=n>0). Switching to x264+denoise, closing, and reopening reverts everything to nvenc/off.

**Почему так.** st._rSeeded is the existing session guard; gating the reseed on it makes within-session edits sticky while keeping per-file fields (filename/out_dir) and enrich availability fresh. The reset button restores discoverability. updateEnrichRenderCheckbox (app.js:3880) already disables+unchecks when no suggestions apply, so availability stays correct without force-checking.

**Текущий код:**
```
function openRenderModal() {
  if (!st.hasSession) return
  seedRenderModal()
  openOverlay('#renderModal')
}
```

**Изменение:**
```
Seed once per session (st._rSeeded already set at end of seedRenderModal:2564); on later opens refresh only per-file/dynamic bits + add a reset button:
function openRenderModal() {
  if (!st.hasSession) return
  if (!st._rSeeded) seedRenderModal()
  else {
    $('#rFilename').value = ($('#filename').textContent || 'output').replace(/\.[^.]+$/, '')
    $('#rOutDir').value = st.outDir || ''
    updateEnrichRenderCheckbox()
  }
  openOverlay('#renderModal')
}
Add #btnRenderReset in index.html, wired in bindUI mirroring btnDetectReset (app.js:4176): $('#btnRenderReset').onclick = () => { seedRenderModal(); flash('render settings reset to defaults') }.
```

**Риск / безопасность.** LOW-MEDIUM. Behavior change: enrich no longer auto-checks when suggestions appear AFTER the first open this session (mitigated by updateEnrichRenderCheckbox keeping it enabled+labelled). No reload persistence (config returns on reload) — if wanted, layer a localStorage snapshot keyed by st._inpPath (app.js:207) inside seedRenderModal like RENDER_FMT_KEY/CAP_PRESET_KEY; spec as follow-up to keep this low-risk. Needs a small index.html addition.

**Тест (зафиксировать поведение).** jsdom (tests/js/renderModal): open twice, mutate #rEncoder between opens, assert it persists on second open and resets after #btnRenderReset. No JS harness yet — manual repro: modal -> x264 -> close -> reopen -> still x264.

---
## 39. config.yaml subtitles/chapters enabled must seed the modal checkboxes instead of hardcoded true
🟡 MEDIUM · симптом 3 · effort S · `subs-chapters-defaults`

- **Файл / якорь:** `web/app.js` — seedRenderModal() line 2499 + serve.py /api/state defaults dict ~1959
- **Закрывает находки:** web/app.js:2499, serve.py:1959, serve.py:911

**Проблема.** seedRenderModal hardcodes rSubs.checked=true; rChapters.checked=true, and /api/state defaults never exposes cfg.subtitles.enabled/cfg.chapters.enabled, so the UI cannot reflect them. collectRenderOpts then always sends explicit booleans that _resolve_render_opts applies over session config — a user who set chapters.enabled:false in config.yaml still gets the slow LLM chapters stage every render, and the box re-checks itself on every open.

**Почему так.** Mirrors how denoise/deess/loudnorm/music are already seeded from st.rdefaults (app.js:2508/2521/2522/2538). Exposing the two flags is the missing half; the !== false guard preserves back-compat across client/serve skew.

**Текущий код:**
```
web/app.js:2499:  $('#rSubs').checked = true; $('#rChapters').checked = true

serve.py defaults dict ~1956-1959:
            "encoder": s.cfg.render.encoder,
            "quality": s.cfg.render.nvenc.qp if s.cfg.render.encoder == "nvenc" else s.cfg.render.x264.crf,
            "audio_bitrate": s.cfg.render.audio_bitrate,
            "censor_method": s.cfg.censor.method,
```

**Изменение:**
```
(A) serve.py defaults dict — add two keys after censor_method: "subtitles": s.cfg.subtitles.enabled, and "chapters": s.cfg.chapters.enabled. (B) web/app.js:2499 — seed from defaults (only explicit false unchecks, so an absent key keeps historical on): $('#rSubs').checked = d.subtitles !== false; $('#rChapters').checked = d.chapters !== false
```

**Риск / безопасность.** LOW. !== false keeps subs/chapters ON when the key is missing (old server) — no default-config behavior change. Only an explicit enabled:false unchecks the box. No render fast-path affected.

**Тест (зафиксировать поведение).** tests/test_serve_render.py (or test_health.py): test_state_defaults_expose_subs_chapters — set cfg.chapters.enabled=False, GET /api/state, assert body['defaults']['chapters'] is False and ['subtitles'] reflects cfg. Checkbox-seeding jsdom test desirable but no harness.

---
## 40. CTA detector sends the ENTIRE marked transcript in one prompt — exceeds num_ctx=16384 on ~45+ min videos, silently truncated by Ollama
🟡 MEDIUM · симптом 2 · effort ? · `cta-windowed-prompt`

- **Файл / якорь:** `vpipe/enrich_llm.py` — _detect_cta() single-prompt build/call (~lines 838-860) + CTA constants (~line 66)
- **Закрывает находки:** vpipe/enrich_llm.py:846, vpipe/enrich_llm.py:844
- **Зависит от:** `keepalive-warm-then-explicit-unload`

**Проблема.** Unlike lists (400-word windows) and illustrations (600-word windows), _detect_cta formats ALL n effective words into one user message; the >45-min branch only thins markers (25→50) and never bounds the word count. At ~45-60+ min of Russian speech the prompt (~2-3 tokens/word) exceeds llm.num_ctx=16384 (config.yaml:205) and Ollama truncates the context head server-side with no client warning — the model sees only part of the video, so CTA suggestions degrade and cluster in the surviving portion. (Word indices stay correct because _marked_text embeds ABSOLUTE indices — truncation blinds, it does not misnumber.)

**Почему так.** Windowing is the same proven pattern as lists/illustrations; because markers carry absolute filtered indices, per-window word_idx values feed the unchanged global dedup(120s)/density(2/10min)/fallback logic without renumbering. The n<=CTA_WINDOW fast-path leaves the common short/medium video on exactly one call (zero behavior change, all existing tests use ~300-word transcripts). Depends on keepalive-unload so calling ka_next once per window no longer misfires the old n_calls counter.

**Текущий код:**
```
    every = CTA_MARK_EVERY_LONG if eff_dur > CTA_LONG_S else CTA_MARK_EVERY
    log("Монтаж: CTA…")
    user = _CTA_USER_TMPL.format(text=_marked_text(eff, 0, n, every))
    # CTA — единственный детектор, которому нужно РАЗНООБРАЗИЕ, а не точный
    # снап: ... (comment kept)
    with _cta_temperature(llm, CTA_TEMPERATURE):
        try:
            data = llm.chat_json(_CTA_SYSTEM, user, _CTA_SCHEMA,
                                 keep_alive=ka_next())
        except Exception as e:  # noqa: BLE001 — сбойный детектор не валит пасс
            log(f"  enrich: CTA-детектор пропущен ({e})")
            return []
    raw = data.get("ctas") if isinstance(data, dict) else None

    cand: list[dict] = []
    for r in (raw if isinstance(raw, list) else []):
        ... [parse body appends to cand] ...
```

**Изменение:**
```
Add a constant next to the CTA marker consts (~line 66):
    CTA_WINDOW = 3200          # слов на ОДИН CTA-вызов: ~≤0.8*num_ctx с маркерами
                              # (>~21 мин ролик режется на окна; короче — 1 окно)

Rewrite the single-call section to window it (fast-path = one window for n<=CTA_WINDOW, byte-identical to today):
    every = CTA_MARK_EVERY_LONG if eff_dur > CTA_LONG_S else CTA_MARK_EVERY
    log("Монтаж: CTA…")
    # Окна, чтобы промпт не превысил num_ctx на длинных роликах: маркеры несут
    # АБСОЛЮТНЫЙ filtered-индекс, поэтому word_idx из окон совместимы с общими
    # гардами/дедупом ниже. Короткий/средний ролик = РОВНО одно окно [0:n]
    # (поведение без изменений). Модель держим тёплой (ka_next → 300; финальный
    # unload — в detect_all).
    wins = (segment_windows(n, _win_cfg(CTA_WINDOW, 0))
            if n > CTA_WINDOW else [(0, n)])
    cand: list[dict] = []
    for lo, hi in wins:
        user = _CTA_USER_TMPL.format(text=_marked_text(eff, lo, hi, every))
        with _cta_temperature(llm, CTA_TEMPERATURE):
            try:
                data = llm.chat_json(_CTA_SYSTEM, user, _CTA_SCHEMA,
                                     keep_alive=ka_next())
            except Exception as e:  # noqa: BLE001 — сбойное окно не валит пасс
                log(f"  enrich: CTA-окно [{lo}:{hi}] пропущено ({e})")
                continue
        raw = data.get("ctas") if isinstance(data, dict) else None
        for r in (raw if isinstance(raw, list) else []):
            ... [MOVE the existing parse body here, still appending to cand] ...

The dedup/density/fallback tail (cand.sort → kept → _fallback_cta → return) stays OUTSIDE the loop, unchanged: it already merges by t_eff so cross-window duplicates are handled. Keep the CTA-temperature comment on the loop body.
```

**Риск / безопасность.** MODERATE. Only videos >~21 min now issue >1 CTA call (benign: dedup/density collapse extras; temperature bump applies per window). Preserve the exact parse body and the outside-the-loop dedup/fallback so single-window output is identical. CTA_WINDOW is a tunable: at ~3 tokens/Cyrillic-word + markers, 3200 words ≈ 10k tokens < 0.8*16384; raise cautiously if you observe boundary CTAs being missed (add a small overlap to _win_cfg). Do NOT window without the keepalive-unload patch (n_calls would zero intermediate windows and unload mid-pass).

**Тест (зафиксировать поведение).** tests/test_enrich_llm.py: add test_cta_windows_long_transcript — _build_tr(CTA_WINDOW*2) words, params with ONLY cta enabled; provide two {'ctas':[...]} responses; assert the number of chat_json calls whose schema is _CTA_SCHEMA == len(segment_windows(n, _win_cfg(CTA_WINDOW,0))) (==2) and that a CTA reported in the SECOND window (absolute word_idx in [CTA_WINDOW, 2*CTA_WINDOW)) survives into the returned items. Keep an assertion that _build_tr(300) still yields exactly ONE CTA call (fast-path).

---
## 41. keep_alive accounting excludes schematic-extractor (and would exclude CTA-window) calls: model unloads mid-pass then cold-reloads, and stays resident 300s after the task
🟡 MEDIUM · симптом none · effort ? · `keepalive-warm-then-explicit-unload`

- **Файл / якорь:** `vpipe/enrich_llm.py` — detect_all() n_calls/ka_next block (~lines 1667-1693) and end-of-pass (~line 1740)
- **Закрывает находки:** vpipe/enrich_llm.py:1681, vpipe/enrich_llm.py:1277, vpipe/enrich_llm.py:1740

**Проблема.** n_calls pre-counts only lists+cta+ill windows (+folder match) to zero the LAST call, but _build_candidates then fires one uncounted schematic-extractor LLM call per schematic point with hardcoded keep_alive=KEEP_ALIVE_BETWEEN. Consequences on 8GB: (a) with no user folder (common), ka_next zeros the last illustrations window → Ollama unloads qwen3 → the next extractor cold-reloads it (tens of s, VRAM churn); (b) the last extractor leaves the model resident 300s after the pass — and when SD is skipped, serve._wait_ollama_unloaded never runs, so a render started within 5 min competes with a ~5-6GB resident LLM. The pre-count is fundamentally unable to count extractor calls (they depend on the illustration detector's output), so the mechanism can't be patched by counting — it must be replaced.

**Почему так.** Keeping the model warm across the whole pass and unloading exactly once at the end is strictly better than the pre-count: no mid-pass unload/reload churn (fixes 2a), guaranteed VRAM release regardless of how many extractor/CTA-window calls fired (fixes 2b), and it makes CTA windowing safe (ka_next can be called any number of times). match_user_assets keeps its hardcoded keep_alive=0 — harmless and still the genuine last call in the folder case; the end unload is idempotent.

**Текущий код:**
```
    _folder_p = Path(str(user_folder)).expanduser() if user_folder else None
    will_match_folder = (
        run_assets and image_source in ("auto", "user_folder")
        and _folder_p is not None and _folder_p.is_dir()
        and bool(_index_assets(_folder_p)))

    # План вызовов известен заранее -> знаем ПОСЛЕДНИЙ (keep_alive=0 на нём,
    # VRAM под Whisper/рендер — §3.4).
    n_calls = ((len(segment_windows(len(eff.words),
                                    _win_cfg(LISTS_WINDOW, LISTS_OVERLAP)))
                if run_lists else 0)
               + (1 if run_cta else 0)
               + (len(segment_windows(len(eff.words), _win_cfg(ILL_WINDOW, 0)))
                  if run_ill else 0)
               + (1 if will_match_folder else 0))
    made = 0

    def ka_next() -> int:
        nonlocal made
        made += 1
        return 0 if made >= n_calls else KEEP_ALIVE_BETWEEN

... (later, at the tail of detect_all) ...
    prog(1.0)

    items: list[EnrichItem] = []
```

**Изменение:**
```
Replace the whole _folder_p/will_match_folder/n_calls/made/ka_next block with a warm-through ka_next:

    # VRAM (§3.4): модель держим ТЁПЛОЙ ВЕСЬ пасс (детекторы → CTA-окна →
    # schematic-экстракторы _build_candidates → матч папки). Их точное число
    # заранее НЕизвестно (экстракторы зависят от вывода детектора иллюстраций),
    # поэтому НЕ угадываем «последний вызов» (старый n_calls промахивался:
    # экстракторы не считались → модель выгружалась посреди пасса и грузилась
    # заново). Вместо этого — ОДИН явный llm.unload() в конце пасса (ниже):
    # робастно и работает даже когда SD-этап пропущен.
    def ka_next() -> int:
        return KEEP_ALIVE_BETWEEN

Then at the tail, replace `    prog(1.0)\n\n    items: list[EnrichItem] = []` with:

    prog(1.0)

    # VRAM освобождаем ЯВНО в конце пасса (см. ka_next): единственная точка
    # выгрузки qwen3 — работает и когда SD-этап пропущен. best-effort: у боевого
    # OllamaClient unload() никогда не бросает; тестовые моки без unload()
    # просто пропускаем (getattr-гард).
    _unload = getattr(llm, "unload", None)
    if callable(_unload):
        try:
            _unload()
        except Exception as e:  # noqa: BLE001 — выгрузка best-effort
            log(f"enrich: не удалось выгрузить модель в конце пасса ({e})")

    items: list[EnrichItem] = []

(_index_assets stays imported — still used by match_user_assets; segment_windows still used by the detectors.)
```

**Риск / безопасность.** MODERATE — updates two tests that currently LOCK the buggy 'zero on last' behavior. test_enrich_llm.py:822 must become [300,300,300]; test_enrich_llm.py:829 must become [300] and [300,300]. test_enrich_assets.py:386 is UNAFFECTED (match still keep_alive=0; the other calls were already 300). The getattr guard means the many MockLLMs lacking unload() don't raise. test_api_enrich RecordingLLM tests monkeypatch _run_enrich_detectors, so detect_all never runs there — llm.events==[] assertions (lines 425,446) stay valid.

**Тест (зафиксировать поведение).** Update tests/test_enrich_llm.py:822 → assert [c['keep_alive'] for c in llm.calls]==[300,300,300]; update :829 → [300] and [300,300]. ADD test_pass_unloads_model_once_at_end: a MockLLM subclass recording .unload() calls; run detect_all; assert exactly one unload() after the last chat_json and that NO chat_json used keep_alive=0 (warm-through). Optionally assert the schematic-extractor path (test_enrich_assets scenario) no longer sees a keep_alive=0 detector call preceding the extractor.

---
## 42. VRAM guard is comment-only: sd-cli is never invoked with --max-vram though llm.unload/_wait_ollama_unloaded docstrings promise it as the OOM fallback
🟡 MEDIUM · симптом none · effort ? · `sd-max-vram-guard`

- **Файл / якорь:** `vpipe/imagegen.py` — generate_image() sd-cli command assembly (~lines 193-209) + vpipe/config.py ImagegenCfg (~line 303)
- **Закрывает находки:** vpipe/imagegen.py:193, vpipe/config.py:303, serve.py:3411

**Проблема.** llm.unload() (vpipe/llm.py:89) and _wait_ollama_unloaded (serve.py:3411,3420-3421) both tell the user «sd-cli с --max-vram подстрахует», but generate_image builds the command with NO --max-vram (nor --offload-to-cpu). Verified via tools/sd-cli.exe --help that the vendored build DOES support `--max-vram <float>` (0=disable graph split, negative=auto-detect free VRAM). When Ollama fails to free VRAM within the 10s poll, SDXL-Turbo on the 8GB RTX 3080 competes with the still-resident ~5-6GB qwen3:8b, sd-cli OOMs (non-zero exit), and every diffusion candidate silently degrades to emoji/none while the log falsely claims a guard is active.

**Почему так.** The safeguard the code already advertises simply needs wiring. Auto-detect (-1.0) lets sd-cli use available free VRAM and graph-split under pressure, converting a hard OOM (all photos lost) into a slower-but-succeeding generation. Command flags are NOT part of _cache_key, so no cache invalidation. cfg=0 is an escape hatch to restore unrestricted behavior.

**Текущий код:**
```
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

---- and in vpipe/config.py ImagegenCfg ----
    imagegen_vae: str = ""                   # путь к sdxl_vae_fp16fix.safetensors (--vae); ""=без
```

**Изменение:**
```
In vpipe/config.py ImagegenCfg, add a field (default makes the promised guard real):
    imagegen_max_vram: float = -1.0          # GiB бюджет VRAM для sd-cli graph-split; <0=авто-детект свободной, 0=выкл

In vpipe/imagegen.py generate_image, append after the --vae-on-cpu block:
    # VRAM-гард (§2): sd-cli режет граф под доступную VRAM, когда qwen3 не
    # успела выгрузиться (8 ГБ карта). <0 = авто-детект свободной; 0 = выкл.
    try:
        max_vram = float(getattr(cfg, "imagegen_max_vram", -1.0))
    except (TypeError, ValueError):
        max_vram = -1.0
    if max_vram != 0.0:
        cmd += ["--max-vram", str(max_vram)]

(Optional truth-in-logging: no change needed to serve.py:3411/3420-3421 docstrings — they become accurate once the flag is wired.)
```

**Риск / безопасность.** LOW. Adding a trailing flag doesn't shift any index-based test lookups (test_imagegen uses cmd.index('-o')/'--vae'/'-n' — all still resolve). The test _cfg SimpleNamespace lacks imagegen_max_vram → getattr default -1.0 applies → --max-vram appears; no membership test breaks. Small perf caveat: on a fully-free GPU, --max-vram -1 may graph-split slightly more than unrestricted; acceptable trade for not losing the whole batch. Coordinate with sd-batch (wave5): the new batch command builder must include this same flag (put it in the shared _sd_cmd_core helper).

**Тест (зафиксировать поведение).** tests/test_imagegen.py: extend a generate_image mock-run test (pattern of test_generate_image_vae_flag_when_configured) to assert '--max-vram' in seen['cmd'] and the value == '-1.0' by default; add a case with _cfg(imagegen_max_vram=0.0) asserting '--max-vram' NOT in cmd (disable path).

---
## 43. Queueing the open clip re-detects and overwrites the user's curated cutlist.json
🟡 MEDIUM · симптом 4 · effort M · `queue-detect-overwrite`

- **Файл / якорь:** `serve.py` — _queue_process_one, detection step, serve.py ~1560-1568
- **Закрывает находки:** serve.py:1565

**Проблема.** _queue_process_one builds a throwaway Session on the same file+out_dir as the editor, then unconditionally re-detects (unless the ctor freshly detected). _detect() saves fresh detection over out/<stem>.cutlist.json (models save_json at Session._detect line 281), keeping only TYPE_MANUAL cuts and silently reverting the user's enable/disable curation of auto cuts. The 'add current clip to queue' UI button saves the curated cutlist to disk first, so the queue then destroys exactly what the user just saved; the loss surfaces a day later on reopen.

**Почему так.** put_cutlist sets cl.source=str(s.inp) on every save (serve.py:2233) and Session._detect passes source=str(self.inp) to run_detection, so a cutlist whose source matches this input is provably the user's own curation of THIS file — trusting it is more correct than re-detecting, and it stops the destructive save. The FRESH-detect fast-path (_ctor_fresh_detect) is untouched, so first-time clips still detect exactly once.

**Текущий код:**
```
    # (2) detection. If the ctor LOADED an on-disk cutlist (prior editor
    #     session), re-detect so stale edits can't leak in. If the ctor just
    #     generated a FRESH cutlist from a cached transcript, skip the
    #     redundant (LLM-heavy) second pass.
    _chk()
    if not getattr(ls, "_ctor_fresh_detect", False):
        stage("Детекция вырезов…")
        ls._detect()
    prog(0.45)
```

**Изменение:**
```
    # (2) detection. Skip the redundant second pass when the ctor already
    #     produced a FRESH cutlist from a cached transcript. ALSO skip (and do
    #     NOT overwrite the file) when the ctor loaded an on-disk cutlist that
    #     BELONGS to this exact input (its `source` == this file): that is the
    #     user's curated cutlist, and _detect() would save over it, reverting
    #     every enable/disable toggle (only TYPE_MANUAL cuts survive its merge).
    #     Re-detect only for a genuinely foreign/stale cutlist (missing or
    #     mismatched `source` — e.g. a legacy stem-collision in out_dir).
    _chk()
    own_cutlist = (
        ls.cutlist is not None
        and os.path.normcase(str(getattr(ls.cutlist, "source", "") or ""))
            == os.path.normcase(str(ls.inp))
    )
    if not getattr(ls, "_ctor_fresh_detect", False) and not own_cutlist:
        stage("Детекция вырезов…")
        ls._detect()
    prog(0.45)
```

**Риск / безопасность.** Behavior change: any queued clip that already has a matching-source cutlist.json now renders the SAVED curation instead of re-running detection — intended, and matches the UI which saves-before-queue so disk==live. A clip whose on-disk cutlist has empty/mismatched source (copied file, pre-source-field legacy) still re-detects as before, so no silent skip for genuinely stale data. os is already imported. No unit test exercises _queue_process_one (it drives real ffmpeg/whisper), so no regression in the suite; add the targeted test below. Out of scope (note as follow-up): the sibling issue that the queue render overwrites out/<stem>.mp4 from a prior editor render.

**Тест (зафиксировать поведение).** tests/test_queue.py: monkeypatch serve.Session with a fake whose ctor sets .cutlist (source == its inp), _ctor_fresh_detect=False, and a .`_detect` MagicMock; stub _resolve_render_opts and _run_render_pipeline. Call serve._queue_process_one(job) and assert fake._detect was NEVER called AND the on-disk cutlist.json bytes are unchanged. Second case: fake.cutlist.source != inp -> assert _detect WAS called. This locks 'own cutlist is not clobbered'.

---
## 44. POST /api/enrich/select doesn't resync flat asset_kind/asset_path to the newly-selected candidate — selection is a silent no-op for what renders, contradicting its own docstring
🟡 MEDIUM · симптом 4 · effort S · `enrich-select-flat-resync`

- **Файл / якорь:** `serve.py` — enrich_select() body (~line 3832-3836)
- **Закрывает находки:** serve.py:3833

**Проблема.** The handler sets d["selected"]=idx and re-runs ImagePayload.sanitize, which recomputes only the `source` string (enrich.py:447-450). The flat asset_kind/asset_path/gen_* fields the render/flat path consumes stay pinned to the previously materialized candidate, so selecting a different diffusion seed / stock / schematic / none does not change what renders. The docstring (serve.py:3803-3805) explicitly promises this flat resync. The exact sync already exists as enrich_llm._legacy_sync_from_candidate but is only invoked during analysis.

**Почему так.** Reusing the analysis-time sync makes `selected` actually drive the flat fields the render path reads, honoring the endpoint's contract with a one-line call to existing, tested logic.

**Текущий код:**
```
    # Меняем selected и пересобираем item через санитайзер (клампы/синк source).
    d = target.payload.to_dict()
    d["selected"] = idx
    new_pl = enrich_mod.ImagePayload.sanitize(d)
    target.payload = new_pl
```

**Изменение:**
```
    # Меняем selected И пересинхронизируем ПЛОСКИЕ поля (asset_kind/asset_path/
    # gen_prompt_en/gen_seed) под выбранный кандидат — иначе рендер/стадия images
    # продолжат смотреть на ПРЕДЫДУЩИЙ материализованный ассет (docstring обещает
    # ровно этот синк). _legacy_sync_from_candidate — тот же синк, что делает
    # анализ; читает d["selected"]/d["candidates"], поэтому зовём ПОСЛЕ set idx.
    d = target.payload.to_dict()
    d["selected"] = idx
    enrich_llm._legacy_sync_from_candidate(d)
    new_pl = enrich_mod.ImagePayload.sanitize(d)
    target.payload = new_pl
```

**Риск / безопасность.** enrich_llm is already imported in serve.py (line 68). _legacy_sync_from_candidate mutates only asset_kind/gen_prompt_en/gen_seed/asset_path per the selected candidate's source (diffusion->generate, stock->user+path, icon->emoji, schematic/none->none); sanitize then clamps. The 400/404/409 guards and strict save are untouched. PARTIAL FIX: full 'selection changes the rendered pixels' also requires plan_render to consume resolved_asset() (finding 1, R1/enrich-render cluster) — otherwise a newly-selected diffusion candidate whose PNG was never materialized still has no flat asset_path. Note this dependency; do not duplicate that fix. The shipped UI selects via /api/enrich/save (visual patch), not this endpoint, so end-user blast radius is nil; this repairs the documented UI-agent/API contract.

**Тест (зафиксировать поведение).** Extend tests/test_api_enrich.py::test_select_sets_selected_and_resyncs_source: after selecting the 'none' candidate assert payload.asset_kind=='none'. Add test_select_resyncs_flat_fields with candidates=[{source:'diffusion',prompt:'p',seed:7},{source:'diffusion',prompt:'q',seed:42}]; POST select idx=1; GET /api/enrich (or read plan file) and assert payload.asset_kind=='generate', payload.gen_seed==42, payload.gen_prompt_en=='q'. Add a stock case: select a {source:'stock',asset_path:X} candidate -> asset_kind=='user', asset_path==X.

---
## 45. Clip renders must use the live «Настройки рендера» values (codec/quality/loudnorm/cut_fade/censor), not config defaults
⚪ LOW · симптом 3 · effort S · `clips-render-live-modal-opts`

- **Файл / якорь:** `web/app.js` — clipsRenderOpts() — app.js:1096-1121
- **Закрывает находки:** web/app.js:1096

**Проблема.** clipsRenderOpts() builds from currentRenderOpts() + st.rdefaults (config.yaml session defaults), so a user who switches to x264/CRF20 and enables «Громкость под YouTube» in the render modal still gets clips encoded with the config-default nvenc/QP and config loudnorm. Yet the clips modal's own hint (app.js:1072-1076) promises the render-modal caption preset is used, setting the expectation that render-modal settings apply. The in-code comment even claims «текущие кодек/качество/громкость редактора».

**Почему так.** Makes clip renders honor the settings the UI tells the user apply, using the same seed-then-collect path autopack already trusts, instead of a diverging config-default subset.

**Текущий код:**
```
function clipsRenderOpts() {
  const o = clipsOptsLoad()
  const d = st.rdefaults || {}
  const opts = {
    ...currentRenderOpts(),
    subtitles: false, chapters: false, metadata: false,
    vertical: !!o.vertical,
    denoise_loudnorm: !!d.denoise_loudnorm,
    loudnorm_mode: d.loudnorm_mode === '2pass' ? '2pass' : 'dynamic',
    out_dir: o.out_dir || st.outDir || '',
  }
  if (d.cut_fade != null) opts.cut_fade = d.cut_fade
  if (o.vertical) {
    opts.vertical_target = '1080x1920'
    opts.vertical_center = o.center === 'manual' ? o.center_pos : 'auto'
  }
  if (o.burn) {
    opts.burn_subtitles = true
    const p = capPresetFind(capPresetSaved())
    opts.burn_style = p ? { ...p.style } : { karaoke: true }
  } else opts.burn_subtitles = false
  return opts
}
```

**Изменение:**
```
function clipsRenderOpts() {
  const o = clipsOptsLoad()
  if (!st._rSeeded) seedRenderModal()   // fill render-modal DOM from session defaults if the user never opened it (same pattern as openAutopackModal, app.js:2937)
  const opts = collectRenderOpts()      // LIVE encoder/quality/censor/loudnorm/cut_fade/denoise from «Настройки рендера»
  // Clips are their own artifact: no chapter/metadata/subtitle sidecars, no main-video
  // formats/music/background-enrich/filename — the server also forces these off; we just
  // don't send conflicting values.
  opts.subtitles = false; opts.chapters = false; opts.metadata = false
  delete opts.formats; delete opts.music; delete opts.enrich; delete opts.filename
  opts.vertical = !!o.vertical
  opts.out_dir = o.out_dir || st.outDir || ''
  if (o.vertical) {
    opts.vertical_target = '1080x1920'
    opts.vertical_center = o.center === 'manual' ? o.center_pos : 'auto'
    delete opts.scale_h   // vertical_target defines geometry; a stray render-modal scale_h would fight it
  }
  if (o.burn) {
    opts.burn_subtitles = true
    // caption style stays the render-modal preset (клипы-модалка has no style fields)
    const p = capPresetFind(capPresetSaved())
    opts.burn_style = p ? { ...p.style } : { karaoke: true }
  } else opts.burn_subtitles = false
  return opts
}
```

**Риск / безопасность.** collectRenderOpts() reads render-modal DOM fields that always exist in index.html; seedRenderModal() populates them from st.rdefaults without needing the modal open (openAutopackModal relies on this). New fields now forwarded (censor from #rCensor, denoise_*, scale_h/fps) match what autopack already sends to clip rendering, so the server's _resolve_render_opts tolerates them. Keep the explicit burn override AFTER collectRenderOpts (which sets burn from the render modal's own checkbox) so the clips-modal burn choice wins. Verify the server still forces music=null for clips (it does, per the finding). Preserve out_dir precedence (clips modal > session).

**Тест (зафиксировать поведение).** Needs jsdom+vitest (web/tests/): seed st.rdefaults, call seedRenderModal(), set #rEncoder='libx264', check #rLoudnorm, set #rQuality=20 → assert clipsRenderOpts() returns {encoder:'libx264', quality:20, denoise_loudnorm:true} and has no `formats`/`music`/`filename` keys, subtitles/chapters/metadata all false, and burn_style from the saved preset when o.burn. A backend test cannot catch this (the bug is which opts the client sends).

---
## 46. Debounced enrich save must deep-merge nested payload so two payload edits in one flush don't lose the first
⚪ LOW · симптом 5 · effort S · `enrich-shallow-merge`

- **Файл / якорь:** `web/app.js` — enrichQueueSave() ~3667-3672
- **Закрывает находки:** web/app.js:3669, web/app.js:3667

**Проблема.** enrichQueueSave coalesces pending patches with a shallow spread {...prev,...patch}. payload is nested, so a second {payload:{items}} wholesale replaces a pending {payload:{title}} — the title edit is dropped server-side and the UI reverts it on the save response. The single global enrSaveTimer is reset by every queued edit on any item, widening the window past 350 ms.

**Почему так.** One-level-deep merge of the only nested field (payload) preserves disjoint fragments (title vs items vs schematic_style) from separate producers in one debounce window; the server merge (serve.py:3759 {**d['payload'],**e['payload']}) is already per-field, so the client just stops clobbering before sending.

**Текущий код:**
```
function enrichQueueSave(id, patch) {
  const prev = enrPending.get(id) || { id }
  enrPending.set(id, { ...prev, ...patch })
  clearTimeout(enrSaveTimer)
  enrSaveTimer = setTimeout(enrichFlushSave, 350)
}
```

**Изменение:**
```
function enrichQueueSave(id, patch) {
  const prev = enrPending.get(id) || { id }
  const merged = { ...prev, ...patch }
  if (prev.payload || patch.payload) {
    merged.payload = { ...(prev.payload || {}), ...(patch.payload || {}) }
  }
  enrPending.set(id, merged)
  clearTimeout(enrSaveTimer)
  enrSaveTimer = setTimeout(enrichFlushSave, 350)
}
```

**Риск / безопасность.** LOW. Deep merge only when either side has payload; top-level keys (enabled/t_start/t_end) keep shallow semantics. No new async. Pairs with enrich-beforeunload (both touch enrPending).

**Тест (зафиксировать поведение).** jsdom (tests/js/enrichQueueSave): queue {payload:{title:'T'}} then {payload:{items:[...]}} for same id, assert enrPending.get(id).payload has BOTH title and items. No harness yet — server half already correct (documentable in test_api_enrich.py by posting two disjoint payload fragments).

---
## 47. Encoder switch must use config qp/crf per encoder, and the quality label must match what is sent
⚪ LOW · симптом 3 · effort S · `quality-per-encoder`

- **Файл / якорь:** `web/app.js` — seedQuality() ~2441, encoder onchange ~4117, seedRenderModal ~2493 + serve.py defaults ~1957 + web/index.html:499
- **Закрывает находки:** web/app.js:2441, web/app.js:4117, serve.py:1957, web/index.html:499

**Проблема.** Switching encoder calls seedQuality(enc,null) which falls back to hardcoded QUAL_DEFAULT {nvenc:19,x264:17} instead of config nvenc.qp/x264.crf (only one 'quality' number ships). Also the slider is min=14 max=30 while the server accepts 0-51: a config qp of 12 seeds value=12 -> browser clamps to 14, but the label uses the raw 12 -> UI shows 'QP 12' while 14 is sent.

**Почему так.** Shipping quality_nvenc/quality_x264 lets the encoder toggle reflect the real config value; clamping val to the slider's live min/max before writing the label guarantees the displayed number equals what collectRenderOpts (parseInt of the slider) sends.

**Текущий код:**
```
const QUAL_DEFAULT = { nvenc: 19, x264: 17 }
function seedQuality(encoder, q) {
  const val = (q != null) ? q : (QUAL_DEFAULT[encoder] != null ? QUAL_DEFAULT[encoder] : 19)
  $('#rQuality').value = val
  $('#rQualVal').textContent = `${qualLabel(encoder)} ${val}`
}

serve.py:1957:  "quality": s.cfg.render.nvenc.qp if s.cfg.render.encoder == "nvenc" else s.cfg.render.x264.crf,
web/index.html:499:  <input id="rQuality" type="range" min="14" max="30" step="1">
```

**Изменение:**
```
(A) serve.py defaults — add both numbers next to 'quality' (keep 'quality' for back-compat): "quality_nvenc": s.cfg.render.nvenc.qp, and "quality_x264": s.cfg.render.x264.crf. (B) web/app.js seedQuality — resolve per encoder from rdefaults and clamp into the slider's live range so the label cannot lie:
function seedQuality(encoder, q) {
  const d = st.rdefaults || {}
  const cfgQ = (encoder === 'x264') ? d.quality_x264 : d.quality_nvenc
  let val = (q != null) ? q : (cfgQ != null ? cfgQ : (QUAL_DEFAULT[encoder] != null ? QUAL_DEFAULT[encoder] : 19))
  const el = $('#rQuality'); const lo = +el.min || 14, hi = +el.max || 30
  val = Math.max(lo, Math.min(hi, val))
  el.value = val
  $('#rQualVal').textContent = `${qualLabel(encoder)} ${val}`
}
(C) web/app.js seedRenderModal:2493 -> seedQuality(enc, null). (D) OPTIONAL web/index.html:499 -> min="10" so more config values are representable (the clamp in B fixes the mismatch regardless).
```

**Риск / безопасность.** LOW. 'quality' stays in defaults so un-migrated clients keep working. Clamp reads the slider's own min/max at runtime, so widening in (D) auto-relaxes it. _resolve_render_opts clamps 0-51 server-side anyway.

**Тест (зафиксировать поведение).** tests/test_serve_render.py: test_state_defaults_expose_both_qualities — set nvenc.qp=20, x264.crf=24, GET /api/state, assert defaults['quality_nvenc']==20 and ['quality_x264']==24. jsdom test for the clamp (config 12, slider min 14 -> value 14 AND label 'QP 14') — no harness yet.

---
## 48. config render.codegfx.codegfx_style is dead: detect_all never forwards it, so every schematic candidate is hardcoded to 'minimal'
⚪ LOW · симптом 3 · effort ? · `thread-codegfx-style`

- **Файл / якорь:** `vpipe/enrich_llm.py` — detect_all() signature (~line 1626) + _build_candidates call (~line 1724); serve.py _run_enrich_detectors (~line 3390)
- **Закрывает находки:** vpipe/enrich_llm.py:1724, serve.py:3390, vpipe/config.py:324

**Проблема.** CodegfxCfg.codegfx_style (config.py:324, «канальный дефолт-тема, одна из 4») has zero production consumers: detect_all calls _build_candidates without default_style, so every schematic candidate gets SCHEMATIC_STYLE_DEF='minimal'. Setting codegfx_style: neon/business/whiteboard in config has no effect — there is currently no working way to change the code-graphics theme.

**Почему так.** _build_candidates already validates default_style (falls back to 'minimal' if not in SCHEMATIC_STYLES), so an invalid config value is safe. This is the single missing wire from config → candidate.style. Note the UI-side style override is a separate contract (finding covered by the R3 UI cluster); this fixes the config default.

**Текущий код:**
```
def detect_all(transcript: Optional[Transcript], cutlist: Optional[CutList],
               params: Optional[dict], llm, log: LogFn = _noop,
               on_progress=None, *, user_folder: Optional[str] = None) -> list[EnrichItem]:
...
    if run_ill:
        try:
            _build_candidates(dicts, eff, llm, log,
                              run_diffusion=image_source in ("auto", "generate"))

---- serve.py _run_enrich_detectors ----
    return enrich_llm.detect_all(
        s.transcript, s.cutlist, enrich_mod.sanitize_params(params), s.llm,
        log=log, on_progress=s.set_progress,
        user_folder=(params or {}).get("user_folder", ""))
```

**Изменение:**
```
1) detect_all signature — add kwarg (SCHEMATIC_STYLE_DEF is already imported):
               on_progress=None, *, user_folder: Optional[str] = None,
               codegfx_style: str = SCHEMATIC_STYLE_DEF) -> list[EnrichItem]:

2) _build_candidates call — forward it:
            _build_candidates(dicts, eff, llm, log,
                              default_style=codegfx_style,
                              run_diffusion=image_source in ("auto", "generate"))

3) serve.py _run_enrich_detectors — read cfg and pass it:
    _cg = getattr(s.cfg.render, "codegfx", None)
    _cg_style = getattr(_cg, "codegfx_style", enrich_mod.SCHEMATIC_STYLE_DEF)
    return enrich_llm.detect_all(
        s.transcript, s.cutlist, enrich_mod.sanitize_params(params), s.llm,
        log=log, on_progress=s.set_progress,
        user_folder=(params or {}).get("user_folder", ""),
        codegfx_style=_cg_style)
```

**Риск / безопасность.** LOW. Default kwarg keeps every existing detect_all caller (and its tests) identical (SCHEMATIC_STYLE_DEF='minimal'). getattr on cfg tolerates a missing codegfx block. No schema/persistence change.

**Тест (зафиксировать поведение).** tests/test_enrich_llm.py (or test_enrich_assets.py): unit-test _build_candidates directly — points with one schematic-routed _route, call _build_candidates(points, eff, llm, log, default_style='neon'); assert points[0]['payload']['candidates'][0]['style']=='neon' and schematic_style=='neon'. Add an integration assert: detect_all(..., codegfx_style='business') yields a schematic candidate with style 'business'.

---
## 49. Watch-folder registry write-back races watch_set seeding — a folder change can enqueue the whole pre-existing archive
⚪ LOW · симптом 4 · effort M · `watch-registry-gen`

- **Файл / якорь:** `serve.py` — _watch_tick write-back (serve.py ~1799-1834) + watch_set seed apply (~3018-3024) + WATCH globals (~1650)
- **Закрывает находки:** serve.py:1828

**Проблема.** _watch_tick copies WATCH_PROCESSED under lock, scans the disk WITHOUT the lock (deliberate), then unconditionally does WATCH_PROCESSED.clear(); WATCH_PROCESSED.update(registry). If an old thread's tick is in flight when POST /api/watch seeds the registry for a new folder (WATCH_PROCESSED.update(seed)), the old tick's write-back — built from the pre-seed copy — clobbers the seed. The new watcher then scans the new folder against an unseeded registry and, after the two-phase stability check, enqueues every old video in it — exactly what seeding promises to prevent.

**Почему так.** A monotonically increasing generation, read with the 'before' copy and re-checked before the write-back (all under WATCH_LOCK), lets an in-flight tick detect that its snapshot is stale and abandon the clobber, while preserving the deliberate lock-free disk scan. The finding's own recommended fix (version the registry).

**Текущий код:**
```
    log = logging.getLogger("fastvideoedit.watch")
    with WATCH_LOCK:
        before = {k: dict(v) for k, v in WATCH_PROCESSED.items()}
    registry = {k: dict(v) for k, v in before.items()}
    ...
    with WATCH_LOCK:
        WATCH_PROCESSED.clear()
        WATCH_PROCESSED.update(registry)
        WATCH_STATUS["error"] = None
        WATCH_STATUS["last_scan"] = time.time()
    if registry != before:
        _save_watch()
    return enqueued
```

**Изменение:**
```
Three coordinated edits guarded by WATCH_LOCK (a generation counter):

(1) Near the WATCH globals (~line 1651, beside WATCH_STATUS) add:
    WATCH_GEN = 0   # bumped under WATCH_LOCK on every watch_set (folder/seed change)

(2) In _watch_tick, capture the generation together with the 'before' snapshot:
    with WATCH_LOCK:
        gen = WATCH_GEN
        before = {k: dict(v) for k, v in WATCH_PROCESSED.items()}
    registry = {k: dict(v) for k, v in before.items()}

(3) In _watch_tick's write-back block, refuse to clobber if the generation moved under us:
    with WATCH_LOCK:
        if gen != WATCH_GEN:
            # watch_set re-seeded the registry (folder change) while our
            # lock-free scan ran. `registry` is based on the PRE-seed copy;
            # writing it back would clobber the seed and the new watcher
            # would enqueue the whole pre-existing archive. Drop our
            # write-back — the new generation owns the registry now.
            WATCH_STATUS["last_scan"] = time.time()
            return enqueued
        WATCH_PROCESSED.clear()
        WATCH_PROCESSED.update(registry)
        WATCH_STATUS["error"] = None
        WATCH_STATUS["last_scan"] = time.time()
    if registry != before:
        _save_watch()
    return enqueued

(4) In watch_set (serve.py ~3018-3024), declare `global WATCH_GEN` at the top of the function body and bump it inside the existing WATCH_LOCK block where the seed is applied:
    with WATCH_LOCK:
        WATCH["enabled"] = enabled
        WATCH["folder"] = folder
        WATCH_GEN += 1
        WATCH_PROCESSED.update(seed)
        if not enabled:
            WATCH_STATUS["error"] = None
```

**Риск / безопасность.** WATCH_GEN is an int guarded by WATCH_LOCK; read/increment are cheap and uncontended. Startup path (_watch_apply from main, serve.py:4400) has no competing watch_set, so gen stays 0 and the first tick writes back normally — byte-for-byte identical to today. The only behavior delta: an OLD tick finishing after a folder change no longer persists its stale registry; its already-enqueued files (legit old-folder new files) still stand and are deduped by _watch_busy_paths. Alternative if you prefer no global: merge instead of clear — only pop keys under the scanned folder's prefix — but that is more code and still needs a folder-changed guard.

**Тест (зафиксировать поведение).** tests/test_watch.py: monkeypatch serve.scan_once so that, mid-call, it simulates watch_set by doing `serve.WATCH_GEN += 1; serve.WATCH_PROCESSED.update(SEED)` and then returns []. Pre-seed WATCH_PROCESSED empty, call serve._watch_tick(folder, {}) and assert afterwards WATCH_PROCESSED == SEED (the tick's write-back was skipped, the seed survived). Add a non-racing control: no gen bump -> WATCH_PROCESSED reflects the tick's registry as before.

---
## 50. burn_style.karaoke defaults to True on an absent key, overriding config.yaml's karaoke:false
⚪ LOW · симптом 3 · effort ? · `burn-karaoke-config-default`

- **Файл / якорь:** `serve.py` — _resolve_render_opts, burn_style block (~L941)
- **Закрывает находки:** serve.py:941

**Проблема.** Unlike every sibling burn_style field (applied only `if bs.get(...)`), karaoke is written unconditionally: `b.karaoke = bool(bs.get('karaoke', True))`. Any /api/render with burn_subtitles=true and no 'karaoke' key (older clients, queue jobs with partial render_opts, curl) force-enables karaoke even when the user set subtitles.burn.karaoke:false in config.yaml (config default is True, so it's silently un-overridable via config).

**Почему так.** Matches the plumbing contract of every other field: an absent key keeps the config.yaml value; an explicit true/false overrides. The web UI always sends the key, so its behavior is unchanged.

**Текущий код:**
```
        if bs.get("position") in ("bottom", "top", "center"):
            b.position = bs["position"]
        b.karaoke = bool(bs.get("karaoke", True))
```

**Изменение:**
```
        if bs.get("position") in ("bottom", "top", "center"):
            b.position = bs["position"]
        if "karaoke" in bs:
            b.karaoke = bool(bs["karaoke"])
```

**Риск / безопасность.** Only affects callers that OMIT karaoke while config sets it false — the exact bug. The web UI seeds the checkbox and always sends karaoke, so interactive renders are byte-for-byte unchanged. No test currently exercises the absent-key path (test_serve_render.py:202 always passes karaoke:true).

**Тест (зафиксировать поведение).** In tests/test_serve_render.py: test_karaoke_absent_keeps_config_default — load config, set cfg.subtitles.burn.karaoke=False on the session cfg; cfg2,*_ = _resolve_render_opts(s, {'burn_subtitles':True,'burn_style':{'font':'Arial'}}); assert cfg2.subtitles.burn.karaoke is False. test_karaoke_explicit_true_overrides / test_karaoke_explicit_false_overrides lock both directions.

---
## 51. Fail-fast validation for API/queue render-opts: reject JSON bool quality/fps, garbage audio_bitrate, and round odd scale_h to even
⚪ LOW · симптом none · effort ? · `harden-render-opts-validation`

- **Файл / якорь:** `serve.py` — _resolve_render_opts: quality (~L901-903), audio_bitrate (~L907-908), scale_h (~L1083-1092), fps (~L1097-1098)
- **Закрывает находки:** serve.py:901, serve.py:907, serve.py:1091, serve.py:1097

**Проблема.** Merges three non-UI-caller holes in _resolve_render_opts (queue.json/curl reach it unvalidated). (#4) quality:true passes `isinstance(q,(int,float))` (bool is an int) → QP/CRF 1 (near-lossless, giant file); fps:true → float(True)=1.0 passes 0<fps<=120 → 1-fps render. (#6) audio_bitrate accepts any truthy value verbatim into ffmpeg -b:a → task-time failure on 'fast'/'-320k'. (#5) odd scale_h (735) passes the [144,4320] check then dies at task time with 'height not divisible by 2' because yuv420p needs even dims (render.py:990 scale=-2:H only fixes width). All violate the fail-fast 400 contract the codebase already honors for min_score (serve.py:1071) and formats.

**Почему так.** One coherent 'harden the shared helper for non-UI callers' change: all four live in _resolve_render_opts, all convert a silent-bad-value or task-time crash into an immediate, clear 400 (or a safe even-round), matching the fail-fast promise the endpoint comment makes. Rounding scale_h (vs 400) is friendlier and matches what an even preset would produce.

**Текущий код:**
```
    q = opts.get("quality")
    if isinstance(q, (int, float)):
        q = max(0, min(51, int(q)))   # H.264/HEVC QP/CRF range
...
    if opts.get("audio_bitrate"):
        cfg.render.audio_bitrate = str(opts["audio_bitrate"])
...
        if not (144 <= scale_h <= 4320):
            raise HTTPException(400, "scale_h must be between 144 and 4320")
        if scale_h == s.media.height:            # identity -> let copy fast-path win
            scale_h = None
...
    fps = opts.get("fps") or None                # None = source fps
    if fps is not None:
```

**Изменение:**
```
(a) quality bool — exclude bool from the numeric branch (mirror the min_score guard):
    if isinstance(q, (int, float)) and not isinstance(q, bool):

(b) audio_bitrate — whitelist Nk (re is already imported):
    if opts.get("audio_bitrate"):
        ab = str(opts["audio_bitrate"]).strip()
        if not re.fullmatch(r"\d{2,4}k", ab):
            raise HTTPException(400, "audio_bitrate: ожидается вид «128k»…«320k» / e.g. '192k'")
        cfg.render.audio_bitrate = ab

(c) scale_h — round odd height down to even AFTER the range check (before the identity check), so yuv420p never sees an odd height:
        if not (144 <= scale_h <= 4320):
            raise HTTPException(400, "scale_h must be between 144 and 4320")
        if scale_h % 2:                          # yuv420p requires an even height
            scale_h -= 1
        if scale_h == s.media.height:            # identity -> let copy fast-path win
            scale_h = None

(d) fps bool — reject JSON true/false right after `fps = opts.get("fps") or None`:
    fps = opts.get("fps") or None                # None = source fps
    if isinstance(fps, bool):                    # JSON true/false is not an fps
        raise HTTPException(400, "fps must be a number (0 < fps <= 120)")
    if fps is not None:
```

**Риск / безопасность.** The web UI only sends valid presets, so interactive renders are unchanged. quality/fps bool: previously-'working' bool callers now 400 — intended (they were producing 1-QP/1-fps garbage). audio_bitrate: any caller sending a non-Nk string now 400s instead of failing mid-render — intended; confirm no internal caller passes e.g. '320k ' with spaces (the .strip() covers it) or a bare integer (would now 400 — none found; the UI/config use 'Nk'). scale_h rounding changes an odd request by 1px (imperceptible) rather than erroring. Add `not isinstance(q,bool)` only to the quality branch to avoid touching the clamp.

**Тест (зафиксировать поведение).** New tests/test_render_opts_validation.py driving _resolve_render_opts directly (fixture from test_multiformat _mk_session): test_quality_true_ignored (quality:true → cfg qp/crf unchanged from default, NOT 1); test_fps_true_400 and test_fps_false_is_source (pytest.raises HTTPException 400 for True; False→None); test_audio_bitrate_garbage_400 ('fast','-320k','320' → 400) and test_audio_bitrate_valid ('192k' accepted); test_scale_h_odd_rounded (735 → returned scale_h==734) and test_scale_h_even_unchanged (720→720). Assert status_code==400 like test_multiformat.py:148-150.

---
## 52. Enrich CTA/card geometry hardcodes the DEFAULT burn style — «Неон»/«Крупный» presets overlap CTA text and cards with subtitles
⚪ LOW · симптом 1 · effort M · `enrich-subs-collision-hardcoded-top`

- **Файл / якорь:** `vpipe/enrich_cards.py` — SUBS_TOP_1080 const (line 62), cta_mv (line 470), scrim y_limit (line 285); build/write_enrich_ass sigs; serve.py:1261 call
- **Закрывает находки:** vpipe/enrich_cards.py:62, vpipe/enrich_cards.py:285, vpipe/enrich_cards.py:470, serve.py:1261

**Проблема.** SUBS_TOP_1080=915 is derived from AssStyleCfg defaults (size 52, margin_v 40, bottom, 2 lines) and is never recomputed from the user-selected burn style: serve.py:1261 passes no style into write_enrich_ass and build_enrich_ass has no param for it. Preset «Неон» (margin_v 160, size 62) raises the real subs-zone top to ~771px, so the CTA text (bottom fixed at y≈845 via MarginV=235) draws on the karaoke line; «Крупный» (position center) renders subs over the panel. Enrich and burn are separate subtitles= passes with no cross-file collision resolution (burn.ass is last, so subs stay on top — impact is clutter, not lost subtitles).

**Почему так.** SUBS_TOP now tracks the ACTUAL burn style: «Неон» → ~772 (CTA/cards lift above it), default → ~916 (byte-identical to today's 915 within rounding), top/center → 1080 so bottom-anchored CTA sits near the true bottom (clear of non-bottom subs). Default param keeps every existing caller and test unchanged.

**Текущий код:**
```
# Верхняя кромка зоны burn-сабов @1080 ... ≈ 915 (дефолты AssStyleCfg: 52/40/bottom/2).
SUBS_TOP_1080 = 915
...
    y_limit = _px(SUBS_TOP_1080 - CARD_SUBS_CLEAR_1080, k)   # ~line 285
...
    cta_mv = _px(1080 - SUBS_TOP_1080 + CTA_GAP_OVER_SUBS_1080, k)   # ~line 470
...
def build_enrich_ass(cards, ctas, W, H, style_overrides=None) -> str:
def write_enrich_ass(cards, ctas, W, H, path, style_overrides=None) -> Path:
...
                    enrich_cards.write_enrich_ass(
                        re_plan.cards, re_plan.cta_texts, out_w, out_h,
                        enr_ass)
```

**Изменение:**
```
1) Add a helper (enrich_cards.py, near SUBS_TOP_1080):
def subs_zone_top_1080(style, max_lines: int = 2) -> float:
    """Top edge (px @1080) of the burn-subtitle zone for the given style.
    Bottom-anchored: 1080 - margin_v - max_lines*(size*1.2). Top/center subs
    do not sit in the bottom zone -> 1080 (fully clear)."""
    if getattr(style, 'position', 'bottom') != 'bottom':
        return 1080.0
    return max(0.0, 1080.0 - float(style.margin_v) - max_lines * float(style.size) * 1.2)
2) build_enrich_ass gains param subs_top_1080: float | None = None; at top do subs_top = SUBS_TOP_1080 if subs_top_1080 is None else subs_top_1080, and thread it into the card-items builder (the function owning line 285). Replace the two SUBS_TOP_1080 uses (285, 470) with subs_top.
3) write_enrich_ass gains the same param and forwards it to build_enrich_ass.
4) serve.py:1261 call:
                    enrich_cards.write_enrich_ass(
                        re_plan.cards, re_plan.cta_texts, out_w, out_h, enr_ass,
                        subs_top_1080=(enrich_cards.subs_zone_top_1080(
                            cfg.subtitles.burn, cfg.subtitles.max_lines)
                            if cfg.subtitles.burn.enabled else None))
```

**Риск / безопасность.** Low, but touches two files. Keep back-compat: param default None ⇒ SUBS_TOP_1080 ⇒ existing test_enrich_cards_ass.py output unchanged. KNOWN RESIDUAL (out of scope, flag as follow-up): the DEFAULT PANEL card geometry (py+ph=920) never consults SUBS_TOP at all, so «Крупный» center-subs-over-panel is only partially mitigated (CTA/scrim fixed, panel not) — a larger geometry change. Note P3 must land first so burn size/margin are truly @1080 values that subs_zone_top_1080 reads correctly.

**Тест (зафиксировать поведение).** Add tests/test_enrich_cards_ass.py::test_subs_zone_top_from_style: subs_zone_top_1080(AssStyleCfg())≈916; AssStyleCfg(size=62,margin_v=160)≈771; AssStyleCfg(position='center')==1080. And ::test_cta_margin_v_follows_subs_top: build_enrich_ass with subs_top_1080=771 vs default → the CtaText MarginV field is larger (CTA lifted higher) for the smaller subs_top.

---

# Волна 4 — Устойчивость UI + краевые случаи

## 53. start_task error-vs-cancel masking and the /api/events SSE terminal-snapshot flush are untested — a real failure after cancel is reported as a clean cancellation and the stream's done/error delivery is unverified
🟠 HIGH · симптом 4 · effort M · `sse-and-start-task-tests`

- **Файл / якорь:** `tests/test_events_sse.py` — serve.Session.start_task worker (serve.py:308-321) + /api/events SSE (serve.py:4286-4316)
- **Закрывает находки:** serve.py:317, serve.py:4286, serve.py:4305

**Проблема.** start_task's worker sets task['error']='cancelled' if task['cancelled'] else str(e) (serve.py:317) — a GENUINE failure occurring while the cancelled flag is set hides the real error. /api/events (4286-4316) is the UI's only progress/done/error channel incl. the final-snapshot flush (4305-4311); grep 'api/events' in tests = nothing. A regression that ends the stream before the terminal state ships green and the UI shows a render 'running' forever or silently done.

**Текущий код:**
```
# tests/test_queue.py:159-168 — only start_task's 409 guard is covered; the worker body isn't:
def test_editor_start_task_blocked_while_queue_running(monkeypatch):
    monkeypatch.setattr(serve, '_queue_running', True)
    fake = SimpleNamespace(task={'running': False})
    with pytest.raises(HTTPException) as ei:
        serve.Session.start_task(fake, 'render', lambda: None)
    assert ei.value.status_code == 409
# No test requests /api/events. No test drives the worker's success/error/cancel branches.
```

**Изменение:**
````
New file tests/test_events_sse.py.

```python
import json, time
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
import serve

def _run_worker(fake, fn):
    # drive start_task's worker synchronously: patch Thread to run inline
    import threading
    real=threading.Thread
    try:
        threading.Thread=lambda target,daemon=None: SimpleNamespace(start=target)
        serve.Session.start_task(fake, 'render', fn)
    finally:
        threading.Thread=real

# 1) genuine success -> percent 100, done True, no error

def test_start_task_success(monkeypatch):
    monkeypatch.setattr(serve, '_queue_running', False)
    fake=SimpleNamespace(task={'running':False})
    _run_worker(fake, lambda: None)
    assert fake.task['done'] is True and fake.task['percent']==100.0 and fake.task['error'] is None

# 2) genuine failure, NOT cancelled -> real message surfaced

def test_start_task_reports_real_error(monkeypatch):
    monkeypatch.setattr(serve, '_queue_running', False)
    fake=SimpleNamespace(task={'running':False})
    def fn(): raise RuntimeError('ffmpeg exit 1: no such filter')
    _run_worker(fake, fn)
    assert fake.task['error']=='ffmpeg exit 1: no such filter' and fake.task['running'] is False

# 3) DOCUMENTED masking: a real failure AFTER cancel is reported as 'cancelled' (bug the fix must address)
@pytest.mark.xfail(strict=True, reason='serve.py:317 masks a genuine post-cancel error as cancelled; fix should distinguish a clean cancel from a real crash')
def test_post_cancel_real_error_not_masked(monkeypatch):
    monkeypatch.setattr(serve, '_queue_running', False)
    fake=SimpleNamespace(task={'running':False})
    def fn():
        fake.task['cancelled']=True                    # user cancel flips the flag...
        raise RuntimeError('disk full')                # ...but a REAL crash follows
    _run_worker(fake, fn)
    assert 'disk full' in (fake.task['error'] or '')

# 4) SSE terminal-snapshot flush: an already-finished task yields a done/error frame then closes

def test_events_flushes_terminal_snapshot(monkeypatch):
    fake=SimpleNamespace(task={'name':'render','running':False,'percent':100.0,'stage':'',
                               'error':None,'done':True,'results':{'mp4':'o.mp4'},'cancelled':False})
    monkeypatch.setattr(serve, 'SESSION', fake)
    client=TestClient(serve.app)
    with client.stream('GET','/api/events') as r:
        frames=[]
        for line in r.iter_lines():
            if line.startswith('data: '): frames.append(json.loads(line[6:]))
            if frames: break
    assert frames and frames[-1]['done'] is True and frames[-1]['results']=={'mp4':'o.mp4'}
```
````

**Риск / безопасность.** Patching threading.Thread globally is intrusive — scope it tightly (try/finally restore) and only in _run_worker; alternatively spawn the real thread and poll `while fake.task['running']: time.sleep(0.01)` with a timeout. The xfail(strict=True) test (#3) locks the masking bug and flips when serve.py:317 is fixed to not swallow a real post-cancel error. For the SSE test, TestClient.stream iterates the async generator; ensure the fake task starts with running=False so gen() emits one snapshot and breaks immediately (no hang). Do not leave SESSION monkeypatched across tests.

**Тест (зафиксировать поведение).** test_events_flushes_terminal_snapshot: set SESSION.task to a finished done=True snapshot, open /api/events via TestClient.stream, assert the last data: frame carries done=True and the results payload — locks the terminal-flush the UI depends on. Plus test_post_cancel_real_error_not_masked (xfail-strict) to track the error-masking fix.

---
## 54. Stop the scary autosave-error loop and the lost word-edit when the user edits during a background task
🟡 MEDIUM · симптом 4 · effort M · `calm-409-during-task`

- **Файл / якорь:** `web/app.js` — doSave() error path (app.js:2049-2066); startWordEdit() entry (app.js:1849)
- **Закрывает находки:** web/app.js:2076, web/app.js:2049, web/app.js:1879

**Проблема.** setRunning() disables only ~11 launch buttons; region drag/resize, cut checkboxes, hotkeys and dblclick word editing stay live during a task. But the backend 409s every mutation (_guard_no_task: PUT /api/cutlist serve.py:2230, PUT /api/transcript/word serve.py:2177). A cutlist edit during a render therefore enters doSave()'s failure path — sticky red toast + 'Не сохранено' pill + a 2s retry loop hammering 409 for the whole render (edits DO flush after, so it only LOOKS like data loss). Word edits are worse: commitWordEdit reverts the text and toasts an error with no retry — the edit is silently lost.

**Почему так.** Removes the two data-loss-LOOKING behaviors (red retry-loop, reverted word) during tasks with a minimal, choke-point change, without disabling the whole editing surface.

**Текущий код:**
```
      if (!res.ok) {
        st.dirty = true                                   // unsaved -> keep dirty
        let detail = 'HTTP ' + res.status
        try { const j = await res.json(); if (j && j.detail) detail = j.detail } catch {}
        throw new Error(detail)
      }
    }
    ...
  } catch (e) {
    st.saveError = true   // keep dirty
    setSavePill('error', 'Не сохранено')
    showSaveError('Не удалось сохранить правки: ' + e.message + '. Изменения не потеряны — повтор автоматически.')
    clearTimeout(saveTimer); saveTimer = setTimeout(save, 2000)   // auto-retry
  } finally {

// startWordEdit (app.js:1849):
function startWordEdit(sp) {
  if (editingSpan === sp) return
  if (editingSpan) cancelWordEdit(editingSpan)        // одна правка за раз
  const w = st.words[+sp.dataset.i]; if (!w) return
```

**Изменение:**
```
1) Carry the status on the thrown error (in the !res.ok block):
        const err = new Error(detail); err.status = res.status; throw err

2) Calm branch for 409 in the catch (edits are deferred, not lost):
  } catch (e) {
    st.dirty = true; st.saveError = true   // keep dirty either way
    if (e.status === 409) {
      // Бэк отклоняет запись, пока идёт фоновая задача. Правки не потеряны —
      // авто-повтор допишет их после. Спокойный статус вместо красной ошибки.
      setSavePill('saving', 'Отложено — идёт задача')
      clearSaveError()
      clearTimeout(saveTimer); saveTimer = setTimeout(save, 2000)
    } else {
      setSavePill('error', 'Не сохранено')
      showSaveError('Не удалось сохранить правки: ' + e.message + '. Изменения не потеряны — повтор автоматически.')
      clearTimeout(saveTimer); saveTimer = setTimeout(save, 2000)
    }
  } finally {

3) Block word editing during a task at the entry point (prevents the silent revert):
function startWordEdit(sp) {
  if (st.task) { toast('Идёт фоновая задача — правка текста будет доступна после её завершения', 'info'); return }
  if (editingSpan === sp) return
  ...
```

**Риск / безопасность.** The 409 branch keeps st.dirty (beforeunload still warns) and keeps the retry loop, so edits still flush when the task ends — only the presentation changes from red-error to a calm deferred pill. Network errors (no e.status) fall through to the existing error path unchanged. Guarding startWordEdit is the smallest fix for the WORSE word-edit case (silent loss) — it blocks starting an edit rather than letting it hit 409 and revert. Alternative (heavier) fix of making all editing surfaces read-only during a task is unnecessary given the doSave calm-branch already neutralizes the cutlist case. Preserve the existing non-409 error UX exactly.

**Тест (зафиксировать поведение).** Backend 409 contract is already locked (tests/test_transcript_edit.py:154 sets task['running']=True → PUT /api/transcript/word → 409; add the analogous assertion for PUT /api/cutlist). Frontend (jsdom+vitest): mock PUT /api/cutlist → 409{detail} with st.segs dirty → assert savePill text 'Отложено — идёт задача' and NO sticky error toast created; call startWordEdit with st.task set → assert it returns without setting contenteditable and shows an info toast.

---
## 55. Guard init()/loadData() so a transient /api/state or /api/transcript failure can't silently abort app bootstrap
🟡 MEDIUM · симптом 2 · effort M · `init-error-handling`

- **Файл / якорь:** `web/app.js` — init() top-level call + body (app.js:183, 185-261); loadData() fetch (app.js:378)
- **Закрывает находки:** web/app.js:192, web/app.js:378

**Проблема.** init() is fired with no .catch (app.js:183) and has no try/catch; `const s = await (await fetch('/api/state')).json()` (192) and loadData()'s `const tr = await (await fetch('/api/transcript')).json()` (378) are unguarded — while the adjacent cutlist (401) and peaks (241-242) fetches ARE guarded. A /api/transcript 404/500 (e.g. another tab swapped the session between the state and transcript fetches) aborts loadData mid-way and skips everything after line 247: clips/enrich cache loads, the running-task re-attach (250), and all video listeners (252-260), leaving buttons enabled with zero feedback.

**Почему так.** Converts a silent bootstrap abort into a visible, self-healing failure and prevents one flaky sub-fetch from taking down clips/enrich/task-reattach.

**Текущий код:**
```
init()

async function init() {
  bindFiles()
  ...
  const s = await (await fetch('/api/state')).json()
  ...
  if (s.has_transcript) await loadData()
  loadClipsFromCache()
  loadEnrichFromCache()
  if (s.task && s.task.running) followTask(s.task.name)
  ...
}

// loadData():
async function loadData() {
  const tr = await (await fetch('/api/transcript')).json()
  ...
```

**Изменение:**
```
1) Catch init() rejection with a retry (app.js:183):
init().catch((e) => {
  toast('Сервер недоступен — повторяю…', 'error')
  setTimeout(() => { init().catch(() => {}) }, 3000)
})

2) Guard the state bootstrap fetch (app.js:192) so a bad response throws a clean error the catch above can act on:
  let s
  try { const r = await fetch('/api/state'); if (!r.ok) throw new Error('HTTP ' + r.status); s = await r.json() }
  catch (e) { toast('Не удалось загрузить состояние: ' + e.message, 'error'); throw e }

3) Guard loadData()'s transcript fetch like the sibling cutlist fetch, so a transcript failure does NOT abort the rest of init (clips/enrich/task re-attach still run):
async function loadData() {
  let tr
  try { const r = await fetch('/api/transcript'); if (!r.ok) throw new Error('HTTP ' + r.status); tr = await r.json() }
  catch (e) { toast('Не удалось загрузить транскрипт: ' + e.message, 'error'); return }
  ...
```

**Риск / безопасность.** Low. loadData()'s early return on failure leaves st.words empty (same as a not-yet-transcribed session) — the rest of init proceeds, which is the whole point. The init().catch retry loop is bounded per-failure (single setTimeout, not a tight loop) and stops as soon as one init() succeeds. Do not swallow the state-fetch error silently — rethrow so the retry fires. Keep the guarded style identical to the existing peaks (241-242) and cutlist (401) guards for consistency.

**Тест (зафиксировать поведение).** Needs jsdom+vitest (web/tests/): mock fetch('/api/transcript') → 500 with fetch('/api/state') OK and s.task.running=true; assert loadData returns without throwing, followTask('...') still gets called, and video event listeners are attached. Also: mock fetch('/api/state') → reject on first call, resolve on second; assert init retries and eventually binds. Backend already returns well-formed 404/409 for these (verified in serve.py).

---
## 56. On reload, surface a task that FINISHED while the tab was reloaded/discarded (render output + terminal error)
🟡 MEDIUM · симптом 2 · effort S · `init-reattach-finished-task`

- **Файл / якорь:** `web/app.js` — init() re-attach line (app.js:250)
- **Закрывает находки:** web/app.js:250

**Проблема.** init() re-attaches only when `s.task.running` is true. If a task finished (or errored) while the tab was reloaded or discarded during a long render on a still-alive server, the persisted s.task.done/results/error from /api/state is ignored: no «Рендер завершён», no results panel with output links, no error toast — the user finds neither confirmation nor output. /api/state returns the full task dict (serve.py:1982) including done/results/error, so the data is available.

**Почему так.** Recovers the one class of results that has no cache-restore path (the main render's output links) plus terminal errors, closing the 'reload during a multi-hour render loses the outcome' gap.

**Текущий код:**
```
  if (s.task && s.task.running) followTask(s.task.name)
```

**Изменение:**
```
  if (s.task && s.task.running) followTask(s.task.name)
  // Задача завершилась, пока вкладка была перезагружена/усыплена — показать исход один раз.
  // Клипы/enrich уже восстановлены из кэша выше; главы/мета — своими load*FromCache;
  // здесь остаётся рендер основного ролика (его ссылки иначе теряются) и терминальная ошибка.
  else if (s.task && s.task.done && s.task.name === 'render' && s.task.results) {
    showResults(s.task.results); toast('Рендер завершён', 'success')
  }
  else if (s.task && s.task.error && s.task.error !== 'cancelled') {
    toast('Прошлая задача завершилась с ошибкой: ' + s.task.error, 'error')
  }
```

**Риск / безопасность.** Scoped narrowly to avoid double-rendering: this only fires when the server is still alive (if it had restarted, s.no_session is true and init returned at line 194, so s.task wouldn't exist). Cancelled tasks are excluded. showResults(s.task.results) reads the render results shape (serve.py:2472 _render_formats output) which showResults already handles (r.formats or r.mp4, app.js:2895-2902). Depends on the chapters/metadata restore patch for those two task types (do not also re-render them here or they'll double up). Keep clips/enrich out — they restore via loadClipsFromCache/loadEnrichFromCache at app.js:248-249.

**Тест (зафиксировать поведение).** Backend (lockable now): after a fake render task completes, GET /api/state.task has {done:true,name:'render',results:{...}} — assert the shape. Frontend (jsdom+vitest): call init() with mocked /api/state returning a done render task → assert showResults called with the results and a success toast; with task.error set → assert error toast; with error==='cancelled' → assert no toast.

---
## 57. Queue poll must back off and stop toast-flooding when the server is unreachable
🟡 MEDIUM · симптом 5 · effort S · `queue-poll-flood`

- **Файл / якорь:** `web/app.js` — loadQueueList() ~4590-4600, startQueuePoll ~4666, stopQueuePoll ~4671
- **Закрывает находки:** web/app.js:4593, web/app.js:4599, web/app.js:4669

**Проблема.** startQueuePoll runs loadQueueList every 1.5 s; its catch unconditionally toasts (6 s TTL) and early-returns BEFORE the only stopQueuePoll() call (4599). If serve.py dies mid-batch the interval keeps firing — a new red toast every 1.5 s, stacking permanently, and the poll never stops.

**Почему так.** A failure counter turns an unbounded toast storm into a single sticky message and halts the interval after ~4.5 s, while a one-off manual loadQueueList (timer 0) still surfaces its error immediately. Reset-on-success recovers from a transient hiccup.

**Текущий код:**
```
async function loadQueueList() {
  let j
  try { const res = await fetch('/api/queue'); if (!res.ok) throw new Error('HTTP ' + res.status); j = await res.json() }
  catch (e) { toast('failed to load queue: ' + e.message, 'error'); return }
  st.queueJobs = j.jobs || []
  st.queueRunning = !!j.running
  renderQueueList()
  updateQueueBadge()
  if (!st.queueRunning && !st.queueJobs.some((x) => x.status === 'running')) stopQueuePoll()
}
```

**Изменение:**
```
Add a consecutive-failure counter: swallow first failures silently while polling, stop after 3 and show ONE sticky error; reset on success; a manual call (no active timer) still toasts once as before:
catch (e) {
    st.queuePollFails = (st.queuePollFails || 0) + 1
    if (st.queuePollTimer && st.queuePollFails >= 3) {
      stopQueuePoll()
      toast('queue unreachable — polling stopped (' + e.message + '); reload after restarting the server', 'error')
    } else if (!st.queuePollTimer) {
      toast('failed to load queue: ' + e.message, 'error')
    }
    return
  }
Add st.queuePollFails = 0 on the success path (right after the try, before assigning st.queueJobs).
```

**Риск / безопасность.** LOW. New st.queuePollFails defaults via || 0. Manual-call path (openQueueModal/add/remove) keeps its immediate single toast. Success path and stopQueuePoll unchanged.

**Тест (зафиксировать поведение).** jsdom (tests/js/queuePoll): stub fetch to reject, start poll, tick fake timer >=3x, assert stopQueuePoll ran (st.queuePollTimer===0) and exactly one error toast. No harness yet — manual repro: start worker, kill serve.py, watch the 1.5 s toast stream.

---
## 58. Pin transcription and render to the SAME (first) audio track: add -map 0:a:0 to extract_audio and the no-cut copy fast-path
🟡 MEDIUM · симптом 4 · effort S · `probe-multiaudio-map`

- **Файл / якорь:** `vpipe/probe.py` — extract_audio(), line 93 (ffmpeg -i src -vn ... with NO -map) + render.py:1109 no-cut fast-path -map 0:a
- **Закрывает находки:** vpipe/probe.py:93, vpipe/render.py:1109

**Проблема.** extract_audio runs `ffmpeg -i src -vn -ac 1 ...` with NO -map, so ffmpeg's default selection picks the audio stream with the MOST channels (ties → lowest index). The render/censor paths hard-code the FIRST audio stream ([0:a], asrc_idx '0'). For a typical OBS recording (track1=mono mic, track2=stereo desktop), Whisper transcribes the DESKTOP track (stereo wins) while the render keeps the MIC track — pause/filler/profanity/hesitation timestamps are detected against audio that is not in the output, producing seemingly wrong cuts and mis-placed censor beeps. Separately the no-cuts copy fast-path maps ALL audio tracks (render.py:1109 `-map 0:a`) while every filtered path keeps exactly one, so output track count varies with render settings.

**Текущий код:**
```
    ff.run(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", out_wav],
           total=total, on_progress=on_progress, desc="audio extraction")
```

**Изменение:**
```
In extract_audio, add an explicit first-track map so Whisper analyzes the same stream render keeps:

    ff.run(["-i", str(src), "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", out_wav],
           total=total, on_progress=on_progress, desc="audio extraction")

(`0:a:0?` = first audio stream, `?` makes it non-fatal if the source has no audio, matching the current tolerant behavior.) Also update the no-cut copy fast-path in render.py:1109 from `"-map", "0:v", "-map", "0:a"` to `"-map", "0:v", "-map", "0:a:0"` so it keeps exactly one track like every other path. Add a one-line comment: '# FVE uses the FIRST audio track everywhere (extract/censor/render).'
```

**Риск / безопасность.** Low and strictly consistency-improving. `[0:a]` in the render filtergraphs already resolves to the first audio stream in modern ffmpeg, so aligning extract_audio to 0:a:0 makes the analyzed track match the rendered one — no filtergraph change needed. The `?` suffix preserves the current no-audio tolerance (video-only sources). The render.py:1109 change reduces a multi-track copy to single-track: acceptable because every filtered render already outputs one track, and multi-track output was an undocumented side effect of the copy path. If a user genuinely wanted all tracks preserved on lossless copy, note it in the PR — but single-track is the consistent contract. Single-audio sources (the common case) are entirely unaffected.

**Тест (зафиксировать поведение).** Add tests/test_extract_audio_map.py::test_extract_audio_maps_first_track — a FakeFF (pattern from tests/test_music_duck.py:128) whose .run records args; call extract_audio and assert the arg list contains '0:a:0?' immediately after '-map'. And in test_serve_render.py add ::test_nocut_copy_maps_single_audio asserting the fast-path args contain '0:a:0' not bare '0:a'.

---
## 59. Apostrophe in the video/output filename breaks the subtitles= filter escaping and fails the whole burn-in render
🟡 MEDIUM · симптом 3 · effort S · `ass-path-quote-idiom`

- **Файл / якорь:** `vpipe/render.py` — _ass_path_for_filter() ~604-622 and callers ~997-1003
- **Закрывает находки:** vpipe/render.py:621, vpipe/render.py:997, vpipe/render.py:1003

**Проблема.** _ass_path_for_filter escapes a single quote as `\'` (render.py:621) but the callers wrap the value in single quotes: `subtitles='{...}'` (render.py:997-998, 1003). Inside an ffmpeg single-quoted filter value a backslash is literal and the `'` still CLOSES the quote, so the path is mangled and libass cannot open the ASS — render() errors out. The ASS path embeds the input video stem (work_dir = work/<stem>-<hash8>) and the per-clip/per-format ASS names embed the user-chosen output filename, so a video named "Don't panic.mp4" (or an output name with an apostrophe) kills every burn-subtitles / enrich-cards render. The burn-block try/except in serve.py does NOT save it — the failure is inside render(), not ASS prep.

**Почему так.** One-line correctness fix to the escaper that all burn-in/enrich renders depend on; the alternative sanitized-name approach is noted but deferred to keep blast radius minimal.

**Текущий код:**
```
    p = p.replace(",", "\\,").replace("[", "\\[").replace("]", "\\]")
    p = p.replace(";", "\\;")
    p = p.replace("'", "\\'")   # a quote in the path must not close the value
    return p
```

**Изменение:**
```
Emit the standard ffmpeg close-escape-reopen idiom for an embedded single quote instead of the broken backslash-quote. Under a single-quoted filter value, `'` must be written as `'\''` (end quote, literal escaped quote, reopen quote):

    p = p.replace(",", "\\,").replace("[", "\\[").replace("]", "\\]")
    p = p.replace(";", "\\;")
    # A literal single quote inside a single-quoted filter value must be written
    # as the close-escape-reopen idiom '\'' — a backslash alone does NOT escape
    # it (the quote still terminates the value). Callers wrap the result in '...'.
    p = p.replace("'", "'\\''")
    return p

(Belt-and-suspenders alternative, if a fully quote-proof path is preferred: also route ASS files through a sanitized ASCII/hash-keyed name so no user stem/output name ever reaches the filter — but the quoting fix above is the minimal correct change and keeps existing filenames.)
```

**Риск / безопасность.** Low and localized. The close-escape-reopen idiom is the documented ffmpeg quoting for single quotes and is inert for paths without a quote (no `'` → no change → byte-for-byte identical filter string for the overwhelmingly common case). Both call sites (subtitles= and cards_ass subtitles=) already wrap in single quotes, so the idiom composes correctly. The fontsdir= value at render.py:998 goes through the same function, so it is fixed too. Verify the drive-colon and comma escaping still precede this (they do; order is preserved).

**Тест (зафиксировать поведение).** tests/test_write_ass.py (or a new tests/test_ass_path.py)::test_ass_path_quote_idiom — assert _ass_path_for_filter("C:/a/Don't panic/burn.ass") produces "C\\:/a/Don'\\''t panic/burn.ass" and, wrapped in single quotes and split by a minimal ffmpeg-quote parser, round-trips back to the original path. Add a serve-level test rendering a session whose stem contains an apostrophe with burn.enabled and assert the ass_path is non-None and render() is invoked (mock ff) without an escaping-mangled path.

---
## 60. HDR/10-bit (PQ/HLG) input is force-truncated to 8-bit yuv420p with no tonemap — washed-out gray output; also distorts burned subtitles/overlays
🟡 MEDIUM · симптом 3 · effort M · `hdr-tonemap-to-sdr`

- **Файл / якорь:** `vpipe/render.py` — video_encoder_args pix_fmt ~148/152; vpre build ~983-992; probe.py MediaInfo ~12-51
- **Закрывает находки:** vpipe/render.py:148, vpipe/render.py:152, vpipe/probe.py:41

**Проблема.** Both encoder-arg builders end with `-pix_fmt yuv420p` (render.py:148 NVENC, 152 x264) and nothing probes color_transfer/primaries or inserts a tonemap. A BT.2020/PQ (HDR10) or HLG source (iPhone HDR, some screen recorders) has its 10-bit values bit-depth-truncated to 8-bit while retaining PQ/BT.2020 coding — on SDR playback paths (YouTube treats 8-bit AVC as SDR; most players ignore PQ tags on 8-bit H.264) the result is the classic washed-out, desaturated gray. Burned subtitles and RGBA enrich overlays are composited assuming SDR onto PQ-coded frames, so overlay colors are distorted too. probe.py doesn't even capture color metadata, so nothing downstream can react.

**Почему так.** Turns silently-wrong washed-out HDR output into correctly tonemapped SDR, and (as a side effect of forcing re-encode through the chain) makes burned subtitles/overlays composite on correct colors.

**Текущий код:**
```
        args += ["-rc-lookahead", "32", "-spatial-aq", "1", "-temporal-aq", "1",
                 "-b_ref_mode", "middle", "-bf", "3",
                 "-profile:v", "high", "-pix_fmt", "yuv420p", *_CFR]
        return args
    x = r.x264
    return ["-c:v", "libx264", "-crf", str(x.crf), "-preset", x.preset,
            "-profile:v", "high", "-pix_fmt", "yuv420p", *_CFR]

    # probe.py MediaInfo has no color/pix_fmt fields
    # render.py vpre build (983-992): crop_filter/scale/fps only
```

**Изменение:**
```
1) probe.py — add `color_trc: str = ''`, `color_primaries: str = ''`, `pix_fmt: str = ''` to the MediaInfo dataclass and read them in probe_media: `color_trc=v.get('color_transfer','') if v else ''`, `color_primaries=v.get('color_primaries','') if v else ''`, `pix_fmt=v.get('pix_fmt','') if v else ''`.

2) render.py — add a helper and prepend a tonemap to the video chain when the source is HDR. Near the vpre build (render.py:983):

    _HDR_TRC = {"smpte2084", "arib-std-b67"}   # PQ (HDR10) / HLG
    def _is_hdr(media) -> bool:
        return str(getattr(media, "color_trc", "")).lower() in _HDR_TRC

    # inside render(), before appending crop_filter/scale to vpre:
    if _is_hdr(media):
        vpre.append("zscale=t=linear:npl=100,tonemap=hable,"
                    "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
        log("  HDR-источник (PQ/HLG) — тонмаппинг в SDR (BT.709).")

   Because a non-empty vpre makes vpost non-empty, this correctly forces the re-encode path and bypasses the video-copy fast-path (render.py:1078/1083) — otherwise an HDR source with no cuts would copy the untonemapped stream. If ff lacks the zscale filter (ff.has_filter('zscale') is False), fall back to logging a UI warning and skip (do not hard-fail). tonemap requires libzimg; guard with has_filter.
```

**Риск / безопасность.** Medium. Only triggers when color_trc is exactly PQ/HLG, so all SDR (BT.709) sources — the vast majority — get zero change and keep every fast-path byte-for-byte. Adding fields to MediaInfo is additive (defaults empty) and safe for existing constructions, but check for any positional MediaInfo(...) instantiations in tests/fixtures and update them (keyword construction in probe_media is fine). zscale/tonemap can be slow and needs libzimg — gate on ff.has_filter('zscale') and degrade to a warning so a build without zimg doesn't crash. npl=100 is a reasonable default; expose later if needed. Verify the tonemap sits BEFORE crop/scale so it operates on full-range source, and format=yuv420p at its end keeps the downstream chain 8-bit.

**Тест (зафиксировать поведение).** tests/test_serve_render.py::test_hdr_source_gets_tonemap — construct a MediaInfo with color_trc='smpte2084' and assert render() (via FakeFF, has_filter('zscale')=True) emits a graph whose video chain begins with 'zscale=t=linear' and 'tonemap=hable', and that the copy fast-path is bypassed (encoder != 'copy'). test_sdr_source_unchanged — color_trc='bt709' → no zscale in the graph, fast-paths intact. test_hdr_without_zscale_warns — has_filter False → no crash, warning logged, no tonemap.

---
## 61. PUT /api/cutlist accepts NaN/Infinity/out-of-range/missing-key segments; a NaN poisons cutlist.json and bricks GET /api/cutlist with a permanent 500
🟡 MEDIUM · симптом 4 · effort M · `cutlist-validate-put`

- **Файл / якорь:** `serve.py` — put_cutlist, serve.py 2227-2237
- **Закрывает находки:** serve.py:2231, serve.py:2231 (edge-cases/medium dup)

**Проблема.** put_cutlist feeds the raw body to CutList.from_dict, which does d['id'], float(d['start']), float(d['end']) with no checks (models.py:130-134). A missing key -> KeyError -> raw 500; a JSON-literal NaN/Infinity (json.loads permits it) is accepted and re-serialized into cutlist.json by save_json (allow_nan default True). Every later GET /api/cutlist then renders via Starlette JSONResponse (allow_nan=False) -> ValueError -> 500 that survives restart (Session.__init__ reloads the file at serve.py:255-256) and even survives re-detection if the poisoned segment is type 'manual' (carried forward at serve.py:279). Sibling endpoints /api/clips/save, /api/clips/render, /api/enrich/save all guard this exact class; this one does not.

**Почему так.** Pre-validating the raw dicts (rather than after from_dict) gives clean per-index 400s and prevents the KeyError-500 and the persisted-NaN brick. Whitelisting id/type/action as non-empty strings (not an enum) prevents the KeyError without over-constraining the many detection types the frontend legitimately sends. math is already imported (line 17).

**Текущий код:**
```
@app.put("/api/cutlist")
def put_cutlist(payload: dict = Body(...)):
    s = S()
    _guard_no_task()
    cl = CutList.from_dict(payload)
    cl.duration = s.media.duration
    cl.source = str(s.inp)
    s.cutlist = cl
    cl.save_json(s.cutlist_path)
    save_txt(cl, s.out_dir / f"{s.inp.stem}.cutlist.txt")
    return {"ok": True, "segments": len(cl.segments)}
```

**Изменение:**
```
@app.put("/api/cutlist")
def put_cutlist(payload: dict = Body(...)):
    s = S()
    _guard_no_task()
    # Validate every segment BEFORE from_dict, which does d["id"]/float(d[...])
    # with no checks (KeyError-500 on a missing key) and silently accepts the
    # NaN/±Infinity literals json.loads permits. An unguarded NaN is written
    # into cutlist.json by save_json (allow_nan default True) and then bricks
    # every later GET /api/cutlist (JSONResponse uses allow_nan=False -> 500),
    # a 500 that survives restart. Mirror the /api/clips/render guard.
    raw_segs = payload.get("segments", [])
    if not isinstance(raw_segs, list):
        raise HTTPException(400, "segments должен быть списком")
    duration = float(s.media.duration)
    for i, d in enumerate(raw_segs):
        if not isinstance(d, dict):
            raise HTTPException(400, f"Сегмент №{i + 1}: ожидается объект")
        for k in ("id", "type", "action"):
            if not isinstance(d.get(k), str) or not d.get(k):
                raise HTTPException(
                    400, f"Сегмент №{i + 1}: поле «{k}» должно быть непустой строкой")
        try:
            start, end = float(d.get("start")), float(d.get("end"))
        except (TypeError, ValueError):
            raise HTTPException(
                400, f"Сегмент №{i + 1}: start/end должны быть числами (секунды)")
        if not (math.isfinite(start) and math.isfinite(end)):
            raise HTTPException(
                400, f"Сегмент №{i + 1}: start/end должны быть конечными числами")
        if end <= start:
            raise HTTPException(400, f"Сегмент №{i + 1}: пустой/обратный диапазон")
        if start < -0.001 or end > duration + 0.001:
            raise HTTPException(400, f"Сегмент №{i + 1}: границы вне ролика")
        # Clamp to [0, duration] so past-EOF trims can't reach ffmpeg; from_dict
        # re-reads these mutated values below.
        d["start"], d["end"] = max(0.0, start), min(duration, end)
    cl = CutList.from_dict(payload)
    cl.duration = s.media.duration
    cl.source = str(s.inp)
    s.cutlist = cl
    cl.save_json(s.cutlist_path)
    save_txt(cl, s.out_dir / f"{s.inp.stem}.cutlist.txt")
    return {"ok": True, "segments": len(cl.segments)}
```

**Риск / безопасность.** New 400s where garbage previously slipped to 200 — the intended tightening, matching /api/clips/render. Browser UI round-trips (finite, ordered, keyed segments) are unaffected. Two safety notes: (1) end<=start is rejected; merge_intervals/resolve already dropped such segments from render math (b>a guard), so no render path relied on accepting them — if you fear a legit zero-length placeholder, DROP such segments instead of 400 (audit found none). (2) The clamp mutates the incoming dict; from_dict re-reads it so persisted JSON holds clamped values — desired. No existing test PUTs NaN to /api/cutlist.

**Тест (зафиксировать поведение).** Add tests mirroring tests/test_api_clips.py:test_render_bad_body_400. Use client.put('/api/cutlist', content='<raw>', headers={'Content-Type':'application/json'}) for the literals json= cannot emit: {"segments":[{"id":"x","start":NaN,"end":10,"type":"manual","action":"remove"}]}, ..."start":-Infinity..., ..."end":Infinity...; plus json= bodies for missing 'id', missing 'type', start>=end, start<0, end>duration -> ALL 400. Then a valid single-segment PUT -> 200 AND a follow-up GET /api/cutlist -> 200 (proves no poisoning). Needs a fake session with media.duration, inp, out_dir, cutlist_path, task={'running':False} like test_api_clips FakeSession.

---
## 62. Same-stem outputs collide: fixed-name chapters.txt/metadata.txt and same-stem queue mp4s silently overwrite across videos
🟡 MEDIUM · симптом 4 · effort ? · `same-stem-output-collision`

- **Файл / якорь:** `serve.py` — _run_render_pipeline chapters/metadata paths (~L1303, L1320, L1326); _queue_process_one before render (~L1572)
- **Закрывает находки:** serve.py:1110, serve.py:1303, serve.py:1326
- **Зависит от:** `output-filename-double-stem`

**Проблема.** Even with the cutlist guarded (cutlist-source-guard patch) and stems un-truncated (output-filename patch), two different videos sharing a stem still overwrite each other's outputs. Two vectors: (a) chapters.txt and metadata.txt use FIXED names in out_dir (out_dir/'chapters.txt', out_dir/'metadata.txt') — they collide for ANY two videos rendered into the same out_dir, not just same-stem ones. (b) The batch queue defaults every job to the shared out_dir (serve.py:2899) and derives the mp4 base purely from the stem, so job 2 destroys job 1's out/<stem>.mp4 and both jobs report 'done' with the identical path.

**Почему так.** The fixed-name txt collision is a clear, contained correctness bug fixed by stem-keying (reusing _with_ext from the output-filename patch). The mp4 overwrite is confined to the real vector (the back-to-back batch queue) via a per-output source marker + hash-suffix uniquify, so the interactive editor keeps its expected overwrite-on-re-render semantics and no output filename changes unless a genuine cross-video collision is detected.

**Текущий код:**
```
            cr = chapters_mod.generate(tr, removed, cfg.chapters,
                                       out_dir / "chapters.txt", llm=s.llm,
...
            chapters_txt = out_dir / "chapters.txt"
            mr = metadata_mod.generate(
                tr, removed, cfg.metadata, s.llm,
                chapters_path=chapters_txt if chapters_txt.exists() else None,
...
                meta_path = out_dir / "metadata.txt"
```

**Изменение:**
```
(a) Stem-key the txt sidecars so they can't collide across videos. Introduce a local `sc = sidecar_base or base` at the top of the subtitles/chapters section and derive:
    chapters_out = _with_ext(sc, ".chapters.txt")   # was out_dir/'chapters.txt'
    ... chapters_mod.generate(..., chapters_out, ...)
    ... chapters_txt = chapters_out
    ... meta_path = _with_ext(sc, ".metadata.txt")   # was out_dir/'metadata.txt'
(The results dict already returns the real path via cr.get('path')/meta_path.resolve(), and the frontend link() uses that path with a static 'chapters.txt' label — app.js:2910-2911 — so no UI contract breaks.)

(b) Queue-side collision guard in _queue_process_one, right after `cfg, scale_h, fps, out_dir, base = _resolve_render_opts(ls, job.render_opts)` (~L1572):
    mp4 = _with_ext(base, ".mp4")
    marker = _with_ext(base, ".source")
    prior = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    if mp4.exists() and prior and prior != ls.audio_hash:
        base = base.parent / f"{base.name}-{ls.audio_hash[:6]}"   # uniquify, don't clobber
        job.stage = f"Имя занято другим видео — сохраняю как {base.name}.mp4"
    try:
        _with_ext(base, ".source").write_text(ls.audio_hash, encoding="utf-8")
    except OSError:
        pass
Pass the (possibly uniquified) base into _run_render_pipeline as today, and add the note to job.result (e.g. result['renamed']=base.name when uniquified) so the queue surfaces it.
```

**Риск / безопасность.** Medium blast radius. (a) Renaming chapters.txt/metadata.txt: verify no consumer hardcodes those names — checked app.js (uses the returned path, label only) and /api/output (serves by path); the preview endpoints use work_dir/preview_chapters.txt (separate). Old out/chapters.txt files become orphaned (harmless). (b) The queue marker adds two tiny sidecar writes per job; a uniquified base means the mp4 lands at <stem>-<hash6>.mp4 — surfaced in job.stage/result so it's not silent. Keep the editor path (do_render) unchanged: it must still overwrite on re-render of the same clip. depends_on output-filename for _with_ext.

**Тест (зафиксировать поведение).** tests/test_queue.py: test_same_stem_different_source_no_clobber — enqueue two jobs whose paths differ but share a stem (stub probe_media to give distinct audio_hash), monkeypatch render to touch the out mp4; run the worker; assert two distinct mp4 files exist and job2.result flags the rename. tests/test_serve_render.py: test_chapters_metadata_stem_keyed — enable chapters+metadata (stub generators), render with sidecar_base=out/'vidA'; assert files are out/'vidA.chapters.txt' / out/'vidA.metadata.txt', not out/'chapters.txt'.

---
## 63. Session lifecycle (probe, cached transcript, stale cutlist reload, auto-detect on open, manual-cut preservation on re-detect) and /api/open, /api/upload have zero tests
🟡 MEDIUM · симптом 4 · effort M · `session-lifecycle-tests`

- **Файл / якорь:** `tests/test_session_lifecycle.py` — serve.Session.__init__ (serve.py:210-259) + Session._detect manual-cut preservation (serve.py:277-283) + /api/open (2039) /api/upload (2054)
- **Закрывает находки:** serve.py:210, serve.py:255, serve.py:277, serve.py:2039, serve.py:2054

**Проблема.** Every API test builds a SimpleNamespace stand-in or monkeypatches serve.SESSION; nothing touches Session.__init__ or open_session. So cached-cutlist reuse vs fresh detect (serve.py:255-259, _ctor_fresh_detect), re-open after edit, and manual-cut preservation on re-detect (277-283, which extends the NEW cutlist with old TYPE_MANUAL segments regardless of validity) are all unexercised — symptom-4 'stale state / re-runs' lives here.

**Текущий код:**
```
# No test constructs a real Session. Session.__init__ calls FFmpeg(cfg.ffmpeg) (resolve_bin needs a
# real binary), probe_media, hash_input — all must be monkeypatched to test hermetically.
# The closest existing pattern is the SimpleNamespace stand-ins in test_caption_presets.py:143-150.
```

**Изменение:**
````
New file tests/test_session_lifecycle.py. Monkeypatch the ctor's heavy collaborators so no ffmpeg/probe runs.

```python
from pathlib import Path
from types import SimpleNamespace
import pytest
import serve
from vpipe.config import load_config
from vpipe.models import CutList, CutSegment, TYPE_MANUAL, TYPE_PAUSE, ACTION_REMOVE, Transcript, Segment, Word

@pytest.fixture()
def ctor_env(tmp_path, monkeypatch):
    cfg=load_config('config.yaml')
    cfg.paths.cache_dir=str(tmp_path/'cache'); cfg.paths.work_dir=str(tmp_path/'work')
    monkeypatch.setattr(serve, 'FFmpeg', lambda c: SimpleNamespace())
    monkeypatch.setattr(serve, 'probe_media', lambda ff,p: SimpleNamespace(duration=100.0,width=1920,height=1080,fps=30.0,has_audio=True))
    monkeypatch.setattr(serve, 'hash_input', lambda p: 'deadbeefcafe')
    monkeypatch.setattr(serve, 'get_client', lambda c: None)
    vid=tmp_path/'clip.mp4'; vid.write_bytes(b'\x00')
    return cfg, vid, tmp_path

def _tr():
    w=[Word('раз',0.0,0.4),Word('два',0.5,0.9)]
    return Transcript(language='ru',duration=100.0,model='t',audio_hash='deadbeefcafe',
                      segments=[Segment(0.0,0.9,'раз два',w)])

# 1) fresh session, no cache -> transcript None, no cutlist, no detect

def test_ctor_blank(ctor_env, monkeypatch):
    cfg,vid,tmp=ctor_env
    s=serve.Session(str(vid), cfg, str(tmp/'out'), False)
    assert s.transcript is None and s.cutlist is None and s._ctor_fresh_detect is False

# 2) cached transcript present, no cutlist -> ctor auto-detects once (fresh flag True)

def test_ctor_fresh_detect_from_cached_transcript(ctor_env, monkeypatch):
    cfg,vid,tmp=ctor_env
    (Path(cfg.paths.cache_dir)).mkdir(parents=True,exist_ok=True)
    _tr().save(Path(cfg.paths.cache_dir)/'deadbeefcafe.transcript.json')
    called={'n':0}
    def fake_detect(tr,c,fill,prof,**kw):
        called['n']+=1; return CutList(source='clip.mp4',duration=100.0,segments=[])
    monkeypatch.setattr(serve,'run_detection',fake_detect)
    s=serve.Session(str(vid), cfg, str(tmp/'out'), False)
    assert s.transcript is not None and s._ctor_fresh_detect is True and called['n']==1

# 3) on-disk cutlist present -> ctor loads it, does NOT detect

def test_ctor_loads_existing_cutlist(ctor_env, monkeypatch):
    cfg,vid,tmp=ctor_env
    out=Path(tmp/'out'); out.mkdir(parents=True,exist_ok=True)
    CutList(source='clip.mp4',duration=100.0,segments=[
        CutSegment(id='m0',start=10.0,end=12.0,type=TYPE_MANUAL,action=ACTION_REMOVE,enabled=True)]
        ).save_json(out/'clip.cutlist.json')
    monkeypatch.setattr(serve,'run_detection',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not detect')))
    s=serve.Session(str(vid), cfg, str(tmp/'out'), False)
    assert s.cutlist is not None and s._ctor_fresh_detect is False
    assert any(seg.type==TYPE_MANUAL for seg in s.cutlist.segments)

# 4) _detect preserves manual cuts across a re-detection

def test_detect_preserves_manual_cuts(ctor_env, monkeypatch):
    cfg,vid,tmp=ctor_env
    s=serve.Session(str(vid), cfg, str(tmp/'out'), False)
    s.transcript=_tr()
    s.cutlist=CutList(source='clip.mp4',duration=100.0,segments=[
        CutSegment(id='m0',start=50.0,end=52.0,type=TYPE_MANUAL,action=ACTION_REMOVE,enabled=True)])
    monkeypatch.setattr(serve,'run_detection',lambda *a,**k: CutList(source='clip.mp4',duration=100.0,
        segments=[CutSegment(id='p0',start=5.0,end=6.0,type=TYPE_PAUSE,action=ACTION_REMOVE,enabled=True)]))
    cl=s._detect()
    types={seg.type for seg in cl.segments}
    assert TYPE_MANUAL in types and TYPE_PAUSE in types      # manual survived the re-detect
    starts=[seg.start for seg in cl.segments]
    assert starts==sorted(starts)                            # re-sorted by start
```
Also add lightweight endpoint tests for /api/open (404 on missing path, 400 on non-video ext, ok path calls open_session) and /api/upload (415 on bad ext) using TestClient with open_session monkeypatched.
````

**Риск / безопасность.** Session.__init__ has several collaborators to intercept (FFmpeg, probe_media, hash_input, get_client, run_detection via _detect); miss one and the test hits a real binary. Verify the exact imported names in serve.py (FFmpeg from vpipe.ffmpeg_utils, probe_media/hash_input/extract_audio from vpipe.probe, run_detection from vpipe.detect). _detect writes cutlist.json + a .txt via save_txt into out_dir/work_dir — use tmp_path so nothing lands in the repo. Do not assert on save side effects beyond existence.

**Тест (зафиксировать поведение).** test_detect_preserves_manual_cuts: seed a session cutlist with a TYPE_MANUAL cut, monkeypatch run_detection to return only auto (TYPE_PAUSE) cuts, call _detect(), assert the result contains BOTH the manual and auto segments and is sorted by start — locks serve.py:277-283 against a regression that drops user cuts on re-detect.

---
## 64. Make «Очистить завершённые» clear only finished jobs and report them truthfully
🟡 MEDIUM · симптом 5 · effort S · `ui-clear-frontend-send-statuses`

- **Файл / якорь:** `web/app.js` — clearQueue(), ~line 4580-4583
- **Закрывает находки:** web/app.js:4580, web/app.js:4515, web/index.html:700
- **Зависит от:** `ui-clear-server-statuses-filter`

**Проблема.** clearQueue() posts /api/queue/clear with an empty body, invoking the wipe-all-non-running behavior, and only reports it after the fact via the toast `Удалено заданий: ${r.removed}` — with no confirm(). The button label promises 'clear completed', not 'clear the whole queue'.

**Почему так.** Sends the new `statuses` filter so the button matches its label — done/error jobs are pruned, pending/running survive. queuePost already JSON-stringifies the body (app.js:4543-4550), so no plumbing changes. Once behavior is honest, no destructive confirm() is needed (no data loss), and the label «Очистить завершённые» at index.html:700 becomes accurate — so index.html requires NO edit. The toast now also gives positive feedback when there was nothing to clear, instead of silently doing nothing.

**Текущий код:**
```
async function clearQueue() {
  const r = await queuePost('/api/queue/clear', {})
  if (r && r.ok) { if (r.removed) toast(`Удалено заданий: ${r.removed}`, 'info'); loadQueueList() }
}
```

**Изменение:**
```
async function clearQueue() {
  // «Очистить завершённые» — clear ONLY finished jobs (done/error); pending
  // batch jobs the user queued (e.g. for an overnight run) stay untouched.
  const r = await queuePost('/api/queue/clear', { statuses: ['done', 'error'] })
  if (r && r.ok) {
    toast(r.removed ? `Очищено завершённых: ${r.removed}` : 'Завершённых заданий нет', 'info')
    loadQueueList()
  }
}
```

**Риск / безопасность.** Depends on ui-clear-server-statuses-filter being deployed first; if the frontend ships against an old server, the old server ignores the body and wipes all non-running jobs (i.e. reverts to today's buggy behavior — no NEW breakage, just no fix). No frontend tests exist to break. Individual pending-job removal still works via removeQueueJob()/#btnQueueClear per-row delete, so users retain a way to drop pending jobs deliberately. If product later wants a true 'nuke everything' control, add a separate explicit button/confirm rather than overloading this one.

**Тест (зафиксировать поведение).** No JS test harness exists in the repo, so lock behavior at the API layer via the backend test in ui-clear-server-statuses-filter. Optionally add a lightweight assertion in a future frontend smoke test that clearQueue() posts body {statuses:['done','error']} (e.g. stub queuePost and assert the second arg). Manual verification: queue 3 pending + render 1 to done, click «Очистить завершённые» → only the done job disappears, pending jobs remain and survive reload.

---
## 65. Give /api/queue/clear an optional statuses filter so «clear completed» can spare pending jobs
🟡 MEDIUM · симптом 5 · effort S · `ui-clear-server-statuses-filter`

- **Файл / якорь:** `serve.py` — queue_clear() endpoint, ~line 2945-2953 (@app.post("/api/queue/clear"))
- **Закрывает находки:** web/index.html:700, web/app.js:4580, serve.py:2945

**Проблема.** POST /api/queue/clear unconditionally drops EVERY non-running job (pending/done/error). The only UI that calls it is the button labelled «Очистить завершённые» (clear completed), so a user pruning finished entries before starting an overnight batch silently loses all queued PENDING jobs, and the change is persisted to disk (_save_queue) so it is not undoable.

**Почему так.** Adds a filter path without changing the default. When `statuses` is absent (existing callers, existing tests that POST with no JSON body), the exact original list comprehension runs, so removed counts and persisted state are byte-for-byte identical. When `statuses` is supplied, only jobs whose status is in the set are dropped, and the explicit `j.status == "running"` guard keeps a running job safe even if a caller foolishly includes "running". `Body` is already imported (used by queue_remove at ~line 2935) and `Optional` is imported at serve.py:35, so no new imports.

**Текущий код:**
```
@app.post("/api/queue/clear")
def queue_clear():
    """Drop every job that isn't currently running (pending/done/error)."""
    with QUEUE_LOCK:
        before = len(QUEUE)
        QUEUE[:] = [j for j in QUEUE if j.status == "running"]
        removed = before - len(QUEUE)
    _save_queue()
    return {"ok": True, "removed": removed}
```

**Изменение:**
```
@app.post("/api/queue/clear")
def queue_clear(body: Optional[dict] = Body(default=None)):
    """Drop finished/queued jobs.

    Back-compat: with no body, clears every non-running job (pending/done/error).
    Pass {"statuses": ["done", "error"]} to clear ONLY those statuses, so pending
    batch jobs the user queued survive — this is what the «Очистить завершённые»
    button sends. Running jobs are never removed.
    """
    statuses = None
    if isinstance(body, dict) and isinstance(body.get("statuses"), list):
        statuses = {str(s) for s in body["statuses"]}
    with QUEUE_LOCK:
        before = len(QUEUE)
        if statuses is None:
            QUEUE[:] = [j for j in QUEUE if j.status == "running"]
        else:
            QUEUE[:] = [j for j in QUEUE
                        if j.status == "running" or j.status not in statuses]
        removed = before - len(QUEUE)
    _save_queue()
    return {"ok": True, "removed": removed}
```

**Риск / безопасность.** Must preserve the no-body fast-path exactly — two tests lock it: tests/test_queue.py:139 (test_queue_clear_keeps_running_only, no body → removed==3, keeps only 'r') and tests/test_queue_persist.py:278 (test_clear_persists_to_disk, no body → queue becomes []). Both hit the `statuses is None` branch unchanged, so they pass. Only real risk: FastAPI treating the added body param as required and 422-ing an empty POST — Body(default=None) makes it optional so TestClient's bodyless POST yields None; verify by running those two tests. This is a mutating POST /api/* (same CSRF posture as before); adding a JSON body does not change it, and the frontend already sends Content-Type: application/json via queuePost.

**Тест (зафиксировать поведение).** Add tests/test_queue.py::test_queue_clear_statuses_filter_keeps_pending — build QUEUE = [pending 'p', running 'r', done 'd', error 'e'] via monkeypatch (mirroring test_queue_clear_keeps_running_only), POST /api/queue/clear with json={"statuses": ["done", "error"]}, assert r.json()["removed"] == 2 and [j.id for j in serve.QUEUE] == ["p", "r"]. Keep the existing test_queue_clear_keeps_running_only (no body) unchanged to lock the default branch.

---
## 66. Clamp the «Вырезать» floating button into the viewport so it doesn't vanish on large/auto-scrolled selections
⚪ LOW · симптом 5 · effort S · `cutfloat-clamp-viewport`

- **Файл / якорь:** `web/app.js` — onSelectionChange() positioning (app.js:1832-1834)
- **Закрывает находки:** web/app.js:1833

**Проблема.** onSelectionChange places the position:fixed #cutFloat at (r.top - 42) with no clamping. range.getBoundingClientRect() ignores scroll-container clipping, so when the user drag-selects a passage taller than the visible transcript pane and #transcript autoscrolls, the selection's first line scrolls above the viewport, r.top goes negative, and the button renders entirely off-screen — the primary discoverable cut action disappears exactly on large selections, leaving only the non-obvious X hotkey. Right-edge clipping is also possible in the stacked ≤900px layout.

**Почему так.** Keeps the primary discoverable text-cut affordance on-screen for exactly the large-selection case where it currently disappears, with a self-contained clamp.

**Текущий код:**
```
  st.selRange = [lo, hi]
  const r = range.getBoundingClientRect()
  float.style.left = (r.left + r.width / 2 - 45) + 'px'; float.style.top = (r.top - 42) + 'px'
  float.classList.remove('hidden')
```

**Изменение:**
```
  st.selRange = [lo, hi]
  const r = range.getBoundingClientRect()
  float.classList.remove('hidden')            // un-hide first so offsetWidth/Height are measurable
  const fw = float.offsetWidth || 90, fh = float.offsetHeight || 34
  let left = r.left + r.width / 2 - fw / 2
  let top = r.top - fh - 8
  if (top < 8) top = r.bottom + 8             // clipped above → flip below the selection
  left = Math.max(8, Math.min(left, window.innerWidth - fw - 8))
  top = Math.max(8, Math.min(top, window.innerHeight - fh - 8))
  float.style.left = left + 'px'; float.style.top = top + 'px'
```

**Риск / безопасность.** Reordering remove('hidden') before measuring is required for a non-zero offsetWidth; the fallback (90/34) covers the one frame before layout. Uses fw/2 instead of the hardcoded 45 offset, generalizing the original centering. position:fixed with no ancestor transform (style.css:218 confirms) means viewport clamping is correct. Runs only on selectionchange, so the extra layout read is negligible. Preserve the existing hide branches (collapsed/out-of-transcript selections) untouched.

**Тест (зафиксировать поведение).** Layout math is not reliably testable in jsdom (offsetWidth=0). Extract the clamp into a pure helper `clampFloat({top,bottom,left,width}, vw, vh, fw, fh) => {left, top}` and unit-test it (vitest): a negative r.top flips below (top===bottom+8); left never < 8 nor > vw-fw-8. A Playwright smoke test (select a tall passage that autoscrolls, assert #cutFloat is within the viewport) would lock the real integration.

---
## 67. Close the double-click window on task-start buttons so the 2nd POST doesn't produce a spurious red 409 toast
⚪ LOW · симптом 5 · effort S · `task-start-double-click-guard`

- **Файл / якорь:** `web/app.js` — clipsRender (app.js:1504), loadChapters (812), loadMetadata (971), loadClips (1124)
- **Закрывает находки:** web/app.js:1504, web/app.js:812, web/app.js:971, web/app.js:1124

**Проблема.** clipsRender/loadChapters/loadMetadata/loadClips guard with `if (st.task) return`, but st.task is set only in setRunning() inside followTask() AFTER the POST resolves, and the buttons are never disabled at click time. A fast double-click or Enter auto-repeat passes the guard twice; the server rejects the 2nd with 409 (no duplicate task actually starts — start_task's TASK_LOCK at serve.py:302 and _guard_no_task both hold) but failToast surfaces it as a scary red 'Не удалось запустить…'. clipsRender is worst because `await save()` widens the window. (runEnrichSuggest is excluded — closeOverlay hides its trigger before the first await.)

**Почему так.** Eliminates a confusing red error on a race that the server already handles correctly, with a one-flag guard that needs no debounce timing.

**Текущий код:**
```
// clipsRender (app.js:1504):
async function clipsRender() {
  if (st.task) { toast('Дождись завершения текущей задачи', 'info'); return }
  const chosen = st.clipsData.filter((c) => st.clipsSel.has(String(c.id)))
  if (!chosen.length) { toast('Выбери хотя бы один клип — чекбокс на карточке', 'info'); return }
  await save()
  ...
  followTask('render_clips')
}

// loadChapters (app.js:812) — representative of the 3 preview loaders:
async function loadChapters() {
  if (st.task) { toast('Дождись завершения задачи перед предпросмотром глав', 'info'); return }
  let res
  try { res = await fetch('/api/preview/chapters', { method: 'POST' }) }
  ...
  followTask('preview_chapters')
}
```

**Изменение:**
```
Add a synchronous in-flight flag `st._starting` set the moment the guard passes and cleared in a finally (followTask sets st.task synchronously, so after the finally st.task guards further clicks).

clipsRender:
async function clipsRender() {
  if (st.task || st._starting) { toast('Дождись завершения текущей задачи', 'info'); return }
  const chosen = st.clipsData.filter((c) => st.clipsSel.has(String(c.id)))
  if (!chosen.length) { toast('Выбери хотя бы один клип — чекбокс на карточке', 'info'); return }
  st._starting = true
  try {
    await save()
    const stem = ...
    const body = { ... }
    st._clipsRenderIds = ...
    let res
    try { res = await fetch('/api/clips/render', {...}) }
    catch (e) { toast('Сеть: не удалось запустить рендер клипов (' + e.message + ')', 'error'); return }
    if (!res.ok) { await failToast(res, 'Не удалось запустить рендер клипов'); return }
    followTask('render_clips')
  } finally { st._starting = false }
}

Apply the IDENTICAL two-line treatment to loadChapters (812), loadMetadata (971), loadClips (1124): change the guard to `if (st.task || st._starting) {…}`, add `st._starting = true` right after it, and wrap the remaining body (from `let res` through `followTask(...)`) in `try { … } finally { st._starting = false }`.
```

**Риск / безопасность.** Minimal — one boolean. The finally clears _starting on every exit including success (safe: st.task is already set by followTask on success, and stays null on failure so the button is usable again immediately). Do NOT add this to runEnrichSuggest (its overlay-close already prevents re-entry). No change to the server. Keep the existing toast/failToast messages so genuine failures still report.

**Тест (зафиксировать поведение).** Needs jsdom+vitest (web/tests/): mock POST with a delayed resolve; invoke clipsRender() twice synchronously → assert fetch('/api/clips/render') is called exactly once and no error toast is produced. Repeat for loadChapters (Enter auto-repeat scenario).

---
## 68. beforeunload must guard pending/in-flight enrich edits, and a failed flush must not silently drop them
⚪ LOW · симптом 5 · effort S · `enrich-beforeunload`

- **Файл / якорь:** `web/app.js` — beforeunload handler line 180 + enrichFlushSave ~3673-3688 + enrSaveTimer decl ~3665
- **Закрывает находки:** web/app.js:180, web/app.js:3673, web/app.js:3676
- **Зависит от:** `enrich-shallow-merge`

**Проблема.** Enrich edits accumulate in enrPending and flush 350 ms later; the beforeunload guard only checks st.dirty/st.saving, so reloading inside the debounce window (or during the POST) drops edits with no warning. enrichFlushSave calls enrPending.clear() BEFORE the fetch, so a network failure loses the patches (it just toasts).

**Почему так.** Including enrPending.size||enrSaveTimer closes the silent-loss window; resetting enrSaveTimer=0 at entry prevents a stale (fired) timer id from making the guard fire spuriously after a clean save; requeue() routes through enrichQueueSave so patches are preserved AND the debounce re-arms to retry, instead of clear-then-drop.

**Текущий код:**
```
line 180:  if (st.dirty || st.saving) { e.preventDefault(); e.returnValue = ''; return '' }

async function enrichFlushSave() {
  if (!enrPending.size) return
  const items = [...enrPending.values()]
  enrPending.clear()
  let res
  try { res = await fetch('/api/enrich/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ items }) }) }
  catch (e) { toast('network: failed to save montage edits (' + e.message + ')', 'error'); return }
  if (!res.ok) { await failToast(res, 'failed to save montage edits'); return }
```

**Изменение:**
```
(A) line 180 predicate — also warn on pending/armed enrich edits: if (st.dirty || st.saving || enrPending.size || enrSaveTimer) { ... }. (B) enrichFlushSave — reset the armed-timer marker at entry (so a clean flush doesn't leave enrSaveTimer truthy and trip the guard) and re-queue on BOTH failure paths: at top enrSaveTimer = 0 before the size check; define const requeue = () => { for (const it of items) enrichQueueSave(it.id, it) }; on catch -> requeue(); toast('...retrying'); return; on !res.ok -> requeue(); await failToast(...); return.
```

**Риск / безопасность.** LOW. requeue reuses enrichQueueSave coalescing (deep-merge if enrich-shallow-merge is applied; still works shallow otherwise). Slightly more aggressive unload prompts (now also enrich edits) — intended. Success path unchanged.

**Тест (зафиксировать поведение).** jsdom (tests/js/enrichFlush): queue an edit, assert beforeunload predicate true while enrPending.size>0; stub fetch reject, flush, assert items back in enrPending and enrSaveTimer re-armed; stub success, assert enrSaveTimer===0 and predicate false. No harness yet.

---
## 69. Global Enter hotkey must not also re-fire a focused button/link
⚪ LOW · симптом 5 · effort S · `enter-double-fire`

- **Файл / якорь:** `web/app.js` — bindKeys Enter branch ~4316-4320
- **Закрывает находки:** web/app.js:4319, web/app.js:4316

**Проблема.** The Enter branch runs toggleEnabled(st.selected) for any non-field focus and does not preventDefault. A button keeps focus after a click and natively dispatches click on Enter, so after clicking a cut row's jump button (or Redetect), pressing Enter toggles the cut AND re-fires the focused button — re-seeking or re-opening a confirm dialog. inField exempts only inputs, not buttons.

**Почему так.** When an actionable control is focused, Enter belongs to that control (its native click). Skipping the global toggle there removes the double action while keeping the documented Enter=toggle hotkey working whenever focus is on a non-interactive element (the common case after selecting a cut row body).

**Текущий код:**
```
    else if (k === 'Enter') {
      if (activeTab === 'enrich' && st.enrichActive) { e.preventDefault(); enrichPreviewActive() }
      else if (st.selected) toggleEnabled(st.selected)
    }
```

**Изменение:**
```
Bail on the toggle when a button/link has focus (let the control's native Enter be the single action), mirroring the tablist guard at :4303:
    else if (k === 'Enter') {
      if (activeTab === 'enrich' && st.enrichActive) { e.preventDefault(); enrichPreviewActive() }
      else if (st.selected) {
        const ae2 = document.activeElement
        if (ae2 && (ae2.tagName === 'BUTTON' || ae2.tagName === 'A')) return
        toggleEnabled(st.selected)
      }
    }
```

**Риск / безопасность.** LOW. Only suppresses the toggle when focus is literally on BUTTON/A; keyboard users can blur/click the row body to toggle. The enrich-preview Enter branch is untouched.

**Тест (зафиксировать поведение).** jsdom (tests/js/bindKeys): set st.selected, focus a button, dispatch keydown Enter, assert toggleEnabled NOT called and button click fired once; focus body, Enter -> toggleEnabled once. No harness yet.

---
## 70. Transcript search must fold Russian yo->ye so it finds both spellings
⚪ LOW · симптом 5 · effort S · `search-yo-fold`

- **Файл / якорь:** `web/app.js` — doSearch() ~4072-4096 (dictKey at ~2329)
- **Закрывает находки:** web/app.js:4073, web/app.js:4078, web/app.js:4089

**Проблема.** doSearch matches with plain toLowerCase().includes(q) and never folds the Russian yo->ye, while the filler dictionary in the same file normalizes with dictKey (lowercase + yo->ye). Whisper transcripts intermix the two, so searching one spelling silently misses every occurrence of the other in both the virtualized and non-virtualized paths.

**Почему так.** dictKey already encapsulates the exact fold the dedup path uses; reusing it makes search insensitive to the yo/ye variance symmetrically without a new helper. dictKey (2329) is initialized before doSearch runs at user-input time.

**Текущий код:**
```
function doSearch(q) {
  q = q.trim().toLowerCase()
  ...
      const hit = q && sp.textContent.toLowerCase().includes(q)
  ...
  if (q) for (const w of st.words) {
    if (w.word.toLowerCase().includes(q)) { virtMatches.add(w.i); if (firstIdx < 0) firstIdx = w.i }
  }
```

**Изменение:**
```
Route needle and both haystacks through the existing dictKey() (const at app.js:2329, lowercase + yo->ye): line 4073 -> q = dictKey(q.trim()); line 4078 -> const hit = q && dictKey(sp.textContent).includes(q); line 4089 -> if (dictKey(w.word).includes(q)) { ... }.
```

**Риск / безопасность.** MINIMAL. dictKey also lowercases, so case-insensitivity is preserved; the only added effect is the yo/ye fold. No effect on highlight/scroll logic. Pure client, no server contract.

**Тест (зафиксировать поведение).** jsdom (tests/js/doSearch): transcript containing the yo-spelling, doSearch of the ye-spelling, assert the span gets class 'match' and scrolls into view; repeat virt path. No harness yet — manual repro is the given one.

---
## 71. Very short videos (< ~50s): fixed 30s/20s clean zones auto-reject every suggestion; density budget int(2.5*dur/60) is 0 under 24s — the whole plan saves disabled with no UI-level explanation
⚪ LOW · симптом 5 · effort ? · `short-video-clean-zones-and-budget-floor`

- **Файл / якорь:** `vpipe/enrich.py` — plan_render() step 2 clean-zone gate (~line 1205) and step 4c density budget (~line 1264)
- **Закрывает находки:** vpipe/enrich.py:1205, vpipe/enrich.py:1264

**Проблема.** plan_render rejects any overlay with f0<CLEAN_HEAD_S(30) or f1>dur-CLEAN_TAIL_S(20); for dur<=50s the two zones overlap and NO window can ever be accepted. Under 24s the density budget int(2.5*dur/60) is also 0, rejecting everything. Because enrich_suggest runs plan_render before saving, every suggestion persists as off_limits/enabled semantics with only per-item notes — nothing tells the user the clip is simply too short, and re-enabling by hand gets re-disabled at render.

**Почему так.** For any dur>=200s, min(30, 0.15*dur)=30 and min(20, 0.1*dur)=20 — identical to today, so every existing test (durations 210/300) is byte-for-byte unchanged. Only sub-200s clips get proportionally smaller zones, guaranteeing a placeable window; the budget floor lets at least one overlay through on <24s clips. This is the least-invasive of the finding's options (no new toast/banner plumbing).

**Текущий код:**
```
    # 2) чистые зоны / CTA>=60 c / отступ от швов ------------------------------
    placed: list[_Cand] = []
    for c in cands:
        it = c.item
        if c.f0 < CLEAN_HEAD_S or c.f1 > dur - CLEAN_TAIL_S:
            _reject(it, ST_OFF_LIMITS,
                    f"чистая зона: первые {CLEAN_HEAD_S:.0f} c и последние "
                    f"{CLEAN_TAIL_S:.0f} c без оверлеев")
            continue
...
    # 4c. общая плотность: <=2.5 оверлея/мин (жёсткий потолок 4) — трим по score.
    budget = int(min(OVERLAYS_PER_MIN, OVERLAYS_PER_MIN_HARD) * dur / 60.0)
```

**Изменение:**
```
Add two fraction constants next to CLEAN_HEAD_S/CLEAN_TAIL_S (~lines 67-68):
    CLEAN_HEAD_FRAC = 0.15    # на коротком ролике зона головы = min(30 c, 15% ролика)
    CLEAN_TAIL_FRAC = 0.10    # хвоста = min(20 c, 10% ролика)

Step 2 — scale the zones by duration:
    # 2) чистые зоны / CTA>=60 c / отступ от швов ------------------------------
    # На КОРОТКИХ роликах фиксированные 30/20 c зоны перекрываются и отвергают
    # ЛЮБОЙ оверлей (весь план off_limits, «Включено 0 из N»). Масштабируем зоны
    # под длину: не больше 15%/10% ролика — окно всегда существует.
    clean_head = min(CLEAN_HEAD_S, CLEAN_HEAD_FRAC * dur)
    clean_tail = min(CLEAN_TAIL_S, CLEAN_TAIL_FRAC * dur)
    placed: list[_Cand] = []
    for c in cands:
        it = c.item
        if c.f0 < clean_head or c.f1 > dur - clean_tail:
            _reject(it, ST_OFF_LIMITS,
                    f"чистая зона: первые {clean_head:.0f} c и последние "
                    f"{clean_tail:.0f} c без оверлеев")
            continue

Step 4c — floor the budget at 1:
    # 4c. общая плотность: <=2.5 оверлея/мин (жёсткий потолок 4) — трим по score.
    #     max(1,…): на очень коротком ролике (<24 c) budget иначе = 0 и режет ВСЁ.
    budget = max(1, int(min(OVERLAYS_PER_MIN, OVERLAYS_PER_MIN_HARD) * dur / 60.0))
```

**Риск / безопасность.** LOW. Zero regression for dur>=200s (the regime of all current tests). Verify no test asserts the literal 30/20 note string for a <200s duration. _plan_punches still uses the fixed CLEAN_HEAD_S/CLEAN_TAIL_S — intentionally left (punches are optional polish and simply won't appear on tiny clips). If product wants the explicit 'ролик слишком короткий' banner, that's a follow-up UI patch, not this one.

**Тест (зафиксировать поведение).** tests/test_enrich_plan.py: add test_short_video_scaled_clean_zones — Timeline([], duration=45), one img at t=22 (dur small so it fits the middle); assert it is accepted (re.stills len 1, status ST_OK) rather than off_limits. Add test_budget_floor_on_tiny_clip — duration=20, one candidate → budget max(1,0)=1 → the single item survives density trim. Add a guard test that duration=300 still uses 30/20 zones (an item at t=25 is rejected).

---
## 72. Cap the filter_complex graph embedded in FFmpegError so a long-video failure produces a readable toast, not a 25-30KB wall
⚪ LOW · симптом 2 · effort S · `ffmpegerror-graph-cap`

- **Файл / якорь:** `vpipe/ffmpeg_utils.py` — FFmpeg.run() error assembly, ~line 197-204
- **Закрывает находки:** vpipe/ffmpeg_utils.py:198
- **Зависит от:** `r2-filter-complex-script`

**Проблема.** On any ffmpeg failure, run() appends the FULL filtergraph to the exception text (ffmpeg_utils.py:198-204). For a long many-cut video the graph is ~18-30KB; it becomes task['error'] verbatim, is emitted as a ~25-30KB SSE payload, and app.js renders it as a bottom-anchored toast with no max-height/overflow — the actual one-line ffmpeg error is pushed unscrollably off-screen and the toast auto-dismisses after 6s. Real long-video failures become undiagnosable from the UI.

**Почему так.** Keeps the failure toast usable on exactly the long-video renders that are most likely to fail, without dropping the real ffmpeg error line.

**Текущий код:**
```
            parts = [f"{desc} failed (exit {proc.returncode}). ffmpeg said:\n{detail}"]
            if "-filter_complex" in args:
                try:
                    graph = args[args.index("-filter_complex") + 1]
                    parts.append(f"\nfilter_complex graph:\n{graph}")
                except (IndexError, ValueError):
                    pass
            raise FFmpegError("".join(parts))
```

**Изменение:**
```
Cap the embedded graph to a head+tail excerpt (and, since P1 already writes big graphs to a temp script, optionally reference that path). Minimal change:

            parts = [f"{desc} failed (exit {proc.returncode}). ffmpeg said:\n{detail}"]
            if "-filter_complex" in args:
                try:
                    graph = args[args.index("-filter_complex") + 1]
                    if len(graph) > 1200:
                        graph = (graph[:600] + f"\n… [{len(graph) - 1200} chars omitted] …\n"
                                 + graph[-600:])
                    parts.append(f"\nfilter_complex graph:\n{graph}")
                except (IndexError, ValueError):
                    pass
            raise FFmpegError("".join(parts))

Because `detail` (the actual ffmpeg stderr head+tail) is placed FIRST, the diagnostic one-liner is always at the top of the message even after truncation.
```

**Риск / безопасность.** Very low. Short graphs (<=1200 chars) are unchanged, so existing error-message tests and small-render diagnostics are untouched. The ffmpeg stderr `detail` is never truncated — only the reconstructable graph is capped — so no real diagnostic information is lost (and with P1 the full graph is also on disk as the .ffscript temp during the run, and could be referenced). Consider also capping task['error'] length server-side before SSE serialization as a defense-in-depth (serve.py), but the run()-side cap is sufficient for this cluster.

**Тест (зафиксировать поведение).** tests/test_ffmpeg_runner.py::test_ffmpegerror_caps_graph — invoke run() against a failing fake ffmpeg with a >5000-char -filter_complex; assert the raised FFmpegError message length is bounded (< ~3KB), contains the omitted-chars marker, and that the ffmpeg stderr line appears BEFORE the graph excerpt. Short-graph case asserts the graph is included verbatim.

---
## 73. Wrap the final os.replace with retry + timestamped fallback so a locked target mp4 doesn't fail an hours-long render at 100%
⚪ LOW · симптом 2 · effort S · `run-atomic-replace-retry`

- **Файл / якорь:** `vpipe/render.py` — _run_atomic() final os.replace, ~line 131
- **Закрывает находки:** vpipe/render.py:131
- **Зависит от:** `r2-validate-output-duration`

**Проблема.** In _run_atomic only ff.run is wrapped by the cleanup handler (render.py:123-130); the concluding `os.replace(tmp, out_path)` (line 131) is outside it. On Windows, replacing a file another process holds open without FILE_SHARE_DELETE raises PermissionError — and the previous render's mp4 at the same path is exactly what a user opens in a media player (or what an in-flight /api/output FileResponse is streaming) while re-rendering with tweaked settings. Result: the entire successful encode is reported as a task error at the very end, and the finished multi-GB .part is neither promoted nor deleted (only the next server start sweeps it).

**Почему так.** Converts a 'render failed at 100%, output lost' data-loss edge into 'render saved under an alternate name', matching how cloud editors avoid clobbering open files.

**Текущий код:**
```
    try:
        ff.run(run_args, total=total, on_progress=on_progress, desc=desc)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, out_path)
```

**Изменение:**
```
Replace the bare os.replace (which by P2 is now preceded by the duration check) with a short retry, then a timestamped-name fallback, and RETURN the actual saved path so the caller can report it:

    # os.replace can fail with PermissionError on Windows if the previous output
    # at out_path is held open (media player / in-flight /api/output stream).
    # Retry briefly, then fall back to a sibling timestamped name rather than
    # discarding a finished multi-GB encode.
    import time
    for attempt in range(5):
        try:
            os.replace(tmp, out_path)
            return out_path
        except PermissionError:
            if attempt < 4:
                time.sleep(0.4)
                continue
            p = Path(out_path)
            alt = str(p.with_name(f"{p.stem} ({int(time.time())}){p.suffix}"))
            os.replace(tmp, alt)
            return alt

Change _run_atomic to return the final path (str). Update its call sites (render.py:1114, 1157, 1229) to capture it — e.g. `saved = _run_atomic(...)` and use `saved` as `out` in the returned dict (`"out": saved` instead of `"out": out_path`) so the UI opens the file that was actually written.
```

**Риск / безопасность.** Low. Happy path (target not locked) is unchanged — os.replace succeeds on attempt 0 and returns out_path exactly as before. The fallback only fires after 5 locked attempts (~1.6s), a genuinely stuck target; the timestamped sibling keeps the finished encode instead of losing hours of work. Returning the path is a small signature change — audit the three call sites (all in render()) to thread `saved` into the result dict's 'out'; serve.py reads rr.get('out') so it will surface the real path automatically. Keep the P2 duration-validate BEFORE this block so a truncated file is never promoted, locked or not.

**Тест (зафиксировать поведение).** tests/test_serve_render.py::test_run_atomic_replace_fallback — monkeypatch os.replace to raise PermissionError for the first N calls then succeed / then force the timestamped branch; assert _run_atomic returns the alt path, the .part is consumed, and no exception escapes. test_run_atomic_replace_happy — normal case returns out_path on first try.

---
## 74. Clamp removed intervals to [0, duration] so a past-EOF cut (unvalidated API / hand-edited cutlist) can't emit degenerate trim segments
⚪ LOW · симптом none · effort S · `timeline-clamp-intervals`

- **Файл / якорь:** `vpipe/timeline.py` — Timeline.__init__ / merge_intervals, ~line 16-41 and kept_segments ~84-94
- **Закрывает находки:** vpipe/timeline.py:89, vpipe/timeline.py:32

**Проблема.** kept_segments appends (cursor, a) for any removed interval starting after cursor without clamping a/cursor to duration (timeline.py:88-91). A cutlist with removed intervals past EOF (possible via the unvalidated PUT /api/cutlist or a hand-edited cutlist.json — CutSegment.from_dict takes start/end verbatim) yields a kept segment entirely past EOF; render() then emits trim=start/end beyond the file end, feeding an empty stream to concat (error) or silently shortening output. new_duration also miscomputes because total_removed counts non-existent media.

**Почему так.** Single-chokepoint defensive clamp that makes every Timeline consumer (render trim build, subtitle remap, chapters) robust to unvalidated/hand-edited cutlists without waiting on API-layer validation.

**Текущий код:**
```
    def __init__(self, removed: Iterable[tuple[float, float]], duration: float):
        self.removed = merge_intervals(removed)
        self.duration = float(duration)
        self.starts = [a for a, _ in self.removed]
```

**Изменение:**
```
Clamp intervals into [0, duration] in Timeline.__init__ before all derived arrays are built (defense-in-depth, complements API-layer validation):

    def __init__(self, removed: Iterable[tuple[float, float]], duration: float):
        self.duration = float(duration)
        d = self.duration
        clamped = [(max(0.0, min(float(a), d)), max(0.0, min(float(b), d)))
                   for (a, b) in removed]
        self.removed = merge_intervals([(a, b) for (a, b) in clamped if b > a])
        self.starts = [a for a, _ in self.removed]
        ...  # cum/total_removed built from the clamped self.removed as before

kept_segments (line 84) then needs no change — with clamped removed, cursor never exceeds duration and the trailing `if cursor < self.duration` guard already caps the last kept segment at duration.
```

**Риск / безопасность.** Very low. For valid cutlists (every interval already within [0,duration]) the clamp is a no-op and merge_intervals output is identical, so all existing timeline/subtitle/render behavior is byte-for-byte unchanged. Only degenerate past-EOF or negative intervals are affected, and only to bring them into range (the correct behavior). The `if b > a` filter after clamping drops intervals that collapse to zero width once clamped (e.g. entirely past EOF), which is exactly the fix. Keep clamping in __init__ (single chokepoint) rather than in kept_segments so removed_before/remap/total_removed also benefit.

**Тест (зафиксировать поведение).** tests/test_timeline.py::test_clamp_past_eof — Timeline([(50, dur+2),(dur+5, dur+10)], dur) with dur=60: assert self.removed == merged clamp of [(50,60)], kept_segments()==[(0,50)] with no segment past 60, new_duration()==50, and no interval starts >= duration. test_clamp_negative_start — (−5, 10) clamps to (0,10).

---
## 75. _guard_no_task reads task flags without TASK_LOCK, and _queue_worker's drain-exit isn't atomic with clearing _queue_running (lost queue wakeup)
⚪ LOW · симптом 4 · effort M · `guard-queue-toctou`

- **Файл / якорь:** `serve.py` — _guard_no_task (serve.py 1905-1911) + _queue_worker drain-exit (1582-1610)
- **Закрывает находки:** serve.py:1905

**Проблема.** (a) _guard_no_task reads SESSION.task['running'] and _queue_running with no lock while start_task/_start_queue_worker set them under TASK_LOCK, so a mutating endpoint (esp. POST /api/models, which rebuilds SESSION.llm) can pass its guard mid-mutation of a just-starting task. (b) _queue_worker decides to exit on 'no pending job' then only later clears _queue_running under TASK_LOCK; a queue/add + queue/start landing in that gap is a lost wakeup — queue_start no-ops (flag still True) while the worker is committed to exit, so the new job sits pending until the user presses Старт again.

**Почему так.** Making the flag-clear atomic with the final pending re-check, under the same lock queue_start uses to test+set the flag, closes the lost-wakeup: queue_start can never see flag=True while the worker is irreversibly exiting. Taking TASK_LOCK in the read-only guard removes the torn read of SESSION.llm that POST /api/models can cause.

**Текущий код:**
```
def _guard_no_task() -> None:
    """409 if a background task is in flight (mutating endpoints use this)."""
    if SESSION is not None and SESSION.task["running"]:
        raise HTTPException(409, "Идёт фоновая задача — дождитесь её завершения")
    if _queue_running:
        raise HTTPException(409, "Очередь обрабатывает ролики — дождитесь её "
                                 "завершения или остановите очередь")

# ... and in _queue_worker:
        if job is None:
            break
        ...
    # Release the GPU flag under the same lock that the editor checks it.
    with TASK_LOCK:
        _queue_running = False
```

**Изменение:**
```
Edit A — _guard_no_task reads the flags under TASK_LOCK:

def _guard_no_task() -> None:
    """409 if a background task is in flight (mutating endpoints use this).

    Read the running flags under TASK_LOCK (the same lock start_task /
    _start_queue_worker set them under) so a mutating endpoint can't observe a
    half-committed start."""
    with TASK_LOCK:
        task_running = SESSION is not None and SESSION.task["running"]
        queue_running = _queue_running
    if task_running:
        raise HTTPException(409, "Идёт фоновая задача — дождитесь её завершения")
    if queue_running:
        raise HTTPException(409, "Очередь обрабатывает ролики — дождитесь её "
                                 "завершения или остановите очередь")

Edit B — in _queue_worker, replace the `if job is None: break` branch with an atomic drain-exit that re-checks pending under TASK_LOCK (the lock queue_start checks the flag under):

        if job is None:
            # Drain: no pending job. Clear _queue_running atomically with a
            # final pending re-check under TASK_LOCK — the SAME lock
            # _start_queue_worker checks the flag under. Without this a
            # queue/add + queue/start landing between 'saw no pending' and
            # 'cleared the flag' is a lost wakeup. Lock order TASK_LOCK->
            # QUEUE_LOCK matches every other nesting (verified: nothing takes
            # QUEUE_LOCK then TASK_LOCK).
            with TASK_LOCK:
                with QUEUE_LOCK:
                    if any(j.status == "pending" for j in QUEUE):
                        continue
                _queue_running = False
            return

Keep the existing trailing `with TASK_LOCK: _queue_running = False` — it now only covers the cancel-exit path (while condition false via _queue_cancel).
```

**Риск / безопасность.** Per the audit's own note, wrapping the guard read in TASK_LOCK narrows but does not fully close (a) — the endpoint's mutation still runs after the lock releases; accept as improvement, not a complete fix. (b) introduces a TASK_LOCK->QUEUE_LOCK nesting; grep-verified no site acquires QUEUE_LOCK then TASK_LOCK, so no deadlock — preserve this ordering. The main-loop body, _save_queue transitions, and cancel path are untouched. TASK_LOCK is uncontended so the extra acquisition per mutating request is negligible.

**Тест (зафиксировать поведение).** tests/test_queue.py: (b) drive serve._queue_worker() directly with a stubbed _queue_process_one that, while running the FIRST job, appends a SECOND pending QueueJob to serve.QUEUE (simulating queue/add mid-drain); assert BOTH jobs reach status 'done' and _queue_running is False only after the second drains (worker did not exit with a pending job). (a) assert _guard_no_task() raises 409 when serve._queue_running=True and returns None when both flags are False — a semantics-unchanged smoke test proving the lock wrap didn't alter the guard's outcome.

---
## 76. Watch folder promotes a size-stable-but-still-locked file to 'processed'; the one job errors on probe and it is never retried
⚪ LOW · симптом 4 · effort S · `watch-locked-file-open`

- **Файл / якорь:** `serve.py` — scan_once stable-promotion branch, serve.py ~1765-1769
- **Закрывает находки:** serve.py:1765

**Проблема.** scan_once moves a candidate from pending into the registry the moment its size/mtime were stable for one interval — before any open attempt. If the writing program still holds an exclusive lock after the size stops changing (a muxer finalizing a long file), _queue_process_one's Session/probe/hash raises, the job goes to error, and because the file is already in the registry no future scan re-enqueues it. There is no retry path for errored watch jobs.

**Почему так.** A shared-read open is the cheapest reliable readiness probe: on Windows an exclusive writer lock makes open() raise PermissionError/OSError, so we keep the candidate pending and retry next scan; on an unlocked file the open succeeds and behavior is byte-for-byte the old promotion. The finding's recommended fix.

**Текущий код:**
```
        if pending.get(key) == (size, mtime):
            # Двухфазная стабильность: размер/время не менялись целый скан.
            pending.pop(key, None)
            registry[key] = {"size": size, "mtime": mtime}
            new_files.append(path)
        else:
            pending[key] = (size, mtime)    # ждём подтверждения следующим сканом
```

**Изменение:**
```
        if pending.get(key) == (size, mtime):
            # Двухфазная стабильность пройдена. Но «размер стабилен» ≠ «файл
            # разлочен»: некоторые рекордеры/копиры держат ЭКСКЛЮЗИВНЫЙ лок уже
            # после того, как файл перестал расти (мультиплексор финализирует
            # длинный ролик). Промоутить сейчас — первый же job упадёт на
            # probe/hash, уедет в error, а файл уже в registry и НИКОГДА не
            # переедет снова. Пробуем открыть на shared-read: залочен —
            # оставляем в pending, ждём следующего скана (ключ не трогаем,
            # ретрай бесплатный).
            try:
                with open(path, "rb"):
                    pass
            except OSError:
                continue
            pending.pop(key, None)
            registry[key] = {"size": size, "mtime": mtime}
            new_files.append(path)
        else:
            pending[key] = (size, mtime)    # ждём подтверждения следующим сканом
```

**Риск / безопасность.** One extra open()+close() per newly-stable file, once, at promotion — negligible. Read-open bumps atime, never mtime, so the two-phase stability check is unaffected. Does not help a file that opens but still fails ffprobe (truly corrupt) — that correctly errors once and stays registered (not a regression). Defense-in-depth follow-up (out of scope): also drop the registry entry / add a bounded retry counter when a watch job errors.

**Тест (зафиксировать поведение).** tests/test_watch.py scan_once unit: create a real file; scan 1 -> pending, not returned. Monkeypatch builtins.open to raise PermissionError for that exact path; scan 2 (same size/mtime) -> assert NOT promoted (still pending, not in new_files, key absent from registry). Restore open; scan 3 -> asserted promoted exactly once. Keep existing growing-file and normal-stable-file cases green (they open fine).

---
## 77. Autopack publishes the LIVE results dict into s.task then inserts keys from the worker thread while /api/state serializes it — intermittent RuntimeError('dictionary changed size during iteration') 500
⚪ LOW · симптом 4 · effort M · `autopack-results-snapshot`

- **Файл / якорь:** `serve.py` — autopack run() worker (~line 4143-4281)
- **Закрывает находки:** serve.py:4148

**Проблема.** run() publishes the live dict early (s.task["results"] = results, line 4148) and keeps inserting NEW top-level keys from the worker thread (enrich/main/totals/clips/ok). GET /api/state returns "task": s.task by live reference (line 1982); FastAPI's jsonable_encoder iterates the nested results dict in pure Python on the event loop with no lock/copy. A size-changing insert during that iteration raises RuntimeError -> 500 on /api/state, breaking a page load mid-autopack. (Only the ~ new-key inserts can raise; list appends and nested-dict key overwrites are safe. /api/events is unaffected — bare json.dumps is GIL-atomic.)

**Почему так.** Writer-side snapshotting is the only fully-safe fix (reader-side copy still iterates the live dict during dict()); publishing a frozen copy after each insert eliminates the size-change race while preserving the exact observable results shape.

**Текущий код:**
```
    def run():
        warnings: list[str] = []
        skipped: list[str] = []
        results: dict = {"warnings": warnings, "skipped": skipped}
        # Сразу в task: отмена/падение на любой стадии сохраняет уже сделанное.
        s.task["results"] = results
        ...
        results["enrich"] = {"applied": True, "min_score": AUTOPACK_ENRICH_MIN_SCORE, "count": n_apply}
        ...
        results["enrich"] = {"applied": False}      # (x2 in the stale / no-plan branches)
        ...
        results["main"] = res_main
        ...
        results["totals"] = totals
        ...
        results["clips"] = []                       # (llm None branch)
        results["clips"] = {"error": str(e)}        # (suggest failed)
        results["clips"] = []                       # (empty cands)
        results["clips"] = clip_results             # (render branch)
        ...
        results["ok"] = True
```

**Изменение:**
```
Republish a FROZEN shallow snapshot after every top-level key insert so no reader ever iterates a dict this thread is mutating. Replace the initial publish + route every top-level `results[...] =` insert through a helper:

    def run():
        warnings: list[str] = []
        skipped: list[str] = []
        results: dict = {"warnings": warnings, "skipped": skipped}
        def _publish() -> None:
            # /api/state serializes s.task["results"] by LIVE reference on the
            # event loop while this worker inserts keys; a size-changing insert
            # during jsonable_encoder's dict iteration raises RuntimeError -> 500.
            # Publish dict(results) (frozen key set) so readers only ever iterate
            # a dict we won't mutate. Nested lists (warnings/clip_results) are
            # append-only and nested dict (totals) is key-overwrite -> safe to share.
            s.task["results"] = dict(results)
        def _set_result(key: str, value) -> None:
            results[key] = value
            _publish()
        _publish()                                   # replaces `s.task["results"] = results`

Then change each top-level insert to _set_result:
  results["enrich"] = {...}            -> _set_result("enrich", {...})     (all 3 branches: line ~4194, ~4202, ~4206)
  results["main"] = res_main           -> _set_result("main", res_main)   (~4215)
  results["totals"] = totals           -> _set_result("totals", totals)   (~4221)
  results["clips"] = []                -> _set_result("clips", [])        (~4234, ~4253)
  results["clips"] = {"error": str(e)} -> _set_result("clips", {"error": str(e)}) (~4246)
  results["clips"] = clip_results      -> _set_result("clips", clip_results)      (~4261)
  results["ok"] = True                 -> _set_result("ok", True)         (~4280)

Leave `totals["clips_rendered"] = ...` (~4277) and `clip_results.append(...)` (~4276) as-is (nested overwrite / list append can't raise, and `totals`/`clip_results` are shared with the last published snapshot by reference — that's intended).
```

**Риск / безопасность.** Preserves exact key-presence semantics (a key appears in the published snapshot iff it was set) — no frontend contract change, unlike a pre-seed-all-keys shortcut which would turn absent keys into null. Reference reassignment is GIL-atomic; a reader sees either the pre- or post-insert snapshot, never a half-mutated dict. Cancel/partial-success still preserved: s.task["results"] always holds the latest snapshot including partials. Only caveat: every top-level insert site must be routed through _set_result — a missed site leaves a (rare) residual race, not a new bug; enumerate all 10 sites as listed.

**Тест (зафиксировать поведение).** tests/test_api_autopack.py::test_autopack_results_published_as_fresh_snapshots — drive /api/autopack with stubbed stages (monkeypatch _render_formats etc. to no-op quickly), poll GET /api/state repeatedly from the test thread until task done, and assert (a) every response is 200 (no 500) and (b) `id(results)` differs across two polls captured at different stages (proving fresh snapshots, not a live dict). For a direct race repro, add an optional stress variant: run run() in a thread while a second thread hammers `json.dumps(s.task)` in a loop and assert no RuntimeError over N iterations (documented as best-effort — the race is low-probability without instrumentation).

---
## 78. Shared fixed-name .json.tmp on clips/enrich saves reproduces the Windows PermissionError race already fixed for the transcript PUT (audit D-1) — two overlapping saves -> spurious 500
⚪ LOW · симптом 4 · effort S · `enrich-clips-uuid-tmp`

- **Файл / якорь:** `serve.py, vpipe/enrich.py` — clip-bounds save (~serve.py:3341-3348), _save_clips_json (~serve.py:3190-3198), save_enrich (~vpipe/enrich.py:829-841)
- **Закрывает находки:** serve.py:3343, serve.py:3193, vpipe/enrich.py:831

**Проблема.** The transcript-word PUT was fixed to use a per-call uuid tmp because two parallel writes to a shared .tmp on Windows hit PermissionError on os.replace -> false 500 (comment serve.py:2205-2209). The same fixed-name pattern remains in the STRICT writers clip-bounds save (serve.py:3343, 500 on failure) and save_enrich (enrich.py:831, used by strict /api/enrich/save and /api/enrich/select), plus best-effort _save_clips_json (serve.py:3193). The «Монтаж» UI can overlap saves (a save slower than the 350ms debounce, a bulk enrich_save overlapping a pending flush, a double-clicked «Сохранить границы», or a UI-agent /api/enrich/select racing a save).

**Почему так.** Uniform application of the already-proven D-1 fix removes the Windows os.replace contention on every strict save path in the «Монтаж»/clips flow.

**Текущий код:**
```
    # serve.py:3341-3348 (clip-bounds save, STRICT)
    p = _clips_json_path(s)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        raise HTTPException(500, f"Не удалось сохранить clips.json: {e}")
    return {"ok": True, "clip": clip}

    # vpipe/enrich.py:829-841 (save_enrich, STRICT+best-effort callers)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.replace(tmp, p)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

    # serve.py:3190-3198 (_save_clips_json, best-effort)
    p = _clips_json_path(s)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass  # non-fatal: candidates are already in task['results']
```

**Изменение:**
```
Apply the D-1 unique-tmp pattern (uuid per call + finally cleanup) at all three sites. uuid is already imported in serve.py (line 24) and enrich.py (used by new_item_id).

# serve.py:3341-3348
    p = _clips_json_path(s)
    tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")   # audit D-1: общий .tmp
    try:                                                     # -> PermissionError на Windows
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        raise HTTPException(500, f"Не удалось сохранить clips.json: {e}")
    finally:
        tmp.unlink(missing_ok=True)
    return {"ok": True, "clip": clip}

# vpipe/enrich.py:829-841
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # audit D-1: уникальное имя tmp на вызов — параллельные save с общим .tmp на
    # Windows ловили PermissionError на os.replace -> ложный 500. Последний
    # replace побеждает; tmp чистится в finally.
    tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)

# serve.py:3190-3198
    p = _clips_json_path(s)
    tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass  # non-fatal: candidates are already in task['results']
    finally:
        tmp.unlink(missing_ok=True)
```

**Риск / безопасность.** Behavior preserved: strict sites still raise on failure, best-effort still swallows; finally-unlink(missing_ok=True) is a no-op after a successful replace (tmp already moved). The uuid tmp lands in the same dir; no other code globs for *.json.tmp (the startup sweep only targets *.part) except the existing test assertion below. Separate KNOWN LIMITATION (out of scope, note only): enrich_save reloads plan state from file per request (serve.py:3720), so two concurrent saves are still last-writer-wins on CONTENT even with unique tmp names — the uuid fix removes the spurious 500 but not the lost-update; a real fix would serialize enrich writes under a lock.

**Тест (зафиксировать поведение).** Cross-platform lock-in via unique-name assertion (POSIX os.replace won't raise the Windows error, so don't test the 500 directly): tests/test_enrich_models.py::test_save_enrich_uses_unique_tmp_per_call — monkeypatch enrich_mod.os.replace to capture the src (tmp) path on each call; call save_enrich twice; assert the two captured tmp names differ and both match the pattern `*.json.<hex>.tmp`. Mirror for clips in tests/test_api_clips. Also UPDATE tests/test_api_enrich.py:175 which asserts `not p.with_suffix('.json.tmp').exists()` — change to assert no leftover matches glob `<stem>.enrich.json.*.tmp` (the fixed-name never exists now; the real invariant is no uuid-tmp scrap remains).

---
## 79. GET /api/events raises KeyError('cancelled') mid-stream on any session that has never started a task
⚪ LOW · симптом 4 · effort S · `events-ctor-cancelled-key`

- **Файл / якорь:** `serve.py` — Session.__init__ task dict (~247-248) + _payload in GET /api/events (~4292)
- **Закрывает находки:** serve.py:4292, serve.py:247

**Проблема.** The SSE _payload hard-indexes 8 keys including "cancelled" (t[k]), but the ctor task dict (serve.py:247-248) has only 7 keys — "cancelled" is added only by start_task's replacement dict (serve.py:304-306). Any /api/events while the ctor dict is live sends 200 headers then aborts the stream with KeyError('cancelled') inside the generator. Every other reader of task uses .get(); _payload is the lone hard-index.

**Почему так.** The streaming payload is the only hard-indexer of a key not present at ctor time; aligning the ctor and softening the index removes the mid-stream 500 without changing any observed field.

**Текущий код:**
```
    # ctor (serve.py:247-248)
    self.task = {"name": None, "running": False, "percent": 0.0,
                 "stage": "", "error": None, "done": False, "results": None}

    # _payload (serve.py:4290-4294)
    def _payload() -> str:
        t = s.task
        return json.dumps({k: t[k] for k in
                           ("name", "running", "percent", "stage",
                            "error", "done", "results", "cancelled")})
```

**Изменение:**
```
1) Align the ctor dict with start_task's shape (source-of-truth fix):
    self.task = {"name": None, "running": False, "percent": 0.0,
                 "stage": "", "error": None, "done": False, "results": None,
                 "cancelled": False}

2) Make _payload defensive so a hand-built/legacy task dict can never abort the stream:
    def _payload() -> str:
        t = s.task
        return json.dumps({k: t.get(k) for k in
                           ("name", "running", "percent", "stage",
                            "error", "done", "results", "cancelled")})

Apply BOTH: (1) fixes the divergence at the source (and keeps /api/state's s.task shape consistent), (2) is a belt-and-braces one-char change.
```

**Риск / безопасность.** Purely additive/defensive. Adding "cancelled": False to the ctor cannot affect /api/cancel (which requires task["running"], only true post-start_task) or any .get() reader. t.get(k) returns None for a truly-absent key instead of raising — strictly safer serialization.

**Тест (зафиксировать поведение).** tests/test_api_events.py::test_events_before_any_task_no_keyerror — build a real serve.Session-backed test client (or a session whose task dict is the fresh ctor shape), GET /api/events, assert status 200 and the streamed body parses as JSON containing "cancelled": false with no server exception. NOTE: tests/test_api_enrich.py FakeSession (line 90-91) mirrors the OLD 7-key ctor dict — either drive a real Session for this test or add "cancelled": False to that fake, else the test reproduces the KeyError it's meant to guard.

---
## 80. The wav-missing branch of Session._detect is untested: acoustic hesitation detection is silently skipped when audio16k.wav is absent, so cut results differ between runs with no user-visible signal
⚪ LOW · симптом 4 · effort S · `hesitations-wav-missing-skip`

- **Файл / якорь:** `tests/test_hesitations.py` — serve.Session._detect wav gate (serve.py:271-276) + vpipe.detect.run_detection audio_path skip
- **Закрывает находки:** serve.py:273, tests/test_hesitations.py:169

**Проблема.** serve.py:271-276 passes audio_path only 'if wav.exists()'. If a session's work_dir/audio16k.wav was cleaned (or detection runs on a cache-restored session before extraction), hesitations are silently not detected — same video, different cuts between runs (symptom 4), no error. test_hesitations.py already covers run_detection with/without audio_path in ISOLATION but nothing tests Session._detect's wav gate or asserts the skip is surfaced.

**Текущий код:**
```
# tests/test_hesitations.py:169-174 — tests run_detection directly, not Session._detect's wav gate:
def test_run_detection_skips_hesitations_without_audio_path():
    cfg = _cfg_only_hesitations()
    cl = run_detection(_transcript(), cfg, FillerLists(), ProfanityLists(),
                       source='x', llm=None, log=lambda *_: None)
    assert all(s.type != TYPE_HESITATION for s in cl.segments)
```

**Изменение:**
````
Append to tests/test_hesitations.py a test that drives Session._detect and asserts the wav presence controls whether audio_path is forwarded to run_detection.

```python
import serve
from types import SimpleNamespace

def _detect_session(tmp_path, wav_present):
    work=tmp_path/'work'; work.mkdir(parents=True, exist_ok=True)
    if wav_present:
        (work/'audio16k.wav').write_bytes(b'RIFF....')
    s=SimpleNamespace(cfg=Config(), transcript=_transcript(), fillers=FillerLists(),
                      profanity=ProfanityLists(), inp=SimpleNamespace(stem='clip', __fspath__=lambda:'clip'),
                      llm=None, work_dir=work, out_dir=tmp_path/'out', cutlist=None)
    s.inp = type('P', (), {'stem':'clip'})()
    (tmp_path/'out').mkdir(parents=True, exist_ok=True)
    s.cutlist_path = tmp_path/'out'/'clip.cutlist.json'
    return s

def test_session_detect_forwards_audio_path_only_when_wav_exists(tmp_path, monkeypatch):
    seen={}
    def spy(tr, cfg, fill, prof, **kw):
        seen['audio_path']=kw.get('audio_path')
        from vpipe.models import CutList
        return CutList(source='clip', duration=10.0, segments=[])
    monkeypatch.setattr(serve, 'run_detection', spy)
    monkeypatch.setattr(serve, 'save_txt', lambda *a, **k: None)
    # wav present -> audio_path is the wav
    s=_detect_session(tmp_path, wav_present=True)
    serve.Session._detect(s)
    assert seen['audio_path'] is not None and str(seen['audio_path']).endswith('audio16k.wav')
    # wav absent -> audio_path is None (hesitation detector silently skipped)
    s2=_detect_session(tmp_path/'b', wav_present=False)
    serve.Session._detect(s2)
    assert seen['audio_path'] is None
```
If the team wants the skip SURFACED (finding's real ask), the follow-up source change is to log/flag when wav is absent; add an assertion on that signal here once it exists (leave a TODO comment referencing serve.py:273).
````

**Риск / безопасность.** Low: Session._detect is called unbound on a SimpleNamespace, so the stand-in must carry every attribute _detect touches (cfg, transcript, work_dir, cutlist, cutlist_path, out_dir, inp.stem). Verify against serve.py:262-284 before committing — _detect also calls cl.save_json(self.cutlist_path) and save_txt(...), both of which must be given tmp paths / a no-op. The test asserts CURRENT behavior (skip is silent); the surfacing signal is a source change out of scope here.

**Тест (зафиксировать поведение).** test_session_detect_forwards_audio_path_only_when_wav_exists: spy on run_detection, call Session._detect once with work_dir/audio16k.wav present (assert audio_path forwarded) and once absent (assert audio_path is None) — locks the wav gate so a refactor can't silently change which runs get acoustic hesitation detection.

---

# Волна 5 — Производительность + долги тестов

## 81. No end-to-end ffmpeg execution tier exists: every render test pins graph strings against FakeFF, CI bans the real binary, and the one real media fixture (tests/_media/test.mp4) is referenced by no test
🔴 CRITICAL · симптом 2 · effort L · `integration-real-ffmpeg-tier`

- **Файл / якорь:** `tests/test_integration_render.py` — tests/_media/test.mp4 (orphaned fixture) driven through vpipe.render.render end-to-end + tests/conftest.py marker
- **Закрывает находки:** tests/test_edge_fade.py:79, .github/workflows/ci.yml:3, scripts/make_smoke_clip.py:24

**Проблема.** All render-path tests use FakeFF: run() records args and writes an empty file, then assertions string-match the graph byte-for-byte. This pins current behavior but validates nothing against ffmpeg's actual parser/encoders — an ffmpeg-illegal graph (subtitles='...' quoting, concat A/V stream-count mismatch, a renamed filter option in 8.1.1) passes 773 green tests. CI (.github/workflows/ci.yml:3-6) permanently bans ffmpeg with no compensating local tier. tests/_media/test.mp4 (170 KB) is used only by manual scripts. The exact user failure (render dies/truncates on a real machine) is structurally invisible.

**Текущий код:**
```
# tests/test_edge_fade.py:89-95 — the FakeFF that never runs ffmpeg:
def run(self, args, total=None, on_progress=None, desc='ffmpeg'):
    self.runs.append(list(args))
    if args:
        try: Path(args[-1]).write_bytes(b'')   # writes an EMPTY file; never parsed by ffmpeg
        except OSError: pass
# scripts/make_smoke_clip.py:28-34 already knows how to synthesize a real clip via lavfi testsrc+sine.
```

**Изменение:**
````
Two artifacts, both opt-in so CI's ffmpeg ban is untouched. (1) tests/conftest.py registering the marker and gating on an env var:

```python
import os, shutil, pytest
def pytest_configure(config):
    config.addinivalue_line('markers', 'ffmpeg: opt-in; runs the REAL ffmpeg binary (skipped in CI)')
_RUN = os.environ.get('FVE_FFMPEG_TESTS') == '1' and shutil.which('ffmpeg') is not None
requires_ffmpeg = pytest.mark.skipif(not _RUN, reason='set FVE_FFMPEG_TESTS=1 with ffmpeg on PATH to run the integration tier')
```
(2) tests/test_integration_render.py:

```python
import subprocess, json
from pathlib import Path
import pytest
from conftest import requires_ffmpeg
from vpipe.config import load_config
from vpipe.ffmpeg_utils import FFmpeg
from vpipe.probe import probe_media
from vpipe.render import render
from vpipe.models import CutList, CutSegment, TYPE_MANUAL, ACTION_REMOVE

MEDIA=Path(__file__).parent/'_media'/'test.mp4'

@pytest.fixture(scope='module')
def ff():
    cfg=load_config('config.yaml'); return FFmpeg(cfg.ffmpeg), cfg

def _dur(ff, path):
    d=ff.probe(path); return float(d['format']['duration']), len(d['streams'])

def _cl(dur):
    # drop [1.0,2.0] and [4.0,5.0] -> expect dur-2.0 encoded
    return CutList(source=str(MEDIA), duration=dur, segments=[
        CutSegment(id='a',start=1.0,end=2.0,type=TYPE_MANUAL,action=ACTION_REMOVE,enabled=True),
        CutSegment(id='b',start=4.0,end=5.0,type=TYPE_MANUAL,action=ACTION_REMOVE,enabled=True)])

@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize('opts', [
    {},                                   # plain cuts
    {'scale_h':720},                      # cuts + rescale
    {'loudnorm':True, 'deess':True},      # cuts + mastering chain
])
def test_real_render_duration_and_streams(ff, tmp_path, opts):
    f, cfg = ff
    src_dur, _ = _dur(f, MEDIA)
    media=probe_media(f, MEDIA)
    if opts.get('loudnorm'): cfg.render.denoise.loudnorm=True
    if opts.get('deess'): cfg.render.denoise.deess=True
    out=tmp_path/'out.mp4'
    rr=render(f, media, _cl(media.duration), cfg, str(out), str(tmp_path),
              log=lambda *a,**k:None, scale_h=opts.get('scale_h'))
    assert out.exists() and out.stat().st_size>0
    got_dur, nstreams = _dur(f, out)
    assert abs(got_dur - (media.duration-2.0)) < 0.5     # truncation/overrun guard
    assert nstreams >= 2                                 # v+a survived concat
```
Document in the file header: run locally with `FVE_FFMPEG_TESTS=1 python -m pytest tests/test_integration_render.py`. If tests/_media/test.mp4 is too short for the cut points, regenerate/extend via scripts/make_smoke_clip.py (12s testsrc) and commit a slightly longer fixture, or scale the cut points to the fixture's real duration read at runtime.
````

**Риск / безопасность.** This is the ONE tier allowed to touch a real binary — it MUST stay skipped on CI (env-var gate + skipif), preserving .github/workflows/ci.yml's hermeticity guarantee (do NOT add an ffmpeg install step to the workflow). Keep assertions on OUTPUT INVARIANTS (duration within tolerance, stream count, non-empty) not exact bytes/codecs, so ffmpeg version drift doesn't false-fail. Verify tests/_media/test.mp4 is long enough for the [4.0,5.0] cut (it is ~seconds; read media.duration at runtime and skip/clamp cut points if shorter). ASS-burn case can be added once a fixture ASS is available; the mastering case already exercises the audio filter chain that most often breaks on a real parser.

**Тест (зафиксировать поведение).** test_real_render_duration_and_streams (marked ffmpeg + skipif): render tests/_media/test.mp4 through {plain cuts, cuts+720p, cuts+loudnorm+deess}, then ffprobe the output and assert duration == source-2.0s within 0.5s and >=2 streams — the first test in the suite that would catch an ffmpeg-illegal graph or a truncated render on the real 8.1.1 binary.

---
## 82. SD generation spawns a fresh sd-cli process per candidate, reloading the ~4GB SDXL-Turbo GGUF for each of the 4 seeds of every illustration point
🟠 HIGH · симптом none · effort L · `sd-batch-single-process`

- **Файл / якорь:** `vpipe/imagegen.py` — generate_candidates() candidate loop (~lines 243-276)
- **Закрывает находки:** vpipe/imagegen.py:263
- **Зависит от:** `sd-max-vram-guard`

**Проблема.** generate_candidates loops `for i in range(count)` calling generate_image, each a brand-new sd-cli process that parses the 3.94GB GGUF, uploads weights over PCIe and inits CUDA/cuBLAS from scratch. serve._generate_point_candidates calls this per diffusion point, so 8 photo points = 32 full model loads. On the RTX 3080 the per-process load (~10-30s) dominates the 8-step Turbo render (~3-8s), so the «Монтаж: генерация фото» stage spends the majority of wall time re-initializing the same model. sd-cli supports `-b N` (verified in --help) to emit N seeds in ONE process/one load — a ~2-3x stage speedup.

**Почему так.** -b N amortizes the dominant model-load cost across all N seeds in one process. Consecutive deterministic seeds (base+i) keep the 'N distinct + reproducible' property and let each output map to the SAME per-seed cache key generate_image uses, so re-runs hit the all-cached fast path (zero subprocess). The per-seed fallback guarantees no correctness regression on sd-cli builds whose -b/%d output naming differs from this one.

**Текущий код:**
```
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
```

**Изменение:**
```
Switch to sd-cli `-b N` batch with DETERMINISTIC CONSECUTIVE seeds (base..base+N-1) so one process/one load emits all N, while preserving determinism, per-seed caching, and a safe fallback. Concrete design:

(a) Factor the command+key core out of generate_image into a shared helper (also used by sd-max-vram-guard):
    def _sd_cmd_core(prompt, negative, seed, W, H, steps, cfg_scale, vae, cfg) -> list[str]:
        returns [ '-M','img_gen','-m',model,'-p',prompt,'-n',negative,'--steps',str(steps),
                  '--cfg-scale',str(cfg_scale),'--sampling-method',SAMPLING_METHOD,'--diffusion-fa',
                  '-W',str(W),'-H',str(H),'-s',str(seed) ] + vae/vae-on-cpu/max-vram flags
        (generate_image builds its cmd from this + ['-o',tmp]; keeps its cache/negative/people logic.)

(b) generate_candidates:
    prompt = ' '.join((prompt or '').split())
    if not prompt: return []
    size = max(64, int(getattr(cfg,'imagegen_size',768)))
    count = n if (isinstance(n,int) and n>0) else max(1,int(getattr(cfg,'imagegen_candidates',DIFFUSION_CANDIDATES)))
    base = _seed_for(prompt, -1)
    seeds = [(base+i) & 0x7FFFFFFF for i in range(count)]
    # planned final cache path per seed (SAME key generate_image would use):
    planned = [(_planned_out_png(prompt, STYLE_SUFFIX, s, size, size, cfg, cache_dir), s) for s in seeds]
    cached = [str(p) for (p,_s) in planned if p and p.is_file() and p.stat().st_size>0]
    if len(cached) == count:                 # FAST PATH: 0 subprocess (idempotent re-run)
        return list(dict.fromkeys(cached))
    # one batch process: -b count -s base -o <tmp>/b_%03d.tmp.png (%d ⇒ image sequence, begin-idx 0)
    produced = _run_sd_batch(prompt, base, count, size, cfg, cache_dir, log)   # returns ordered list or None
    if produced is not None and len(produced) == count:
        out = []
        for (final_png,_s), src in zip(planned, produced):
            try: os.replace(src, final_png); out.append(str(final_png))
            except OSError: pass
        if out: return list(dict.fromkeys(out))
    # FALLBACK (batch unsupported/failed/wrong count): current per-seed loop — correctness > 1 load
    out=[]; seen=set()
    for s in seeds:
        try: p = generate_image(prompt, STYLE_SUFFIX, s, size, size, cfg=cfg, cache_dir=cache_dir, log=log)
        except Exception as e:  # noqa: BLE001
            log(f'  SD: кандидат {s} упал ({e}).'); p=None
        if p and p not in seen: seen.add(p); out.append(p)
    return out

(c) _run_sd_batch builds `binp + _sd_cmd_core(prompt+STYLE_SUFFIX, negative, base, size, size, steps, cfg_scale, vae, cfg)` but replaces the single '-s' with '-b', str(count), '-s', str(base) and '-o', str(tmpdir/'b_%03d.tmp.png'); subprocess.run with SD_TIMEOUT_S*count; on returncode!=0/timeout/OSError → None; else glob the produced b_000.tmp.png..b_{count-1:03d}.tmp.png in index order and return the list (or None if the count mismatches).
```

**Риск / безопасность.** HIGH / needs real-binary validation (no real-SD tests exist). The exact multi-file output naming (`%03d` sequence, --output-begin-idx default 0 when %d present) and the -b/-s seed semantics MUST be smoke-tested against tools/sd-cli.exe before merge — hence the mandatory fallback path. Seed scheme changes from hash('prompt#i') to base+i, so cache keys change once (old cached candidates re-generate) — acceptable. The batch cmd MUST include the --max-vram flag from sd-max-vram-guard (shared _sd_cmd_core), otherwise the batch path reintroduces the OOM. Preserve generate_image unchanged for the single-image (enrich_image_batch legacy) path.

**Тест (зафиксировать поведение).** tests/test_imagegen.py: rewrite test_generate_candidates_n4_distinct_seeds to the batch contract — monkeypatch subprocess.run to (i) assert exactly ONE call, (ii) assert '-b' in cmd with value '4' and '-s' with the deterministic base and an '-o' path containing '%03d', (iii) create 4 fake PNGs matching the %03d pattern; assert 4 returned paths. ADD test_generate_candidates_all_cached_no_subprocess — pre-create the 4 final cache PNGs, monkeypatch subprocess.run to raise if called, assert it's NEVER called and 4 paths returned. ADD test_generate_candidates_batch_failure_falls_back — subprocess.run returns non-zero; assert the per-seed generate_image fallback runs and yields paths. Keep test_generate_candidates_empty_prompt_returns_empty.

---
## 83. Keep qwen3 warm between bad-take windows (keep_alive) instead of reloading the ~6GB model from disk every window
🟠 HIGH · симптом 2 · effort S · `badtakes-keepalive-warm`

- **Файл / якорь:** `vpipe/detect/badtakes.py` — detect() window loop, lines 52-65 (segment_windows loop + chat_json call)
- **Закрывает находки:** vpipe/detect/badtakes.py:60

**Проблема.** badtakes.detect calls llm.chat_json(_SYSTEM, user, _SCHEMA) with NO keep_alive, so llm.py:133 falls back to config keep_alive=0 (unload-after-call). Every other multi-window LLM caller keeps the model warm between windows (chapters.py:215 uses 60s, clips.py:603 uses 300s) and unloads only on the last. On a 2h video (~20-27 windows of 80 segments) that is ~20-27 gratuitous unload/cold-load cycles of a ~5-6GB model per detection pass — which runs after every transcription, every /api/detect, every queue job, and in the Session ctor — while the UI stage sits on «Детекция вырезов…» with no per-window progress, feeding the perceived-hang half of symptom 2.

**Текущий код:**
```
    chosen: dict[int, str] = {}
    for win_start, win_end in segment_windows(n, cfg.llm):
        window = segments[win_start:win_end]
        lines = []
        for local_i, s in enumerate(window):
            lines.append(f"{local_i} | {s.start:.1f}-{s.end:.1f} | {s.text}")
        user = ("Сегменты расшифровки:\n" + "\n".join(lines) +
                "\n\nВерни JSON: {\"removals\": [{\"index\": <номер>, \"reason\": <причина>}]}")
        try:
            data = llm.chat_json(_SYSTEM, user, _SCHEMA)
        except (LLMUnavailable, Exception) as e:  # noqa: BLE001
```

**Изменение:**
```
Materialize the window list to know the last index, enumerate, and pass keep_alive mirroring chapters.py (0 on the last window, 60 between):

    chosen: dict[int, str] = {}
    windows = segment_windows(n, cfg.llm)
    n_win = len(windows)
    for wi, (win_start, win_end) in enumerate(windows):
        window = segments[win_start:win_end]
        lines = []
        for local_i, s in enumerate(window):
            lines.append(f"{local_i} | {s.start:.1f}-{s.end:.1f} | {s.text}")
        user = ("Сегменты расшифровки:\n" + "\n".join(lines) +
                "\n\nВерни JSON: {\"removals\": [{\"index\": <номер>, \"reason\": <причина>}]}")
        try:
            # Keep qwen3 warm BETWEEN windows (default keep_alive=0 reloads the
            # ~6 GB model once per window on a long video); unload after the last
            # so it frees VRAM for the next stage (chapters.py pattern).
            ka = 0 if wi == n_win - 1 else 60
            data = llm.chat_json(_SYSTEM, user, _SCHEMA, keep_alive=ka)
        except (LLMUnavailable, Exception) as e:  # noqa: BLE001

(rest of the loop body — the removals parsing — is unchanged.)
```

**Риск / безопасность.** Very low. chat_json already accepts keep_alive (llm.py:188) and clips/chapters use exactly this. Only behavioral change is the model stays resident up to 60s between windows — deliberate and matches siblings; the LAST window still passes 0 so VRAM is freed before Whisper/SD/render (the 8GB-card contract is preserved). Single-window videos (n_win==1) pass ka=0 → byte-for-byte identical to today. The broad `except (LLMUnavailable, Exception)` per-window guard is retained so a warm-model failure still degrades gracefully.

**Тест (зафиксировать поведение).** Add tests/test_detect.py::test_badtakes_keeps_model_warm_between_windows — reuse the MockLLM from tests/test_clips.py (records keep_alive per chat_json call). Build a Transcript with enough segments to force 2 windows (Config with cfg.llm.max_segments_per_call small, e.g. 10, and >12 segments), call badtakes.detect, and assert llm.calls[0]['keep_alive']==60 and llm.calls[-1]['keep_alive']==0. Add a single-window case asserting the lone call gets keep_alive==0.

---
## 84. Replace the two O(words) generator scans per VAD gap with a single bisect over precomputed sorted starts + prefix-max ends (behavior-identical)
🟡 MEDIUM · симптом none · effort M · `hesitations-bisect-clamp`

- **Файл / якорь:** `vpipe/detect/hesitations.py` — detect(), lines 153-156 (word-safe clamp: max/min generator passes) + precompute before the gap loop (~line 140)
- **Закрывает находки:** vpipe/detect/hesitations.py:155

**Проблема.** For each candidate gap the word-safe clamp does prev_end = max(w.end for w in words if w.start < mid) and next_start = min(w.start for w in words if w.start >= mid) — two full O(words) passes per gap. A 2h recording (~15k words, 1000-3000 in-band gaps) is 30-90M Python iterations ≈ 10-30s, paid on every detection run (transcribe finish, re-detect, queue job, session ctor). Words are already sorted by start, so one bisect gives both neighbours in O(log n).

**Текущий код:**
```
        if words:
            mid = 0.5 * (g_start + g_end)
            prev_end = max((w.end for w in words if w.start < mid), default=None)
            next_start = min((w.start for w in words if w.start >= mid), default=None)
            if prev_end is not None:
                a = max(a, prev_end)
            if next_start is not None:
                b = min(b, next_start)
```

**Изменение:**
```
Add `from bisect import bisect_left` at the top (pattern already used in clips.py:23). Before the `for g_start, g_end in gaps:` loop, precompute once from a defensively-sorted word list:

    words = words or []
    words = sorted(words, key=lambda w: w.start)   # Whisper is already sorted; defensive
    w_starts = [w.start for w in words]
    # prefix-max of word ENDS: pmax_end[i] = max end over words[:i] (None for i==0).
    # Ends aren't guaranteed monotonic if words overlap, so we can't just take
    # words[idx-1].end — the prefix max reproduces the old max(...) exactly.
    pmax_end: list = [None]
    for w in words:
        pmax_end.append(w.end if pmax_end[-1] is None else max(pmax_end[-1], w.end))

Then inside the loop replace the two generator scans with:

        if words:
            mid = 0.5 * (g_start + g_end)
            idx = bisect_left(w_starts, mid)        # count of words with start < mid
            prev_end = pmax_end[idx]                 # None if idx == 0
            next_start = w_starts[idx] if idx < len(w_starts) else None
            if prev_end is not None:
                a = max(a, prev_end)
            if next_start is not None:
                b = min(b, next_start)

(Delete the two `words or []` / prior `words = words or []` duplicate — keep the single sorted assignment above.)
```

**Риск / безопасность.** This MUST be behavior-identical (perf-only). Equivalence: bisect_left(w_starts, mid) returns the first index whose start >= mid, so words[:idx] are exactly those with start < mid (matching the old `w.start < mid` predicate) and w_starts[idx] is exactly min(start where start >= mid). pmax_end[idx] equals max(w.end for words[:idx]) — reproducing the old max even under overlapping word ends. A word with start==mid falls in the 'next' group in both versions. The one precondition is sorted-by-start, which we enforce with the defensive sort (Whisper already returns sorted, so the sort is a no-op in practice). The existing word-clamp tests (test_cut_smoothing.py:66-88 test_hesitation_clamped_off_a_word / test_genuine_interword_hesitation_kept / test_word_clip_dropped_when_clamp_collapses) must pass UNCHANGED — they are the regression guard. The _overlaps_existing O(existing) scan flagged in the finding is left as-is (correctness-critical, secondary cost); optionally sort existing once and bisect, but that is not required for the dominant win and adds risk.

**Тест (зафиксировать поведение).** Add tests/test_hesitations.py::test_word_clamp_bisect_equals_linear — a property-style test: generate a sorted list of ~200 Word(start,end) and ~50 gaps, compute (a,b) with a local re-implementation of the OLD max/min generator clamp and with detect() (gaps monkeypatched via _patch_gaps), and assert the emitted CutSegment start/end are identical to 3 decimals. Include an overlapping-ends case (w[i].end > w[i+1].start) to lock the prefix-max branch. The existing clamp tests remain as-is and must stay green.

---
## 85. Multi-format render re-runs face detection and the whole audio pipeline (censor FLAC, DeepFilterNet, 2-pass loudnorm) per format although they're identical across formats
🟡 MEDIUM · симптом 3 · effort ? · `multiformat-hoist-audio-facecrop`

- **Файл / якорь:** `serve.py` — _render_formats loop (~L1435-1477); render.py audio stage (censor/DFN/loudnorm) is the deferred half
- **Закрывает находки:** serve.py:1435, serve.py:1474

**Проблема.** _render_formats loops formats, each a full _resolve_render_opts + _run_render_pipeline. The video re-encode per format is inherent, but everything that depends only on the cutlist+source audio is not: detect_center re-samples frames + runs face detection per cropped format; censor_audio rewrites work_dir/censored.flac; the opt-in DeepFilterNet pass re-runs CPU neural denoise over the full 1-2h track per format (minutes each); measure_loudness replays the full final audio per format. For source+9x16+1x1 with DFN on, a 2h video pays ~Nx the audio-pipeline cost it needs.

**Почему так.** The face-center hoist is a safe, serve-only change that removes N-1 face-detection passes with no behavioral change (same center fed explicitly). The audio-pipeline reuse is the larger saving but lives in render.py and must be verified with a real 3-format DFN benchmark — separated so the cluster ships the safe half now.

**Текущий код:**
```
    for i, f in enumerate(formats):
        if is_cancelled():
            break                                  # cancel между форматами
        label = _FORMAT_LABEL[f]
        opts_i = {**opts, "filename": stem + _FORMAT_SUFFIX[f]}
...
        try:
            cfg, scale_h, fps, out_dir, base = _resolve_render_opts(s, opts_i)
            res = _run_render_pipeline(
                s, cfg, scale_h, fps, out_dir, base,
                on_progress=prog, on_stage=stage,
                sidecar_base=out_dir / stem)
```

**Изменение:**
```
SERVE-SIDE (concrete, do now): hoist the face-center out of the per-format loop. Before the loop, if any requested non-source format needs a vertical/aspect crop and the client left center on 'auto', compute it ONCE (the kept range and source are identical across formats — cutlist_override is None in the formats path, so detect_center already scans the whole file):
    fixed_center = None
    wants_crop = any(f != "source" for f in formats)
    if wants_crop and str(opts.get("vertical_center", "auto") or "auto") == "auto":
        try:
            cx = facecrop_mod.detect_center(
                s.media.path, s.ff, s.media.duration,
                samples=s.cfg.render.vertical.samples, start=0.0, end=None)
            fixed_center = f"{min(1.0, max(0.0, float(cx))):.4f}"
        except Exception:  # noqa: BLE001 — detection is best-effort (0.5 fallback)
            fixed_center = None
Then in the loop, when building opts_i for a cropped format, inject the precomputed center so _run_render_pipeline takes the explicit-float branch (no re-detection):
    if fixed_center is not None:
        opts_i["vertical_center"] = fixed_center

RENDER-SIDE (deferred, needs verification + coordination with the render/R2 cluster): make render() reuse the cutlist-scoped audio artifacts instead of regenerating them per format. Key work_dir/censored.flac and the DeepFilterNet wav and the loudnorm measurement by a signature of (audio_hash + effective cutlist + censor/denoise cfg); on a matching signature, skip regeneration and reuse the file/stats. This is the bulk of the win but touches render.py's audio stage — spec it as a follow-up owned jointly with the render cluster; benchmark before/after (finding is 'unverified').
```

**Риск / безопасность.** Face hoist: detect_center is deterministic given (source, range, samples); feeding the precomputed float yields the SAME crop as per-format auto-detection — verify by asserting identical crop_filter across formats in a test. Only skip re-detection when center=='auto' (respect an explicit user center). detect_center never raises (0.5 fallback), but the try/except is belt-and-suspenders. Audio reuse is the risky part: a stale reuse across a changed cutlist would silently ship wrong audio — hence the strict signature key and a verification gate; do NOT land it without a real-ffmpeg test and a benchmark confirming the win. Unverified finding: quantify before investing in the render.py half.

**Тест (зафиксировать поведение).** tests/test_multiformat.py: test_facecrop_detected_once_across_formats — monkeypatch facecrop_mod.detect_center with a call-counter returning 0.33; run _render_formats for ['9x16','1x1','16x9'] with the render recorder; assert detect_center called exactly ONCE and every recorded crop_filter uses the same center. test_explicit_center_not_hoisted — pass vertical_center:'0.7'; assert detect_center NOT called. For the deferred audio half: add a slow real-ffmpeg benchmark (render 3 formats, DFN on) asserting censored.flac/DFN wav are produced once — plus a wall-clock assertion vs the per-format baseline to confirm the finding before shipping.

---
## 86. No property-based / fuzz testing for the interval math the whole pipeline stands on; 38 lines of hand-picked 2-interval cases guard Timeline
🟡 MEDIUM · симптом 1 · effort M · `timeline-property-fuzz`

- **Файл / якорь:** `tests/test_timeline_property.py` — vpipe.timeline (merge_intervals/Timeline/remap_words) + vpipe.subtitles.build_cues invariants
- **Закрывает находки:** tests/test_timeline.py:1, vpipe/timeline.py:84, vpipe/subtitles.py:120

**Проблема.** requirements.txt has no hypothesis; nothing generates randomized inputs. Timeline (merge/remap/kept_segments/removed_overlap) is covered by 5 tiny 2-interval tests; build_cues invariants (cues never overlap, end<=total, monotonic, all non-dropped words represented) are asserted for single hand cases only. These functions consume adversarial real input — hundreds of overlapping/touching/inverted/float-jittered intervals plus overlapping whisper word timestamps — where hand cases miss interactions. Symptoms 1 and 2 are both 'works on small inputs, breaks on real ones'; the absence of invariant fuzzing is a structural reason bugs slip.

**Текущий код:**
```
# tests/test_timeline.py:22-30 — the whole Timeline structural coverage is 2-interval cases:
def test_kept_segments():
    tl = Timeline([(1, 2), (5, 6)], duration=10)
    assert tl.kept_segments() == [(0, 1), (2, 5), (6, 10)]
def test_kept_full_when_no_cuts():
    tl = Timeline([], duration=10)
    assert tl.kept_segments() == [(0, 10)]
```

**Изменение:**
````
Add hypothesis to a dev-only requirements-dev.txt (comment: `hypothesis>=6,<7  # dev-only property tests; NOT installed in CI (see ci.yml)`), and guard the import so CI (which installs only requirements.txt + pytest + httpx) skips gracefully. New file tests/test_timeline_property.py:

```python
import pytest
hypothesis = pytest.importorskip('hypothesis')   # CI has no hypothesis -> whole module skipped, stays green
from hypothesis import given, strategies as st, settings
from vpipe.timeline import Timeline, merge_intervals, remap_words
from vpipe.subtitles import build_cues
from vpipe.config import SubsCfg, MaskingCfg, ProfanityLists
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Word

_ivs = st.lists(st.tuples(st.floats(0,100), st.floats(0,100)), max_size=40)

@given(_ivs)
def test_kept_and_removed_partition_duration(raw):
    D=100.0
    tl=Timeline(raw, D)
    kept=tl.kept_segments()
    # kept + removed cover [0,D] with no overlap, and durations sum to D
    total=sum(b-a for a,b in kept)+tl.total_removed
    assert abs(total-D)<1e-6
    for a,b in kept: assert 0<=a<=b<=D
    # kept segments are disjoint and ordered
    for (a1,b1),(a2,b2) in zip(kept, kept[1:]): assert b1<=a2

@given(_ivs, st.floats(0,100))
def test_remap_monotonic_and_in_range(raw, t):
    tl=Timeline(raw, 100.0)
    c=tl.remap_clamped(t)
    assert 0.0<=c<=tl.new_duration()+1e-9

@given(st.lists(st.tuples(st.floats(0,50), st.floats(0,50)), max_size=30))
def test_merge_intervals_disjoint_sorted(raw):
    m=merge_intervals(raw)
    for (a1,b1),(a2,b2) in zip(m, m[1:]):
        assert b1<a2 or b1<=a2       # strictly non-overlapping (touching merges)
        assert a1<=a2

# words strategy: sorted, positive-width
_words = st.lists(st.tuples(st.floats(0,90), st.floats(0.01,2.0)), min_size=1, max_size=40)

@settings(max_examples=200)
@given(_words, _ivs)
def test_build_cues_invariants(wspec, raw):
    words=[]
    for s,d in sorted(wspec):
        words.append(Word('слово', s, s+d))
    tl=Timeline(raw, 100.0)
    remapped=remap_words(words, tl)
    m=ProfanityMatcher(ProfanityLists(roots=[],allow=[]))
    total=tl.new_duration()
    cues=build_cues(remapped, m, SubsCfg(), MaskingCfg(), total)
    for c in cues:
        assert c.start<=c.end and c.end<=total+1e-6 and c.start>=0.0
    for c1,c2 in zip(cues, cues[1:]):
        assert c1.end<=c2.start+1e-6       # monotonic, non-overlapping
```
````

**Риск / безопасность.** hypothesis is behind importorskip so `python -m pytest -q` on CI (no hypothesis installed) skips this file entirely — CI stays green and the ffmpeg-ban philosophy is untouched. Some invariants may FAIL on discovery (that is the point — they surface real interaction bugs); when they do, capture the minimal counterexample hypothesis prints and either (a) file it against timeline/build_cues or (b) tighten the strategy if the input is genuinely impossible in production (e.g. words never exceed duration by >epsilon). Do NOT loosen an assertion to hide a real failure. Keep max_examples modest (<=200) so local runs stay fast.

**Тест (зафиксировать поведение).** test_kept_and_removed_partition_duration: for random interval sets, assert sum(kept durations)+total_removed == duration and kept segments are disjoint/ordered/in-range — a single property that fuzzes the partition math the entire cut pipeline depends on, far past the 2-interval hand cases.

---
## 87. Show Russian task labels in the progress strip and compute ETA from observed progress (fixes bogus ETA after mid-task reload)
⚪ LOW · симптом 5 · effort S · `progress-ru-labels-eta`

- **Файл / якорь:** `web/app.js` — TASK_TAB map (app.js:125); followTask() progress seed + onmessage ETA (app.js:2097-2119)
- **Закрывает находки:** web/app.js:2099

**Проблема.** followTask seeds the strip with the raw English task id (`setProgress(35, `${name}…`)`) and falls back to `t.stage || t.name || name`, so before the server sends a stage the otherwise-Russian UI shows 'render_clips…', 'preview_metadata…', 'autopack…'. Separately, ETA is computed from taskStart=Date.now() set at followTask time; after a reload re-attaches to a task already ~80% done, elapsed is measured from the reconnect, so 'осталось ~…' is wildly wrong (e.g. 30s at 80% claims ~7s left for a 20-minute render tail).

**Почему так.** Removes English task-id leakage from a Russian UI and fixes the nonsensical post-reload ETA by measuring the real observed rate instead of reconnect wall-clock.

**Текущий код:**
```
const TASK_TAB = { transcribe: 'cuts', detect: 'cuts', preview_chapters: 'chapters', preview_metadata: 'meta', preview_clips: 'clips', render_clips: 'clips', enrich: 'enrich' }

... (followTask, app.js:2096-2119)
function followTask(name) {
  setRunning(name); taskStart = Date.now()
  $('#progress').classList.remove('hidden'); $('#progress').classList.add('indeterminate')
  setProgress(35, `${name}…`)
  ...
  es.onmessage = (ev) => {
    let t; try { t = JSON.parse(ev.data) } catch { return }
    const pct = Math.max(0, Math.min(100, t.percent || 0))
    if (pct > 0) { $('#progress').classList.remove('indeterminate'); setProgress(pct) }
    let eta = ''
    if (pct >= 3 && pct < 100) {
      const elapsed = (Date.now() - taskStart) / 1000
      const remain = elapsed * (100 - pct) / pct
      if (remain >= 60) eta = `  · осталось ~${Math.round(remain / 60)}м`
      else if (remain > 1) eta = `  · осталось ~${Math.max(1, Math.round(remain))}с`
    }
    setProgress(pct, `${t.stage || t.name || name || ''} ${Math.round(pct)}%${eta}`)
```

**Изменение:**
```
1) Add a RU label map next to TASK_TAB (app.js:125):
const TASK_RU = { transcribe: 'Транскрипция', detect: 'Поиск вырезов', render: 'Рендер', preview_chapters: 'Главы', preview_metadata: 'Метаданные', preview_clips: 'Подбор клипов', render_clips: 'Рендер клипов', enrich: 'Монтаж', autopack: 'Авто-пак' }

2) Russian seed + reset ETA baseline in followTask:
  setRunning(name); taskStart = Date.now(); st._etaBase = null
  $('#progress').classList.remove('hidden'); $('#progress').classList.add('indeterminate')
  setProgress(35, `${TASK_RU[name] || name}…`)

3) ETA from observed delta since first message (correct after re-attach) + RU fallback label:
    if (pct > 0) { $('#progress').classList.remove('indeterminate'); setProgress(pct) }
    if (st._etaBase == null && pct > 0 && pct < 100) st._etaBase = { pct, t: Date.now() }
    let eta = ''
    if (st._etaBase && pct > st._etaBase.pct && pct < 100) {
      const dP = pct - st._etaBase.pct
      const dT = (Date.now() - st._etaBase.t) / 1000
      const remain = dT * (100 - pct) / dP
      if (remain >= 60) eta = `  · осталось ~${Math.round(remain / 60)}м`
      else if (remain > 1) eta = `  · осталось ~${Math.max(1, Math.round(remain))}с`
    }
    const label = t.stage || TASK_RU[t.name] || TASK_RU[name] || t.name || name || ''
    setProgress(pct, `${label} ${Math.round(pct)}%${eta}`)
```

**Риск / безопасность.** TASK_RU is additive; unknown ids fall back to the raw name (no worse than today). The delta-based ETA needs two data points, so no ETA shows on the very first message — acceptable and strictly better than a wrong number. Server-sent t.stage (already Russian, e.g. 'Главы…') still takes precedence. taskStart can be left declared (now unused for ETA) to avoid touching other refs. Alternative/more robust: have the server include started_at in /api/state.task and compute ETA from that — note as a follow-up, not required here.

**Тест (зафиксировать поведение).** Needs jsdom+vitest (web/tests/): call followTask('render_clips') → assert progressText seed contains 'Рендер клипов' not 'render_clips'. Feed two onmessage frames (pct 80 then 82, ~1s apart) simulating a re-attach → assert the ETA reflects the 2%/s observed rate (minutes-scale), not a ~7s value derived from reconnect time.

---
## 88. SD prompt gets the photographic style tail twice: _art_direction_prompt appends _ARTDIR_SUFFIX, then generate_image appends STYLE_SUFFIX on top
⚪ LOW · симптом none · effort ? · `artdir-drop-duplicate-style-suffix`

- **Файл / якорь:** `vpipe/enrich_llm.py` — _ARTDIR_SUFFIX constant + _art_direction_prompt() (~lines 1155-1165)
- **Закрывает находки:** vpipe/imagegen.py:168, vpipe/enrich_llm.py:1165

**Проблема.** _art_direction_prompt() appends _ARTDIR_SUFFIX (', cinematic lighting, shallow depth of field, photorealistic, professional photography, ultra detailed'), and generate_image() then appends STYLE_SUFFIX (a superset: cinematic/photorealistic/professional photography/shallow depth of field/ultra detailed + moody dark scene + teal/cyan grade). So every SD prompt carries ~5 duplicated style tokens, contradicting the code's own comment that imagegen owns the photo tail («здесь только subject-обогащение»). Not a hard 77-token overflow (subject is 2-6 words), but redundant weighting on both the candidate path and the legacy path.

**Почему так.** STYLE_SUFFIX already contains every token _ARTDIR_SUFFIX added (plus the grade), so dropping it loses nothing and removes the duplication the V2 router was built to avoid. Matches the module's stated design (imagegen owns the photo tail).

**Текущий код:**
```
_ARTDIR_SUFFIX = (", cinematic lighting, shallow depth of field, "
                  "photorealistic, professional photography, ultra detailed")


def _art_direction_prompt(query_en: str) -> str:
    """Голый английский query_en → БОГАТЫЙ арт-дирекшн-промпт (§5). Код-путь
    (без LLM): subject как есть + кинематографичный хвост. Пусто/русский → ""."""
    q = " ".join((query_en or "").split())
    if not q or _CYRILLIC.search(q):
        return ""
    return q + _ARTDIR_SUFFIX
```

**Изменение:**
```
Delete _ARTDIR_SUFFIX and return the bare subject (imagegen's STYLE_SUFFIX — a superset — owns the photo tail):

def _art_direction_prompt(query_en: str) -> str:
    """Голый английский query_en → subject для диффузии (§5). Фото-хвост
    (свет/линза/grade) добавляет imagegen.generate_image через STYLE_SUFFIX —
    здесь НЕ дублируем его. Пусто/русский → ""."""
    q = " ".join((query_en or "").split())
    if not q or _CYRILLIC.search(q):
        return ""
    return q
```

**Риск / безопасность.** LOW. Changes the stored candidate prompt (and thus the SD cache key), so cached diffusion images regenerate once — acceptable. Empty/Cyrillic guard preserved, so the `if prompt and run_diffusion` gate in _build_candidates is unaffected.

**Тест (зафиксировать поведение).** Update tests/test_enrich_llm.py:749 test_art_direction_prompt_rich_or_empty — its line 752 asserts 'photorealistic'/'depth of field' are IN the prompt; change to assert _art_direction_prompt('server room datacenter') == 'server room datacenter' (bare subject, no duplicated style tail), keep the ''→'' and Cyrillic→'' cases. Optionally add an imagegen assertion that the final -p (query + STYLE_SUFFIX) contains 'photorealistic' exactly once.

---
## 89. SD model at an external temp-ish path (D:\tmp\enrich2\sd\...) — a cleanup silently disables the whole photo tier; imagegen_vae is unset so the color-stability VAE is never used
⚪ LOW · симптом none · effort ? · `sd-model-vae-repo-local`

- **Файл / якорь:** `config.yaml` — render.imagegen block (~lines 91-94) + .gitignore
- **Закрывает находки:** config.yaml:94, vpipe/config.py:303

**Проблема.** imagegen_model points outside the repo at D:\tmp\enrich2\sd\models\sdxl-turbo-Q4_0.gguf (uncommitted local edit; file exists today). If D:\tmp is cleaned, _resolve_model returns None, _sd_configured flips false, and all photo points degrade to none/emoji. (The UI DOES warn via #enGenWarnRow, so it's signposted — but the model lives in a cleanup-prone dir.) Separately, imagegen_vae is unset (default '' in config.py:303), so sdxl_vae_fp16fix.safetensors — which exists next to the model — is never passed via --vae, leaving a free color-stability win on the table.

**Почему так.** Repo-relative paths resolve via _REPO_ROOT and are immune to temp-dir cleanup; a single `/models/` gitignore entry keeps the big binaries out of git while making the config portable. Wiring imagegen_vae activates the fp16fix VAE the code already supports (--vae, imagegen.py:206) for more stable color.

**Текущий код:**
```
  # LOCAL-ONLY (do not commit absolute paths): enable AI image generation.
  imagegen:
    imagegen_enabled: true
    imagegen_model: D:\tmp\enrich2\sd\models\sdxl-turbo-Q4_0.gguf
```

**Изменение:**
```
Relocate both files into a repo-local, gitignored dir and wire the VAE (config.yaml stays LOCAL-ONLY / uncommitted). _resolve_model already resolves repo-root-relative paths (imagegen.py:97, _REPO_ROOT / configured), so:

1) Ops step: move sdxl-turbo-Q4_0.gguf AND sdxl_vae_fp16fix.safetensors from D:\tmp\enrich2\sd\models\ to <repo>/models/.
2) .gitignore — add one line (the ONLY committable change here):
    /models/
3) config.yaml imagegen block:
  # LOCAL-ONLY (do not commit absolute paths): enable AI image generation.
  imagegen:
    imagegen_enabled: true
    imagegen_model: models/sdxl-turbo-Q4_0.gguf        # repo-local (models/ gitignored)
    imagegen_vae: models/sdxl_vae_fp16fix.safetensors  # §5.6 стабильный цвет
```

**Риск / безопасность.** LOW. config.yaml is an uncommitted machine edit — keep it that way (do NOT commit the imagegen block). The relative-path resolution is already exercised by _resolve_sd_bin('tools/sd-cli.exe'); verify _resolve_model('models/...') returns the moved file after relocation. If the files are not moved, SD simply stays gracefully disabled as today.

**Тест (зафиксировать поведение).** tests/test_imagegen.py already covers repo-root-relative resolution for the binary (test_resolve_sd_bin_repo_root_relative); add test_resolve_model_repo_root_relative mirroring it — monkeypatch _REPO_ROOT to tmp_path, create tmp_path/'models'/'m.gguf', assert imagegen._resolve_model('models/m.gguf') == str(that file). (No committed config change to test; the relocation is operational.)

---
## 90. Document cut_fade and min_segment in config.yaml's render section (min_segment is otherwise tunable only via an undocumented key)
⚪ LOW · симптом none · effort S · `config-yaml-cutfade-minseg-docs`

- **Файл / якорь:** `config.yaml` — render: section, after `faststart: true` (line 106), before the commented `vertical:` block
- **Закрывает находки:** vpipe/config.py:347

**Проблема.** RenderCfg defines cut_fade=0.015 (config.py:347) and min_segment=0.04 (config.py:353) — the seam de-click / sliver-merge fix — and render.py consumes both (render.py:874, 1170), but config.yaml's render section documents every other knob (vertical/denoise/music each get detailed commented blocks) yet contains neither key. A user tuning cut quality from the file cannot discover them; min_segment is worst off — it has NO UI control and NO serve.py render_opt, so it is tunable exclusively via an undocumented yaml key. (Note: the denoise_highpass render_opt at serve.py:987 has no UI element but is NOT dead — it is reachable via config.yaml's commented highpass_hz and via direct API — so leave it intact; this patch only adds the missing cut_fade/min_segment docs.)

**Текущий код:**
```
  audio_bitrate: 320k
  faststart: true         # move moov atom to front for web/YouTube
```

**Изменение:**
```
Insert a commented block (matching the style of the surrounding vertical/denoise blocks) immediately after the faststart line:

  audio_bitrate: 320k
  faststart: true         # move moov atom to front for web/YouTube
  # Сглаживание стыков резов (de-click). Короткий fade-out/fade-in на каждом
  # ВНУТРЕННЕМ шве убирает щелчок и ощущение «рвано». Длино-сохраняющий (без
  # наложения) → A/V не рассинхронятся. Применяется только к внутренним швам,
  # не к истинному началу/концу клипа. UI-слайдер #rCutFade шлёт мс (кламп
  # 0..0.06 с). 0 = старые жёсткие резы.
  # cut_fade: 0.015        # секунд фейда с каждой стороны шва
  # Оставленные «огрызки» короче этого (сек) выбрасываются, чтобы два близких
  # реза слились, а не оставляли заикающийся блип. Держите НИЖЕ самого короткого
  # реального слова (~60-80 мс): убирает только вдохи/VAD-хвосты, не речь. UI-
  # контрола НЕТ — настраивается только отсюда. 0 = дропать лишь суб-кадровые.
  # min_segment: 0.04
```

**Риск / безопасность.** Zero functional risk — commented YAML is inert; defaults come from config.py regardless. Only requirement: keep the values (0.015 / 0.04) in sync with config.py:347/353 so the documentation isn't misleading. Optionally, as a SEPARATE follow-up (not this patch), add a min_segment slider next to #rCutFade and/or a highpass control next to the denoise strength slider to close the UI gap — out of scope for a docs-only change.

**Тест (зафиксировать поведение).** No code test (config comment). Optional guard: add tests/test_config_docs.py::test_render_yaml_documents_cut_knobs asserting config.yaml text contains 'cut_fade:' and 'min_segment:' (even commented) so the docs can't silently regress; and asserting the documented default numerals match RenderCfg().cut_fade / .min_segment to keep them in sync.

---
## 91. Optional (default-OFF) module-level WhisperModel cache so a batch queue of same-model clips can skip repeated construction — with explicit eviction before VRAM-competing stages
⚪ LOW · симптом none · effort M · `whisper-model-cache-optin`

- **Файл / якорь:** `vpipe/transcribe.py` — _run_once(), lines 248-296 (WhisperModel construct + finally del) — add module cache gated by a default-OFF config flag
- **Закрывает находки:** vpipe/transcribe.py:294

**Проблема.** _run_once constructs WhisperModel per call and its finally block does `del model; _free_gpu()`, so a sequential queue of N same-model clips pays N cold constructions. HOWEVER (per the finding's own corrected verdict) the payoff is weak on the 8GB card: each queue job also runs an LLM-heavy detect stage on qwen3:8b (~5-6GB) which cannot coexist with a resident large-v3 (~3GB), so a naive cross-job cache would OOM or would have to be evicted before each job's detect anyway. Steady-state cost is also lower than a cold disk read (the 3GB file sits in the 64GB-RAM page cache), so this is CTranslate2 init + VRAM upload, not 15-45s.

**Текущий код:**
```
    try:
        model = _load_model(size, device, ctype, log)
    finally:
        if stop_watch is not None:
            try:
                stop_watch()
            except Exception:
                pass  # never let the watcher mask the real outcome
    ...
    finally:
        del model
        _free_gpu()
```

**Изменение:**
```
Add a config flag `cache_model: bool = False` to TranscribeCfg (config.py) with a comment: '# держать модель Whisper в VRAM между заданиями очереди (эконом на N клипов);\n# по умолчанию ВЫКЛ — на 8ГБ карте она конфликтует с qwen3 detect/SD. Требует\n# явной выгрузки перед LLM/рендером.' Then, gated by that flag, add a module cache:

    _MODEL_CACHE: dict = {}   # (size, device, ctype) -> WhisperModel

    def release_whisper_cache():
        """Evict any cached WhisperModel + free VRAM. MUST be called before any
        VRAM-competing stage (LLM detect, SD, render), mirroring the Ollama
        _wait_ollama_unloaded pattern."""
        _MODEL_CACHE.clear()
        _free_gpu()

In _run_once, when cfg.cache_model is True: look up `_MODEL_CACHE.get((size,device,ctype))`; construct + store on miss; and in the finally block SKIP `del model; _free_gpu()` when caching is on (leave the cache to own the lifetime). When cfg.cache_model is False the code path is byte-for-byte the current behavior. The queue worker (serve.py:_queue_process_one, before the detect stage) and the render/SD entrypoints must call transcribe.release_whisper_cache() so a warm model never collides with qwen3/SD.
```

**Риск / безопасность.** Default-OFF makes the default behavior byte-for-byte identical — zero regression risk unless a user opts in. The real hazard when enabled is OOM: a resident large-v3 next to qwen3:8b on 8GB will fail. This is why release_whisper_cache() MUST be wired into every VRAM-competing entry (LLM detect at serve.py:1536/262/275, SD, NVENC render) — treat that wiring as part of the patch, not optional. Because the corrected verdict shows the win only materializes when LLM detection is disabled, keep the flag off in shipped config.yaml and document it as advanced. Do NOT cache unconditionally.

**Тест (зафиксировать поведение).** Add tests/test_transcribe_cache.py::test_no_cache_constructs_each_call — monkeypatch _load_model to count constructions; run two _run_once with cfg.cache_model=False; assert 2 constructions and the cache dict empty. ::test_cache_reuses_when_enabled — cfg.cache_model=True, two runs same (size,device,ctype); assert _load_model called once and the second reuses. ::test_release_evicts — after a cached run call release_whisper_cache(); assert cache empty and a subsequent run reconstructs. (All with a fake WhisperModel whose .transcribe returns an empty iterator + info stub — no CUDA needed.)

---
## 92. Every single-word edit re-serializes and rewrites the entire multi-MB transcript JSON synchronously (UNVERIFIED — benchmark first)
⚪ LOW · симптом 5 · effort L · `transcript-flush-debounce`

- **Файл / якорь:** `serve.py` — put_transcript_word cache write, serve.py 2203-2216
- **Закрывает находки:** serve.py:2212

**Проблема.** PUT /api/transcript/word mutates one word then does a full json dump of all ~15k words (3-8 MB for a 2h video) via tmp+os.replace on every edit. Rapid proofreading pays a few hundred ms of serialize+write per keystroke-fix. Correctness is fine (atomic); the cost is redundant I/O. NOTE: this finding's verdict is 'unverified' — confirm the cost on the real hardware before adding lifecycle machinery.

**Почему так.** The in-memory mutation (w.word / segs[si].text) is already applied before the write, so the disk flush can be coalesced without affecting the response or GET /api/transcript (which reads memory, not disk). Forced flushes on task-start and shutdown bound the loss window to a crash within ~1s.

**Текущий код:**
```
    cache_file = s.cache_dir / f"{s.audio_hash}.transcript.json"
    tmp = cache_file.with_name(f"{cache_file.name}.{uuid.uuid4().hex}.tmp")
    try:
        s.transcript.save(tmp)
        os.replace(tmp, cache_file)
    finally:
        tmp.unlink(missing_ok=True)   # no-op после удачного replace
    return {"ok": True, "text": w.word, "segment_text": segs[si].text}
```

**Изменение:**
```
GATE ON A BENCHMARK FIRST (see test), then debounce the flush. Extract the atomic write into a helper and schedule it instead of running it inline:

# module scope, near TASK_LOCK/QUEUE_LOCK:
_TR_FLUSH_LOCK = threading.Lock()
_tr_flush_timer: Optional[threading.Timer] = None

def _flush_transcript_now(s) -> None:
    if s is None or s.transcript is None:
        return
    cache_file = s.cache_dir / f"{s.audio_hash}.transcript.json"
    tmp = cache_file.with_name(f"{cache_file.name}.{uuid.uuid4().hex}.tmp")
    try:
        s.transcript.save(tmp)
        os.replace(tmp, cache_file)
    finally:
        tmp.unlink(missing_ok=True)

def _schedule_transcript_flush(s, delay: float = 1.0) -> None:
    global _tr_flush_timer
    with _TR_FLUSH_LOCK:
        if _tr_flush_timer is not None:
            _tr_flush_timer.cancel()
        _tr_flush_timer = threading.Timer(delay, _flush_transcript_now, args=(s,))
        _tr_flush_timer.daemon = True
        _tr_flush_timer.start()

# in put_transcript_word, REPLACE the inline tmp/save/replace block with:
    # In-memory edit already applied above; coalesce the multi-MB disk flush
    # across a burst of edits (~1s idle). Forced flush on task start and
    # server shutdown guarantees no edit is lost.
    _schedule_transcript_flush(s)
    return {"ok": True, "text": w.word, "segment_text": segs[si].text}

# REQUIRED force-flush hooks (else edits within the debounce window are lost on a hard exit):
#  * at the top of start_task(), before launching the worker: cancel the pending timer and call _flush_transcript_now(SESSION);
#  * on server shutdown (register an atexit handler or a FastAPI shutdown event): _flush_transcript_now(SESSION).

# INTERIM zero-risk win you may ship immediately instead: skip the write when the edit is a no-op — guard the block with `if (prefix + new) != old:` (old computed at serve.py:2199).
```

**Риск / безопасность.** UNVERIFIED finding — do not ship the timer machinery until the benchmark confirms the cost is real on the RTX-3080 box; the debounce adds genuine lifecycle complexity (timer lifetime, a Session that could be replaced by open_video between schedule and fire — mitigate by capturing SESSION at schedule time and no-oping if s.transcript is None, and by force-flushing in open_video/close paths). Failure mode if hooks are missed: an edit made <1s before a hard kill is lost (acceptable for proofreading, but only with the shutdown+task-start flushes wired). The interim no-op-skip is safe and independent. Preserve the atomic tmp->os.replace and the uuid-unique tmp name (fixes the parallel-PUT PermissionError noted at serve.py:2205-2208).

**Тест (зафиксировать поведение).** tests/test_transcript_edit.py: (1) BENCHMARK/CONFIRM — add a test that PUTs 20 consecutive word edits and counts how many times the cache file is rewritten (monkeypatch _flush_transcript_now / Transcript.save to increment a counter); today it is 20, assert the debounced version collapses a rapid burst to <20 while the final on-disk content still reflects all 20 edits after a forced flush. (2) CORRECTNESS — after edits, call the shutdown/force-flush hook and assert the on-disk JSON contains the last edit (survives 'restart'); and assert GET /api/transcript returns every edit immediately (memory path) even before the timer fires. (3) keep the existing test_edit_word_updates_cache_on_disk green by having it trigger a forced flush (or await the timer) before reading the file.

---
## 93. async upload does synchronous f.write on the event-loop thread — a multi-GB upload stalls all other responses in bursts
⚪ LOW · симптом 2 · effort S · `upload-async-write`

- **Файл / якорь:** `serve.py` — upload() stream loop, serve.py 2078-2086
- **Закрывает находки:** serve.py:2085

**Проблема.** upload() is the one async endpoint doing sustained blocking I/O: `async for chunk in request.stream(): ...; f.write(chunk)`. On a single-loop uvicorn, loopback receive outruns disk write, so a multi-GB upload blocks the loop in bursts and every other in-flight response (video-seek Range requests, polls, static files) stalls, making the app feel hung.

**Почему так.** anyio is already imported (serve.py:62, a hard Starlette dependency), so run_sync offloads each write to the threadpool and keeps the loop free with a one-line, structure-preserving change.

**Текущий код:**
```
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Файл слишком большой (>30 ГБ). "
                                             "/ File too large (>30 GB).")
                f.write(chunk)
        os.replace(tmp, dest)
```

**Изменение:**
```
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Файл слишком большой (>30 ГБ). "
                                             "/ File too large (>30 GB).")
                # f.write is a blocking syscall; on a single-loop uvicorn a
                # multi-GB upload would otherwise stall every other in-flight
                # response in bursts (loopback receive outruns disk write).
                # Offload it so the event loop stays responsive.
                await anyio.to_thread.run_sync(f.write, chunk)
        os.replace(tmp, dest)
```

**Риск / безопасность.** No new dependency. One threadpool hop per chunk adds minor overhead; if that matters for tens of thousands of tiny chunks, buffer to ~4-8 MB before offloading, or switch to `f = await anyio.open_file(tmp,'wb'); await f.write(chunk)`. Corrected-verdict caveat: upload is gated by _guard_no_task (409 while any task runs) and the SSE stream self-closes when idle, so this NEVER freezes render progress — impact is only a sluggish UI during large uploads on an otherwise idle server; keep severity low. The 413 cap and the .part->os.replace atomic pattern are unchanged.

**Тест (зафиксировать поведение).** Async regression (correctness, not timing): POST a small multi-chunk body to /api/upload via Starlette TestClient with work_dir monkeypatched to tmp; assert 200 and the written file bytes exactly equal the posted bytes in order (proves the anyio offload didn't reorder/corrupt the stream). Optionally monkeypatch anyio.to_thread.run_sync to a counting passthrough and assert it was awaited once per chunk. A true non-blocking-latency assertion is timing-flaky — cover with a manual/perf note instead.

---
## 94. Uploaded videos in work/_uploads are never garbage-collected — multi-GB disk leak per upload
⚪ LOW · симптом 2 · effort M · `upload-gc-janitor`

- **Файл / якорь:** `serve.py` — new helper near startup sweep + call in main() after the *.part sweep (serve.py ~4420-4430)
- **Закрывает находки:** serve.py:2070

**Проблема.** Every drag-and-drop upload streams up to 30 GB into work/_uploads/<name> and stays forever; the only deletion is a failed upload's .part, and the startup sweep removes only *.part scraps. Over weeks the drive fills, and ffmpeg/NVENC/SD then fail mid-render with opaque errors (feeds the 'long renders crash' perception).

**Почему так.** A startup janitor that protects the open session's input and every queued job's path deletes only genuinely stale uploads. Placing the call after _load_queue()/open_session guarantees the keep-set is complete. Wrapped in try/except so it can never block startup.

**Текущий код:**
```
    # Sweep stale *.part scraps left by an interrupted render/upload.
    try:
        for d in (Path(cfg.paths.work_dir), Path(cfg.paths.out_dir)):
            if d.exists():
                for p in d.rglob("*.part"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
    except Exception:  # noqa: BLE001 — best-effort cleanup, never block startup
        pass
```

**Изменение:**
```
Add a helper (near the other module-level helpers, e.g. just above main()):

def _sweep_uploads(cfg, keep_days: float = 7.0) -> None:
    """Best-effort GC for work/_uploads: delete files older than keep_days that
    are NOT open in the editor and NOT referenced by any queue job. Never
    raises — a cleanup failure must not block startup."""
    try:
        updir = Path(cfg.paths.work_dir) / "_uploads"
        if not updir.exists():
            return
        keep: set[str] = set()
        s = SESSION
        if s is not None:
            try:
                keep.add(os.path.normcase(str(s.inp.resolve())))
            except OSError:
                pass
        with QUEUE_LOCK:
            for j in QUEUE:
                try:
                    keep.add(os.path.normcase(str(Path(j.path).resolve())))
                except OSError:
                    continue
        cutoff = time.time() - keep_days * 86400.0
        for p in updir.iterdir():
            try:
                if not p.is_file() or p.suffix.lower() == ".part":
                    continue
                if os.path.normcase(str(p.resolve())) in keep:
                    continue
                if p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass
    except Exception:  # noqa: BLE001 — never block startup
        pass

Then call it in main() immediately AFTER the existing *.part sweep block (so _load_queue() at ~4395 and open_session at ~4415 have already populated QUEUE and SESSION):

    _sweep_uploads(cfg)
```

**Риск / безопасность.** Only touches work/_uploads, only files older than keep_days (default 7), never the open clip or a queued clip. Runs once at startup (no request-path cost). Scope is deliberately the multi-GB _uploads leak; the finding's secondary items (per-video work/<stem>-<hash>/ dirs, cache/ LRU, /api/health disk usage) are a larger design — note as follow-up, do not bundle. Consider promoting keep_days to a config knob (paths.upload_keep_days) if the user wants control.

**Тест (зафиксировать поведение).** tests/test_upload_gc.py (new, no ffmpeg): build work/_uploads with (a) old unreferenced file (os.utime mtime=now-30d) -> deleted; (b) old file whose path is a QUEUE job -> kept; (c) old file == SESSION.inp -> kept; (d) recent file -> kept; (e) an old *.part -> untouched by _sweep_uploads. Wire serve.APP cfg + serve.QUEUE + serve.SESSION via monkeypatch (like test_queue fixtures), call serve._sweep_uploads(cfg, keep_days=7) and assert the surviving set is exactly {b,c,d,e}.

---
## 95. No eviction for cache/enrich_img + cache/codegfx image caches (grow unboundedly) — add an OPT-IN, default-off age-based startup sweep
⚪ LOW · симптом none · effort M · `cache-lru-sweep`

- **Файл / якорь:** `serve.py, vpipe/config.py` — startup sweep block in main() (~serve.py:4420-4430); PathsCfg (~vpipe/config.py:16-19)
- **Закрывает находки:** serve.py:4424

**Проблема.** The only cleanup is the startup *.part sweep + DFN temp wavs. cache/enrich_img (768x768 PNGs, 4 candidates per point per prompt/seed signature) and cache/codegfx PNGs accumulate forever — old keys are never pruned when prompts/fields change. A user processing a video/day slowly loses GB. (Verdict: UNVERIFIED — the disk-growth magnitude was not reproduced; ship conservatively and confirm before enabling.)

**Почему так.** An opt-in, correctness-cache-preserving age sweep bounds the only genuinely unbounded caches while defaulting to today's behavior until the growth claim is validated.

**Текущий код:**
```
    # Sweep stale *.part scraps left by an interrupted render/upload.
    try:
        for d in (Path(cfg.paths.work_dir), Path(cfg.paths.out_dir)):
            if d.exists():
                for p in d.rglob("*.part"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
    except Exception:  # noqa: BLE001 — best-effort cleanup, never block startup
        pass

    # PathsCfg (vpipe/config.py:16-19)
    class PathsCfg(_Base):
        out_dir: str = "./out"
        cache_dir: str = "./cache"
        work_dir: str = "./work"
```

**Изменение:**
```
1) Add an opt-in, default-OFF config knob to PathsCfg (default 0 -> never evict, preserving current behavior):

    class PathsCfg(_Base):
        out_dir: str = "./out"
        cache_dir: str = "./cache"
        work_dir: str = "./work"
        # Startup LRU sweep of the IMAGE caches only (enrich_img/codegfx). 0 =
        # disabled (default; unbounded, as today). >0 deletes PNGs older than N
        # days by mtime. Never touches transcript/peaks (correctness caches) or
        # active work dirs.
        cache_img_max_age_days: int = 0

2) After the *.part sweep in main(), add a guarded best-effort sweep:

    # Opt-in LRU sweep of image caches (enrich_img/codegfx). Default off.
    try:
        max_age = int(getattr(cfg.paths, "cache_img_max_age_days", 0) or 0)
        if max_age > 0:
            cutoff = time.time() - max_age * 86400
            for sub in ("enrich_img", "codegfx"):
                d = Path(cfg.paths.cache_dir) / sub
                if not d.exists():
                    continue
                for p in d.rglob("*.png"):
                    try:
                        if p.stat().st_mtime < cutoff:
                            p.unlink()
                    except OSError:
                        pass
    except Exception:  # noqa: BLE001 — best-effort, never block startup
        pass

(time is already imported at serve.py:23.)
```

**Риск / безопасность.** HIGHEST-uncertainty patch in the cluster (verdict unverified). Kept safe by: default 0 = exact current behavior (no deletion); scope limited to the two IMAGE caches (never transcripts/peaks, whose loss forces expensive re-transcription; never work/ dirs, which may be active for an open session); mtime-only, guarded, best-effort. Deleting a PNG still referenced by a saved enrich.json only forces a re-render on next use (resolved_asset drops a missing path gracefully), not a crash. Recommend: confirm the multi-GB growth on the user's machine and pick a sane default (e.g. 30) only after validating; consider surfacing a UI hint before auto-deleting.

**Тест (зафиксировать поведение).** tests/test_cache_sweep.py::test_img_cache_sweep_opt_in — create cache/enrich_img/old.png (mtime = now-40d) and fresh.png (now); with cache_img_max_age_days=0 assert BOTH survive after the sweep; with =30 assert old.png removed and fresh.png kept; assert cache/<hash>.transcript.json and cache/<hash>.peaks.json are NEVER touched regardless of age. Drive the sweep via a small extracted helper (e.g. _sweep_img_cache(cfg)) so it's unit-testable without booting uvicorn.

---
## 96. Schematic PNGs re-render from scratch (headless Chrome) on every suggest run — serve computes a cache path but never checks it before invoking render_schematic
⚪ LOW · симптом none · effort S · `schematic-cached-render`

- **Файл / якорь:** `serve.py` — _run_schematic_candidates() (~line 3457-3482) + delete _codegfx_cache_path (~3424-3432)
- **Закрывает находки:** serve.py:3470, serve.py:3424

**Проблема.** _run_schematic_candidates builds a deterministic path via _codegfx_cache_path then calls render_schematic UNCONDITIONALLY; render_schematic has no idempotence check, so every suggest re-launches headless Chrome for every schematic candidate even when the identical PNG already exists (~0.5-1s + Chrome startup each). serve's sha1 key ({i,s,f}) also differs from codegfx.cache_key (which adds normalized style + WxH), so the same candidate could occupy two cache entries and a size change would serve a stale-sized PNG (latent today only because it always re-renders).

**Почему так.** Routing through the existing idempotent, size-aware cache wrapper eliminates redundant headless-Chrome launches and collapses the two divergent cache keys into one.

**Текущий код:**
```
    try:
        from vpipe.codegfx.render import render_schematic
    except Exception as e:  # noqa: BLE001 — codegfx ещё не готов / нет Chrome
        log(f"  Код-схема: движок недоступен ({e}) — schematic-кандидаты без "
            "превью (UI покажет none/диффузию).")
        return 0
    cache_root = Path("cache").resolve()
    s.stage(f"Монтаж: код-схемы ({len(targets)})…")
    made = 0
    for c, _pl in targets:
        intent = c.get("intent", "")
        style = c.get("style", "minimal")
        fields = c.get("fields") or {}
        out_png = _codegfx_cache_path(cache_root, intent, style, fields)
        try:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            path = render_schematic(intent, fields, style, out_png,
                                    cfg=cfg_cg, log=log)
        except Exception as e:  # noqa: BLE001 — одна схема не валит остальные
            log(f"  Код-схема «{intent}» упала ({e}).")
            path = None
        if path:
            c["asset_path"] = str(path)
            c["preview"] = str(path)
            made += 1
    return made
```

**Изменение:**
```
Use the idempotent cached variant (it computes cache/codegfx/<sha1>.png with a WxH-aware key and skips Chrome when a real PNG already exists) and drop serve's duplicate path helper:

    try:
        from vpipe.codegfx.render import render_schematic_cached
    except Exception as e:  # noqa: BLE001 — codegfx ещё не готов / нет Chrome
        log(f"  Код-схема: движок недоступен ({e}) — schematic-кандидаты без "
            "превью (UI покажет none/диффузию).")
        return 0
    s.stage(f"Монтаж: код-схемы ({len(targets)})…")
    made = 0
    for c, _pl in targets:
        intent = c.get("intent", "")
        style = c.get("style", "minimal")
        fields = c.get("fields") or {}
        try:
            path = render_schematic_cached(intent, fields, style, cfg=cfg_cg, log=log)
        except Exception as e:  # noqa: BLE001 — одна схема не валит остальные
            log(f"  Код-схема «{intent}» упала ({e}).")
            path = None
        if path:
            c["asset_path"] = str(path)
            c["preview"] = str(path)
            made += 1
    return made

Also DELETE the now-unused _codegfx_cache_path helper (serve.py:3424-3432). Verify no other reference: `grep -n _codegfx_cache_path serve.py` should show only the definition.
```

**Риск / безопасность.** render_schematic_cached returns an ABSOLUTE path (base=_REPO_ROOT/cache/codegfx resolved), matching the old code's absolute str(out_png) — resolved_asset()'s is_absolute() guard (enrich.py:504) stays satisfied. Location is the same cache/codegfx dir (module CACHE_DIR = Path('cache')/'codegfx', anchored to repo root vs the old CWD-relative resolve — equivalent when CWD==repo root, and more robust otherwise). One-time cost: the WxH-aware key differs from serve's old {i,s,f} key, so the first post-change run re-renders once, then caches. Fixes the latent dual-entry/stale-size bug for free. Preserve the ImportError guard and the per-candidate try/except.

**Тест (зафиксировать поведение).** tests/test_codegfx_render.py::test_render_schematic_cached_idempotent (or extend existing codegfx tests): monkeypatch render._screenshot to write a valid PNG and increment a counter; call render_schematic_cached twice with identical (intent, fields, style, cfg); assert the counter == 1 (second call served from cache, no Chrome). Integration: a suggest-level test asserting _run_schematic_candidates on a plan whose schematic PNG already exists does not call render_schematic (monkeypatch render.render_schematic to fail the test if invoked when _is_real_png is true).

---

# Приложение — заметки по кластерам (порядок / общие хелперы / мержи)

### serve-render-settings

10 findings → 7 patches (three merges). MERGES: (a) #7 cutlist-source-validation + #8 defer-ctor-detection are ONE patch — they rewrite the SAME Session.__init__ load block (serve.py:255-259) and only compose correctly together (a rejected stale cutlist must leave self.cutlist=None so the deferred fresh detect can't merge stale manual cuts). (b) #4 bool-quality/fps + #5 odd-scale_h + #6 audio_bitrate are ONE patch — all are "harden _resolve_render_opts against non-UI (API/queue) callers" in the same function. (c) #3 double-stem and #9 same-stem-collision stay SEPARATE (different fixes at the same anchor) but #9 depends_on #3 because both use the new _with_ext() helper.

SHARED HELPERS introduced (define once in serve.py, reused across patches): _clean_stem(raw, fallback) [patch output-filename], _with_ext(base, ext) [patch output-filename, reused by same-stem-collision], _cutlist_matches() method [patch cutlist-guard].

CROSS-CLUSTER / THEME DEPENDENCIES:
- R1 (single effective-cutlist helper): render-output-validate compares the probed mp4 duration against render()'s reported new_duration. render.py returns tl.new_duration() PRE-sliver-drop (render.py:868/1234) while it actually encodes the sliver-FILTERED kept set (render.py:876). Until R1 makes render() return the effective (sliver-filtered) duration, the expected value overstates by the dropped slivers (each < min_segment=0.04s or <1 frame); the 0.5s+1frame tolerance absorbs a handful but a pathological many-sliver video could false-alarm. Patch is written one-sided (only raise when SHORTER) and depends_on the R1 render.py:876 fix for exactness.
- models.py: cutlist-guard adds an optional `audio_hash: str = ""` field to CutList (to_dict/from_dict). Low-conflict additive change; flagged for the implementer.
- subtitles.py:228/232: output-filename patch needs a companion 2-line edit (with_suffix → string append) so dotted-stem sidecar .srt/.vtt names aren't re-truncated; identical output for non-dotted stems.

APPLY ORDER: wave 1 first (output-filename introduces _with_ext/_clean_stem used by later patches), then wave 3, 4, 5. same-stem-output-collision (wave 4) depends_on output-filename (wave 1).

### serve-flow-1500-2500

MERGES: the two serve.py:2231 findings (low "edge-case" + medium "edge-cases") are one and the same defect and are merged into a single patch `cutlist-validate-put` (source_lines lists both). ORTHOGONAL TO R1/R2/R3: none of these findings touch the shared effective-cutlist helper, the filter-graph-to-file work, or the UI<->backend contract — this cluster is queue/watch/upload/guard flow, so every patch is self-contained (no shared helper to introduce). LOCK-ORDER INVARIANT (verified by grep of every TASK_LOCK/QUEUE_LOCK/WATCH_LOCK site in serve.py): acquisition order is TASK_LOCK -> QUEUE_LOCK, and WATCH_LOCK is never nested with the other two. No code path takes QUEUE_LOCK then TASK_LOCK, so patch `guard-queue-toctou` introducing a TASK_LOCK->QUEUE_LOCK nesting in _queue_worker cannot deadlock; keep this invariant when applying. ORDER: apply wave 3 (queue-detect-overwrite, watch-registry-gen) then wave 4 (cutlist-validate-put, guard-queue-toctou, watch-locked-file-open) then wave 5 (upload-gc-janitor, upload-async-write, transcript-flush-debounce). The two upload patches (upload-gc-janitor, upload-async-write) both live in/near the /api/upload endpoint but edit disjoint regions — no depends_on. transcript-flush-debounce is the only UNVERIFIED finding (verdict=unverified): gate it behind a confirming benchmark before shipping the debounce machinery.

### serve-flow-2500plus

9 findings → 9 patches (no merges; each is a distinct fix). Ordering by wave then severity: WAVE 1 stability = [serve-shutdown-cancel-all, queue-stop-guard]; WAVE 3 Монтаж = [enrich-merge-content-key (HIGH), enrich-select-flat-resync]; WAVE 4 UI-resilience = [events-ctor-cancelled-key, autopack-results-snapshot, enrich-clips-uuid-tmp]; WAVE 5 perf = [schematic-cached-render, cache-lru-sweep].

Cross-patch notes:
- `ffmpeg_utils.cancel_all()` is the shared kill primitive touched by BOTH serve-shutdown-cancel-all (add owners: atexit+shutdown) and queue-stop-guard (gate on _queue_running). Land them together; they don't conflict (one adds callers, one guards a caller).
- The task dict (Session.__init__, serve.py:247-248) is touched by events-ctor-cancelled-key (add "cancelled": False). autopack-results-snapshot touches s.task["results"] publishing but not the ctor — independent, but both are "task-dict contract" hygiene; review as a pair.
- `vpipe/enrich.py:save_enrich` is edited by enrich-clips-uuid-tmp (uuid tmp). It is the writer used by enrich-select-flat-resync and enrich-merge-content-key's persistence — apply uuid-tmp first so the other two inherit the race fix for free. No code conflict (different lines).
- enrich-select-flat-resync is a PARTIAL fix: it makes the flat asset_kind track the selected candidate, but full "selection changes what renders" correctness also needs the R1/enrich-render cluster's plan_render→resolved_asset() change (finding 1, different cluster). Note the dependency in the PR description; do not duplicate that fix here.
- FakeSession in tests/test_api_enrich.py mirrors the production ctor task dict (line 90-91) WITHOUT "cancelled" — after events-ctor-cancelled-key lands, the new /api/events test must drive a real serve.Session (or the test must add the key to its fake), else it reproduces the very KeyError. Call this out to the test author.
- cache-lru-sweep is verdict=unverified (disk-growth claim not reproduced). Ship opt-in/default-off; recommend confirming the growth on the user's machine before enabling by default.

### appjs-early

SHARED HELPERS introduced by this cluster (define once, reuse): (a) `taskLost()` + `st._sseErrs` counter — patch sse-dead-server-watchdog; (b) `TASK_RU` label map next to TASK_TAB (app.js:125) — patch progress-ru-labels-eta; (c) `loadChaptersFromCache`/`loadMetadataFromCache` — patch chapters-metadata-restore-client, which DEPENDS on the two new GET endpoints from chapters-metadata-restore-server; (d) `st._starting` in-flight flag — patch task-start-double-click-guard.

ORDERING/OVERLAP the implementer must respect:
- sse-dead-server-watchdog (wave 1) subsumes the "spinner forever after server restart" half of finding line-250; init-reattach-finished-task (wave 4) covers only the OTHER half of that finding (reload after a task FINISHED on a still-alive server). Land the watchdog first.
- chapters-metadata restore (server + client) and init-reattach-finished-task both surface post-reload results. Once the two chapters/metadata GET endpoints land, init-reattach only needs to re-surface the RENDER result (clips/enrich already restore via loadClipsFromCache/loadEnrichFromCache at app.js:248-249; chapters/metadata restore via the new caches). Keep init-reattach scoped to render+error to avoid duplicate rendering.
- calm-409-during-task and init-error-handling both touch save/doSave-adjacent flow but are independent edits.

THEME LINKS: none of these belong to R1 (sliver cutlist) or R2 (filter_complex_script). Only chapters-metadata-restore-* touches R3 "one UI<->backend contract" (adds the missing GET side of a POST-only contract) — coordinate the new /api/chapters + /api/metadata route names with whoever owns the R3/serve.py cluster so they aren't duplicated.

TEST-DEBT REALITY: there is NO JS test harness (no package.json/vitest/jest/playwright; tests/ is pytest-only). Every frontend behavior here is currently untestable. Two of these patches (clips-render-live-modal-opts, calm-409, cutfloat-clamp, progress-eta) are only truly lockable by introducing a jsdom+vitest harness under web/tests/ — I flag that per-patch. Backend-contract halves (events 409, cutlist/word 409, new GET endpoints) ARE lockable today via tests/ TestClient (pattern: tests/test_transcript_edit.py — SimpleNamespace session with task={"running":bool}, monkeypatch serve.SESSION).

### appjs-late

SHARED SEED PATH: subs-chapters-defaults(#2), quality-per-encoder(#4) and queue-render-opts(#1) converge on seedRenderModal()/collectRenderOpts() in web/app.js and the "defaults" dict in serve.py (/api/state ~1955-1978). Touch that defaults dict ONCE (add subtitles, chapters, quality_nvenc, quality_x264). render-modal-persist(#3) also edits openRenderModal()/seedRenderModal(); apply AFTER #2/#4. Wave-3 order: #2+#4 -> #1 -> #3. MERGE: findings #5 (app.js:3380) and #10 (app.js:3379) are the same schema-mismatch defect -> patch montage-visual-schema. CROSS-CLUSTER HARD DEPENDENCY: montage-visual-schema fixes the review-UI half only; plan_render (vpipe/enrich.py:1332-1349) reads only flat asset_kind and _legacy_sync_from_candidate sets asset_kind='none' for schematic, so a chosen schematic/diffusion candidate still won't render until a companion backend patch makes plan_render use resolved_asset()/chosen_candidate() (render/enrich cluster) — ship in one PR. TEST DEBT: repo has NO JS test tooling (all pytest; no package.json/jest/vitest/jsdom). Pure-client patches need a new jsdom harness (tests/js/); where the fix crosses the wire I give a concrete pytest as the achievable lock.

### subtitles

All 12 findings resolved in 8 patches (4 merges). MERGES: P1 = #1(356 crit)+#11(357 high) same karaoke last-word drop; P3 = #2(438 high)+#9(441 med) same ASS-size-vs-PlayRes; P8 = #6(enrich 62 low)+#10(enrich 62 low) same hardcoded SUBS_TOP. R1 note: the .srt/.vtt sidecar (serve.py:1290 subs_mod.generate→build_cues→write_srt) and the burn ASS (serve.py:1211 build_cues→write_ass) BOTH call build_cues, so P2 (build_cues max_cps) fixes cue END for both consistently; the burned-vs-srt DIVERGENCE in #1 is purely inside _karaoke_text's time-re-selection (P1) — it is the single canonical fix and every subtitle generator that shares build_cues already stays in sync. APPLY ORDER inside _karaoke_text: P1 (line ~356 filter), P4 (line ~400 return), P5 (line ~395 call + signature) touch the SAME function on different lines — apply all three, re-locate by content not line number. P3 must land before the golden test in test_caption_presets is updated (that test at play_res=(1080,1920) currently pins the UNSCALED size/margin — it flips from asserting the bug to asserting the fix). Waves: 1=content-truncation (P1,P2), 2=subtitle render/layout (P3,P4,P5,P6,P7), 3=enrich geometry (P8).

### render-ffmpeg

14 findings → 12 patches. Two root-cause merges dominate this cluster:

R2 (WinError-206 / inline giant graph) merges FOUR findings — ffmpeg_utils.py:149, render.py:1173, render.py:1205, censor.py:88 — into ONE central fix (P1): spill any large -filter_complex to a temp file and pass -filter_complex_script from INSIDE FFmpeg.run(). Because every pass (cut encode, video-only, no-cut audio, measure, censor) funnels through ff.run, one edit fixes them all. Critically, the caller's `args` list is left untouched (only the local `cmd` is rewritten), so (a) every FakeFF test that does `args[args.index("-filter_complex")+1]` keeps working, and (b) the FFmpegError graph-embed at 198-204 still finds the graph. P2 (ffprobe-validate the produced .part) and P11 (cap the graph in FFmpegError) are the other two halves of R2 and edit the SAME two functions — apply P1+P2+P11 together, one visit to ffmpeg_utils.run/_run_atomic.

R1 (shared effective cutlist) merges finding #1 (sliver-drop desync) into ONE helper P6 `effective_cut()` that both render() AND serve.py's burn.ass / enrich / subtitles / chapters / metadata builders consume. P7 (frame-snap, finding #2) EXTENDS the same helper — do P6 first, then P7. Do NOT invent separate trims in subtitles.py/serve.py; they must all call render_mod.effective_cut.

P3 (stall watchdog) and P1's spill both edit FFmpeg.run — coordinate. P4 (cancel flag) and P5 (DFN Popen+register) both touch cancellation/enhance_audio — P5 supplies the registered-Popen the DFN needs, P4 threads the should_cancel callback; apply together and pass one `should_cancel=lambda: bool(s.task.get("cancelled"))` from serve.py:1278.

Wave order within cluster: 1 (stability) P1,P2,P3,P4,P5 → 2 (subtitles) P6,P7 → 4 (edge cases) P8,P9,P10,P11,P12. No wave-3/5 patches in this cluster.

### enrich-backend

SHARED-CODE COORDINATION (apply these in wave order, minding overlaps):

1) imagegen.py sd-cli command is edited by THREE patches (artdir-suffix touches the prompt fed to it; vram-max-vram adds a flag; sd-batch rewrites generate_candidates). To avoid three-way conflict, apply in this order: vram-max-vram (wave3) → artdir-suffix (wave5) → sd-batch (wave5). The batch rewrite MUST carry the --max-vram flag added by the vram patch (best: factor a shared `_sd_cmd_core(prompt,negative,seed,W,H,steps,cfg_scale,vae,cfg)->list[str]` used by BOTH generate_image and the new batch path — this is the R2-style "one command builder" so the two never diverge). sd-batch depends_on vram-max-vram.

2) enrich_llm.detect_all is edited by keepalive-unload (removes n_calls, adds end-of-pass llm.unload) and cta-window (per-window ka_next calls). cta-window DEPENDS ON keepalive-unload: once ka_next stops trying to zero the "last" call, calling it N times for N CTA windows is safe. Apply keepalive-unload first. codegfx-style also edits detect_all (signature + the _build_candidates call) — independent, no conflict.

3) THE CRITICAL PATCH (schematic-render) is the flagship fix: it routes plan_render's image branch through ImagePayload.resolved_asset() — the SAME single-source resolver the .kadr export (serve.py:2567) and _legacy_sync_from_candidate already treat as canonical (that helper DELIBERATELY sets flat asset_kind="none" for schematic, documenting that render must use resolved_asset). This also fixes a latent diffusion bug: after /api/enrich/select changes `selected`, sanitize does NOT resync flat asset_path, so the old flat-field render path would show the WRONG photo; resolved_asset reads candidates[selected] and is always correct.

4) All existing test_enrich_plan.py tests build ImagePayload with flat asset_kind and NO candidates → chosen_candidate() is None → they keep the legacy flat path unchanged (zero regression). Only NEW candidate-carrying payloads route through resolved_asset.

5) Two existing tests LOCK the buggy keep_alive behavior and MUST be updated by keepalive-unload (test_enrich_llm.py:822,829). test_enrich_assets.py:386 is NOT affected (match_user_assets keeps its hardcoded keep_alive=0, which remains genuinely-last in the folder case). test_api_enrich RecordingLLM tests monkeypatch _run_enrich_detectors so detect_all never runs there — unaffected.

6) config-sd-path-vae edits are LOCAL-ONLY (config.yaml is an uncommitted machine edit annotated "do not commit absolute paths") — keep the relocation local; the only committable change is adding `/models/` to .gitignore.

### probe-detect

SHARED SURFACE — patches #1 (probe-perstream-duration), #4 (probe-rotation-swap), #5 (probe-multiaudio-map) all edit the SAME function vpipe/probe.py:probe_media() and the MediaInfo dataclass. Land them in one editing pass in this order: rotation swap (#4, changes width/height), then per-stream duration clamp (#1, changes duration), then multi-audio (#5 touches extract_audio, a separate function in the same file). None conflict textually but a senior dev should open probe.py once and apply all three. Keep MediaInfo backward-compatible: if any new field is added it MUST have a default (dozens of tests build MediaInfo(...) positionally / partially).

ROOT-CAUSE TIES: Patch #1's post-render duration verification is the R2 theme ("validate produced output duration"). This cluster owns the probe.py:37 half (per-stream clamp) and the render.py post-_run_atomic ffprobe check; the temp-file `-filter_complex_script` half of R2 is owned by another cluster — do NOT duplicate it here, just add the duration probe after the existing _run_atomic call. Patch #1's duration clamp also feeds R1 (shared effective cutlist): because it mutates media.duration at the single probe source, every downstream consumer (Timeline, subtitles, chapters, enrich) automatically sees the corrected shorter duration — this is the correct single-source location, do not re-clamp per-consumer.

FINDING SPLIT: the models.py:164 finding is realized as TWO coordinated patches — #2 (atomic save in vpipe/models.py) + #3 (corrupt-file guard in serve.py Session.__init__). They share source_lines; #3 makes /api/open survive a file that #2 now prevents from being created. Apply both.

VERDICT CORRECTIONS folded in: (a) the models.py finding claims Transcript.save is already atomic — it is NOT (models.py:98 is also a direct write_text); patch #2 fixes save_json (the hot path) and notes Transcript.save shares the flaw. (b) The transcribe cache finding's own corrected verdict shows the payoff is weak on the 8GB card (LLM detect + Whisper can't coexist, so a cross-job cache must be evicted before each job's detect anyway) — patch #8 is therefore spec'd as DEFAULT-OFF opt-in to keep default behavior byte-for-byte. (c) denoise_highpass is NOT dead (reachable via config.yaml + direct API) — patch #9 documents cut_fade/min_segment and leaves denoise_highpass intact rather than deleting it.

TEST DEBT: the audit notes these bugs slip because there are no real-ffmpeg tests and only <=3-cut scenarios. Patches #1/#4/#5 are lockable with pure-unit fake-ff tests (inject a crafted ffprobe JSON dict — no ffmpeg needed); #1 additionally wants one real-ffmpeg integration test for the truncation case. #6 mirrors the existing test_clips.py MockLLM keep_alive assertion. #7 needs an equivalence (behavior-identical) test since it is perf-only.

### ui-html

Single finding, one coordinated fix across serve.py + app.js (index.html needs NO change — the label «Очистить завершённые» becomes truthful once behavior is fixed). Chosen resolution = Option A from the finding's `corrected` note (clear only finished jobs) rather than Option B (rename + confirm), because it eliminates the silent data-loss entirely instead of merely warning about it, and makes the existing label accurate. Apply ui-clear-1 (server) FIRST, then ui-clear-2 (frontend depends on the new `statuses` param). The server default path (no body) is left byte-for-byte identical so the two behavior/persistence tests that lock it (tests/test_queue.py:139 test_queue_clear_keeps_running_only → removed==3; tests/test_queue_persist.py:278 test_clear_persists_to_disk → []) keep passing untouched. There are no frontend tests in the repo, so the lock lives in a new backend test that exercises the `statuses` filter. If FastAPI rejects the optional body on an empty POST (it should not — Body(default=None) makes it optional), fall back to `body: dict = Body(default_factory=dict)` and treat `{}` as "no filter".

### test-debt

All 12 findings are distinct test-authoring tasks (no true merges: each targets a different untested function/surface), so each becomes one patch. Ordered by wave, then severity. Shared infrastructure to introduce ONCE and reuse across the new files: (1) a tiny `_media()` FakeFF/MediaInfo helper already exists in tests/test_edge_fade.py:79-101 and a `_fake_session()` in tests/test_queue.py:172-177 — new render/pipeline tests should import or copy these rather than reinvent; (2) create tests/conftest.py (does not exist today) to register custom markers (`ffmpeg`, `slow`) so P11's opt-in tier and any importorskip pattern have a home and CI stays clean. Cross-cluster dependencies to respect: P3 (render-sliver desync) and P5 (_wrap word-drop) and P6 (caption scale) each encode the CORRECT contract as an xfail(strict=True) that auto-flips to green when the corresponding SOURCE fix lands in the R1 (shared effective cutlist) / subtitles clusters — do NOT weaken these to match today's buggy output, and do NOT put the source edits here (this cluster is READ-ONLY test spec). CI ban (.github/workflows/ci.yml:3-6 "Deliberately NO ffmpeg") stays intact: every new test is either fully mocked (FakeFF/monkeypatch) or gated behind an env-var + skip marker (P11) or importorskip (P12) so `python -m pytest -q` on the runners never touches a real binary. Naming: keep the existing `sys.path.insert(0, parents[1])` shim only if the file lives in tests/ and imports serve at top level (see test_edge_fade.py:31) — most existing files rely on pytest rootdir; match the neighbours in the same dir.
