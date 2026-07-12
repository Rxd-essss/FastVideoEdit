# -*- coding: utf-8 -*-
"""Wave-4a #79: clips/enrich saves use a UNIQUE per-call .tmp (uuid).

The shared fixed-name .json.tmp reproduced the Windows PermissionError race
already fixed for the transcript-word PUT: two overlapping saves collided on
os.replace -> spurious 500. POSIX os.replace won't raise that, so we lock the
invariant directly: two saves get DIFFERENT tmp names of the form
<name>.<hex>.tmp, and no scrap is left behind. Fail-before: both calls used the
same fixed <name>.tmp (no hex segment).
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import serve
from vpipe import enrich as enrich_mod
from vpipe.enrich import EnrichPlan, save_enrich


def test_save_clips_json_unique_tmp_per_call(tmp_path, monkeypatch):
    s = SimpleNamespace(out_dir=tmp_path, inp=SimpleNamespace(stem="vid"),
                        audio_hash="h",
                        cfg=SimpleNamespace(llm=SimpleNamespace(model="m")))
    seen = []
    real = serve.os.replace

    def capture(src, dst):
        seen.append(str(src))
        return real(src, dst)

    monkeypatch.setattr(serve.os, "replace", capture)
    serve._save_clips_json(s, [])
    serve._save_clips_json(s, [])
    assert len(seen) == 2 and seen[0] != seen[1]
    for x in seen:
        assert re.search(r"vid\.clips\.json\.[0-9a-f]+\.tmp$", x)
    assert list(tmp_path.glob("*.tmp")) == []       # scrap cleaned up


def test_save_enrich_unique_tmp_per_call(tmp_path, monkeypatch):
    seen = []
    real = enrich_mod.os.replace

    def capture(src, dst):
        seen.append(str(src))
        return real(src, dst)

    monkeypatch.setattr(enrich_mod.os, "replace", capture)
    p = tmp_path / "video.enrich.json"
    save_enrich(EnrichPlan(hash="h1"), p)
    save_enrich(EnrichPlan(hash="h2"), p)
    assert len(seen) == 2 and seen[0] != seen[1]
    for x in seen:
        assert re.search(r"video\.enrich\.json\.[0-9a-f]+\.tmp$", x)
    assert list(tmp_path.glob("*.tmp")) == []
