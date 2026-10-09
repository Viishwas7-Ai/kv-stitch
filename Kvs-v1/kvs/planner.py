"""KVPlanner: the one call the app makes instead of its Ollama call.

The router (SMR) decides the model and the app's builder makes the prompt as pieces:

    header   the base prompt every request starts with
    pieces   [(name, text, is_module), ...] in prompt order: modules and fixed parts
             (rules, glue, OS text). Nothing per user or per day in here.
    tail     the per-request text: user/date line, context, the command

    planner.plan("granite4:micro", header, pieces, tail) -> PlanResult

Paths, all exact (same plan as reading the whole prompt from scratch):
    full     the whole prefix is cached: only the tail is read
    partial  the longest cached start is loaded, the rest is read after it
    miss     nothing cached: everything is read
    fallback something failed: the app's own function (its Ollama call) answers

Speed mode, only when asked (plan(..., fast=True)), and only when the whole prefix is not
cached:
    fast     the longest cached start (at least base + first module) is loaded exactly; every
             module after it is stitched in from its own first-run start (base + that module);
             fixed parts after the modules and the tail are read. Faster, NOT always the same
             plan: a stitched module never saw the modules before it.

After a partial or a miss the whole prefix is saved, plus the start up to the first and the
second module, so later requests that share that start reuse it. A changed text (module,
rule, base prompt) gets a new cache and the old version is deleted.

First run: build_starts() / add_workflows() build and pin what the app knows it will need.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from .cache import PrefixCache
from .engine import Engine
from .workflows import Workflow

log = logging.getLogger("kvs")

Piece = tuple   # (name, text) or (name, text, is_module)


@dataclass
class PlanResult:
    text: str
    path: str                  # full | partial | miss | fallback
    seconds: float
    detail: dict = field(default_factory=dict)


def ollama_blob(model: str, ollama_dir: str = "~/.ollama/models") -> str:
    """The GGUF file Ollama stores for a model name like "granite4:micro"."""
    base = os.path.expanduser(ollama_dir)
    name, _, tag = model.partition(":")
    ns, _, repo = name.rpartition("/")
    man = os.path.join(base, "manifests", "registry.ollama.ai", ns or "library", repo, tag or "latest")
    with open(man) as f:
        layers = json.load(f)["layers"]
    digest = next(l["digest"] for l in layers if l["mediaType"].endswith(".model"))
    return os.path.join(base, "blobs", digest.replace(":", "-"))


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


class KVPlanner:
    def __init__(self, models: dict[str, str] | None, cache_root: str, *, n_ctx: int = 8192,
                 n_gpu_layers: int = -1, n_batch: int = 512, n_ubatch: int = 512,
                 flash_attn: bool = False, chat: bool = True, max_tokens: int = 600,
                 max_disk_mb: float | None = 3000, max_in_memory: int = 2,
                 versions_per_slot: int = 1, checkpoints: int = 2,
                 exact_only: set[str] | None = None,
                 fallback: Callable[[str, str], str] | None = None):
        """
        models:      {"granite4:micro": "/path/to/gguf", ...}; a model not listed here is
                     looked up in Ollama's folder by name (ollama_blob)
        cache_root:  one sub-folder per model is made inside it
        n_batch, n_ubatch, flash_attn: reading settings (see Engine). Any non-default setting
                     gets its own cache folder (model_fa_b1024u1024), so the default caches are
                     never replaced by them
        chat:        wrap the prompt in the model's chat template (as Ollama does)
        checkpoints: also save the start up to the first N modules on the way
        exact_only:  module names that always take the exact path, even when speed mode is
                     asked for (e.g. {"DELETE", "RENAME", "FILES"}: a wrong step is costly)
        fallback:    fallback(model_name, prompt_text) -> plan text, used if anything fails
        """
        self.models = dict(models or {})
        self.cache_root = os.path.expanduser(cache_root)
        self.n_ctx, self.n_gpu_layers, self.chat = n_ctx, n_gpu_layers, chat
        self.n_batch, self.n_ubatch, self.flash_attn = n_batch, n_ubatch, flash_attn
        self.max_tokens = max_tokens
        self.max_disk_mb, self.max_in_memory = max_disk_mb, max_in_memory
        self.versions_per_slot, self.checkpoints = versions_per_slot, checkpoints
        self.fallback = fallback
        self.exact_only = set(exact_only or ())
        self.model: str | None = None
        self.eng: Engine | None = None
        self.cache: PrefixCache | None = None
        self.wrap = ("", "")
        self._tok: dict[tuple[str, bool], list[int]] = {}

    # ---------- model (one in memory at a time: 8 GB machines) ----------
    def load(self, model: str):
        if self.model == model:
            return
        self.close()
        path = self.models.get(model) or ollama_blob(model)
        log.info("kvs: loading %s", model)
        self.eng = Engine(path, n_ctx=self.n_ctx, n_gpu_layers=self.n_gpu_layers, n_batch=self.n_batch,
                          n_ubatch=self.n_ubatch, flash_attn=self.flash_attn)
        self.wrap = self.eng.chat_wrap() if self.chat else ("", "")
        folder = _safe(model) + ("_" + self.eng.settings_tag if self.eng.settings_tag else "")
        self.cache = PrefixCache(self.eng, os.path.join(self.cache_root, folder),
                                 max_in_memory=self.max_in_memory, max_disk_mb=self.max_disk_mb,
                                 versions_per_slot=self.versions_per_slot)
        self.model, self._tok = model, {}

    def close(self):
        if self.eng:
            self.eng.close()
        self.model = self.eng = self.cache = None

    # ---------- prompt → token runs ----------
    def _t(self, text: str, bos: bool = False) -> list[int]:
        k = (text, bos)
        if k not in self._tok:
            self._tok[k] = self.eng.tok(text, bos=bos)
        return self._tok[k]

    def _runs(self, header: str, pieces: list[Piece]):
        """Token runs: [header] + one per piece, with names, texts and module positions."""
        h = self.wrap[0] + header
        runs = [self._t(h, bos=True)] + [self._t(p[1]) for p in pieces]
        names = ["HEADER"] + [p[0] for p in pieces]
        texts = [h] + [p[1] for p in pieces]
        mods = [i + 1 for i, p in enumerate(pieces) if len(p) > 2 and p[2]]
        return runs, names, texts, mods

    def _entry(self, runs, names, texts, upto: int):
        toks = [t for r in runs[:upto] for t in r]
        return self.cache.key(toks), "|".join(names[:upto]), "".join(texts[:upto]), len(toks)

    def _put_prefix(self, runs, names, texts, mods) -> tuple[int, str, dict]:
        """Get the whole prefix into the context, as cheaply as possible. Returns
        (next position, path, info)."""
        n = len(runs)
        keys = {L: self._entry(runs, names, texts, L)[0] for L in range(1, n + 1)}
        start = next((L for L in range(n, 0, -1) if self.cache.has(keys[L])), 0)
        if start == n:
            pos = self.cache.load(keys[n])
            self.cache.touch(keys[n])
            return pos, "full", {"cached_pieces": n}
        if start:
            pos = self.cache.load(keys[start])
            self.cache.touch(keys[start])
        else:
            self.eng.clear()
            pos = 0
        save_at = {r + 1 for r in mods[:self.checkpoints]} | {n}
        removed = 0
        for L in range(start + 1, n + 1):
            self.eng.decode(runs[L - 1], pos)
            pos += len(runs[L - 1])
            if L in save_at and not self.cache.has(keys[L]):
                k, slot, text, _ = self._entry(runs, names, texts, L)
                removed += self.cache.write(k, slot, self.eng.save(), pos, text)
        return pos, ("partial" if start else "miss"), {"cached_pieces": start, "removed_old": removed}

    # ---------- speed mode ----------
    def _can_stitch(self, runs, names, texts, mods) -> bool:
        """Worth it only with 2+ modules and when the whole prefix is not cached already."""
        return len(mods) >= 2 and not self.cache.has(self._entry(runs, names, texts, len(runs))[0])

    def _module_entry(self, runs, names, texts, base: int, r: int):
        """The first-run start of module r alone: runs[:base] (header + base pieces) + runs[r]."""
        toks = [t for x in runs[:base] for t in x] + runs[r]
        return (self.cache.key(toks), "|".join(names[:base] + [names[r]]),
                "".join(texts[:base]) + texts[r], toks)

    def _put_prefix_fast(self, runs, names, texts, mods) -> tuple[int, str, dict]:
        n, base = len(runs), mods[0]                    # runs[:base] = header + pieces before module 1
        keys = {L: self._entry(runs, names, texts, L)[0] for L in range(base + 1, n + 1)}
        start = next((L for L in range(n - 1, base, -1) if self.cache.has(keys[L])), base + 1)
        stitch = [r for r in mods if r >= start]
        if not stitch:                                  # nothing left to stitch: exact path
            return self._put_prefix(runs, names, texts, mods)
        # every module's own start must exist; building one clears the context, so first
        mod_keys = {}
        for r in stitch:
            k, slot, text, toks = self._module_entry(runs, names, texts, base, r)
            if not self.cache.has(k):
                self._base(runs, names, texts, base, pin=False)
                bk = self._entry(runs, names, texts, base)[0]
                b0 = sum(len(x) for x in runs[:base])
                self.cache.load(bk)
                self.eng.decode(runs[r], b0)
                self.cache.write(k, slot, self.eng.save_from(b0), len(toks), text, parent=bk)
            mod_keys[r] = k
        # the exact start
        if self.cache.has(keys[start]):
            pos = self.cache.load(keys[start])
            self.cache.touch(keys[start])
        else:
            toks = [t for x in runs[:start] for t in x]
            self.eng.clear()
            self.eng.decode(toks, 0)
            pos = len(toks)
            k, slot, text, _ = self._entry(runs, names, texts, start)
            self.cache.write(k, slot, self.eng.save(), pos, text)
        b = sum(len(x) for x in runs[:base])
        for L in range(start + 1, n + 1):
            r = L - 1
            if r in mod_keys:
                state, _ = self.cache.read(mod_keys[r])
                self.eng.stitch(state, b, b + len(runs[r]), pos)
                self.cache.touch(mod_keys[r])
            else:
                self.eng.decode(runs[r], pos)
            pos += len(runs[r])
        return pos, "fast", {"cached_pieces": start, "stitched": len(stitch)}

    # ---------- public ----------
    def plan(self, model: str, header: str, pieces: list[Piece], tail: str,
             stop: list[str] | None = None, fast: bool = False) -> PlanResult:
        t0 = time.perf_counter()
        try:
            self.load(model)
            runs, names, texts, mods = self._runs(header, pieces)
            if fast and any(names[r] in self.exact_only for r in mods):
                fast = False                         # a destructive module: never approximate
            if fast and self._can_stitch(runs, names, texts, mods):
                pos, path, info = self._put_prefix_fast(runs, names, texts, mods)
            else:
                pos, path, info = self._put_prefix(runs, names, texts, mods)
            t_prompt = time.perf_counter()
            tt = self._t(tail + self.wrap[1])
            logits = self.eng.decode(tt, pos, want_last=True)
            text, n_gen = self.eng.generate(logits, pos + len(tt), self.max_tokens, stop)
            info.update(prompt_s=round(t_prompt - t0, 2), gen_tokens=n_gen,
                        total_s=round(time.perf_counter() - t0, 2))
            self.cache._save_index()
            return PlanResult(text, path, time.perf_counter() - t0, info)
        except Exception as e:                       # never leave the app without a plan
            log.exception("kvs failed, using the fallback")
            if not self.fallback:
                raise
            text = self.fallback(model, header + "".join(p[1] for p in pieces) + tail)
            return PlanResult(text, "fallback", time.perf_counter() - t0, {"error": repr(e)})

    def build(self, model: str, header: str, pieces: list[Piece], pin: bool = True) -> bool:
        """Build and keep the cache for exactly this prefix. Returns True if it was built,
        False if it was already there (unchanged). A start (base + one module, the module
        last) is stored as its base, kept once, plus only the module's cells."""
        self.load(model)
        runs, names, texts, mods = self._runs(header, pieces)
        n = len(runs)
        k, slot, text, _ = self._entry(runs, names, texts, n)
        if self.cache.has(k):
            if pin and not self.cache.index[k].get("pinned"):
                self.cache.index[k]["pinned"] = True
                self.cache._save_index()
            return False
        if mods == [n - 1] and n > 1:                    # a start: base + this module
            self._base(runs, names, texts, n - 1, pin)
            base_tokens = sum(len(r) for r in runs[:n - 1])
            bk = self._entry(runs, names, texts, n - 1)[0]
            self.cache.load(bk)
            self.eng.decode(runs[-1], base_tokens)
            self.cache.write(k, slot, self.eng.save_from(base_tokens), base_tokens + len(runs[-1]),
                             text, pin=pin, parent=bk)
            return True
        toks = [t for r in runs for t in r]
        self.eng.clear()
        self.eng.decode(toks, 0)
        self.cache.write(k, slot, self.eng.save(), len(toks), text, pin=pin)
        return True

    def _base(self, runs, names, texts, upto: int, pin: bool):
        """Make sure the base runs[:upto] is cached (stored whole, once)."""
        k, slot, text, n = self._entry(runs, names, texts, upto)
        if not self.cache.has(k):
            self.eng.clear()
            self.eng.decode([t for r in runs[:upto] for t in r], 0)
            self.cache.write(k, slot, self.eng.save(), n, text, pin=pin)
        elif pin and not self.cache.index[k].get("pinned"):
            self.cache.index[k]["pinned"] = True

    def build_starts(self, model: str, header: str, start_lists: list[list[Piece]],
                     progress: Callable[[str], None] | None = None) -> int:
        """First run: build and pin each start, e.g. [base pieces + one core module] for
        every core module. Unchanged ones are skipped. Returns how many were built."""
        built = 0
        for i, pl in enumerate(start_lists, 1):
            if progress:
                progress(f"  building {'+'.join(p[0] for p in pl)} ...")
            t0 = time.perf_counter()
            done = self.build(model, header, pl)
            built += done
            if progress:
                progress(f"  [{i}/{len(start_lists)}] {'+'.join(p[0] for p in pl)}: "
                         f"{'built in %.1fs' % (time.perf_counter() - t0) if done else 'already cached'}")
        return built

    def add_workflows(self, model: str, workflows: list[Workflow],
                      build: Callable[[list[str]], tuple[str, list[Piece]]],
                      progress: Callable[[str], None] | None = None) -> dict:
        """Build and pin every workflow for `model`: its whole prefix, and its start up to its
        last module (so the same workflow + extra modules reuses it).
        build(module_names) -> (header, pieces): the app's own builder."""
        rep = {"workflows": 0, "built": 0, "skipped": []}
        for wf in workflows:
            if wf.model and wf.model != model:
                continue
            try:
                header, pieces = build(list(wf.modules))
            except Exception as e:
                rep["skipped"].append(f"{wf.name}: {e}")
                continue
            if progress:
                progress(f"  building workflow {wf.name} ...")
            t0 = time.perf_counter()
            n = self.build(model, header, pieces)
            last = max((i for i, p in enumerate(pieces) if len(p) > 2 and p[2]), default=None)
            if last is not None and last + 1 < len(pieces):
                n += self.build(model, header, pieces[:last + 1])
            rep["built"] += n
            rep["workflows"] += 1
            if progress:
                progress(f"  {wf.name} ({', '.join(wf.modules)}): "
                         f"{'built %d cache(s)' % n if n else 'already cached'} "
                         f"in {time.perf_counter() - t0:.1f}s")
        return rep

    def stats(self) -> dict:
        if not self.cache:
            return {}
        idx = self.cache.index
        return {"model": self.model, "folder": self.cache.dir, "entries": len(idx),
                "pinned": sum(1 for e in idx.values() if e.get("pinned")),
                "disk_mb": round(self.cache.disk_mb(), 1)}
