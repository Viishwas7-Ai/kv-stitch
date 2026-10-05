"""PlannerBackend: one call for an app's planning step.

The app keeps building its prompt the way it already does, and hands it over as pieces:

    header   fixed text every prompt starts with
    pieces   [(name, text, is_module), ...] in prompt order: tool modules and fixed parts
             (a terminal section, glue text, rules ...). Anything that changes per user or per
             day does NOT belong here.
    tail     per-request text: user/OS line, today's date, context, the command

For each request the backend picks the cheapest path that is still right:

    full     the whole prefix is cached                    exact
    partial  the longest cached start is loaded,           exact
             only the rest is computed
    fast     (only when fast_mode is on and <= fast_max_modules modules)
             modules stitched from separate caches,        approximate, may make mistakes
             the last module computed fresh
    miss     computed from scratch, then remembered        exact
    fallback the backend failed; the app's own function    whatever the app did before
             (e.g. its Ollama call) is used instead

Cache keys cover the model file, the context size and the exact tokens, so changing a module,
a rule or the model never reuses a stale cache: the old entry just stops matching.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from .core import Stitcher
from .prefix import PrefixCache
from .workflows import Workflow

log = logging.getLogger("kvstitch.backend")

Piece = tuple  # (name, text) or (name, text, is_module)


@dataclass
class PlanResult:
    text: str
    path: str                       # full | partial | miss | fast | fallback
    seconds: float
    prompt_seconds: float = 0.0
    detail: dict = field(default_factory=dict)


def _key(name: str, text: str) -> str:
    return f"{name}#{hashlib.sha1(text.encode()).hexdigest()[:8]}"


class PlannerBackend:
    def __init__(self, models: dict[str, str], cache_root: str, *, n_ctx: int = 8192,
                 n_gpu_layers: int = -1, chat: bool = True, max_tokens: int = 600,
                 save_after: int = 2, max_disk_mb: float | None = 3000, max_in_memory: int = 4,
                 fast_mode: bool = False, fast_max_modules: int = 3,
                 fallback: Callable[[str, str], str] | None = None):
        """
        models:     {"granite3.1-moe:3b": "/path/to/blob", ...}
        cache_root: one sub-folder per model is made inside it
        chat:       wrap header/tail in the model's own chat template (needed for chat models)
        fallback:   fallback(model_name, full_prompt_text) -> plan text, used if anything fails
        """
        self.models = dict(models)
        self.cache_root = os.path.expanduser(cache_root)
        self.n_ctx, self.n_gpu_layers, self.chat = n_ctx, n_gpu_layers, chat
        self.max_tokens = max_tokens
        self.save_after, self.max_disk_mb, self.max_in_memory = save_after, max_disk_mb, max_in_memory
        self.fast_mode, self.fast_max_modules = fast_mode, fast_max_modules
        self.fallback = fallback
        self._model = None            # name of the loaded model
        self._st: Stitcher | None = None
        self._pc: PrefixCache | None = None
        self._wrap = ("", "")
        self._header_text = None

    # ---------- model handling (one model in memory at a time: 8 GB machines) ----------
    def _load(self, model: str):
        if self._model == model:
            return
        self.close()
        path = self.models[model]
        log.info("loading %s", model)
        self._st = Stitcher(path, n_ctx=self.n_ctx, n_gpu_layers=self.n_gpu_layers)
        self._wrap = self._st.chat_wrap() if self.chat else ("", "")
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        self._pc = PrefixCache(self._st, cache_dir=os.path.join(self.cache_root, safe),
                               max_in_memory=self.max_in_memory, save_after=self.save_after,
                               max_disk_mb=self.max_disk_mb)
        self._model, self._header_text = model, None

    def close(self):
        if self._st:
            self._st.close()
        self._model = self._st = self._pc = None

    # ---------- prompt pieces ----------
    def _prepare(self, header: str, pieces: list[Piece]) -> tuple[list[str], list[int]]:
        st = self._st
        h = self._wrap[0] + header
        if h != self._header_text:
            st.set_header(h)                       # a changed header resets the piece table
            self._header_text = h
        names, module_idx = [], []
        for i, p in enumerate(pieces):
            name, text = p[0], p[1]
            k = _key(name, text)
            st.register(k, text)
            names.append(k)
            if len(p) > 2 and p[2]:
                module_idx.append(i)
        return names, module_idx

    def full_prompt(self, header: str, pieces: list[Piece], tail: str) -> str:
        """The plain text the model would see, for a fallback or for logging."""
        pre, suf = self._wrap if self._st else ("", "")
        return pre + header + "".join(p[1] for p in pieces) + tail + suf

    # ---------- public ----------
    def plan(self, model: str, header: str, pieces: list[Piece], tail: str,
             fast: bool | None = None) -> PlanResult:
        t0 = time.perf_counter()
        try:
            self._load(model)
            names, module_idx = self._prepare(header, pieces)
            tail_full = tail + self._wrap[1]
            # save the start after each module on the way, so later requests that share any
            # leading run of modules (a workflow + extras, or a new mix) can reuse it
            first_module_end = tuple(i + 1 for i in module_idx[:2])
            use_fast = self.fast_mode if fast is None else fast
            if (use_fast and not self._pc.has(names) and module_idx
                    and len(module_idx) <= self.fast_max_modules):
                text, tm = self._st.run(names, tail_full, self.max_tokens,
                                        fresh_idx={module_idx[-1]})
                path = "fast"
            else:
                text, tm = self._pc.run(names, tail_full, self.max_tokens,
                                        checkpoints=first_module_end)
                path = tm.extra.get("prefix_kind", "miss")
            return PlanResult(text, path, time.perf_counter() - t0, tm.load_s + tm.tail_s,
                              {"gen_tokens": tm.gen_tokens, "tail_tokens": tm.tail_tokens})
        except Exception as e:                     # never leave the app without a plan
            log.exception("kv-stitch backend failed, using fallback")
            if not self.fallback:
                raise
            text = self.fallback(model, pre_wrap_prompt(header, pieces, tail))
            return PlanResult(text, "fallback", time.perf_counter() - t0, detail={"error": repr(e)})

    def warm(self, model: str, header: str, piece_lists: list[list[Piece]]) -> int:
        """First run: build these prefixes now and pin them (never evicted).
        Pass each core module as its own piece list (and any common combinations)."""
        self._load(model)
        combos = [self._prepare(header, pl)[0] for pl in piece_lists]
        return self._pc.warm(combos, pin=True)

    def add_workflows(self, model: str, workflows: list[Workflow],
                      build: Callable[[list[str]], tuple[str, list[Piece]]],
                      progress: Callable[[str], None] | None = None, prune: bool = True) -> dict:
        """Warm and pin every workflow for `model` (first launch, or after the workflow file changed).

        build(module_names) -> (header, pieces): the app's own prompt builder for those modules,
        in that order, with is_module=True on the module pieces. Returns what was built.

        Unchanged workflows are found in the cache and skipped; a workflow whose modules, rules
        or model changed gets new keys and is rebuilt. With prune=True the old versions
        (pinned caches that no current workflow uses) are deleted. Pruning is skipped if any
        workflow failed to build, so a typo never wipes good caches.
        """
        self._load(model)
        report = {"workflows": 0, "built": 0, "skipped": [], "removed_old": 0}
        keep: set[str] = set()
        for wf in workflows:
            if wf.model and wf.model != model:
                continue
            try:
                header, pieces = build(list(wf.modules))
            except Exception as e:                    # an unknown module name, etc.
                report["skipped"].append(f"{wf.name}: {e}")
                continue
            names, module_idx = self._prepare(header, pieces)
            combos = [names]                           # the whole prefix for exactly this workflow
            if module_idx and module_idx[-1] + 1 < len(names):
                combos.append(names[: module_idx[-1] + 1])   # the start: up to its last module
            keep.update(self._pc.key(c) for c in combos)
            t0 = time.perf_counter()
            n = self._pc.warm(combos, pin=True)
            report["built"] += n
            report["workflows"] += 1
            if progress:
                progress(f"  {wf.name} ({', '.join(wf.modules)}): "
                         f"{'built ' + str(n) + ' cache(s)' if n else 'already cached'} "
                         f"in {time.perf_counter() - t0:.1f}s")
        if prune and not report["skipped"]:
            # anything pinned that is not a current workflow is an old version: remove it
            report["removed_old"] = self._pc.drop_pinned_except(keep)
            if progress and report["removed_old"]:
                progress(f"  removed {report['removed_old']} old cache(s) of changed workflows")
        return report

    def stats(self) -> dict:
        if not self._pc:
            return {}
        return {"model": self._model, "disk_mb": round(self._pc.disk_mb(), 1),
                "entries": len(self._pc.index),
                "pinned": sum(1 for e in self._pc.index.values() if e.get("pinned"))}


def pre_wrap_prompt(header: str, pieces: list[Piece], tail: str) -> str:
    """The prompt text without any chat template (what an Ollama /api/generate call takes)."""
    return header + "".join(p[1] for p in pieces) + tail
