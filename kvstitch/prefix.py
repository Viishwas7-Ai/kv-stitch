"""Exact prefix cache: the whole prompt before the request, computed once per combination.

When everything before the user's command is the same text (same header, same modules, same
rules), the KV cache for that text is the same every time. PrefixCache computes it once,
keeps it (in memory and optionally on disk), and afterwards only decodes the request.
Nothing is approximated: the cached part is exactly what the full prompt computes.

The prefix for a request is fully decided by which modules were picked, so the number of
entries is the number of combinations that actually occur, not every possible one.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict

from .core import MAIN, Stitcher, Timing


class PrefixCache:
    def __init__(self, st: Stitcher, cache_dir: str | None = None, max_in_memory: int = 8):
        self.st = st
        self.dir = cache_dir
        self.max_mem = max_in_memory
        self.mem: OrderedDict[str, tuple[bytes, int]] = OrderedDict()   # key -> (state, n_tokens)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _tokens(self, names: list[str]) -> list[int]:
        toks = list(self.st.header.tokens)
        for n in names:
            toks += self.st.modules[n].tokens
        return toks

    def key(self, names: list[str]) -> str:
        h = hashlib.sha1()
        for t in self._tokens(names):
            h.update(t.to_bytes(4, "little"))
        return h.hexdigest()[:20]

    def _path(self, key: str) -> str | None:
        return os.path.join(self.dir, key + ".kv") if self.dir else None

    def _remember(self, key: str, state: bytes, n: int):
        self.mem[key] = (state, n)
        self.mem.move_to_end(key)
        while len(self.mem) > self.max_mem:
            self.mem.popitem(last=False)

    def has(self, names: list[str]) -> bool:
        k = self.key(names)
        p = self._path(k)
        return k in self.mem or bool(p and os.path.exists(p))

    def load_prefix(self, names: list[str]) -> tuple[int, bool]:
        """Put the prefix for `names` into the context. Returns (next position, was_cached)."""
        k = self.key(names)
        st = self.st
        st.clear()
        if k in self.mem:
            state, n = self.mem[k]
            self.mem.move_to_end(k)
            st._load(state, MAIN)
            return n, True
        p = self._path(k)
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                n = int.from_bytes(f.read(4), "little")
                state = f.read()
            st._load(state, MAIN)
            self._remember(k, state, n)
            return n, True
        toks = self._tokens(names)                     # first time: compute it once, exactly
        st._decode(toks, 0, MAIN)
        state = st._save(MAIN)
        self._remember(k, state, len(toks))
        if p:
            with open(p, "wb") as f:
                f.write(len(toks).to_bytes(4, "little"))
                f.write(state)
        return len(toks), False

    def run(self, names: list[str], tail: str, max_tokens: int = 512,
            stop: list[str] | None = None) -> tuple[str, Timing]:
        tm = Timing()
        t0 = time.perf_counter()
        pos, hit = self.load_prefix(names)
        tm.load_s = time.perf_counter() - t0
        tm.extra["prefix_hit"] = hit
        tt = self.st.tok(tail)
        t0 = time.perf_counter()
        logits = self.st._decode(tt, pos, MAIN, want_last=True)
        tm.tail_s, tm.tail_tokens = time.perf_counter() - t0, len(tt)
        t0 = time.perf_counter()
        text, tm.gen_tokens = self.st._generate(logits, pos + len(tt), max_tokens, stop or [])
        tm.gen_s = time.perf_counter() - t0
        return text, tm
