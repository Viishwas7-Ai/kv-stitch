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
import json
import os
import time
from collections import OrderedDict

from .core import MAIN, Stitcher, Timing


class PrefixCache:
    """
    save_after:  a combination is written to disk once it has been used this many times
                 (1 = immediately). Below that it lives only in the in-memory LRU.
    max_disk_mb: when the folder grows past this, the least recently used caches that are
                 not pinned are deleted. None = no limit.
    warm():      build caches up front (first run) and pin them, so they are never evicted.
    """

    def __init__(self, st: Stitcher, cache_dir: str | None = None, max_in_memory: int = 8,
                 save_after: int = 1, max_disk_mb: float | None = None):
        self.st = st
        self.dir = cache_dir
        self.max_mem = max_in_memory
        self.save_after = max(1, save_after)
        self.max_disk = max_disk_mb * 1024 * 1024 if max_disk_mb else None
        self.mem: OrderedDict[str, tuple[bytes, int]] = OrderedDict()   # key -> (state, n_tokens)
        self.index: dict[str, dict] = {}            # key -> {"uses", "last", "pinned"}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            try:
                with open(self._index_path()) as f:
                    self.index = json.load(f)
            except (OSError, ValueError):
                self.index = {}

    # ---------- bookkeeping ----------
    def _index_path(self) -> str:
        return os.path.join(self.dir, "index.json")

    def _touch(self, key: str, used: bool = True) -> dict:
        e = self.index.setdefault(key, {"uses": 0, "last": 0.0, "pinned": False})
        if used:
            e["uses"] += 1
        e["last"] = time.time()
        return e

    def _save_index(self):
        if self.dir:
            tmp = self._index_path() + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.index, f)
            os.replace(tmp, self._index_path())

    def _write(self, key: str, state: bytes, n: int):
        p = self._path(key)
        if not p:
            return
        with open(p + ".tmp", "wb") as f:
            f.write(n.to_bytes(4, "little"))
            f.write(state)
        os.replace(p + ".tmp", p)
        self._evict()

    def _evict(self):
        """Delete least recently used, unpinned caches until the folder fits max_disk_mb."""
        if not (self.dir and self.max_disk):
            return
        files = {k: self._path(k) for k in self.index if os.path.exists(self._path(k))}
        total = sum(os.path.getsize(p) for p in files.values())
        for k in sorted(files, key=lambda k: self.index[k]["last"]):
            if total <= self.max_disk:
                break
            if self.index[k].get("pinned"):
                continue
            total -= os.path.getsize(files[k])
            os.remove(files[k])

    def drop_pinned_except(self, keep: set[str]) -> int:
        """Delete pinned caches whose key is not in `keep` (an old version of a workflow).
        Returns how many were removed."""
        gone = 0
        for k in [k for k, e in self.index.items() if e.get("pinned") and k not in keep]:
            p = self._path(k)
            if p and os.path.exists(p):
                os.remove(p)
            self.mem.pop(k, None)
            del self.index[k]
            gone += 1
        if gone:
            self._save_index()
        return gone

    def disk_mb(self) -> float:
        if not self.dir:
            return 0.0
        return sum(os.path.getsize(os.path.join(self.dir, f)) for f in os.listdir(self.dir)
                   if f.endswith(".kv")) / 1024 / 1024

    def warm(self, combos: list[list[str]], pin: bool = True) -> int:
        """First run: build and save these combinations now. Returns how many were built."""
        built = 0
        for names in combos:
            k = self.key(names)
            e = self._touch(k, used=False)
            e["pinned"] = e.get("pinned") or pin
            if not (self._path(k) and os.path.exists(self._path(k))):
                toks = self._tokens(names)
                self.st.clear()
                self.st._decode(toks, 0, MAIN)
                self._write(k, self.st._save(MAIN), len(toks))
                built += 1
        self._save_index()
        return built

    def _tokens(self, names: list[str]) -> list[int]:
        toks = list(self.st.header.tokens)
        for n in names:
            toks += self.st.modules[n].tokens
        return toks

    def key(self, names: list[str]) -> str:
        """Same model file, same context size and the exact same tokens -> same cache."""
        h = hashlib.sha1()
        st = self.st
        mf = os.stat(st.model_path)
        h.update(f"{st.model_path}|{mf.st_size}|{int(mf.st_mtime)}|{st.n_ctx}|".encode())
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
        """Is this combination ready without computing it (in memory or on disk)?"""
        k = self.key(names)
        p = self._path(k)
        return k in self.mem or bool(p and os.path.exists(p))

    def _read(self, k: str) -> tuple[bytes, int] | None:
        if k in self.mem:
            self.mem.move_to_end(k)
            return self.mem[k]
        p = self._path(k)
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                n = int.from_bytes(f.read(4), "little")
                state = f.read()
            self._remember(k, state, n)
            self._touch(k, used=False)
            return state, n
        return None

    def load_prefix(self, names: list[str], checkpoints: tuple[int, ...] = ()) -> tuple[int, str]:
        """Put the prefix for `names` into the context. Returns (next position, kind).

        kind: "full"    the whole prefix was cached
              "partial" the longest cached start was loaded and only the rest computed
              "miss"    nothing cached, computed from the start
        Every path gives exactly the full prompt's cache: a start is only ever reused when it
        was computed with exactly the same tokens before it.

        checkpoints: piece counts (e.g. "up to and including the first module") whose state
        is saved on the way, so later requests that share that start can reuse it.
        """
        st = self.st
        st.clear()
        kfull = self.key(names)
        e = self._touch(kfull)
        got = self._read(kfull)
        if got:
            state, n = got
            st._load(state, MAIN)
            p = self._path(kfull)
            if p and not os.path.exists(p) and e["uses"] >= self.save_after:
                self._write(kfull, state, n)           # it became a pattern: keep it
            self._save_index()
            return n, "full"

        start, pos, kind = 0, 0, "miss"
        for L in range(len(names) - 1, 0, -1):         # longest cached start
            g = self._read(self.key(names[:L]))
            if g:
                st._load(g[0], MAIN)
                start, pos, kind = L, g[1], "partial"
                break
        if start == 0:
            ht = self.st.header.tokens
            st._decode(ht, 0, MAIN)
            pos = len(ht)
        for i in range(start, len(names)):             # compute the rest, saving checkpoints
            t = self.st.modules[names[i]].tokens
            st._decode(t, pos, MAIN)
            pos += len(t)
            c = i + 1
            if c in checkpoints and c < len(names):
                kc = self.key(names[:c])
                p = self._path(kc)
                if kc not in self.mem and not (p and os.path.exists(p)):
                    cstate = st._save(MAIN)
                    self._remember(kc, cstate, pos)
                    self._touch(kc, used=False)
                    self._write(kc, cstate, pos)
        state = st._save(MAIN)
        self._remember(kfull, state, pos)
        if e["uses"] >= self.save_after:
            self._write(kfull, state, pos)
        self._save_index()
        return pos, kind

    def run(self, names: list[str], tail: str, max_tokens: int = 512,
            stop: list[str] | None = None, checkpoints: tuple[int, ...] = ()) -> tuple[str, Timing]:
        tm = Timing()
        t0 = time.perf_counter()
        pos, kind = self.load_prefix(names, checkpoints)
        tm.load_s = time.perf_counter() - t0
        tm.extra["prefix_hit"] = kind == "full"
        tm.extra["prefix_kind"] = kind
        tt = self.st.tok(tail)
        t0 = time.perf_counter()
        logits = self.st._decode(tt, pos, MAIN, want_last=True)
        tm.tail_s, tm.tail_tokens = time.perf_counter() - t0, len(tt)
        t0 = time.perf_counter()
        text, tm.gen_tokens = self.st._generate(logits, pos + len(tt), max_tokens, stop or [])
        tm.gen_s = time.perf_counter() - t0
        return text, tm
