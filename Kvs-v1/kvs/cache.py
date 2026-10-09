"""Exact prefix cache, one folder per model.

Every cache entry is the KV state of one exact prompt start (header + some pieces), computed
in one go, so loading it gives exactly what reading that text would.

On disk, in <cache_root>/<model name>/:

    <key>.kv            the KV state
    <key>.prompt.txt    the exact text it holds, to read what is cached
    index.json          per entry: slot, uses, last use, pinned, tokens

key  = hash of (model file, context size, exact tokens). Any change in the text, the model
       or n_ctx gives a new key, so a stale cache is never used.
A first-run start (base + one module) is stored as a DELTA: only the module's cells, plus the
key of its base (header + pieces before the module), which is stored once. Loading puts the
base back and adds the module's cells at the same positions: the same KV, byte for byte,
for a fraction of the disk.

slot = the piece NAMES (e.g. "HEADER|GLUE|WEB|NOTES"). When the text behind a slot changes
       (an edited module, rule or base prompt), the new version is built and the old one
       deleted, keeping at most `versions_per_slot` versions.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict

from .engine import Engine


class PrefixCache:
    def __init__(self, eng: Engine, folder: str, max_in_memory: int = 2,
                 max_disk_mb: float | None = 3000, versions_per_slot: int = 1):
        self.eng = eng
        self.dir = folder
        self.max_mem = max_in_memory
        self.max_disk = max_disk_mb * 1024 * 1024 if max_disk_mb else None
        self.versions = max(1, versions_per_slot)
        self.mem: OrderedDict[str, tuple[bytes, int]] = OrderedDict()
        os.makedirs(folder, exist_ok=True)
        try:
            with open(self._p("index.json")) as f:
                self.index: dict[str, dict] = json.load(f)
        except (OSError, ValueError):
            self.index = {}
        mf = os.stat(eng.model_path)
        self._model_tag = f"{eng.model_path}|{mf.st_size}|{int(mf.st_mtime)}|{eng.n_ctx}|" + \
            (f"{eng.settings_tag}|" if getattr(eng, "settings_tag", "") else "")

    # ---------- files ----------
    def _p(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def _save_index(self):
        tmp = self._p("index.json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.index, f, indent=1)
        os.replace(tmp, self._p("index.json"))

    def key(self, tokens: list[int]) -> str:
        h = hashlib.sha1(self._model_tag.encode())
        for t in tokens:
            h.update(t.to_bytes(4, "little"))
        return h.hexdigest()[:20]

    def has(self, key: str) -> bool:
        if not (key in self.mem or os.path.exists(self._p(key + ".kv"))):
            return False
        parent = self.index.get(key, {}).get("parent")
        return parent is None or self.has(parent)       # a delta needs its base

    def _delete(self, key: str):
        for ext in (".kv", ".prompt.txt"):
            try:
                os.remove(self._p(key + ext))
            except OSError:
                pass
        self.mem.pop(key, None)
        self.index.pop(key, None)
        for k in [k for k, e in self.index.items() if e.get("parent") == key]:
            self._delete(k)                              # deltas of a deleted base go with it

    def load(self, key: str) -> int:
        """Put entry `key` into the engine's prompt. Returns the next position."""
        state, n = self.read(key)
        parent = self.index.get(key, {}).get("parent")
        if parent:
            self.load(parent)
            self.eng.add_cells(state)
        else:
            self.eng.load(state)
        return n

    # ---------- read / write ----------
    def read(self, key: str) -> tuple[bytes, int] | None:
        if key in self.mem:
            self.mem.move_to_end(key)
            return self.mem[key]
        p = self._p(key + ".kv")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(4), "little")
            state = f.read()
        self._remember(key, state, n)
        return state, n

    def _remember(self, key: str, state: bytes, n: int):
        self.mem[key] = (state, n)
        self.mem.move_to_end(key)
        while len(self.mem) > self.max_mem:
            self.mem.popitem(last=False)

    def write(self, key: str, slot: str, state: bytes, n: int, text: str, pin: bool = False,
              parent: str | None = None) -> int:
        """Store an entry. Older versions of the same slot beyond `versions_per_slot` are
        deleted. Returns how many old versions were removed."""
        with open(self._p(key + ".kv.tmp"), "wb") as f:
            f.write(n.to_bytes(4, "little"))
            f.write(state)
        os.replace(self._p(key + ".kv.tmp"), self._p(key + ".kv"))
        with open(self._p(key + ".prompt.txt"), "w") as f:
            f.write(text)
        old = self.index.get(key, {})
        self.index[key] = {"slot": slot, "uses": old.get("uses", 0), "last": time.time(),
                           "pinned": bool(old.get("pinned") or pin), "tokens": n,
                           "created": old.get("created", time.strftime("%Y-%m-%d %H:%M:%S"))}
        if parent:
            self.index[key]["parent"] = parent           # state holds only the cells after it
        self._remember(key, state, n)
        removed = self._drop_old_versions(slot, key)
        self._evict()
        self._save_index()
        return removed

    def touch(self, key: str):
        e = self.index.get(key)
        if e:
            e["uses"] += 1
            e["last"] = time.time()

    def _drop_old_versions(self, slot: str, keep: str) -> int:
        """A pinned (first-run) write replaces every older version of its slot: the text was
        edited. A normal request's write only replaces older UNPINNED versions, so a request
        whose text differs a little never deletes what the first run built."""
        pinned_write = self.index[keep].get("pinned")
        same = sorted((k for k, e in self.index.items() if e.get("slot") == slot and k != keep
                       and (pinned_write or not e.get("pinned"))),
                      key=lambda k: self.index[k]["last"], reverse=True)
        gone = same if pinned_write else same[self.versions - 1:]
        for k in gone:
            self._delete(k)
        return len(gone)

    def _evict(self):
        """Delete least recently used, unpinned entries until the folder fits max_disk_mb."""
        if not self.max_disk:
            return
        sizes = {k: os.path.getsize(self._p(k + ".kv")) for k in self.index
                 if os.path.exists(self._p(k + ".kv"))}
        total = sum(sizes.values())
        for k in sorted(sizes, key=lambda k: self.index[k]["last"]):
            if total <= self.max_disk:
                break
            if not self.index[k].get("pinned"):
                total -= sizes[k]
                self._delete(k)

    def disk_mb(self) -> float:
        return sum(os.path.getsize(self._p(f)) for f in os.listdir(self.dir)
                   if f.endswith(".kv")) / 1024 / 1024

    def clear_all(self):
        for k in list(self.index):
            self._delete(k)
        self._save_index()
