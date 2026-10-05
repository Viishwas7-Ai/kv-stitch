"""Modular prompt KV-cache stitching on top of llama.cpp (via llama-cpp-python).

A prompt is split into:
    header  - fixed text that starts every prompt (computed once)
    modules - named blocks; any subset, in any order, follows the header
    tail    - per-request text (context + user command), always computed fresh

Each module is precomputed ONCE as `header + module` and only the module's
cells are kept. At request time the header and the chosen modules are loaded,
each module is moved (RoPE shift) to where it sits in this prompt, and only
the tail is decoded. Modules never saw each other, which is the one
approximation; `refresh` recomputes the first N tokens of every module after
the first to soften the joins.
"""
from __future__ import annotations

import atexit
import ctypes
import hashlib
import os
import time
from dataclasses import dataclass, field

import llama_cpp as lc

MAIN = 0   # sequence the assembled prompt lives in
TMP = 1    # scratch sequence used while loading a module


@dataclass
class Block:
    tokens: list[int]
    state: bytes          # llama_state_seq_get_data blob (cells for this block only)
    start: int            # position its first token was computed at


@dataclass
class Timing:
    load_s: float = 0.0
    tail_s: float = 0.0
    gen_s: float = 0.0
    tail_tokens: int = 0
    gen_tokens: int = 0
    extra: dict = field(default_factory=dict)


class Stitcher:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1,
                 n_threads: int | None = None, verbose: bool = False):
        # Llama gives us the model, tokenizer and default params; we make our own
        # context because stitching needs 2 sequences sharing one (unified) KV cache.
        self.llm = lc.Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                            n_threads=n_threads, logits_all=False, verbose=verbose)
        p = self.llm.context_params
        p.n_seq_max = 2
        p.kv_unified = True
        self.ctx = lc.llama_init_from_model(self.llm._model.model, p)
        if not self.ctx:
            raise RuntimeError("could not create llama context")
        # Llama made its own context (and KV cache) we never use; free it now so an
        # 8 GB machine doesn't hold two caches. Tokenizing only needs the model.
        self.llm._ctx.close()
        atexit.register(self.close)   # Metal asserts at exit if a context is still alive
        self.mem = lc.llama_get_memory(self.ctx)
        self.vocab = lc.llama_model_get_vocab(self.llm._model.model)
        self.n_vocab = lc.llama_vocab_n_tokens(self.vocab)
        self.n_batch = self.llm.n_batch
        self.header: Block | None = None
        self.modules: dict[str, Block] = {}

    # ---------- low level ----------
    def tok(self, text: str, bos: bool = False) -> list[int]:
        return self.llm.tokenize(text.encode(), add_bos=bos, special=True)

    def _decode(self, tokens: list[int], start: int, seq: int, want_last: bool = False):
        """Decode tokens at positions start.. into `seq`; return last logits if asked."""
        for off in range(0, len(tokens), self.n_batch):
            chunk = tokens[off:off + self.n_batch]
            batch = lc.llama_batch_init(len(chunk), 0, 1)
            try:
                for i, t in enumerate(chunk):
                    batch.token[i] = t
                    batch.pos[i] = start + off + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = seq
                    batch.logits[i] = 0
                last = off + len(chunk) == len(tokens)
                if want_last and last:
                    batch.logits[len(chunk) - 1] = 1
                batch.n_tokens = len(chunk)
                rc = lc.llama_decode(self.ctx, batch)
                if rc != 0:
                    raise RuntimeError(f"llama_decode failed ({rc})")
            finally:
                lc.llama_batch_free(batch)
        if want_last:
            ptr = lc.llama_get_logits_ith(self.ctx, -1)
            return [ptr[i] for i in range(self.n_vocab)]
        return None

    def _save(self, seq: int) -> bytes:
        n = lc.llama_state_seq_get_size(self.ctx, seq)
        buf = (ctypes.c_uint8 * n)()
        w = lc.llama_state_seq_get_data(self.ctx, buf, n, seq)
        return bytes(buf[:w])

    def _load(self, blob: bytes, seq: int):
        buf = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        if lc.llama_state_seq_set_data(self.ctx, buf, len(blob), seq) == 0:
            raise RuntimeError("llama_state_seq_set_data failed")

    def clear(self):
        lc.llama_memory_clear(self.mem, True)

    def chat_wrap(self) -> tuple[str, str]:
        """(prefix, suffix) that put a prompt inside the model's own chat template as
        one user turn followed by the assistant's turn. Chat models (granite, qwen, ...)
        often stop at once on a raw prompt; Ollama applies this template for you."""
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        meta = self.llm.metadata
        tmpl = meta.get("tokenizer.chat_template")
        if not tmpl:
            return "", ""
        tok = lambda i: self.llm.detokenize([i], special=True).decode(errors="ignore") if i >= 0 else ""
        bos = tok(lc.llama_vocab_bos(self.vocab))
        eos = tok(lc.llama_vocab_eos(self.vocab))
        mark = "\u0000KVSTITCH\u0000"
        text = Jinja2ChatFormatter(tmpl, eos, bos)(messages=[{"role": "user", "content": mark}]).prompt
        pre, suf = text.split(mark, 1)
        if bos and pre.startswith(bos):   # set_header adds BOS itself
            pre = pre[len(bos):]
        return pre, suf

    def close(self):
        """Free the context and model. Safe to call more than once."""
        if getattr(self, "ctx", None):
            lc.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "llm", None) is not None:
            self.llm.close()
            self.llm = None

    # ---------- precompute ----------
    def set_header(self, text: str):
        self.clear()
        toks = self.tok(text, bos=True)
        self._decode(toks, 0, MAIN)
        self.header = Block(toks, self._save(MAIN), 0)
        self.modules.clear()   # modules were computed after the old header

    def add_module(self, name: str, text: str):
        """Compute header+module once and keep only the module's cells."""
        assert self.header, "set_header first"
        h = len(self.header.tokens)
        toks = self.tok(text)
        self.clear()
        self._load(self.header.state, MAIN)
        self._decode(toks, h, MAIN)
        lc.llama_memory_seq_rm(self.mem, MAIN, 0, h)      # drop header cells
        self.modules[name] = Block(toks, self._save(MAIN), h)

    # ---------- per request ----------
    def assemble(self, names: list[str], refresh: int = 0,
                 fresh_idx: set[int] | None = None) -> tuple[list[int], int]:
        """Load header + modules into MAIN. Returns (all tokens, next position).

        fresh_idx: positions in `names` that are decoded fresh instead of loaded, so they see
        everything before them exactly. Cached pieces after them are still loaded (shifted).
        """
        self.clear()
        self._load(self.header.state, MAIN)
        toks = list(self.header.tokens)
        pos = len(toks)
        for i, name in enumerate(names):
            b = self.modules[name]
            n = len(b.tokens)
            if fresh_idx and i in fresh_idx:
                self._decode(b.tokens, pos, MAIN)
                toks += b.tokens
                pos += n
                continue
            # refresh: the first r tokens are decoded fresh, so they see every module
            # before them (the first module already saw exactly the header). llama.cpp
            # only accepts new tokens after the last position, so do this before the copy.
            r = min(refresh, n) if refresh and i > 0 else 0
            if r:
                self._decode(b.tokens[:r], pos, MAIN)
            self._load(b.state, TMP)
            if pos != b.start:
                lc.llama_memory_seq_add(self.mem, TMP, b.start, b.start + n, pos - b.start)
            if r:
                lc.llama_memory_seq_rm(self.mem, TMP, pos, pos + r)
            lc.llama_memory_seq_cp(self.mem, TMP, MAIN, -1, -1)
            lc.llama_memory_seq_rm(self.mem, TMP, -1, -1)
            toks += b.tokens
            pos += n
        return toks, pos

    def _generate(self, logits, pos: int, max_tokens: int, stop: list[str]) -> tuple[str, int]:
        out, text = [], ""
        eog = lambda t: lc.llama_vocab_is_eog(self.vocab, t)
        for _ in range(max_tokens):
            t = max(range(self.n_vocab), key=logits.__getitem__)   # greedy = deterministic
            if eog(t):
                break
            out.append(t)
            text = self.llm.detokenize(out, special=False).decode(errors="ignore")
            if any(s in text for s in stop):
                break
            logits = self._decode([t], pos, MAIN, want_last=True)
            pos += 1
        return text, len(out)

    def split_fresh(self, names: list[str], fresh: int) -> int:
        """Index k: modules names[k:] (together at most `fresh` tokens) are computed fresh."""
        k, total = len(names), 0
        while k > 0 and total + len(self.modules[names[k - 1]].tokens) <= fresh:
            k -= 1
            total += len(self.modules[names[k]].tokens)
        return k

    def run(self, names: list[str], tail: str, max_tokens: int = 512,
            stop: list[str] | None = None, refresh: int = 0, fresh: int = 0,
            fresh_idx: set[int] | None = None) -> tuple[str, Timing]:
        """Stitched prompt: header + modules (cached) + tail (fresh).

        refresh: recompute the first N tokens of every module after the first (EPIC-style).
        fresh:   the last modules, up to N tokens in total, are not loaded from the cache but
                 computed fresh together with the tail, so they see everything before them.
        fresh_idx: positions in `names` computed fresh wherever they are; the cached pieces
                 around them are still loaded.
        """
        tm = Timing()
        k = self.split_fresh(names, fresh) if fresh else len(names)
        t0 = time.perf_counter()
        _, pos = self.assemble(names[:k], refresh, fresh_idx)
        tm.load_s = time.perf_counter() - t0
        tt = [t for n in names[k:] for t in self.modules[n].tokens] + self.tok(tail)
        tm.extra["fresh_modules"] = len(names) - k
        t0 = time.perf_counter()
        logits = self._decode(tt, pos, MAIN, want_last=True)
        tm.tail_s, tm.tail_tokens = time.perf_counter() - t0, len(tt)
        t0 = time.perf_counter()
        text, tm.gen_tokens = self._generate(logits, pos + len(tt), max_tokens, stop or [])
        tm.gen_s = time.perf_counter() - t0
        return text, tm

    def run_full(self, names: list[str], tail: str, max_tokens: int = 512,
                 stop: list[str] | None = None) -> tuple[str, Timing]:
        """Baseline: the same token sequence, computed from scratch."""
        toks = list(self.header.tokens)
        for n in names:
            toks += self.modules[n].tokens
        toks += self.tok(tail)
        tm = Timing()
        self.clear()
        t0 = time.perf_counter()
        logits = self._decode(toks, 0, MAIN, want_last=True)
        tm.tail_s, tm.tail_tokens = time.perf_counter() - t0, len(toks)
        t0 = time.perf_counter()
        text, tm.gen_tokens = self._generate(logits, len(toks), max_tokens, stop or [])
        tm.gen_s = time.perf_counter() - t0
        return text, tm

    # ---------- disk cache ----------
    def save_dir(self, path: str):
        import pickle
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "cache.pkl"), "wb") as f:
            pickle.dump({"header": self.header, "modules": self.modules}, f)

    def load_dir(self, path: str):
        import pickle
        with open(os.path.join(path, "cache.pkl"), "rb") as f:
            d = pickle.load(f)
        self.header, self.modules = d["header"], d["modules"]


def text_hash(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:12]
