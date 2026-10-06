"""Engine: one model in llama.cpp, with the few operations the cache needs.

tokenize, decode tokens into the context, save / load the context's KV state, generate
(greedy, so the same prompt always gives the same plan), and the model's chat template.
"""
from __future__ import annotations

import atexit
import ctypes
import os

import llama_cpp as lc

SEQ = 0    # the prompt
TMP = 1    # scratch, used only by speed mode to bring in a module's cells


class Engine:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1,
                 n_threads: int | None = None, verbose: bool = False):
        self.model_path = os.path.abspath(model_path)
        self.n_ctx = n_ctx
        self.llm = lc.Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                            n_threads=n_threads, logits_all=False, verbose=verbose)
        # own context: speed mode needs 2 sequences sharing one KV cache
        p = self.llm.context_params
        p.n_seq_max = 2
        p.kv_unified = True
        self.ctx = lc.llama_init_from_model(self.llm._model.model, p)
        if not self.ctx:
            raise RuntimeError("could not create llama context")
        self.llm._ctx.close()         # the default context is never used; free its KV cache
        self.mem = lc.llama_get_memory(self.ctx)
        self.vocab = lc.llama_model_get_vocab(self.llm._model.model)
        self.n_vocab = lc.llama_vocab_n_tokens(self.vocab)
        self.n_batch = self.llm.n_batch
        atexit.register(self.close)   # Metal asserts at exit if a context is still alive

    # ---------- tokens and KV ----------
    def tok(self, text: str, bos: bool = False) -> list[int]:
        return self.llm.tokenize(text.encode(), add_bos=bos, special=True)

    def clear(self):
        lc.llama_memory_clear(self.mem, True)

    def decode(self, tokens: list[int], start: int, want_last: bool = False):
        """Read tokens at positions start.. into the context; return the last logits if asked."""
        for off in range(0, len(tokens), self.n_batch):
            chunk = tokens[off:off + self.n_batch]
            batch = lc.llama_batch_init(len(chunk), 0, 1)
            try:
                for i, t in enumerate(chunk):
                    batch.token[i] = t
                    batch.pos[i] = start + off + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = SEQ
                    batch.logits[i] = 0
                if want_last and off + len(chunk) == len(tokens):
                    batch.logits[len(chunk) - 1] = 1
                batch.n_tokens = len(chunk)
                if lc.llama_decode(self.ctx, batch) != 0:
                    raise RuntimeError("llama_decode failed")
            finally:
                lc.llama_batch_free(batch)
        if want_last:
            ptr = lc.llama_get_logits_ith(self.ctx, -1)
            return [ptr[i] for i in range(self.n_vocab)]
        return None

    def save(self, seq: int = SEQ) -> bytes:
        n = lc.llama_state_seq_get_size(self.ctx, seq)
        buf = (ctypes.c_uint8 * n)()
        w = lc.llama_state_seq_get_data(self.ctx, buf, n, seq)
        return ctypes.string_at(buf, w)   # bytes(buf[:w]) is ~60x slower

    def load(self, blob: bytes, seq: int = SEQ, clear: bool = True):
        if clear:
            self.clear()
        buf = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        if lc.llama_state_seq_set_data(self.ctx, buf, len(blob), seq) == 0:
            raise RuntimeError("llama_state_seq_set_data failed")

    def stitch(self, blob: bytes, keep_from: int, keep_to: int, at: int):
        """Speed mode: from a saved state, take only the cells at positions keep_from..keep_to
        (one module), move them to start at `at`, and add them to the prompt. The module was
        computed after other text than what is before it now, so this is approximate."""
        self.load(blob, TMP, clear=False)
        lc.llama_memory_seq_rm(self.mem, TMP, 0, keep_from)
        lc.llama_memory_seq_rm(self.mem, TMP, keep_to, -1)
        if at != keep_from:
            lc.llama_memory_seq_add(self.mem, TMP, keep_from, keep_to, at - keep_from)
        lc.llama_memory_seq_cp(self.mem, TMP, SEQ, -1, -1)
        lc.llama_memory_seq_rm(self.mem, TMP, -1, -1)

    def generate(self, logits, pos: int, max_tokens: int, stop: list[str] | None = None) -> tuple[str, int]:
        out, text = [], ""
        for _ in range(max_tokens):
            t = max(range(self.n_vocab), key=logits.__getitem__)   # greedy = deterministic
            if lc.llama_vocab_is_eog(self.vocab, t):
                break
            out.append(t)
            text = self.llm.detokenize(out, special=False).decode(errors="ignore")
            if stop and any(s in text for s in stop):
                break
            logits = self.decode([t], pos, want_last=True)
            pos += 1
        return text, len(out)

    # ---------- chat template ----------
    def chat_wrap(self) -> tuple[str, str]:
        """(prefix, suffix) putting a prompt in the model's own chat template as one user turn.
        Ollama does this for you; chat models often stop at once on a raw prompt."""
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        tmpl = self.llm.metadata.get("tokenizer.chat_template")
        if not tmpl:
            return "", ""
        tok = lambda i: self.llm.detokenize([i], special=True).decode(errors="ignore") if i >= 0 else ""
        bos, eos = tok(lc.llama_vocab_bos(self.vocab)), tok(lc.llama_vocab_eos(self.vocab))
        mark = "\u0000KVS\u0000"
        text = Jinja2ChatFormatter(tmpl, eos, bos)(messages=[{"role": "user", "content": mark}]).prompt
        pre, suf = text.split(mark, 1)
        if bos and pre.startswith(bos):   # the header is tokenized with BOS already
            pre = pre[len(bos):]
        return pre, suf

    def close(self):
        if getattr(self, "ctx", None):
            lc.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "llm", None) is not None:
            self.llm.close()
            self.llm = None
