"""Speculative writing: a cheap drafter guesses the next tokens, the model checks them all in one
pass, and keeps the ones it would have written itself. Reading the prompt is not touched.

Exact: with greedy decoding every kept token is the model's own choice, so the plan is the same,
byte for byte, as writing one token at a time. A wrong guess only costs the extra tokens in that
one pass; the model's own token at the first wrong place comes from the same pass, for free.

Drafters (no training, no second model):
    SkeletonDrafter   the JSON shape of the plan, from the action order the router gives:
                      {"plan": [ {"action": "WebAction", "params": { ... } }, ... ]}
                      Values are left out: those are what the model really has to write.
    LookupDrafter     prompt lookup: find the last few written tokens in the prompt and guess
                      what followed them there (paths, param names, placeholders, action names).
    Combined          the skeleton first, then the prompt.
"""
from __future__ import annotations

import json
import re

import llama_cpp as lc
import numpy as np

from .engine import SEQ, Engine


# ---------------- drafters ----------------
class LookupDrafter:
    """Guess by finding the last n written tokens in `source` (n = max_n .. min_n)."""

    def __init__(self, source: list[int], k: int = 12, max_n: int = 4, min_n: int = 2,
                 forward_only: bool = False):
        self.src, self.k, self.max_n, self.min_n = source, k, max_n, min_n
        self.forward_only = forward_only      # keep moving forward through the source
        self.ptr = 0

    def __call__(self, out: list[int]) -> list[int]:
        src = self.src
        for n in range(min(self.max_n, len(out)), self.min_n - 1, -1):
            tail = out[-n:]
            rng = range(self.ptr, len(src) - n) if self.forward_only else range(len(src) - n - 1, -1, -1)
            for i in rng:
                if src[i:i + n] == tail:
                    draft = src[i + n:i + n + self.k]
                    if draft:
                        if self.forward_only:
                            self.ptr = i + n
                        return draft
        return []


def skeleton_text(actions: list[str], indent: int = 2) -> str:
    """The plan's JSON shape for these actions, written the way json.dumps(indent=2) writes it
    (the layout the model uses), with the param values left out."""
    sp = lambda lvl: " " * (indent * lvl)
    s = "{\n" + sp(1) + '"plan": [\n'
    for i, a in enumerate(actions):
        s += sp(2) + "{\n" + sp(3) + '"action": ' + json.dumps(a) + ",\n" + sp(3) + '"params": {\n' + sp(4) + '"'
        s += "\x00"                                   # the values: unknown, never matched
        s += "\n" + sp(3) + "}\n" + sp(2) + "}" + (",\n" if i < len(actions) - 1 else "\n")
    return s + sp(1) + "]\n}"


class Combined:
    def __init__(self, *drafters):
        self.drafters = drafters

    def __call__(self, out: list[int]) -> list[int]:
        for d in self.drafters:
            g = d(out)
            if g:
                return g
        return []


def skeleton_drafter(eng: Engine, actions: list[str], k: int = 12) -> LookupDrafter:
    toks = []
    for part in skeleton_text(actions).split("\x00"):
        toks += eng.tok(part) + [-1]                 # -1: a gap no written token can match
    return LookupDrafter(toks, k=k, max_n=4, min_n=1, forward_only=True)


def example_steps(texts: list[str]) -> list[dict]:
    """Every step example the module docs show ({"action": ..., "params": {...}}), parsed."""
    out, dec = [], json.JSONDecoder()
    # docs often escape braces for str.format: {{ "action": ... }} -> { "action": ... }
    texts = [t for text in texts for t in (text, text.replace("{{", "{").replace("}}", "}"))]
    for text in texts:
        i = 0
        while True:
            i = text.find("{", i)
            if i < 0:
                break
            try:
                obj, end = dec.raw_decode(text[i:])
            except ValueError:
                i += 1
                continue
            if isinstance(obj, dict) and isinstance(obj.get("action"), str) and obj not in out:
                out.append(obj)
            i += max(end, 1)
    return out


def phrasebook_drafter(eng: Engine, module_texts: list[str], k: int = 12) -> LookupDrafter:
    """Every documented step, written the way the plan writes a step (json indent=2, nested
    inside "plan": [ ... ]), so one guess can cover a whole step: action, param names and,
    when the docs show them, common values. Wrong guesses are simply rejected."""
    toks = []
    seen = set()
    for st in example_steps(module_texts):
        body = json.dumps(st, indent=2, ensure_ascii=False)
        text = "    " + body.replace("\n", "\n    ")
        if text in seen:
            continue
        seen.add(text)
        toks += eng.tok(text) + [-1]
    return LookupDrafter(toks, k=k, max_n=4, min_n=3)


# ---------------- writing ----------------
def _argmax(eng: Engine, i: int) -> int:
    ptr = lc.llama_get_logits_ith(eng.ctx, i)
    return int(np.ctypeslib.as_array(ptr, shape=(eng.n_vocab,)).argmax())


def _decode_all(eng: Engine, tokens: list[int], start: int):
    """Read tokens at start.., keeping the logits of every one of them."""
    batch = lc.llama_batch_init(len(tokens), 0, 1)
    try:
        for i, t in enumerate(tokens):
            batch.token[i] = t
            batch.pos[i] = start + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = SEQ
            batch.logits[i] = 1
        batch.n_tokens = len(tokens)
        if lc.llama_decode(eng.ctx, batch) != 0:
            raise RuntimeError("llama_decode failed")
    finally:
        lc.llama_batch_free(batch)


def generate(eng: Engine, first: int, pos: int, max_tokens: int, drafter=None) -> tuple[str, dict]:
    """Greedy writing, optionally speculative. `first` is the model's first token (argmax of the
    logits after the prompt), `pos` the position it goes to. With drafter=None this is plain
    greedy writing, one token per pass (the fair baseline: same code, same argmax).
    Returns (text, stats)."""
    out: list[int] = []
    t = first
    passes = drafted = accepted = 0
    eog = lambda x: lc.llama_vocab_is_eog(eng.vocab, x)
    while len(out) < max_tokens and not eog(t):
        draft = [d for d in (drafter(out + [t]) if drafter else []) if d >= 0]
        draft = draft[:max(0, max_tokens - len(out) - 1)]
        _decode_all(eng, [t] + draft, pos)
        passes += 1
        drafted += len(draft)
        out.append(t)
        nxt = _argmax(eng, 0)                       # the model's choice after t
        m = 0
        while m < len(draft) and draft[m] == nxt and not eog(nxt):
            out.append(draft[m])                     # the guess is what it would write: keep
            m += 1
            nxt = _argmax(eng, m)
        accepted += m
        if m < len(draft):                           # drop the cells of the rejected guesses
            lc.llama_memory_seq_rm(eng.mem, SEQ, pos + 1 + m, -1)
        pos += 1 + m
        t = nxt
    text = eng.llm.detokenize(out, special=False).decode(errors="ignore")
    return text, {"tokens": len(out), "passes": passes, "drafted": drafted, "accepted": accepted}


# ---------------- structure predictor (v1) ----------------
_VALUE_END = re.compile(r'"([A-Za-z_]\w*)":\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|true|false|null|\[[^\[\]]*\]|\{\})$')


def key_orders(module_texts: list[str]) -> dict[str, list[list[str]]]:
    """For each action: every param-key order its documented examples use, most common first.
    (An action can have several forms: Rename(old_name, new_name, path) and Rename(source_path, clean).)"""
    from collections import Counter
    seqs: dict[str, Counter] = {}
    for st in example_steps(module_texts):
        p = st.get("params")
        if isinstance(p, dict):
            seqs.setdefault(st["action"], Counter())[tuple(p.keys())] += 1
    return {a: [list(t) for t, _ in c.most_common()] for a, c in seqs.items()}


class StructureDrafter:
    """Predicts the fixed parts of the plan and leaves the values to the model.

    Knows the action order (from the router) and each action's usual param keys (from the
    docs). From the text written so far it works out where it is and guesses the next fixed
    piece: the opening and the first step, the next key after a value, or closing the step and
    opening the next one (or closing the plan). Values are never guessed. Layout: json indent=2,
    the way the model writes plans."""

    def __init__(self, eng: Engine, actions: list[str], keys: dict[str, list[str]], k: int = 32):
        self.eng, self.actions, self.keys, self.k = eng, actions, keys, k

    def _orders(self, action: str) -> list[list[str]]:
        o = self.keys.get(action) or []
        return o if (o and isinstance(o[0], list)) else ([o] if o else [])

    def _step_open(self, i: int) -> str:
        a = self.actions[i]
        orders = self._orders(a)
        first = orders[0][0] if orders and orders[0] else None
        s = '    {\n      "action": ' + json.dumps(a) + ',\n      "params": {'
        return s + ('\n        ' + json.dumps(first) + ': ' if first else '}')

    def _text(self, out: list[int]) -> str:
        return self.eng.llm.detokenize(out, special=False).decode(errors="ignore")

    def guess(self, text: str) -> str:
        """The rest of the fixed piece the model is in, if any. The model's tokens do not end
        exactly where a value ends (it may write a quote and a newline as one token), so look a
        few characters back for the last place a fixed piece starts, and continue from there."""
        if not self.actions:
            return ""
        for cut in range(len(text), max(-1, len(text) - 48), -1):
            g = self._guess_at(text[:cut])
            done = text[cut:]
            if g and g.startswith(done) and len(g) > len(done):
                return g[len(done):]
        return ""

    def _guess_at(self, text: str) -> str:
        opening = '{\n  "plan": [\n' + self._step_open(0)
        if len(text) < len(opening) and opening.startswith(text):
            return opening[len(text):]
        m = _VALUE_END.search(text)
        if not m:
            return ""
        step = text.count('"action"') - 1
        if step < 0 or step >= len(self.actions):
            return ""
        last = text.rfind('"params"')
        used = re.findall(r'"([A-Za-z_]\w*)":', text[last + len('"params"'):]) if last >= 0 else []
        # the most common documented form that starts with the keys written so far
        order = next((o for o in self._orders(self.actions[step]) if o[:len(used)] == used), None)
        if order and len(used) < len(order):
            return ",\n        " + json.dumps(order[len(used)]) + ": "
        tail = "\n      }\n    }"
        if step + 1 < len(self.actions):
            return tail + ",\n" + self._step_open(step + 1)
        return tail + "\n  ]\n}"

    def __call__(self, out: list[int]) -> list[int]:
        g = self.guess(self._text(out))
        return self.eng.tok(g)[:self.k] if g else []
