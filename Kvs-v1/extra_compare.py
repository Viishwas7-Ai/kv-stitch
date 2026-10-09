"""The router is not perfect: a raw dump of modules, and the right modules plus extras.

    python Kvs-v1/extra_compare.py granite4:micro --builder my_builder.py --tests tests.txt \
        --dump WEB,NOTES,... [--cache-root kvs_cache] [--show]

tests.txt, one per line:   NEEDED_A, NEEDED_B | EXTRA_X, EXTRA_Y | the user's command

Four ways per test, timed end to end:
    dump         every module in --dump in the prompt, no cache, plain writing (no router)
    full+extra   the needed modules then the extra ones, no cache, plain writing.
                 The reference: ways 3 and 4 read this same prompt.
    exact+pred   Kvs exact path + speculative writing with the structure predictor (+ lookups).
                 The predictor gets the ROUTER'S modules (extras included), turned into one
                 action each, and re-syncs from whatever action the model writes. Must give
                 the same plan as full+extra.
    fast+pred    the same with Kvs speed mode (stitched modules): faster, not always the same.
"""
import argparse
import importlib.util
import os
import re
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvs import KVPlanner  # noqa: E402
from kvs.speculate import (Combined, LookupDrafter, StructureDrafter, generate, key_orders,  # noqa: E402
                           module_actions, phrasebook_drafter, string_keys)

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--gguf")
ap.add_argument("--builder", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--dump", required=True, help="comma list: the modules of the raw dump prompt")
ap.add_argument("--cache-root", default="kvs_cache")
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--n-ctx-dump", type=int, default=16384)
ap.add_argument("--max-tokens", type=int, default=500)
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--show", action="store_true")
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("app_builder", a.builder)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
tests = []
for line in open(a.tests):
    line = line.strip()
    if line and not line.startswith("#"):
        need, extra, cmd = [x.strip() for x in line.split("|", 2)]
        split = lambda s: [m.strip() for m in s.split(",") if m.strip()]
        tests.append((split(need), split(extra), cmd))
DUMP = [m.strip() for m in a.dump.split(",") if m.strip()]
ACT = re.compile(r'"action"\s*:\s*"([^"]+)"')
norm = lambda s: s.strip().replace("\n", " ")
res = {}


def first_token(eng, toks, pos):
    logits = eng.decode(toks, pos, want_last=True)
    return max(range(len(logits)), key=logits.__getitem__)


# ---------- 1) the raw dump ----------
kp = KVPlanner({a.model: a.gguf} if a.gguf else {}, a.cache_root + "_unused", n_ctx=a.n_ctx_dump)
print(f"loading model (context {a.n_ctx_dump}) for the {len(DUMP)}-module dump ...", flush=True)
kp.load(a.model)
eng = kp.eng
for i, (need, extra, cmd) in enumerate(tests, 1):
    header, pieces, tail = builder.build(DUMP, cmd)
    toks = eng.tok(kp.wrap[0] + header, bos=True) + [t for p in pieces for t in eng.tok(p[1])] \
        + eng.tok(tail + kp.wrap[1])
    t0 = time.perf_counter()
    eng.clear()
    text, st = generate(eng, first_token(eng, toks, 0), len(toks), a.max_tokens)
    res[(i, "dump")] = (time.perf_counter() - t0, text, st)
    print(f"[{i}/{len(tests)}] dump ({len(toks)} prompt tokens): {res[(i, 'dump')][0]:.1f}s  "
          f"actions {ACT.findall(text)}", flush=True)
    if a.show:
        print("  plan:", norm(text))
kp.close()

# ---------- 2-4) the router's modules (needed + extra) ----------
kp = KVPlanner({a.model: a.gguf} if a.gguf else {}, a.cache_root, n_ctx=a.n_ctx)
print(f"\nloading model (context {a.n_ctx}) for the router's modules ...", flush=True)
kp.load(a.model)
eng = kp.eng
for i, (need, extra, cmd) in enumerate(tests, 1):
    mods = need + extra
    header, pieces, tail = builder.build(mods, cmd)
    runs, names, texts, midx = kp._runs(header, pieces)
    tt = kp._t(tail + kp.wrap[1])
    mod_texts = [p[1] for p in pieces if len(p) > 2 and p[2]]
    actions = module_actions(mod_texts)                       # the router's view, extras included

    # 2) full + extra, no cache
    toks = [t for r in runs for t in r] + tt
    t0 = time.perf_counter()
    eng.clear()
    ref, st = generate(eng, first_token(eng, toks, 0), len(toks), a.max_tokens)
    res[(i, "full+extra")] = (time.perf_counter() - t0, ref, st)

    for w in ["exact+pred", "fast+pred"]:
        for k in [k for k, e in kp.cache.index.items() if not e.get("pinned")]:
            if k in kp.cache.index:
                kp.cache._delete(k)                           # a first-time combination
        t0 = time.perf_counter()
        if w == "fast+pred" and kp._can_stitch(runs, names, texts, midx):
            pos, path, _ = kp._put_prefix_fast(runs, names, texts, midx)
        else:
            pos, path, _ = kp._put_prefix(runs, names, texts, midx)
        t_read = time.perf_counter() - t0
        prompt = [t for r in runs for t in r] + tt
        drafter = Combined(StructureDrafter(eng, actions, key_orders(mod_texts), str_keys=string_keys(mod_texts)),
                           phrasebook_drafter(eng, mod_texts, k=a.k), LookupDrafter(prompt, k=a.k))
        text, st = generate(eng, first_token(eng, tt, pos), pos + len(tt), a.max_tokens, drafter)
        st["read_s"] = round(t_read, 1)
        res[(i, w)] = (time.perf_counter() - t0, text, st)
        kp.cache._save_index()

    line = f"[{i}/{len(tests)}] {'+'.join(need)} (+{'+'.join(extra)}): full+extra {res[(i, 'full+extra')][0]:.1f}s"
    for w in ["exact+pred", "fast+pred"]:
        secs, text, st = res[(i, w)]
        line += (f" | {w} {secs:.1f}s {'same' if norm(text) == norm(ref) else 'DIFF'} "
                 f"[read {st['read_s']} | write {st['check_s']} | {st['tokens']}tok/{st['passes']}passes]")
    print(line + f"  — {cmd[:40]}", flush=True)
    print(f"  router said: {actions}   model wrote: {ACT.findall(ref)}", flush=True)
    if a.show:
        print("  full+extra:", norm(ref))
        f = res[(i, "fast+pred")][1]
        if norm(f) != norm(ref):
            print("  fast+pred: ", norm(f))
kp.cache._save_index()
kp.close()

print()
for w in ["dump", "full+extra", "exact+pred", "fast+pred"]:
    ts = [res[(i, w)][0] for i in range(1, len(tests) + 1)]
    tpp = [res[(i, w)][2]["tokens"] / max(1, res[(i, w)][2]["passes"]) for i in range(1, len(tests) + 1)]
    same = sum(norm(res[(i, w)][1]) == norm(res[(i, "full+extra")][1]) for i in range(1, len(tests) + 1))
    tag = "" if w == "dump" else f"   same as full+extra {same}/{len(tests)}"
    print(f"{w:11}: median {stats.median(ts):.1f}s   tokens per pass {stats.median(tpp):.2f}{tag}")
