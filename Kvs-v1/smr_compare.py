"""Where we came from vs where we are: all modules + no cache, against the router + Kvs.

    python Kvs-v1/smr_compare.py granite4:micro --builder my_builder.py --tests tests.txt \
        --all WEB,NOTES,REMINDER,... [--cache-root kvs_cache] [--show]

Three ways, each timed end to end (reading + writing):
    all-modules   every module in --all in the prompt, the model picks the actions itself,
                  no cache, plain writing (one token per pass). Before the router.
    exact+spec    the router is perfect: only the test's modules, in its order; Kvs exact path
                  (cached start, rest read) + speculative writing.
    fast+spec     the same with speed mode (start + stitched modules) + speculative writing.

exact+spec gives exactly the full prompt's plan for the router's modules (proven by
check.py / spec_compare.py). The all-modules plans come from a different prompt, so they are
shown for reading, not scored. Each test starts like a first-time combination: only pinned
first-run starts stay cached.
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
from kvs.speculate import Combined, LookupDrafter, generate, skeleton_drafter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--gguf")
ap.add_argument("--builder", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--all", required=True, help="comma list: every module the all-modules prompt holds")
ap.add_argument("--cache-root", default="kvs_cache")
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--n-ctx-all", type=int, default=16384, help="context for the all-modules prompt")
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
        mods, cmd = line.split("|", 1)
        tests.append(([m.strip() for m in mods.split(",") if m.strip()], cmd.strip()))
ALL = [m.strip() for m in a.all.split(",") if m.strip()]
ACT = re.compile(r'"action"\s*:\s*"([^"]+)"')
norm = lambda s: s.strip().replace("\n", " ")
res = {}                                     # (test, way) -> (seconds, text, stats)


def first_token(eng, toks, pos):
    logits = eng.decode(toks, pos, want_last=True)
    return max(range(len(logits)), key=logits.__getitem__)


# ---------- 1) all modules, no cache, plain writing ----------
kp = KVPlanner({a.model: a.gguf} if a.gguf else {}, a.cache_root + "_unused", n_ctx=a.n_ctx_all)
print(f"loading model (context {a.n_ctx_all}) for the all-modules prompt ...", flush=True)
kp.load(a.model)
eng = kp.eng
for i, (mods, cmd) in enumerate(tests, 1):
    header, pieces, tail = builder.build(ALL, cmd)
    toks = eng.tok(kp.wrap[0] + header, bos=True) + [t for p in pieces for t in eng.tok(p[1])] \
        + eng.tok(tail + kp.wrap[1])
    t0 = time.perf_counter()
    eng.clear()
    first = first_token(eng, toks, 0)
    text, st = generate(eng, first, len(toks), a.max_tokens)
    res[(i, "all-modules")] = (time.perf_counter() - t0, text, st)
    print(f"[{i}/{len(tests)}] all-modules ({len(toks)} prompt tokens): {res[(i, 'all-modules')][0]:.1f}s"
          f"  actions {ACT.findall(text)}", flush=True)
    if a.show:
        print("  plan:", norm(text))
kp.close()

# ---------- 2) and 3) the router + Kvs ----------
kp = KVPlanner({a.model: a.gguf} if a.gguf else {}, a.cache_root, n_ctx=a.n_ctx)
print(f"\nloading model (context {a.n_ctx}) for the router + Kvs ...", flush=True)
kp.load(a.model)
eng = kp.eng
for i, (mods, cmd) in enumerate(tests, 1):
    header, pieces, tail = builder.build(mods, cmd)
    runs, names, texts, midx = kp._runs(header, pieces)
    actions = [ACT.findall(p[1])[0] for p in pieces if len(p) > 2 and p[2] and ACT.findall(p[1])]
    for w in ["exact+spec", "fast+spec"]:
        for k in [k for k, e in kp.cache.index.items() if not e.get("pinned")]:
            if k in kp.cache.index:
                kp.cache._delete(k)                      # a first-time combination
        t0 = time.perf_counter()
        if w == "fast+spec" and kp._can_stitch(runs, names, texts, midx):
            pos, path, _ = kp._put_prefix_fast(runs, names, texts, midx)
        else:
            pos, path, _ = kp._put_prefix(runs, names, texts, midx)
        tt = kp._t(tail + kp.wrap[1])
        first = first_token(eng, tt, pos)
        prompt = [t for r in runs for t in r] + tt
        drafter = Combined(skeleton_drafter(eng, actions, k=a.k), LookupDrafter(prompt, k=a.k))
        text, st = generate(eng, first, pos + len(tt), a.max_tokens, drafter)
        res[(i, w)] = (time.perf_counter() - t0, text, st)
        kp.cache._save_index()
    e, f = res[(i, "exact+spec")], res[(i, "fast+spec")]
    agree = "same" if norm(e[1]) == norm(f[1]) else "DIFF"
    print(f"[{i}/{len(tests)}] {'+'.join(mods)}: exact+spec {e[0]:.1f}s | fast+spec {f[0]:.1f}s "
          f"(fast vs exact: {agree})  — {cmd[:45]}", flush=True)
    if a.show:
        print("  exact+spec:", norm(e[1]))
        if agree == "DIFF":
            print("  fast+spec: ", norm(f[1]))
kp.cache._save_index()
kp.close()

print()
for w in ["all-modules", "exact+spec", "fast+spec"]:
    ts = [res[(i, w)][0] for i in range(1, len(tests) + 1)]
    tpp = [res[(i, w)][2]["tokens"] / max(1, res[(i, w)][2]["passes"]) for i in range(1, len(tests) + 1)]
    print(f"{w:12}: median {stats.median(ts):.1f}s end to end   tokens per pass {stats.median(tpp):.2f}")
fast_same = sum(norm(res[(i, 'exact+spec')][1]) == norm(res[(i, 'fast+spec')][1]) for i in range(1, len(tests) + 1))
print(f"fast+spec gave the exact plan on {fast_same}/{len(tests)}")
