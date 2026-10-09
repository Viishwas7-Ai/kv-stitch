"""Exact + speculative writing vs speed mode + speculative writing, against an earlier run.

    python Kvs-v1/spec_compare.py granite4:micro --builder my_builder.py --tests tests.txt \
        --ref kvs_20.txt [--cache-root kvs_cache] [--show]

--ref is the output of an earlier `check.py --show` run on the same tests: its "plan:" lines
are the exact plans (= the full prompt's plans), used to check every plan here and, since the
router is assumed perfect, to give the drafter the right action order. Without --ref the
action order comes from the modules (the first action each one documents).

Two ways, each timed end to end (reading + writing):
    exact+spec   Kvs exact path (cached start, rest read), then speculative writing
    fast+spec    Kvs speed mode (start + stitched modules), then speculative writing
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
                           phrasebook_drafter, skeleton_drafter, string_keys)

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--gguf")
ap.add_argument("--builder", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--ref", help="output of an earlier check.py --show run on the same tests")
ap.add_argument("--cache-root", default="kvs_cache")
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--k", type=int, default=12, help="tokens guessed per pass")
ap.add_argument("--phrasebook", action="store_true",
                help="also draft from every step example in the module docs, in the plan's layout")
ap.add_argument("--drafter", default="lookup", choices=["lookup", "structure", "structure+lookup"],
                help="lookup = phrasebook/skeleton/prompt lookup; structure = the structure predictor "
                     "only; structure+lookup = structure first, then the lookups")
ap.add_argument("--single", default="",
                help="comma list of actions to treat as one-param (predict into the value) by hand")
ap.add_argument("--only", default="", help="comma list: run only these ways (exact+spec,fast+spec)")
ap.add_argument("--limit", type=int, default=0, help="run only the first N tests")
ap.add_argument("--show", action="store_true")
ap.add_argument("--keep-cached", action="store_true",
                help="use whatever is cached; by default each test starts like a first-time "
                     "combination: unpinned caches (left from earlier tests) are removed first")
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

refs = []
if a.ref:
    refs = [l[len("  plan: "):].rstrip("\n") for l in open(a.ref) if l.startswith("  plan: ")]
    if len(refs) != len(tests):
        sys.exit(f"--ref has {len(refs)} plans but there are {len(tests)} tests")
if a.limit:
    tests, refs = tests[:a.limit], refs[:a.limit]

norm = lambda s: s.strip().replace("\n", " ")
ACT = re.compile(r'"action"\s*:\s*"([^"]+)"')

kp = KVPlanner({a.model: a.gguf} if a.gguf else {}, a.cache_root, n_ctx=a.n_ctx, max_tokens=a.max_tokens)
print("loading model ...", flush=True)
kp.load(a.model)
eng = kp.eng

ways = [w.strip() for w in a.only.split(",") if w.strip()] or ["exact+spec", "fast+spec"]
times = {w: [] for w in ways}
same = {w: 0 for w in ways}
tpp = {w: [] for w in ways}
paths = {w: {} for w in ways}
print(f"{len(tests)} tests x {len(ways)} way(s), drafter: {a.drafter}{' + phrasebook' if a.phrasebook else ''}. Each line prints when its test is done.\n", flush=True)
for i, (mods, cmd) in enumerate(tests, 1):
    header, pieces, tail = builder.build(mods, cmd)
    runs, names, texts, midx = kp._runs(header, pieces)
    if refs:
        actions = ACT.findall(refs[i - 1])                      # the perfect router
    else:
        actions = [ACT.findall(p[1])[0] for p in pieces if len(p) > 2 and p[2] and ACT.findall(p[1])]
    row = []
    for w in ways:
        if not a.keep_cached:                                   # a first-time combination
            for k in [k for k, e in kp.cache.index.items() if not e.get("pinned")]:
                if k in kp.cache.index:
                    kp.cache._delete(k)
        t0 = time.perf_counter()
        t_read0 = t0
        if w == "fast+spec" and kp._can_stitch(runs, names, texts, midx):
            pos, path, _ = kp._put_prefix_fast(runs, names, texts, midx)
        else:
            pos, path, _ = kp._put_prefix(runs, names, texts, midx)
        tt = kp._t(tail + kp.wrap[1])
        logits = eng.decode(tt, pos, want_last=True)
        first = max(range(len(logits)), key=logits.__getitem__)
        prompt = [t for r in runs for t in r] + tt
        mod_texts = [p[1] for p in pieces if len(p) > 2 and p[2]]
        parts = []
        if a.drafter.startswith("structure"):
            parts.append(StructureDrafter(eng, actions, key_orders(mod_texts), str_keys=string_keys(mod_texts),
                                          single={x.strip() for x in a.single.split(",") if x.strip()}))
        if a.drafter != "structure":
            if a.phrasebook:
                parts.append(phrasebook_drafter(eng, mod_texts, k=a.k))
            parts += [skeleton_drafter(eng, actions, k=a.k), LookupDrafter(prompt, k=a.k)]
        drafter = Combined(*parts)
        t_read = time.perf_counter() - t_read0
        text, st = generate(eng, first, pos + len(tt), a.max_tokens, drafter)
        secs = time.perf_counter() - t0
        kp.cache._save_index()
        ok = (norm(text) == refs[i - 1]) if refs else None
        same[w] += bool(ok)
        times[w].append(secs)
        tpp[w].append(st["tokens"] / max(1, st["passes"]))
        paths[w][path] = paths[w].get(path, 0) + 1
        mark = "" if ok is None else (" same" if ok else " DIFF")
        row.append(f"{w}({path}) {secs:.1f}s {st['tokens']}tok/{st['passes']}passes{mark} "
                   f"[read {t_read:.1f} | draft {st['draft_s']:.1f} | check {st['check_s']:.1f} | "
                   f"guessed {st['drafted']}, kept {st['accepted']}]")
        if a.show and ok is not True:
            print(f"  {w}:", norm(text))
    print(f"[{i}/{len(tests)}] {'+'.join(mods)}: " + " | ".join(row) + f"  — {cmd[:40]}", flush=True)

print()
for w in ways:
    s = f"same plan {same[w]}/{len(tests)}   " if refs else ""
    print(f"{w:10}: {s}median {stats.median(times[w]):.1f}s end to end   "
          f"tokens per pass {stats.median(tpp[w]):.2f}   paths {paths[w]}")
print("earlier run on these tests: no-cache 36.9s, exact 33.4s (20/20), speed 27.5s (9/20)")
kp.close()
