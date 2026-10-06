"""Speculative writing on your own prompts: same plan? how much faster is the WRITING?

    python Kvs-v1/spec_check.py granite4:micro --builder my_builder.py --tests tests.txt \
        [--cache-root kvs_cache] [--k 12] [--show]

Reading the prompt is done the normal Kvs way and is not timed. For every test the plan is
written three ways, from the same prompt state:

    plain     one token per pass (the baseline)
    lookup    drafts from the prompt only (no router help)
    skeleton  drafts from the plan's JSON shape for the right action order + the prompt.
              "The router is perfect": the action order is taken from the plain plan.

Greedy decoding: every way must give the same plan as plain.
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
from kvs.speculate import Combined, LookupDrafter, generate, skeleton_drafter, _argmax  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--gguf")
ap.add_argument("--builder", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--cache-root", default="kvs_cache")
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--k", type=int, default=12, help="tokens guessed per pass")
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

kp = KVPlanner({a.model: a.gguf} if a.gguf else {}, a.cache_root, n_ctx=a.n_ctx, max_tokens=a.max_tokens)
print("loading model ...", flush=True)
kp.load(a.model)
eng = kp.eng


def ready(header, pieces, tail):
    """Read the prompt the normal Kvs way; return (first token, its position, prompt tokens)."""
    runs, names, texts, mods = kp._runs(header, pieces)
    pos, _, _ = kp._put_prefix(runs, names, texts, mods)
    tt = kp._t(tail + kp.wrap[1])
    logits = eng.decode(tt, pos, want_last=True)
    first = max(range(len(logits)), key=logits.__getitem__)
    return first, pos + len(tt), [t for r in runs for t in r] + tt


ways = ["plain", "lookup", "skeleton"]
times = {w: [] for w in ways}
same = {w: 0 for w in ways}
acc = {w: [] for w in ways}
print(f"{len(tests)} tests x 3 ways of writing. Each line prints when its test is done.\n", flush=True)
for i, (mods, cmd) in enumerate(tests, 1):
    header, pieces, tail = builder.build(mods, cmd)
    row = []
    ref = None
    for w in ways:
        first, pos, prompt = ready(header, pieces, tail)
        if w == "plain":
            drafter = None
        elif w == "lookup":
            drafter = LookupDrafter(prompt, k=a.k)
        else:
            actions = re.findall(r'"action"\s*:\s*"([^"]+)"', ref)       # the perfect router
            drafter = Combined(skeleton_drafter(eng, actions, k=a.k), LookupDrafter(prompt, k=a.k))
        t0 = time.perf_counter()
        text, st = generate(eng, first, pos, a.max_tokens, drafter)
        secs = time.perf_counter() - t0
        if w == "plain":
            ref = text
        ok = text == ref
        same[w] += ok
        times[w].append(secs)
        acc[w].append(st["tokens"] / max(1, st["passes"]))
        row.append(f"{w} {secs:.1f}s {st['tokens']}tok/{st['passes']}passes{'' if ok else ' DIFF'}")
        if a.show and (w == "plain" or not ok):
            print(f"  {w}:", text.strip().replace("\n", " "))
    print(f"[{i}/{len(tests)}] {'+'.join(mods)}: " + " | ".join(row) + f"  — {cmd[:40]}", flush=True)

print()
base = stats.median(times["plain"])
for w in ways:
    m = stats.median(times[w])
    print(f"{w:9}: same plan {same[w]}/{len(tests)}   writing median {m:.1f}s ({base / m:.2f}x)   "
          f"tokens per pass {stats.median(acc[w]):.2f}")
kp.close()
