"""No cache vs exact start vs fast mode, on the same requests.

    python bench/fast_check.py MODEL.gguf --builder my_builder.py --tests tests.txt \
        [--refresh 0,16,32,64] [--model-name granite3.1-moe:3b] [--show]

For every test:

    no-cache  the whole prompt computed from scratch: the reference
    start     header + pieces up to the FIRST module loaded from the exact cache (what an app
              pins on first launch for every core module), everything after it computed
              fresh. Exact: must give the same plan as no-cache.
    fast      the same exact start, then the other modules stitched from their own caches
              (the first N tokens of each re-read in place, one run per --refresh value),
              fixed parts after them and the request computed fresh. Approximate.

my_builder.py defines build(module_names, task) -> (header, pieces, tail), as for
workflow_check.py. tests.txt, one per line:   MOD_A, MOD_B, MOD_C | the user's command
"""
import argparse
import importlib.util
import json
import os
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvstitch import PlannerBackend  # noqa: E402


def parse_json(text):
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except ValueError:
        return None


def actions(obj):
    try:
        return [step.get("action", step.get("act")) for step in obj.get("plan", [])]
    except AttributeError:
        return None


def verdict(out, ref):
    """'text' identical > 'json' same JSON > 'actions' same steps > 'DIFF'."""
    if out.strip() == ref.strip():
        return "text"
    a, b = parse_json(out), parse_json(ref)
    if a is not None and a == b:
        return "json"
    if a is not None and b is not None and actions(a) == actions(b) and actions(a):
        return "actions"
    return "DIFF"


ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--model-name", default="model")
ap.add_argument("--builder", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--refresh", default="0", help="comma list: tokens re-read per stitched module")
ap.add_argument("--cache-dir", default="cache_fast_")
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--no-chat", action="store_true")
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

refresh = [int(x) for x in a.refresh.split(",")]
be = PlannerBackend({a.model_name: a.model}, a.cache_dir, n_ctx=a.n_ctx, chat=not a.no_chat,
                    max_tokens=a.max_tokens, save_after=1, fast_mode=True,
                    fast_max_modules=max(len(m) for m, _ in tests))
print("loading model ...", flush=True)
be._load(a.model_name)
print(f"{len(tests)} tests x (no-cache + start + {len(refresh)} fast run(s)). "
      "Each line prints when its test is done.", flush=True)

pc = be._pc
kinds = ["start"] + [f"r{r}" for r in refresh]
score = {k: {"text": 0, "json": 0, "actions": 0, "DIFF": 0} for k in kinds}
times = {k: [] for k in ["no-cache"] + kinds}
gen_tokens = []


def forget(names):
    """Drop the cached whole prefix so the next request does not just hit it."""
    k = pc.key(names)
    pc.mem.pop(k, None)
    p = pc._path(k)
    if p and os.path.exists(p):
        os.remove(p)
    pc.index.pop(k, None)


for i, (mods, cmd) in enumerate(tests, 1):
    print(f"[{i}/{len(tests)}] {'+'.join(mods)} ...", end="", flush=True)
    header, pieces, tail = builder.build(mods, cmd)
    names, module_idx = be._prepare(header, pieces)
    tail_full = tail + be._wrap[1]
    first = module_idx[0] + 1
    row = {}

    t0 = time.perf_counter()
    ref, rtm = be._st.run_full(names, tail_full, a.max_tokens)
    times["no-cache"].append(time.perf_counter() - t0)
    gen_tokens.append(rtm.gen_tokens)
    row["no-cache"] = f"no-cache {times['no-cache'][-1]:.2f}s ({rtm.gen_tokens} tok written)"
    print(" no-cache", end="", flush=True)
    if a.show:
        print("\n  no-cache:", ref.strip().replace("\n", " "))

    pc.warm([names[:first]], pin=False)                    # the start, as pinned on first launch
    for r in refresh:
        be.fast_refresh = r
        be.plan(a.model_name, header, pieces, tail, fast=True)         # builds the module caches
        res = be.plan(a.model_name, header, pieces, tail, fast=True)   # what a user sees once warm
        k = f"r{r}"
        v = verdict(res.text, ref)
        score[k][v] += 1
        times[k].append(res.seconds)
        row[k] = f"fast {k} {res.seconds:.2f}s {v}"
        print(f" {k}", end="", flush=True)
        if a.show:
            print(f"\n  fast {k}:", res.text.strip().replace("\n", " "))

    forget(names)
    t0 = time.perf_counter()
    text, tm = pc.run(names, tail_full, a.max_tokens)       # start loaded, the rest computed
    secs = time.perf_counter() - t0
    forget(names)
    v = verdict(text, ref)
    score["start"][v] += 1
    times["start"].append(secs)
    row["start"] = f"start({tm.extra['prefix_kind']}) {secs:.2f}s {v}"
    print(" start", end="", flush=True)
    if a.show:
        print("\n  start:", text.strip().replace("\n", " "))

    print("\n  " + " | ".join(row[k] for k in ["no-cache", "start"] + kinds[1:])
          + f"  — {cmd[:50]}", flush=True)

n = len(tests)
print(f"\n{n} tests, median no-cache {stats.median(times['no-cache']):.2f}s, "
      f"median {stats.median(gen_tokens):.0f} tokens written")
for k in kinds:
    s = score[k]
    label = "exact start      " if k == "start" else f"fast refresh {k[1:]:>4}"
    print(f"{label}: same plan {s['text'] + s['json']}/{n} (text {s['text']}, json {s['json']}), "
          f"same actions only {s['actions']}, different {s['DIFF']}   "
          f"median {stats.median(times[k]):.2f}s")
print("cache:", be.stats())
be.close()
