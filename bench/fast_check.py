"""Fast mode check: exact start + stitched modules, against the full prompt.

    python bench/fast_check.py MODEL.gguf --builder my_builder.py --tests tests.txt \
        [--refresh 0,16,32,64] [--model-name granite3.1-moe:3b] [--show]

For every test the full prompt is computed from scratch (the reference). Then, for every
--refresh value, the same request goes through fast mode:

    header + pieces up to the FIRST module   loaded from the exact cache (built once)
    the other modules                        stitched from their own caches, the first
                                             N tokens of each re-read in place
    fixed parts after them, the request      computed fresh

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
        return [step.get("action") for step in obj.get("plan", [])]
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
ap.add_argument("--refresh", default="0,16,32,64", help="comma list: tokens re-read per stitched module")
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
print(f"{len(tests)} tests x (1 full + {len(refresh)} refresh values x 2 runs). "
      "Each line prints when its test is done.", flush=True)

score = {r: {"text": 0, "json": 0, "actions": 0, "DIFF": 0} for r in refresh}
full_t, fast_t = [], {r: [] for r in refresh}
for i, (mods, cmd) in enumerate(tests, 1):
    print(f"[{i}/{len(tests)}] {'+'.join(mods)} ...", end="", flush=True)
    header, pieces, tail = builder.build(mods, cmd)
    names, _ = be._prepare(header, pieces)
    t0 = time.perf_counter()
    ref, _ = be._st.run_full(names, tail + be._wrap[1], a.max_tokens)
    full_t.append(time.perf_counter() - t0)
    print(" full", end="", flush=True)
    line = [f"[{i}/{len(tests)}] {'+'.join(mods)}: full {full_t[-1]:.2f}s"]
    if a.show:
        print("  full:", ref.strip().replace("\n", " "))
    for r in refresh:
        be.fast_refresh = r
        be.plan(a.model_name, header, pieces, tail)         # 1st: builds the start + module caches
        res = be.plan(a.model_name, header, pieces, tail)   # 2nd: what a user sees once warm
        v = verdict(res.text, ref)
        score[r][v] += 1
        fast_t[r].append(res.seconds)
        line.append(f"r{r} {res.path} {res.seconds:.2f}s {v}")
        print(f" r{r}", end="", flush=True)
        if a.show:
            print(f"  r{r}:", res.text.strip().replace("\n", " "))
    print("\n  " + " | ".join(line) + f"  — {cmd[:50]}", flush=True)

print(f"\n{len(tests)} tests, median full {stats.median(full_t):.2f}s")
for r in refresh:
    s = score[r]
    ok = s["text"] + s["json"]
    print(f"refresh {r:>4}: same plan {ok}/{len(tests)} (text {s['text']}, json {s['json']}), "
          f"same actions only {s['actions']}, different {s['DIFF']}   "
          f"median {stats.median(fast_t[r]):.2f}s")
print("cache:", be.stats())
be.close()
