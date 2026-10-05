"""Run a cases file through PlannerBackend the way an app would, and check it against the full prompt.

    python bench/backend_check.py MODEL.gguf cases.json --model-name granite3.1-moe:3b \
        [--cache-dir cache] [--warm] [--fast] [--rounds 2]

Each case is planned `--rounds` times (first = cold, later = warm). The first output of
every case is also computed with the plain full prompt and compared.
--warm  first builds and pins "everything up to the first module" for every case, the way
        an app would on first launch.
"""
import argparse
import json
import os
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvstitch import PlannerBackend  # noqa: E402

FIXED = ("GLUE", "TERMINAL", "OS", "TEXT_RULE", "RULE")

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("cases")
ap.add_argument("--model-name", default="model")
ap.add_argument("--cache-dir", default="cache")
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--warm", action="store_true")
ap.add_argument("--fast", action="store_true", help="allow fast (stitched) mode for uncached prompts")
ap.add_argument("--no-chat", action="store_true")
ap.add_argument("--show", action="store_true")
a = ap.parse_args()

spec = json.load(open(a.cases))
mods = spec["modules"]


def pieces(case):
    out = []
    for n in case["modules"]:
        base = n.split("#")[0]
        out.append((base, mods[n], not base.startswith(FIXED)))
    return out


be = PlannerBackend({a.model_name: a.model}, a.cache_dir, n_ctx=a.n_ctx, chat=not a.no_chat,
                    max_tokens=a.max_tokens, save_after=1, fast_mode=a.fast)
be._load(a.model_name)

if a.warm:
    starts = []
    for c in spec["cases"]:
        p = pieces(c)
        first = next((i for i, x in enumerate(p) if x[2]), None)
        if first is not None:
            starts.append(p[: first + 1])
    t0 = time.perf_counter()
    built = be.warm(a.model_name, spec["header"], starts)
    print(f"warm-up: built {built} start(s) in {time.perf_counter() - t0:.1f}s")

same, rows, times = 0, [], {}
for i, c in enumerate(spec["cases"]):
    p = pieces(c)
    names, _ = be._prepare(spec["header"], p)
    t0 = time.perf_counter()
    ref, _ = be._st.run_full(names, c["tail"] + be._wrap[1], a.max_tokens)
    full_s = time.perf_counter() - t0
    line = [f"[{i + 1}/{len(spec['cases'])}] full {full_s:.2f}s"]
    first_out = None
    for r in range(a.rounds):
        res = be.plan(a.model_name, spec["header"], p, c["tail"])
        times.setdefault(r, []).append(res.seconds)
        first_out = first_out if first_out is not None else res.text
        ok = res.text.strip() == ref.strip()
        line.append(f"round{r + 1}={res.path} {res.seconds:.2f}s {'same' if ok else 'DIFF'}")
        if r == 0:
            same += ok
    print(" | ".join(line))
    if a.show:
        print("  full:", ref.strip().replace("\n", " "))
        print("  plan:", first_out.strip().replace("\n", " "))

n = len(spec["cases"])
print(f"\nfirst-round output identical to the full prompt: {same}/{n}")
for r, ts in times.items():
    print(f"round {r + 1}: median {stats.median(ts):.2f}s")
print("cache:", be.stats())
be.close()
