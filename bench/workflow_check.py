"""Warm a workflows file, then run test commands through PlannerBackend and compare with the full prompt.

    python bench/workflow_check.py MODEL.gguf --model-name granite4:micro \
        --builder my_builder.py --workflows workflows.txt --tests tests.txt [--cache-dir cache]

my_builder.py must define
    build(module_names: list[str], task: str) -> (header, pieces, tail)
using the app's own prompt builder (pieces = [(name, text, is_module), ...]).

tests.txt, one per line:   MOD_A, MOD_B | the user's command
"""
import argparse
import importlib.util
import os
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvstitch import PlannerBackend, load_workflows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--model-name", default="model")
ap.add_argument("--builder", required=True)
ap.add_argument("--workflows", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--cache-dir", default="cache")
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

be = PlannerBackend({a.model_name: a.model}, a.cache_dir, n_ctx=a.n_ctx, chat=not a.no_chat,
                    max_tokens=a.max_tokens, save_after=2)
t0 = time.perf_counter()
rep = be.add_workflows(a.model_name, load_workflows(a.workflows),
                       lambda mods: builder.build(mods, "")[:2])
print(f"workflows warmed: {rep['workflows']}, caches built: {rep['built']} "
      f"in {time.perf_counter() - t0:.1f}s" + (f", skipped: {rep['skipped']}" if rep["skipped"] else ""))

same, full_t, plan_t, paths = 0, [], [], {}
for i, (mods, cmd) in enumerate(tests, 1):
    header, pieces, tail = builder.build(mods, cmd)
    names, _ = be._prepare(header, pieces)
    t0 = time.perf_counter()
    ref, _ = be._st.run_full(names, tail + be._wrap[1], a.max_tokens)
    ft = time.perf_counter() - t0
    res = be.plan(a.model_name, header, pieces, tail)
    ok = res.text.strip() == ref.strip()
    same += ok
    full_t.append(ft)
    plan_t.append(res.seconds)
    paths[res.path] = paths.get(res.path, 0) + 1
    print(f"[{i}/{len(tests)}] {'+'.join(mods)}: full {ft:.2f}s | {res.path} {res.seconds:.2f}s "
          f"{'same' if ok else 'DIFF'}  — {cmd[:60]}")
    if a.show:
        print("  plan:", res.text.strip().replace("\n", " "))

print(f"\nidentical to the full prompt: {same}/{len(tests)}   paths: {paths}")
print(f"median: full {stats.median(full_t):.2f}s -> backend {stats.median(plan_t):.2f}s")
print("cache:", be.stats())
be.close()
