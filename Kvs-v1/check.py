"""Check Kvs-v1 on your own prompts: first run, then every test command vs the plain full prompt.

    python Kvs-v1/check.py granite4:micro --builder my_builder.py --tests tests.txt \
        [--workflows workflows.txt] [--starts WEB,NOTES,RECALL,...] [--cache-root kvs_cache]

The model is found in Ollama's folder by name (or pass --gguf PATH).
my_builder.py defines build(module_names, task) -> (header, pieces, tail).
tests.txt, one per line:   MOD_A, MOD_B | the user's command
--starts builds "everything up to that module" for each listed module (the first-run starts).
"""
import argparse
import importlib.util
import os
import statistics as stats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvs import KVPlanner, load_workflows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--gguf")
ap.add_argument("--builder", required=True)
ap.add_argument("--tests", required=True)
ap.add_argument("--workflows")
ap.add_argument("--starts", default="", help="comma list of core modules to build as starts")
ap.add_argument("--cache-root", default="kvs_cache")
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--max-tokens", type=int, default=400)
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

print("first run (already cached ones are skipped):", flush=True)
t0 = time.perf_counter()
for m in [x.strip() for x in a.starts.split(",") if x.strip()]:
    header, pieces, _ = builder.build([m], "")
    first = next(i for i, p in enumerate(pieces) if len(p) > 2 and p[2])
    kp.build_starts(a.model, header, [pieces[:first + 1]], progress=print)
if a.workflows:
    rep = kp.add_workflows(a.model, load_workflows(a.workflows),
                           lambda mods: builder.build(mods, "")[:2], progress=print)
    if rep["skipped"]:
        print("  skipped:", rep["skipped"])
print(f"first run done in {time.perf_counter() - t0:.1f}s\n", flush=True)

same, full_t, kvs_t, paths = 0, [], [], {}
for i, (mods, cmd) in enumerate(tests, 1):
    header, pieces, tail = builder.build(mods, cmd)
    # the plain full prompt, no cache: the reference
    runs, *_ = kp._runs(header, pieces)
    toks = [t for r in runs for t in r] + kp._t(tail + kp.wrap[1])
    t0 = time.perf_counter()
    kp.eng.clear()
    ref, _ = kp.eng.generate(kp.eng.decode(toks, 0, want_last=True), len(toks), a.max_tokens)
    full_t.append(time.perf_counter() - t0)
    res = kp.plan(a.model, header, pieces, tail)
    ok = res.text.strip() == ref.strip()
    same += ok
    kvs_t.append(res.seconds)
    paths[res.path] = paths.get(res.path, 0) + 1
    print(f"[{i}/{len(tests)}] {'+'.join(mods)}: no-cache {full_t[-1]:.1f}s | {res.path} {res.seconds:.1f}s "
          f"({res.detail.get('cached_pieces', 0)} pieces cached) {'same' if ok else 'DIFF'}  — {cmd[:50]}",
          flush=True)
    if a.show:
        print("  plan:", res.text.strip().replace("\n", " "))

print(f"\nidentical to the full prompt: {same}/{len(tests)}   paths: {paths}")
print(f"median: no-cache {stats.median(full_t):.1f}s -> kvs {stats.median(kvs_t):.1f}s")
print("cache:", kp.stats())
kp.close()
