"""Full prompt vs stitched prompt: same outputs? how much faster?

    python bench/compare.py model.gguf cases.json [--refresh 0,16] [--max-tokens 400]

cases.json:
{
  "header": "fixed text every prompt starts with",
  "modules": {"NAME": "module text", ...},
  "cases": [{"modules": ["NAME", ...], "tail": "context + command + answer prefix"}, ...]
}
Greedy decoding, so any difference in output is caused by the cache, not sampling.
"""
import argparse
import json
import statistics as stats
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvstitch import Stitcher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("cases")
ap.add_argument("--refresh", default="0", help="comma list of tokens to recompute per join")
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--gpu-layers", type=int, default=-1)
ap.add_argument("--out", default="compare_results.json")
ap.add_argument("--show", action="store_true", help="print every output")
a = ap.parse_args()

spec = json.load(open(a.cases))
st = Stitcher(a.model, n_ctx=a.n_ctx, n_gpu_layers=a.gpu_layers)
t0 = time.perf_counter()
st.set_header(spec["header"])
for name, text in spec["modules"].items():
    st.add_module(name, text)
print(f"precomputed header + {len(spec['modules'])} modules in {time.perf_counter() - t0:.1f}s")

refreshes = [int(x) for x in a.refresh.split(",")]
c0 = spec["cases"][0]                     # warm-up so the first timed run isn't paying GPU setup
st.run_full(c0["modules"], c0["tail"], 4)
st.run(c0["modules"], c0["tail"], 4)
rows, same = [], {r: 0 for r in refreshes}
full_t, stitch_t = [], {r: [] for r in refreshes}
full_p, stitch_p = [], {r: [] for r in refreshes}
for i, c in enumerate(spec["cases"]):
    ref, tf = st.run_full(c["modules"], c["tail"], a.max_tokens)
    full_t.append(tf.tail_s + tf.gen_s)
    full_p.append(tf.tail_s)
    row = {"case": i, "modules": c["modules"], "full": ref,
           "full_prompt_s": round(tf.tail_s, 3), "gen_s": round(tf.gen_s, 3)}
    for r in refreshes:
        out, ts = st.run(c["modules"], c["tail"], a.max_tokens, refresh=r)
        stitch_t[r].append(ts.load_s + ts.tail_s + ts.gen_s)
        stitch_p[r].append(ts.load_s + ts.tail_s)
        same[r] += out.strip() == ref.strip()
        row[f"stitched_r{r}"] = out
        row[f"stitched_r{r}_prompt_s"] = round(ts.load_s + ts.tail_s, 3)
    rows.append(row)
    print(f"[{i + 1}/{len(spec['cases'])}] {'+'.join(c['modules']) or '-'} ({tf.tail_tokens} prompt tokens): "
          f"full prompt {tf.tail_s:.2f}s | "
          + " ".join(f"r{r}={'same' if row[f'stitched_r{r}'].strip() == ref.strip() else 'DIFF'}"
                     f" {row[f'stitched_r{r}_prompt_s']:.2f}s" for r in refreshes))
    if a.show:
        print("  full:", ref.strip().replace("\n", " ")[:400])
        for r in refreshes:
            print(f"  r{r}: ", row[f"stitched_r{r}"].strip().replace("\n", " ")[:400])

n = len(rows)
print(f"\n{n} cases | full: prompt {stats.median(full_p):.2f}s, total {stats.median(full_t):.2f}s (medians)")
for r in refreshes:
    print(f"refresh={r:<3} identical outputs: {same[r]}/{n} ({100 * same[r] / n:.1f}%) | "
          f"prompt {stats.median(stitch_p[r]):.2f}s, total {stats.median(stitch_t[r]):.2f}s")
json.dump(rows, open(a.out, "w"), indent=1)
print("details ->", a.out)
st.close()
