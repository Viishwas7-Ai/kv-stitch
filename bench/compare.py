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

def parse_json(text):
    """First JSON object in the output, or None."""
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
    """'text' identical > 'json' same JSON (spacing differs) > 'actions' same steps > 'DIFF'."""
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
ap.add_argument("cases")
ap.add_argument("--refresh", default="0", help="comma list of tokens to recompute per join")
ap.add_argument("--variants", default=None,
                help="comma list like r0,r64,f1500,m1,m2,mall (r=refresh tokens per join, "
                     "f=tokens of the last pieces computed fresh, mN/mall=the last N / all "
                     "pieces whose name is not in --fixed computed fresh); overrides --refresh")
ap.add_argument("--fixed", default="GLUE,TERMINAL,OS,TEXT_RULE,RULE",
                help="piece-name prefixes that are fixed text, always loaded from the cache by mN/mall")
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--gpu-layers", type=int, default=-1)
ap.add_argument("--out", default="compare_results.json")
ap.add_argument("--show", action="store_true", help="print every output")
ap.add_argument("--chat", action="store_true",
                help="wrap header/tail in the model's chat template (needed for chat models like granite)")
ap.add_argument("--prefix", help="with --chat: override the text before the prompt")
ap.add_argument("--suffix", help="with --chat: override the text after the prompt")
a = ap.parse_args()

spec = json.load(open(a.cases))
st = Stitcher(a.model, n_ctx=a.n_ctx, n_gpu_layers=a.gpu_layers)
if a.chat:
    try:
        pre, suf = st.chat_wrap()
    except Exception as e:                      # some templates use features the renderer lacks
        print(f"chat template could not be rendered ({e}); using --prefix/--suffix")
        pre, suf = "", ""
    pre, suf = a.prefix if a.prefix is not None else pre, a.suffix if a.suffix is not None else suf
    print(f"chat template: prefix {pre!r} | suffix {suf!r}")
    spec["header"] = pre + spec["header"]
    for c in spec["cases"]:
        c["tail"] = c["tail"] + suf
t0 = time.perf_counter()
st.set_header(spec["header"])
for name, text in spec["modules"].items():
    st.add_module(name, text)
print(f"precomputed header + {len(spec['modules'])} modules in {time.perf_counter() - t0:.1f}s")

import re as _re
def _variant(v):
    mm = _re.fullmatch(r"m(\d+|all)", v)
    if mm:
        return v, 0, ("all" if mm.group(1) == "all" else int(mm.group(1)))
    m = _re.fullmatch(r"(?:f(\d+))?(?:r(\d+))?", v)
    if not m or not v:
        raise SystemExit(f"bad variant {v!r}")
    return v, int(m.group(2) or 0), int(m.group(1) or 0)

_FIXED = tuple(x.strip() for x in a.fixed.split(",") if x.strip())


def _action_idx(names, want):
    """Positions of the last `want` (or all) pieces that are not fixed text."""
    idx = [i for i, n in enumerate(names) if not n.split("#")[0].startswith(_FIXED)]
    return set(idx if want == "all" else idx[-want:] if want else [])
specs = [_variant(v) for v in a.variants.split(",")] if a.variants else \
        [(f"r{int(x)}", int(x), 0) for x in a.refresh.split(",")]
refreshes = [v for v, _, _ in specs]
_cfg = {v: (r, f) for v, r, f in specs}
c0 = spec["cases"][0]                     # warm-up so the first timed run isn't paying GPU setup
st.run_full(c0["modules"], c0["tail"], 4)
st.run(c0["modules"], c0["tail"], 4)
rows = []
levels = ("text", "json", "actions")
same = {r: {k: 0 for k in levels} for r in refreshes}
full_t, stitch_t = [], {r: [] for r in refreshes}
full_p, stitch_p = [], {r: [] for r in refreshes}
for i, c in enumerate(spec["cases"]):
    ref, tf = st.run_full(c["modules"], c["tail"], a.max_tokens)
    full_t.append(tf.tail_s + tf.gen_s)
    full_p.append(tf.tail_s)
    row = {"case": i, "modules": c["modules"], "full": ref,
           "full_prompt_s": round(tf.tail_s, 3), "gen_s": round(tf.gen_s, 3)}
    for r in refreshes:
        _r, _f = _cfg[r]
        if r.startswith("m"):
            out, ts = st.run(c["modules"], c["tail"], a.max_tokens,
                             fresh_idx=_action_idx(c["modules"], _f))
        else:
            out, ts = st.run(c["modules"], c["tail"], a.max_tokens, refresh=_r, fresh=_f)
        stitch_t[r].append(ts.load_s + ts.tail_s + ts.gen_s)
        stitch_p[r].append(ts.load_s + ts.tail_s)
        v = verdict(out, ref)
        for k in levels[levels.index(v):] if v in levels else ():
            same[r][k] += 1
        row[f"stitched_{r}_verdict"] = v
        row[f"stitched_{r}"] = out
        row[f"stitched_{r}_prompt_s"] = round(ts.load_s + ts.tail_s, 3)
    rows.append(row)
    print(f"[{i + 1}/{len(spec['cases'])}] {'+'.join(c['modules']) or '-'} ({tf.tail_tokens} prompt tokens): "
          f"full prompt {tf.tail_s:.2f}s | "
          + " ".join(f"{r}={row[f'stitched_{r}_verdict']}"
                     f" {row[f'stitched_{r}_prompt_s']:.2f}s" for r in refreshes))
    if a.show:
        print("  full:", ref.strip().replace("\n", " "))
        for r in refreshes:
            print(f"  {r}: ", row[f"stitched_{r}"].strip().replace("\n", " "))

n = len(rows)
empty = sum(1 for r in rows if not r["full"].strip())
if empty:
    print(f"\nWARNING: {empty}/{n} full-prompt outputs are EMPTY, so 'same' means nothing for them. "
          "For chat models (granite, qwen, ...) add --chat.")
print(f"\n{n} cases | full: prompt {stats.median(full_p):.2f}s, total {stats.median(full_t):.2f}s (medians)")
for r in refreshes:
    sr = same[r]
    print(f"{r:<10} same text {sr['text']}/{n} | same JSON {sr['json']}/{n} | "
          f"same actions {sr['actions']}/{n} | "
          f"prompt {stats.median(stitch_p[r]):.2f}s, total {stats.median(stitch_t[r]):.2f}s")
json.dump(rows, open(a.out, "w"), indent=1)
print("details ->", a.out)
st.close()
