"""What can the structure predictor know from the docs alone? Run it on the app's module texts.

    python Kvs-v1/inspect_docs.py modules_file.py [--show]

modules_file.py: any Python file that holds the module docs as triple-quoted strings named
*MODULE_* (e.g. a copy of the app's prompt builder). Nothing is imported or run from it.

Per action it reports:
    first key   CERTAIN if every documented example starts with the same key (predicted like a
                one-param action), else the most common one (still guessed: wrong guesses are cheap)
    next keys   for each first key, whether the rest of the keys is fixed (one form) or not
    predicted   the share of each documented step the predictor writes by itself, simulated by
                walking through the step as the model would write it (values excluded)
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvs.speculate import StructureDrafter, example_steps, key_orders, string_keys  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("modules_file")
ap.add_argument("--show", action="store_true", help="list every action")
a = ap.parse_args()

src = open(a.modules_file).read()
texts = [m.group(1) for m in re.finditer(r'MODULE_[A-Za-z_]+\s*=\s*"""(.*?)"""', src, re.S)]
steps = example_steps(texts)
ko, sk = key_orders(texts), string_keys(texts)
print(f"{len(texts)} modules, {len(steps)} documented steps, {len(ko)} actions\n")


def simulate(step) -> tuple[int, int]:
    """Characters of a one-step plan the predictor writes itself / total."""
    truth = json.dumps({"plan": [step]}, indent=2)
    d = StructureDrafter(None, [step["action"]], ko, str_keys=sk)
    pos = got = 0
    while pos < len(truth):
        g = d.guess(truth[:pos])
        m = 0
        while m < len(g) and pos + m < len(truth) and g[m] == truth[pos + m]:
            m += 1
        if m:
            got += m
            pos += m
        else:
            pos += 1
    return got, len(truth)


per = defaultdict(lambda: [0, 0])
for st in steps:
    g, t = simulate(st)
    per[st["action"]][0] += g
    per[st["action"]][1] += t

certain_first, fixed_rest = [], []
rows = []
for act, forms in sorted(ko.items()):
    firsts = {f[0] for f in forms if f}
    certain = len(firsts) <= 1
    if certain and firsts:
        certain_first.append(act)
    by_first = defaultdict(set)
    for f in forms:
        if f:
            by_first[f[0]].add(tuple(f[1:]))
    rest_fixed = all(len(v) == 1 for v in by_first.values())
    if rest_fixed:
        fixed_rest.append(act)
    g, t = per[act]
    rows.append((act, len(forms), certain, rest_fixed, g / t if t else 0))

all_g = sum(v[0] for v in per.values())
all_t = sum(v[1] for v in per.values())
print(f"first key CERTAIN (same in every example): {len(certain_first)}/{len(ko)}")
print(f"rest of the keys fixed once the first is known: {len(fixed_rest)}/{len(ko)}")
print(f"predicted by structure alone, over all documented steps: {all_g}/{all_t} characters "
      f"({all_g / all_t:.0%})\n")
if a.show:
    print(f"{'action':26} forms  first-key  rest-fixed  predicted")
    for act, nf, c, r, p in sorted(rows, key=lambda r: r[4]):
        print(f"{act:26} {nf:5}  {'certain' if c else 'guessed':9}  {'yes' if r else 'no':10}  {p:.0%}")
