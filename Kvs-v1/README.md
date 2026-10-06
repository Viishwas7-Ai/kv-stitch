# Kvs-v1

The part of kv-stitch an app actually uses: an **exact** KV prefix cache for a planning prompt,
on llama.cpp. Every path gives the same plan as reading the whole prompt from scratch; nothing
is stitched or approximated.

```
Kvs-v1/
  kvs/engine.py      one model in llama.cpp: tokenize, read, save/load KV, generate
  kvs/cache.py       the cache folder: one entry per exact prompt start
  kvs/planner.py     KVPlanner: plan(), first-run builds, workflows, fallback
  kvs/workflows.py   workflows file:  name: MOD_A, MOD_B [@model]
  tests/test_kvs.py  every cached path checked against the plain full prompt
```

Needs `llama-cpp-python`. Copy the `kvs` folder into the app, or add `Kvs-v1` to `sys.path`.

## How a request runs

The router picks the model; the app's builder makes the prompt in pieces:

| Part | What | Cached? |
|---|---|---|
| `header` | the base prompt every request starts with | yes |
| `pieces` | `[(name, text, is_module), ...]`: modules and fixed parts (rules, glue, OS text), in prompt order | yes |
| `tail` | per request: user/date line, context, the command | no, always read |

```python
from kvs import KVPlanner

kp = KVPlanner(
    models={},                                   # empty: models are found in Ollama's folder by name
    cache_root="~/Library/Application Support/MyApp/kvs",
    max_disk_mb=3000,
    fallback=lambda model, prompt: ollama_call(model, prompt),   # used if anything fails
    exact_only={"DELETE", "RENAME", "FILES"},    # these modules never use speed mode
)

res = kp.plan(model_from_router, header, pieces, tail)
res.text      # the plan text, exactly what the model writes for the full prompt
res.path      # full | partial | miss | fast | fallback
res.detail    # cached_pieces, prompt_s, gen_tokens, total_s, removed_old
```

| Path | When | Cost |
|---|---|---|
| `full` | this exact prefix was cached | only the tail is read |
| `partial` | a start of it was cached (first-run start, workflow, earlier request) | the rest is read after it |
| `miss` | nothing matches | everything is read |
| `fallback` | an error | the app's own call answers |

**Speed mode** (`kp.plan(..., fast=True)`, off unless asked): the longest cached start (at least
base + first module) is loaded exactly, and **every module after it is stitched in from its own
first-run start** (base + that module), so it needs no extra cache files. Fixed parts after the
modules and the tail are read. Faster, but **not always the same plan**: a stitched module never
saw the modules before it. If the whole prefix is cached, or there is only one module, the exact
path is used.

After a `partial` or `miss` the whole prefix is saved, and so is the start up to the first and
the second module (`checkpoints=2`), so the next request that shares that start reuses it.

Anything that changes per user or per day must be in the `tail`, after the cached part; one
changing character in `header` or `pieces` means a different prefix.

## First run, and every launch after

```python
# every core module as a start: base pieces + that module, built once and pinned
kp.build_starts(model, header, [base_pieces + [m] for m in core_module_pieces], progress=print)

# workflows people use: whole prefix + start up to its last module, pinned
from kvs import load_workflows
kp.add_workflows(model, load_workflows("workflows.txt"),
                 build=lambda mods: my_builder(mods), progress=print)
```

Call both on every launch: what is already cached is skipped in a moment, and only what changed
is built again. Build for each model the router uses (one model is in memory at a time).

## The cache folder

```
<cache_root>/
  granite4_micro/
    3f9a...c1.kv            KV state (a first-run start holds only its module's cells)
    3f9a...c1.prompt.txt    the exact text it holds — open it to see what is cached
    index.json              slot, uses, last use, pinned, tokens, created
  granite3.1-moe_3b/
    ...
```

- **key** = model file + context size + exact tokens. Edit a module, a rule, the base prompt,
  or update the model, and the old cache simply stops matching.
- **slot** = the piece names (`HEADER|GLUE|WEB|NOTES`). When the first run builds a slot whose
  text changed, the old version is **deleted**. A normal request whose text differs slightly
  (an optional line, a filtered rule) never deletes a pinned version; it replaces older
  unpinned versions only (`versions_per_slot=1`).
- **Starts are stored small.** The base (header + everything before the module) is stored once;
  each first-run start keeps only its module's cells and the key of that base. Loading puts the
  base back and adds the module's cells where they were: the same KV, byte for byte. If a base
  is deleted, its starts go with it.
- `max_disk_mb`: the least recently used **unpinned** entries go first. Pinned ones stay.
- To start over, delete the model's folder.

## v1 benchmark

granite-4.0-micro (3B) on an 8 GB M3 MacBook Air, other apps open, greedy decoding.
20 test commands, 2–3 tool modules each, 21 different modules, typos included. Every
first-run start built (base + each module); no workflows warmed, so every request was a
combination seen for the first time (the hardest case for the cache).

| Mode | Median per command | Same plan as the full prompt |
|---|---|---|
| no cache (whole prompt read) | 36.9 s | reference |
| **exact** (`partial`: start loaded, rest read) | 33.4 s | **20/20, character for character** |
| speed (`fast`: start + stitched modules) | 27.5 s | 9/20 identical |

Speed mode's 11 different plans, judged by whether they would run:
- 6 harmless: an extra optional param, a path made explicit, `.png` vs `png`, wording in a reminder;
- 2 missing an optional param a plan validator can fill;
- 3 wrong: an "open" step aimed at the wrong target, a dropped read-the-screen step, and a
  garbled target on a **delete** step.

So roughly 15–17 of 20 would work, about 25% faster than exact. Use speed mode only where a
wrong step is cheap; requests with destructive actions (delete, rename, move) should always
take the exact path: list them in `exact_only` and speed mode is skipped for them.

Exact mode on a repeated combination (a warmed workflow, `full` path) is the large win:
44.8 s → 19.2 s median on 10 workflows, identical plans (see the main README).

Most of the remaining time is the model writing the plan (~150 tokens of JSON), which a prompt
cache cannot shorten. Asking for compact JSON, or forcing it at decoding time, broke plans on
this model, so the output format is left as the model prefers it.

## Check it on your own prompts

```bash
python Kvs-v1/check.py granite4:micro --builder my_builder.py --tests tests.txt \
    --workflows workflows.txt --starts WEB,NOTES,RECALL --cache-root kvs_cache
```
Runs the first run (starts + workflows), then every test command twice: the plain full prompt
(no cache) and `KVPlanner.plan`, and reports whether the plans are identical, the path taken
and the times. Add `--fast` to also run speed mode on every test. Run it a second time to see the first run skip everything.

## Tests

```bash
cd Kvs-v1 && python -m pytest -q tests          # tiny model from tools/make_tiny_model.py
KVS_MODEL=/path/to/model.gguf python -m pytest -q tests
```
