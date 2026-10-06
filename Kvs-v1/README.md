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
)

res = kp.plan(model_from_router, header, pieces, tail)
res.text      # the plan text, exactly what the model writes for the full prompt
res.path      # full | partial | miss | fallback
res.detail    # cached_pieces, prompt_s, gen_tokens, total_s, removed_old
```

| Path | When | Cost |
|---|---|---|
| `full` | this exact prefix was cached | only the tail is read |
| `partial` | a start of it was cached (first-run start, workflow, earlier request) | the rest is read after it |
| `miss` | nothing matches | everything is read |
| `fallback` | an error | the app's own call answers |

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
    3f9a...c1.kv            KV state
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
- `max_disk_mb`: the least recently used **unpinned** entries go first. Pinned ones stay.
- To start over, delete the model's folder.

## Tests

```bash
cd Kvs-v1 && python -m pytest -q tests          # tiny model from tools/make_tiny_model.py
KVS_MODEL=/path/to/model.gguf python -m pytest -q tests
```
