# kv-stitch

**Modular prompt KV-cache stitching for llama.cpp.** Precompute each prompt module once, then join any combination of them at request time, so the model only reads the part of the prompt that is new.

## The problem

Agents and tool-using apps on small local models often build a large prompt from modules: a fixed header, the docs for whichever tools this request needs, then the user's command. With a ~4,000-token prompt on a laptop, **reading the prompt can take longer than writing the answer**. The prompt changes with every request, so normal prefix caching rarely helps.

| Approach | Covers every combination | Exact output | Storage |
|---|---|---|---|
| No cache | ✅ | ✅ | none |
| Prefix / per-combination cache | ❌ only seen combinations | ✅ | grows with combinations |
| **kv-stitch** | ✅ | ≈ (measured) | one entry per module |

## How it works

```
precompute (once)       header            -> cache H
                        header + module X -> keep only X's part   (X saw the header)
                        header + module Y -> keep only Y's part

per request (X, Y)      load H, load X, load Y shifted to sit after X (RoPE shift)
                        decode only the tail (context + command) -> generate
```

- The header never depends on what follows it, so it is stored once.
- Every module was computed after the real header.
- **The one approximation:** a module never saw the modules placed before it in this request. `refresh=N` recomputes the first N tokens of each later module to smooth the joins. The benchmark measures how much this changes the output.
- **Transformer models only** (Llama, Qwen, Granite dense or MoE, ...). Hybrid and recurrent models such as Mamba keep one running state that can't be split into modules.

Related research: Prompt Cache (Gim et al., 2023), CacheBlend (Yao et al., 2024), Block-Attention (2024).

## Use

```python
from kvstitch import Stitcher

st = Stitcher("granite-4.0-micro-Q4_K_M.gguf", n_ctx=8192)
st.set_header(HEADER)                      # once
for name, text in MODULES.items():
    st.add_module(name, text)              # once per module
st.save_dir("cache/")                      # reuse across restarts with st.load_dir

text, t = st.run(["tool_a", "tool_b"], tail="### User Command ###\n...\n### Your JSON Plan ###\n")
print(text, t)                             # t: load / tail / generation seconds
```

## Measure on your own prompts

```bash
pip install llama-cpp-python            # on Apple Silicon this builds with Metal
python bench/compare.py model.gguf cases.json --chat --variants r0,f1500,m1,m2,mall
```

`cases.json` holds the header, the modules, and a list of requests (module names + tail). It runs every case with the **full prompt** and the **stitched prompt** using greedy decoding, then reports how many outputs are identical and the time each took.

## Tests

```bash
pip install pytest sentencepiece transformers torch gguf
python tools/make_tiny_model.py tiny <path-to>/convert_hf_to_gguf.py
KVSTITCH_MODEL=tiny/tiny.gguf pytest -q -s tests
```

These tests check the cache plumbing on a tiny random model:
- header + one module gives **exactly** the same result as the full prompt
- shifting a module forward and back restores the exact result, and a one-way shift really changes it
- computing every module fresh (`fresh` larger than the modules) gives **exactly** the full-prompt output
- saving and reloading the cache from disk gives the same output
- the multi-module difference is printed for information (it is approximate by design)

## Using it in an app: `PlannerBackend`

```python
from kvstitch import PlannerBackend

be = PlannerBackend(
    {"granite3.1-moe:3b": moe_gguf_path, "granite4:micro": micro_gguf_path},
    cache_root="~/Library/Application Support/MyApp/kvcache",
    save_after=2,            # a module combination becomes a cached pattern on its 2nd use
    max_disk_mb=3000,        # least recently used, unpinned caches are evicted past this
    fast_mode=False,         # opt-in stitching for uncached prompts (approximate)
    fast_max_modules=3,
    fallback=lambda model, prompt: call_ollama(model, prompt),   # used if anything fails
)

# pieces in prompt order: (name, text, is_module). Per-user / per-day text goes in the tail.
res = be.plan("granite4:micro", header, pieces, tail)
res.text, res.path   # path: full | partial | miss | fast | fallback

# first launch: build and pin "everything up to the first module" for the core modules
be.warm("granite3.1-moe:3b", header, [[glue, module] for module in core_modules])
```

**Workflows.** List the module sequences people use in a file and warm them on first launch:

```
# workflows.txt — name: modules in prompt order   [@model]
search_note_remind: WEB, NOTES, REMINDER
```
```python
from kvstitch import load_workflows
be.add_workflows(model, load_workflows("workflows.txt"), build=lambda mods: my_builder(mods))
```
Each workflow pins two caches: the full prefix (an exact match is the fastest path) and its start up to
the last module (the same workflow followed by extra modules reuses it, exactly). The app's builder makes
the pieces, so its own conditions (optional sections, filtered rules, lines that depend on other modules)
are part of what is cached. `bench/workflow_check.py` warms a workflow file and checks test commands
against the full prompt.

Paths, cheapest first: **full** (whole prefix cached, exact) → **partial** (longest cached start loaded,
the rest computed: exact) → **fast** (only if enabled and few modules: stitched, approximate) → **miss**
(computed and remembered, exact) → **fallback** (the app's own call). Check a cases file end to end with
`bench/backend_check.py`.

## Results

**Setup:** MacBook with Apple M3, 8 GB RAM, `granite-4.0-micro` (Q4, 3B), llama-cpp-python with Metal, greedy decoding,
the model's chat template (`--chat`). Prompts are a real **JSON tool-calling** prompt: a fixed header, tool-doc
modules picked per request, OS context and rules, then the user command. The model must answer with a JSON plan
of tool calls. "Same actions" means the stitched plan calls the same tools in the same order as the full prompt.

| Modules per prompt | Prompt size | Cases | Prompt read: full → stitched | Total: full → stitched | Same actions |
|---|---|---|---|---|---|
| 2–3 | ~1.5–1.9k tokens | 6 | 7.3 s → **0.65 s (11×)** | 12.7 s → **5.9 s** | **6/6** |
| 4 | ~3.6–4.7k tokens | 3 | 21.2 s → **7.1 s (3×)** | 30.5 s → **15.9 s** | 2/3 |
| 7 | ~4.8–5.8k tokens | 3 | 30.0 s → **7.4 s (4×)** | 41.9 s → **14.3 s** | 1/3 |

In the 4- and 7-module runs the rules section (~1k tokens) is still computed fresh, which is most of the
remaining prompt time.

**What this shows**
- **Speed:** reading the prompt gets 3–11× faster. Total time roughly halves; writing the answer is now the
  biggest cost.
- **Accuracy depends on how many modules are joined.** With 2–3 modules the plans keep the same tool calls
  (differences are small wording changes in parameters). With 7 modules the model starts **dropping steps**.
  Each module never saw the others, and the error grows with the number of joins, which matches the literature.
- **Refresh** (recompute the first N tokens of each join, the EPIC idea) fixed a generic 12-case test
  (4/12 → 11/12 identical) but did not rescue the 7-module prompts at 16–32 tokens.
- **Caching the rules as many small pieces** made accuracy worse: every extra join costs a little.

**Variants on the 4-module prompts** (same 3 cases; `rN` = recompute N tokens per join,
`fN` = the last modules, up to N tokens, computed fresh with the request):

| Variant | Same actions | Prompt read | Total |
|---|---|---|---|
| full prompt | – | 31.2 s | 45.1 s |
| r0 | 2/3 | 9.2 s | 22.9 s |
| f800 | 2/3 | 7.7 s | 20.7 s |
| r64 / r128 | 2/3 | 13.2–13.4 s | 26.9–30.8 s |
| **f1500** | **3/3** | **17.5 s** | **36.2 s** |
| f2500 | 3/3 | 29.8 s | 48.9 s |

(This run was on a busier machine, so absolute times are higher than in the table above; compare rows with
each other.) Larger refresh alone does not recover the hardest case (a 4-step chain where later steps depend
on earlier ones); computing the last ~1,500 module tokens fresh does, at a much smaller speedup.

**2–3 module prompts, 8 cases** (same model and setup). Here the prompt layout matters: after the tool
modules come fixed pieces (a ~600-token terminal-tool section, then OS context), then the rules and the
command. `fN` spends its fresh budget on those fixed pieces first, so a second family of variants picks
fresh pieces **by type**: `m1` / `m2` = the last 1 / 2 tool modules fresh, `mall` = every tool module fresh,
fixed pieces always loaded from the cache.

| Variant | Same actions | Correct plans (manual check) | Prompt read | Total |
|---|---|---|---|---|
| full prompt | – | – | 21.5–22.4 s | 33.9–34.4 s |
| r0 | 5/8 | ~4/8 | 4.9 s | 16.3 s |
| f800 | 5/8 | ~4/8 | 5.8 s | 17.0 s |
| f1500 | 6/8 | 5/8 | 13.3 s | 24.1 s |
| **m1** | 6/8 | 5/8 (+1 partial) | **9.2 s** | **22.7 s** |
| m2 | 7/8 | 5/8 (+2 partial) | 11.6 s | 22.9 s |
| mall | 7/8 | 5/8 (+2 partial) | 13.5 s | 25.3 s |

"Correct" means the plan would do what was asked; "partial" means the right steps with a wrong or
missing parameter.

- Making the **tool modules** fresh (`m*`) fixed cases that every token-count variant failed (a
  note + reminder request where the reminder step was merged into the note).
- One 4-step chain (find latest files → clean names → find the right folder → move) failed in **every**
  variant, including `mall`, where all tool modules are fresh and only the header, terminal section and OS
  context are cached. Each time the plan switched to shell commands. A fixed section that is stitched
  **after** the tool modules was computed without seeing them, and it pulled the model toward that tool.

**Practical conclusion so far:** on real tool-calling prompts, stitching modules saves about a third of the
total time at a measurable accuracy cost, so it is not yet safe as a default without training. The exact
alternative is to move fixed text **in front of** the variable modules, so it becomes a plain prefix cache:
no approximation at all, and runtimes such as Ollama already reuse a matching prefix automatically.

### Exact prefix cache (`PrefixCache`, variant `p`)

When everything before the user's request is the same text (same header, same modules, same rules), its KV
cache is the same every time. `PrefixCache` computes that whole prefix **once, together**, keeps it (memory
LRU and disk), and afterwards decodes only the request. Nothing is stitched, so nothing is approximated.

| Model | Prompts | Prompt read: full → `p` (warm) | Total: full → `p` | Identical output |
|---|---|---|---|---|
| granite3.1-moe 3B | 23 one-module prompts (~0.9–1.6k tokens) | 2.12 s → **0.68 s** | 3.73 s → **2.38 s** | **23/23** |
| granite-4.0-micro | 8 two/three-module prompts (~2.2–3.9k tokens) | 22.0 s → **4.9 s** | 35.7 s → **15.9 s** | **8/8** |

It reproduced the full prompt's output character for character, including its mistakes. Most of the
remaining prompt time with `p` is copying the saved cache into the context, not reading the request.

The cache only helps when a combination repeats, so it is built for that:
- `warm(combos)` builds a core set up front (for example on first launch) and pins it;
- `save_after=N` writes a new combination to disk only once it has been used N times;
- `max_disk_mb` evicts the least recently used, unpinned entries;
- the key covers the model file, the context size and the exact tokens, so a changed module, rule or model
  never reuses a stale cache.

Anything per user or per day (a username, today's date) should sit after the cached part, next to the request.

## Related work

Prompt Cache (Gim et al., 2023), EPIC (recompute the first tokens of each chunk), CacheBlend (selective
recompute of the most affected ~15% of tokens), KVLink and Block-Attention (train the model to read
independently encoded blocks), APE (training-free attention adjustment). Most of this work targets RAG
documents on server GPUs; this project looks at tool-doc modules for a small model on an 8 GB laptop.

## Status

Prototype. The cache plumbing is tested (exactness tests above). On real tool-calling prompts, naive
stitching is fast but loses steps; type-aware fresh pieces (`m1`/`m2`) recover most of the accuracy for
about a third of the time saved. The exact prefix cache keeps the full prompt's output for repeated
combinations (2.2× faster end to end on the larger model). Next: a prefix tree so combinations that share a
start reuse the shared part exactly, cheaper cache loading, and stitch-aware training for stitching itself.
