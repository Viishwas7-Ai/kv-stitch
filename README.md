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
python bench/compare.py model.gguf cases.json --refresh 0,16,32
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

**Practical rule so far:** stitch when a request needs few modules, use the full prompt for the largest ones.
Larger refresh values and computing the last modules fresh (`--variants r128,f1500,...`) are being measured next.

## Related work

Prompt Cache (Gim et al., 2023), EPIC (recompute the first tokens of each chunk), CacheBlend (selective
recompute of the most affected ~15% of tokens), KVLink and Block-Attention (train the model to read
independently encoded blocks), APE (training-free attention adjustment). Most of this work targets RAG
documents on server GPUs; this project looks at tool-doc modules for a small model on an 8 GB laptop.

## Status

Prototype. The cache plumbing is tested (exactness tests above). Accuracy on real prompts is good for few
modules and degrades with many. Next: adaptive cutoff, fresh-back and larger refresh, stitch-aware training.
