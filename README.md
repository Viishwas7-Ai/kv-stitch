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

text, t = st.run(["CREATE", "NOTES"], tail="### User Command ###\n...\n### Your JSON Plan ###\n")
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
- shifting a module by d positions gives **exactly** the same result as computing it d positions later
- saving and reloading the cache from disk gives the same output
- the multi-module difference is printed for information (it is approximate by design)

## Status

Early prototype. The plumbing is tested; accuracy and speed on real models still need to be measured with `bench/compare.py`.
