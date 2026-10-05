"""Build a tiny random-weight Llama GGUF for offline tests of the cache mechanics.

It generates nonsense, but it is deterministic, which is all the exactness
tests need. Real accuracy has to be measured on a real model (see README).

    python tools/make_tiny_model.py out_dir
"""
import os
import subprocess
import sys

import sentencepiece as spm
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

out = sys.argv[1] if len(sys.argv) > 1 else "tiny"
hf = os.path.join(out, "hf")
os.makedirs(hf, exist_ok=True)

# tokenizer: train a small sentencepiece model on some local text
corpus = os.path.join(out, "corpus.txt")
words = ("create folder file write read open move copy zip notes reminder plan json action "
         "params path desktop user command header module the a to and of in with").split()
with open(corpus, "w") as f:
    for i in range(4000):
        f.write(" ".join(words[(i * k) % len(words)] for k in range(1, 12)) + "\n")
spm.SentencePieceTrainer.train(input=corpus, model_prefix=os.path.join(hf, "tokenizer"),
                               vocab_size=400, model_type="bpe", byte_fallback=True,
                               bos_id=1, eos_id=2, unk_id=0, pad_id=-1)
LlamaTokenizer(vocab_file=os.path.join(hf, "tokenizer.model")).save_pretrained(hf)

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=400, hidden_size=128, intermediate_size=256,
                  num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=4096, bos_token_id=1, eos_token_id=2)
model = LlamaForCausalLM(cfg)
with torch.no_grad():          # sharper attention than the default init
    for n, p in model.named_parameters():
        if "q_proj" in n or "k_proj" in n:
            p.mul_(8)
model.save_pretrained(hf, safe_serialization=True)

conv = sys.argv[2] if len(sys.argv) > 2 else "convert_hf_to_gguf.py"
subprocess.run([sys.executable, conv, hf, "--outtype", "f32",
                "--outfile", os.path.join(out, "tiny.gguf")], check=True)
print("wrote", os.path.join(out, "tiny.gguf"))
