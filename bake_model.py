"""
Run during Docker build to download and serialise:
  - InLegalTrans model weights  (law-ai/InLegalTrans-En2Indic-1B)
  - IndicTrans2 tokeniser       (ai4bharat/indictrans2-en-indic-1B)
  - Custom arch code            (pulled via trust_remote_code into HF cache)

The runtime container sets TRANSFORMERS_OFFLINE=1 and HF_DATASETS_OFFLINE=1
so it never reaches out to HuggingFace.
"""
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "/model"
HF_MODEL = "law-ai/InLegalTrans-En2Indic-1B"
HF_TOKENIZER = "ai4bharat/indictrans2-en-indic-1B"

os.makedirs(f"{MODEL_DIR}/tokenizer", exist_ok=True)
os.makedirs(f"{MODEL_DIR}/weights", exist_ok=True)

print(f"[bake] Downloading tokeniser from {HF_TOKENIZER} ...")
tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER, trust_remote_code=True)
tokenizer.save_pretrained(f"{MODEL_DIR}/tokenizer")
print("[bake] Tokeniser saved.")

print(f"[bake] Downloading model from {HF_MODEL} ...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    HF_MODEL,
    trust_remote_code=True,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
model.save_pretrained(f"{MODEL_DIR}/weights")
print(f"[bake] Model saved. Param count: {sum(p.numel() for p in model.parameters()):,}")
print("[bake] Done — runtime will use local_files_only=True.")
