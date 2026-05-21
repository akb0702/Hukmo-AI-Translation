"""
Model loader and inference for InLegalTrans-En2Indic-1B.

Lazy-loads on first request (or eagerly on startup via load_model()).
Thread-safe for single-worker deployments (Cloud Run containerConcurrency=1).
"""

import logging
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransToolkit import IndicProcessor
except ImportError as exc:
    raise ImportError(
        "IndicTransToolkit is not installed. "
        "Run: pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    ) from exc

logger = logging.getLogger("translate")

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSeq2SeqLM | None = None
_processor: IndicProcessor | None = None


def load_model() -> None:
    """Load tokeniser, model, and IndicProcessor into module-level singletons."""
    global _tokenizer, _model, _processor

    if _model is not None:
        return

    logger.info("Loading tokeniser from %s/tokenizer …", MODEL_DIR)
    _tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_DIR}/tokenizer",
        trust_remote_code=True,
        local_files_only=True,
    )

    logger.info("Loading model from %s/weights on %s …", MODEL_DIR, DEVICE)
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        f"{MODEL_DIR}/weights",
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(DEVICE)
    _model.eval()

    _processor = IndicProcessor(inference=True)
    logger.info("Model ready on %s (dtype=%s).", DEVICE, dtype)


def translate_batch(sentences: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    """
    Translate a batch of pre-chunked sentences.

    Args:
        sentences: List of source strings (each ≤ MAX_CHARS characters).
        src_lang:  FLORES-200 BCP-47 source language code, e.g. "eng_Latn".
        tgt_lang:  FLORES-200 BCP-47 target language code, e.g. "hin_Deva".

    Returns:
        List of translated strings in the same order as *sentences*.
    """
    if not sentences:
        return []

    load_model()

    preprocessed = _processor.preprocess_batch(
        sentences, src_lang=src_lang, tgt_lang=tgt_lang
    )

    inputs = _tokenizer(
        preprocessed,
        max_length=256,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated = _model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            num_return_sequences=1,
            early_stopping=False,
            use_cache=True,
        )

    with _tokenizer.as_target_tokenizer():
        decoded = _tokenizer.batch_decode(
            generated.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    return _processor.postprocess_batch(decoded, lang=tgt_lang)
