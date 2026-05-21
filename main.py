"""
Hukmo Legal Translation API
FastAPI wrapper around InLegalTrans-En2Indic-1B.

Endpoints:
  GET  /health      — Cloud Run health probe
  GET  /languages   — supported language codes
  POST /translate   — translate a block of legal text (any length)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chunker import chunk_text, reassemble
from translate import load_model, translate_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("hukmo.translation")

# FLORES-200 BCP-47 codes supported by InLegalTrans-En2Indic-1B
SUPPORTED_LANGS: dict[str, str] = {
    "eng_Latn": "English",
    "ben_Beng": "Bengali",
    "hin_Deva": "Hindi",
    "mar_Deva": "Marathi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "mal_Mlym": "Malayalam",
    "pan_Guru": "Punjabi",
    "guj_Gujr": "Gujarati",
    "ory_Orya": "Odia",
}

BATCH_SIZE = int(os.environ.get("TRANSLATE_BATCH_SIZE", "8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading translation model …")
    load_model()
    logger.info("Translation model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Hukmo Legal Translation API",
    description=(
        "Translate Indian legal text from English to 9 Indic languages "
        "using InLegalTrans-En2Indic-1B, fine-tuned on the MILPaC corpus."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────


class TranslateRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Source text to translate (any length; chunked internally).",
    )
    src_lang: str = Field(
        default="eng_Latn",
        description="FLORES-200 source language code.",
    )
    tgt_lang: str = Field(
        ...,
        description="FLORES-200 target language code, e.g. 'hin_Deva'.",
    )


class TranslateResponse(BaseModel):
    translation: str
    chunks_processed: int
    src_lang: str
    tgt_lang: str
    char_count_in: int
    char_count_out: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/languages", tags=["meta"])
def languages():
    return {"supported": SUPPORTED_LANGS}


@app.post("/translate", response_model=TranslateResponse, tags=["translation"])
def translate(req: TranslateRequest):
    if req.src_lang not in SUPPORTED_LANGS:
        raise HTTPException(
            400,
            f"Unsupported src_lang '{req.src_lang}'. "
            f"Supported: {list(SUPPORTED_LANGS)}",
        )
    if req.tgt_lang not in SUPPORTED_LANGS:
        raise HTTPException(
            400,
            f"Unsupported tgt_lang '{req.tgt_lang}'. "
            f"Supported: {list(SUPPORTED_LANGS)}",
        )
    if req.src_lang == req.tgt_lang:
        raise HTTPException(400, "src_lang and tgt_lang must differ.")

    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(400, "Input text is empty after processing.")

    logger.info(
        "Translating %d char(s) → %d chunk(s)  %s→%s",
        len(req.text),
        len(chunks),
        req.src_lang,
        req.tgt_lang,
    )

    translations: list[str] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            results = translate_batch(batch, req.src_lang, req.tgt_lang)
            translations.extend(results)
        except Exception as exc:
            logger.exception(
                "Batch %d/%d failed", i // BATCH_SIZE + 1, -(-len(chunks) // BATCH_SIZE)
            )
            raise HTTPException(500, f"Translation failed: {exc}") from exc

    output = reassemble(translations)
    logger.info(
        "Done — output %d char(s) across %d chunk(s).", len(output), len(chunks)
    )

    return TranslateResponse(
        translation=output,
        chunks_processed=len(chunks),
        src_lang=req.src_lang,
        tgt_lang=req.tgt_lang,
        char_count_in=len(req.text),
        char_count_out=len(output),
    )
