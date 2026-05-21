"""
Legal text chunker for InLegalTrans-En2Indic-1B.

The model hard-caps input at max_source_positions=256 tokens.
Legal sentences are dense (~4 chars/token on average for English legal text),
so we target MAX_CHARS=700 as a safe empirical proxy for ~175 tokens,
leaving headroom for the BOS/EOS/lang-tag special tokens.

Split priority:
  1. Sentence boundaries  (. ? ! followed by whitespace)
  2. Legal clause markers (;  or comma before "provided/whereas/subject/that")
  3. Hard word boundary   (space nearest to MAX_CHARS)
"""

import re

MAX_CHARS = 700

_SENTENCE_RE = re.compile(r"(?<=[.?!])\s+")
_CLAUSE_RE = re.compile(
    r"(?<=;)\s+"
    r"|(?<=,)\s+(?=(?:provided|whereas|subject|notwithstanding|that|and|but|or)\b)",
    re.IGNORECASE,
)


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split *text* into a list of chunks each ≤ *max_chars* characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_RE.split(text)
    return _merge(sentences, max_chars)


def _merge(segments: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        if len(seg) > max_chars:
            # Segment still too long — try clause-level split
            sub_segs = _CLAUSE_RE.split(seg)
            if len(sub_segs) > 1:
                for sub in _merge(sub_segs, max_chars):
                    if current and len(current) + 1 + len(sub) <= max_chars:
                        current += " " + sub
                    else:
                        if current:
                            chunks.append(current)
                        current = sub
            else:
                # No clause boundary found — hard-split at word boundary
                for part in _hard_split(seg, max_chars):
                    if current and len(current) + 1 + len(part) <= max_chars:
                        current += " " + part
                    else:
                        if current:
                            chunks.append(current)
                        current = part
        elif current and len(current) + 1 + len(seg) <= max_chars:
            current += " " + seg
        else:
            if current:
                chunks.append(current)
            current = seg

    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split at the last space before max_chars; fall back to hard character cut."""
    parts: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut == -1:
            cut = max_chars
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def reassemble(translations: list[str]) -> str:
    """Join translated chunks back into a single string."""
    return " ".join(t.strip() for t in translations if t.strip())
