# ==================== TEXT NORMALIZATION & PARSING UTILITIES ====================

import re
from typing import Optional

from .annotator_config import STOPWORDS


def _strip_llm_artifacts(text: str) -> str:
    """Remove common LLM-generated noise from a string."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = text
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\[\.\.\.\]|\\n|\\\n", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _line_key(page_num: int, y: float, granularity: float = 2.0) -> tuple[int, int]:
    """Quantize a y-position to a stable per-line key.

    Using raw float y0 is too unstable — quantizing avoids both duplicate marks
    on the same line and accidental de-duplication misses.
    """
    try:
        yy = float(y)
    except Exception:
        yy = 0.0
    return int(page_num), int(yy // float(granularity))


def _normalize_text_for_match(text: str) -> str:
    """Lowercase, strip artifacts, and normalise symbols for fuzzy comparison."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = _strip_llm_artifacts(text)
    cleaned = cleaned.replace("×", "x")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _split_comment_arrow(comment: str) -> Optional[tuple[str, str]]:
    """Split 'anchor → feedback' comment into (anchor, feedback) tuple."""
    if not comment or not isinstance(comment, str):
        return None
    if '→' in comment:
        left, right = comment.split('→', 1)
    elif '->' in comment:
        left, right = comment.split('->', 1)
    else:
        return None
    anchor_part = left.strip().strip('"\'')
    feedback_part = right.strip().strip('"\'')
    if not anchor_part or len(anchor_part) < 3:
        return None
    if not feedback_part:
        return None
    return anchor_part, feedback_part


def _build_anchor_variations(text: str) -> list[str]:
    """Build a small list of surface-form variants for an anchor string.

    Handles percent / per cent / % interchangeability that arises from
    LLM paraphrasing vs. the actual PDF symbol.
    """
    if not text or not isinstance(text, str):
        return []

    base = re.sub(r"\s+", " ", text.replace("|", " ")).strip()
    if not base:
        return []

    variants: list[str] = [base]
    norm = _normalize_text_for_match(base)

    if "percent" in norm or "per cent" in norm:
        v = re.sub(r"\bper\s*cent\b", "%", base, flags=re.IGNORECASE)
        v = re.sub(r"\bpercent\b", "%", v, flags=re.IGNORECASE)
        v = re.sub(r"\s*%\s*", "%", v)
        variants.append(v)

    if "%" in base:
        variants.append(re.sub(r"%", " percent", base))
        variants.append(re.sub(r"%", " per cent", base))

    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        vv = re.sub(r"\s+", " ", v).strip()
        if not vv:
            continue
        key = vv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(vv)
    return out


def _tokenize(text: str) -> list[str]:
    """Extract meaningful tokens for overlap-based matching.

    Strips stopwords and very short tokens; keeps alphanumerics plus a few
    accounting symbols (£, $, %, etc.).
    """
    norm = _normalize_text_for_match(text)
    norm = re.sub(r"[^a-z0-9£$%.,/()\- ]+", " ", norm)
    tokens = [t for t in norm.split() if len(t) > 2 and t not in STOPWORDS]
    return tokens


def _build_candidate_fragments(evidence_text: str) -> list[str]:
    """Split an evidence string into ranked anchor fragments for PDF matching.

    Long evidence strings often contain multiple ';'- or newline-separated
    snippets.  We return them sorted longest-first (more distinctive) and
    de-duplicated, up to 4 candidates.  Falls back to the whole string when
    no valid parts are found.

    DRY helper shared by place_score_near_anchor and the holistic annotation loop.
    """
    parts = [p.strip() for p in re.split(r"\n|;|\|", evidence_text) if p and p.strip()]
    parts = [p for p in parts if len(p) >= 6]
    seen: set[str] = set()
    unique: list[str] = []
    for p in sorted(parts, key=len, reverse=True):
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique[:4] if unique else [evidence_text]
