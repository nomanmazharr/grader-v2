import fitz
import os
import re
import json
from datetime import datetime
from typing import List, Optional, Tuple, Iterable
from bson import ObjectId
from logging_config import logger
from database.mongodb import get_collection


# ==================== CONFIG (unchanged) ====================
CONFIG = {
    'main_score_offset_x': 20,
    'main_score_offset_y': -12,  # More upward for heading
    'main_score_fontsize': 16,   # Larger for visibility
    'main_score_color': (1, 0, 0),

    'criterion_score_offset_x': -18,
    'criterion_score_offset_y': 3,
    'criterion_score_fontsize': 11,
    'criterion_score_color': (1, 0, 0),

    'underline_color': (1, 0, 0),
    'comment_color': (1, 0, 0),
    'comment_offset': 35,
    'y_tolerance': 6,      # Increased for line matching
    'search_tolerance': 0.8,  # Levenshtein ratio for fuzzy
    'max_anchor_words': 6, # Longer anchors
}


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "by", "as", "is", "are",
    "was", "were", "be", "been", "being", "with", "that", "this", "it", "its", "from", "into",
    "will", "would", "should", "can", "could", "may", "might", "there", "therefore",
}


def _normalize_text_for_match(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    # Normalize common OCR/LLM artifacts and symbols
    cleaned = text
    cleaned = cleaned.replace("×", "x")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\[\.\.\.\]|\\n|\\\n", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _build_anchor_variations(text: str) -> list[str]:

    if not text or not isinstance(text, str):
        return []

    base = re.sub(r"\s+", " ", text.replace("|", " ")).strip()
    if not base:
        return []

    variants: list[str] = [base]

    norm = _normalize_text_for_match(base)

    # percent/per cent ↔ % (common LLM phrasing vs PDF symbol)
    if "percent" in norm or "per cent" in norm:
        v = re.sub(r"\bper\s*cent\b", "%", base, flags=re.IGNORECASE)
        v = re.sub(r"\bpercent\b", "%", v, flags=re.IGNORECASE)
        v = re.sub(r"\s*%\s*", "%", v)
        variants.append(v)

    if "%" in base:
        variants.append(re.sub(r"%", " percent", base))
        variants.append(re.sub(r"%", " per cent", base))

    # De-duplicate while preserving order
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
    norm = _normalize_text_for_match(text)
    # Keep alphanumerics and a few accounting symbols; everything else becomes whitespace.
    norm = re.sub(r"[^a-z0-9£$%.,/()\- ]+", " ", norm)
    tokens = [t for t in norm.split() if len(t) > 2 and t not in STOPWORDS]
    return tokens


def _iter_page_lines(page) -> Iterable[tuple[str, fitz.Rect]]:
    """Yield (line_text, line_rect) for each text line on the page."""
    try:
        data = page.get_text("dict")
        for block in data.get("blocks", []) or []:
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                line_text = "".join((s.get("text") or "") for s in spans).strip()
                bbox = line.get("bbox")
                if not line_text or not bbox:
                    continue
                yield line_text, fitz.Rect(bbox)
    except Exception:
        return


def _expand_rect_to_line(page, rect: fitz.Rect) -> fitz.Rect:

    if not rect:
        return rect

    # Prefer using the extracted line bounding boxes from the text dict.
    best_line_rect: Optional[fitz.Rect] = None
    best_overlap = 0.0

    for _, line_rect in _iter_page_lines(page):
        if not line_rect.intersects(rect):
            continue
        overlap = (line_rect & rect).get_area()
        if overlap > best_overlap:
            best_overlap = overlap
            best_line_rect = line_rect

    return best_line_rect or rect


def _expand_rect_to_row(page, rect: fitz.Rect) -> fitz.Rect:

    if not rect:
        return rect

    words = page.get_text("words")
    # Keep this fairly strict to avoid accidentally spanning adjacent lines.
    tol = max(CONFIG.get('y_tolerance', 6), 6)
    target_yc = (rect.y0 + rect.y1) / 2

    row_rect: Optional[fitz.Rect] = None
    for w in words:
        wrect = fitz.Rect(w[:4])
        w_yc = (wrect.y0 + wrect.y1) / 2
        # Same-row/line: vertical centers close.
        if abs(w_yc - target_yc) > tol:
            continue
        if row_rect is None:
            row_rect = wrect
        else:
            row_rect |= wrect

    return row_rect or _expand_rect_to_line(page, rect)


def _line_text_for_rect(page, rect: fitz.Rect) -> str:
    if not rect:
        return ""
    for line_text, line_rect in _iter_page_lines(page):
        if line_rect.intersects(rect):
            return line_text.strip()
    return ""


def _box_overlaps_page_text(page, box: fitz.Rect) -> bool:

    if not box:
        return False
    try:
        for w in page.get_text("words"):
            wrect = fitz.Rect(w[:4])
            if wrect.intersects(box):
                return True
    except Exception:
        return False
    return False


def _is_heading_like(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = _normalize_text_for_match(text)
    if not t:
        return False

    if t.endswith(":"):
        return True
    if "calculated as follows" in t or "is as follows" in t:
        return True
    if t.startswith("revised consolidated statement"):
        return True
    if t.startswith("the basic eps"):
        return True
    # Keep this conservative: many table row labels (e.g. "Goodwill") have no digits.
    # Only treat short non-numeric lines as headings if they contain strong heading-like keywords.
    if not re.search(r"\d", t) and len(t.split()) <= 12:
        heading_keywords = (
            "calculated",
            "as follows",
            "accounting",
            "treatment",
            "journal",
            "statement",
            "revised",
            "electrostatic",
            "revaluation",
        )
        if any(k in t for k in heading_keywords):
            return True
    return False


def _row_has_numeric_content(page, rect: fitz.Rect) -> bool:
    """Heuristic: does the y-band ('row') contain numeric/currency content?

    Useful to avoid underlining table headers / working-paper labels that have no values
    on the same row.
    """
    if not rect:
        return False

    try:
        words = page.get_text("words")
    except Exception:
        return False

    tol = max(CONFIG.get('y_tolerance', 6), 6)

    def _has_nonzero_digit(s: str) -> bool:
        return bool(re.search(r"[1-9]", s or ""))

    def _is_value_like(token: str) -> bool:
        t = (token or "").strip()
        if not t:
            return False

        # Exclude common short code labels (e.g. W1, A12) from counting as numeric content.
        if re.fullmatch(r"[A-Za-z]{1,3}\d{1,3}", t):
            return False

        # Currency/percent symbols alone are often table headers (e.g. "£" "£").
        # Only treat as content if digits are also present AND not just zeros (e.g. "£000").
        if re.search(r"£|\$|%", t) and re.search(r"\d", t):
            digits = re.sub(r"\D", "", t)
            if _has_nonzero_digit(digits):
                return True
            return False

        # Pure numeric-ish values: allow commas, parentheses, leading minus.
        t2 = t.replace(",", "")
        t2 = t2.strip("()")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", t2):
            # Treat 2+ digits as "value-like" (filters out single-digit noise).
            digits = re.sub(r"\D", "", t2)
            return len(digits) >= 2 and _has_nonzero_digit(digits)

        return False

    for w in words:
        wrect = fitz.Rect(w[:4])
        if abs(wrect.y0 - rect.y0) <= tol or abs(wrect.y1 - rect.y1) <= tol:
            txt = (w[4] or "").strip()
            if _is_value_like(txt):
                return True
    return False


def _redirect_if_header_like(page, rect: fitz.Rect, expand_to_line: bool, max_down: float = 240.0) -> Optional[fitz.Rect]:
    """If rect is on a heading/header-like row, redirect to the next numeric line below."""
    if not rect:
        return None

    line_rect = _expand_rect_to_line(page, rect)
    line_text = _line_text_for_rect(page, line_rect)

    # Common header-like cases:
    # - explicit headings detected by text
    # - working-paper refs / table headers where the same row has no numeric values
    norm = _normalize_text_for_match(line_text)
    short_label = (len(norm.split()) <= 3 and len(norm) <= 24)

    # Detect W1/W6-like tokens anywhere in the line (line may also contain "£" headers).
    wp_ref = any(re.fullmatch(r"[a-z]{1,3}\d{1,3}", tok) for tok in norm.split())

    # Header rows often have no digits at all (except the wp ref); don't finalize on them.
    has_any_digit = bool(re.search(r"\d", norm))
    has_digit_besides_wp = False
    for tok in norm.split():
        if re.fullmatch(r"[a-z]{1,3}\d{1,3}", tok):
            continue
        if re.search(r"\d", tok):
            has_digit_besides_wp = True
            break

    row_has_values = _row_has_numeric_content(page, line_rect)
    headerish = (
        _is_heading_like(line_text)
        or ((short_label or wp_ref) and not row_has_values)
        or (wp_ref and (not has_digit_besides_wp) and not row_has_values)
        or ((not has_any_digit) and not row_has_values)
    )

    if not headerish:
        return None

    nxt = _find_next_numeric_line(page, line_rect, max_down=max_down)
    if not nxt:
        return None

    if not expand_to_line:
        refined = _refine_to_numberish_word_on_line(page, nxt)
        return refined or nxt
    return nxt


def _rank_pages_for_anchor(page_token_sets: dict[int, set[str]], allowed_pages: list[int], anchor_text: str) -> list[int]:
    """Rank allowed pages by token overlap with anchor_text (best first)."""
    anchor_tokens = set(_tokenize(anchor_text))
    if not anchor_tokens or not page_token_sets:
        return allowed_pages

    scored: list[tuple[int, int]] = []
    for p in allowed_pages:
        pset = page_token_sets.get(p)
        if not pset:
            scored.append((0, p))
            continue
        score = len(anchor_tokens & pset)
        scored.append((score, p))

    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    ranked = [p for _, p in scored]
    return ranked


def _build_context_words(text: str, max_words: int = 6) -> list[str]:
    """Extract distinctive context tokens for hybrid number+context matching."""
    toks = _tokenize(text)
    if not toks:
        return []
    # Deduplicate, preserve order, keep small list.
    out: list[str] = []
    seen: set[str] = set()
    for tok in toks:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_words:
            break
    return out


def _refine_to_numberish_word_on_line(page, line_rect: fitz.Rect) -> Optional[fitz.Rect]:
    """Pick a tight rect on the given line for the most 'number-like' (usually rightmost) word."""
    if not line_rect:
        return None
    words = page.get_text("words")
    candidates: list[tuple[fitz.Rect, str]] = []
    for w in words:
        rect = fitz.Rect(w[:4])
        if not is_on_same_line(rect, line_rect):
            continue
        txt = (w[4] or "").strip()
        if not txt:
            continue
        # Numbers / currencies / ratios are typically what we want to attach a mark to.
        if re.search(r"\d", txt) or re.search(r"£|\$|%", txt):
            candidates.append((rect, txt))

    if not candidates:
        return None

    # Prefer the rightmost numeric/currency word on the line.
    candidates.sort(key=lambda it: it[0].x0)
    return candidates[-1][0]


def _build_number_variations(num_text: str) -> list[str]:
    clean = (num_text or "").replace(",", "").replace(" ", "")
    if not clean:
        return []

    if re.fullmatch(r"\d+\.\d+", clean):
        int_part, dec_part = clean.split(".", 1)
    else:
        int_part, dec_part = clean, None

    variants: list[str] = [clean]

    def add_thousands(int_only: str) -> None:
        if not int_only.isdigit() or len(int_only) <= 3:
            return
        chunks: list[str] = []
        s = int_only
        while len(s) > 3:
            chunks.append(s[-3:])
            s = s[:-3]
        chunks.append(s)
        chunks = list(reversed(chunks))
        comma = ",".join(chunks)
        space = " ".join(chunks)
        variants.extend([comma, space])
        if dec_part is not None:
            variants.extend([f"{comma}.{dec_part}", f"{space}.{dec_part}"])

    add_thousands(int_part)

    # Common table format: trailing .00
    if dec_part is None and int_part.isdigit():
        variants.append(f"{int_part}.00")
        if len(int_part) > 3:
            chunks: list[str] = []
            s = int_part
            while len(s) > 3:
                chunks.append(s[-3:])
                s = s[:-3]
            chunks.append(s)
            chunks = list(reversed(chunks))
            comma = ",".join(chunks)
            space = " ".join(chunks)
            variants.extend([f"{comma}.00", f"{space}.00"])

    # Deduplicate while preserving order
    dedup: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if v and v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


def _search_number_variations(page, number_str: str) -> list[fitz.Rect]:
    rects: list[fitz.Rect] = []
    for variation in _build_number_variations(number_str):
        hits = page.search_for(variation)
        if hits:
            rects.extend(hits)

    # Deduplicate by coordinates
    dedup: list[fitz.Rect] = []
    seen: set[tuple[float, float, float, float]] = set()
    for r in rects:
        key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def _find_next_numeric_line(page, from_rect: fitz.Rect, max_down: float = 140.0) -> Optional[fitz.Rect]:
    if not from_rect:
        return None

    for line_text, line_rect in _iter_page_lines(page):
        if line_rect.y0 <= from_rect.y1 + 1:
            continue
        if line_rect.y0 > from_rect.y1 + max_down:
            break
        t = _normalize_text_for_match(line_text)
        # Require a meaningful number (not just 0/000/£000 style headers).
        digits = re.sub(r"\D", "", t)
        if digits and re.search(r"[1-9]", digits):
            return line_rect
    return None


def _find_best_line_match(page, anchor_text: str, required_number: Optional[str] = None) -> Optional[fitz.Rect]:
    """Find the best-matching line on the page for anchor_text using token overlap."""
    # For long anchors, first compress to the most distinctive chunk; this prevents overlap ratios
    # from being diluted by lots of irrelevant words/punctuation.
    best_chunk = clean_anchor_text(anchor_text, max_words=CONFIG['max_anchor_words'])
    anchor_tokens = _tokenize(best_chunk or anchor_text)
    if not anchor_tokens:
        return None

    anchor_set = set(anchor_tokens)

    best_rect: Optional[fitz.Rect] = None
    best_score = 0.0

    required_number_norm = None
    if required_number:
        required_number_norm = required_number.replace(",", "").replace(" ", "")

    for line_text, line_rect in _iter_page_lines(page):
        lt_norm = _normalize_text_for_match(line_text)
        line_tokens = _tokenize(line_text)
        if not line_tokens:
            continue

        line_set = set(line_tokens)
        overlap = len(anchor_set & line_set)
        if overlap < 2:
            continue

        score = overlap / max(len(anchor_set), 1)

        if required_number_norm:
            # Bonus if the line contains the expected number (with or without separators)
            lt_digits = lt_norm.replace(",", "").replace(" ", "")
            if required_number_norm in lt_digits:
                score += 0.35

        if score > best_score:
            best_score = score
            best_rect = line_rect

    # Conservative threshold to reduce false positives.
    if best_score >= 0.45:
        return best_rect

    # Fallback: allow a single strong keyword (long alphabetic token) to match a value line.
    # This helps when anchors include dates (e.g. "... 1 June 20X3") but the value line only
    # contains the section keyword + number.
    strong_tokens = [t for t in anchor_set if t.isalpha() and len(t) >= 7]
    if strong_tokens:
        best_rect2: Optional[fitz.Rect] = None
        best_score2 = 0.0
        for line_text, line_rect in _iter_page_lines(page):
            lt_norm = _normalize_text_for_match(line_text)
            line_tokens = _tokenize(line_text)
            if not line_tokens:
                continue

            if not any(st in line_tokens for st in strong_tokens):
                continue

            # Prefer lines that actually have a value.
            has_value = bool(re.search(r"\d", lt_norm)) or bool(re.search(r"£|\$|%", line_text) and re.search(r"\d", lt_norm))
            if not has_value:
                continue

            score2 = 0.30
            # Small bump if more tokens overlap beyond the strong token.
            score2 += (len(anchor_set & set(line_tokens)) / max(len(anchor_set), 1)) * 0.20

            if score2 > best_score2:
                best_score2 = score2
                best_rect2 = line_rect

        if best_rect2 is not None and best_score2 >= 0.30:
            return best_rect2
    return None


# ==================== HELPERS (unchanged) ====================

def clean_anchor_text(text: str, max_words=6):
    if not text or not isinstance(text, str):
        return None
    # Remove LLM artifacts and normalize
    text = re.sub(r'\[\.\.\.\]|\\n|\\\n', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    if len(words) < 2:
        return None
    
    # Prioritize chunks with numbers, symbols, proper nouns (better anchors)
    best_chunk, best_score = None, 0
    for i in range(len(words) - max_words + 1):
        chunk = " ".join(words[i:i+max_words])
        score = (
            chunk.count(',') + chunk.count('£') + chunk.count('$') + chunk.count('%') + chunk.count('×')
            + sum(1 for w in chunk.split() if re.match(r'^\d|\d.*\d|FV|NCI|OCI', w))
            + len([w for w in chunk.split() if w[0].isupper() and len(w) > 2])
        )
        if score > best_score:
            best_score, best_chunk = score, chunk
    return best_chunk or " ".join(words[:max_words])


def find_text_rects_partial(page, search_text, full_match=False):
    """Find text rectangles with fuzzy matching, avoiding duplicates."""
    if not search_text:
        return []
    
    if full_match:
        hits = page.search_for(search_text)
        return hits if hits else []
    
    search_lower = search_text.lower()
    matches = []
    seen = set()  # Avoid duplicates
    
    # 1. Word-level fuzzy matching
    words = search_text.split()
    for w in page.get_text("words"):
        word_lower = w[4].lower()
        for search_word in words:
            if search_word.lower() in word_lower or word_lower in search_word.lower():
                rect = fitz.Rect(w[:4])
                rect_key = (rect.x0, rect.y0, rect.x1, rect.y1)
                if rect_key not in seen:
                    matches.append(rect)
                    seen.add(rect_key)
                break  # Only add once per word
    
    return matches


def find_number_with_context(page, number_str: str, context_words: list) -> Optional[tuple]:
    if not number_str or not context_words:
        return None

    num_rects = _search_number_variations(page, number_str)
    
    if not num_rects:
        logger.debug(f"  [hybrid] Number not found: {number_str}")
        return None
    
    logger.debug(f"  [hybrid] Found {len(num_rects)} occurrence(s) of {number_str}, scoring context...")

    words_cache = page.get_text("words")

    best_rect: Optional[fitz.Rect] = None
    best_score = -1.0
    best_matches: list[str] = []

    # For each number occurrence, score by how many context words appear on the same line.
    for idx, num_rect in enumerate(num_rects):
        line_words: list[str] = []
        for word_obj in words_cache:
            word_rect = fitz.Rect(word_obj[:4])
            if abs(word_rect.y0 - num_rect.y0) <= CONFIG['y_tolerance']:
                wtxt = (word_obj[4] or "").strip().lower()
                if wtxt:
                    line_words.append(wtxt)

        matched_contexts: list[str] = []
        for ctx_word in context_words:
            ctx_lower = (ctx_word or "").lower()
            if not ctx_lower:
                continue
            if any(ctx_lower in w or w in ctx_lower for w in line_words):
                matched_contexts.append(ctx_word)

        # Score: fraction of context words matched, plus a small bias to avoid heading-like lines.
        frac = len(matched_contexts) / max(len(context_words), 1)
        score = frac
        line_text = " ".join(line_words)
        if _is_heading_like(line_text):
            score -= 0.25

        logger.debug(
            f"  [hybrid] Occurrence #{idx+1}: matched={len(matched_contexts)}/{len(context_words)} score={score:.2f}"
        )

        if score > best_score:
            best_score = score
            best_rect = num_rect
            best_matches = matched_contexts

    # Require at least one distinctive token match to avoid picking the wrong repeated number.
    if best_rect is not None and best_score >= (1.0 / max(len(context_words), 1)):
        logger.debug(f"  [hybrid] ✓ Best match: {len(best_matches)} context word(s): {best_matches}")
        return best_rect

    logger.debug("  [hybrid] Number found but no sufficiently strong context match")
    return None


def extract_number_from_text(text: str) -> Optional[str]:
    """Extract a 'useful' number from text for anchoring.

    Avoids tiny numbers that commonly come from dates (e.g., 1 March 20X4 -> 1/20/4)
    which otherwise cause rampant false matches.
    """
    if not text or not isinstance(text, str):
        return None

    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if not numbers:
        return None

    has_currency = bool(re.search(r"£|\$|gbp|usd|eur", text, flags=re.IGNORECASE))

    candidates: list[str] = []
    for raw in numbers:
        norm = raw.replace(",", "").replace(" ", "")
        try:
            val = float(norm)
        except Exception:
            continue

        # Skip tiny decimals (these are often marks like 0.25 or ratios)
        if abs(val) < 1:
            continue

        # Prefer accounting-like magnitudes or formatted values
        if "," in raw or abs(val) >= 1000 or (has_currency and abs(val) >= 100):
            candidates.append(norm)

    # If no strong candidates, do not use number-first anchoring at all.
    if not candidates:
        return None

    # Bias to later numbers (often the result at end of a calculation line)
    return candidates[-1]


def is_on_same_line(r1, r2):
    return abs(r1.y0 - r2.y0) <= CONFIG['y_tolerance']


def find_number_rect_in_text(page, text_rect, number_str):
    if not page or not text_rect or not number_str:
        return text_rect
    
    try:
        # Search on the same y-line but extending to the right
        search_area = fitz.Rect(
            text_rect.x0, 
            text_rect.y0 - 2,
            page.rect.width - 50,
            text_rect.y1 + 2
        )
        
        num_hits = page.search_for(number_str, clip=search_area)
        if num_hits:
            # Rightmost number rect (actual result, not intermediate calc)
            found_rect = num_hits[-1]
            logger.debug(f"      [refined] Number: x={found_rect.x0:.1f} (text was x={text_rect.x0:.1f})\"")
            return found_rect
    except Exception as e:
        logger.debug(f"      [refined] Error: {e}")
    
    return text_rect


def resolve_anchor_rect(
    doc,
    anchor_text,
    allowed_pages,
    placed_marks=None,
    skip_duplicates=True,
    expand_to_line: bool = True,
    page_token_sets: Optional[dict[int, set[str]]] = None,
):
    if not anchor_text or not isinstance(anchor_text, str):
        return None, None
    
    if placed_marks is None:
        placed_marks = set()
    
    evidence_preview = anchor_text[:50] if len(anchor_text) > 50 else anchor_text
    best_rect, best_page = None, -1
    
    ranked_pages = _rank_pages_for_anchor(page_token_sets or {}, list(allowed_pages), anchor_text)

    num = extract_number_from_text(anchor_text)
    if num:
        logger.debug(f"    → Number-first approach: searching for '{num}' (extracted from evidence)")
        
        for page_num in ranked_pages:
            page = doc[page_num - 1]
            context_words = _build_context_words(anchor_text, max_words=6)
            
            # Try HYBRID first: Number + Context (most reliable)
            logger.debug(f"    [number-first] Trying hybrid: num={num}, context={context_words[:3]}")
            hybrid_rect = find_number_with_context(page, num, context_words)
            if hybrid_rect:
                mark_key = (page_num, round(hybrid_rect.y0))
                if not (skip_duplicates and mark_key in placed_marks):
                    logger.debug(f"    [number-first-hybrid] ✓ Found number with context on page {page_num}")
                    chosen = _expand_rect_to_line(page, hybrid_rect) if expand_to_line else hybrid_rect
                    redirected = _redirect_if_header_like(page, chosen, expand_to_line)
                    if redirected is not None:
                        return redirected, page_num
                    return chosen, page_num
            
            # Fallback to number alone (with variations), even if context doesn't match
            num_hits = _search_number_variations(page, num)
            if num_hits:
                for rect in num_hits:
                    mark_key = (page_num, round(rect.y0))
                    if skip_duplicates and mark_key in placed_marks:
                        logger.debug(f"    [number-first-skip] Rect at y={rect.y0:.1f} already marked, trying next...")
                        continue
                    logger.debug(f"    [number-first] ✓ Found number '{num}' on page {page_num}")
                    chosen = _expand_rect_to_line(page, rect) if expand_to_line else rect
                    redirected = _redirect_if_header_like(page, chosen, expand_to_line)
                    if redirected is not None:
                        return redirected, page_num
                    return chosen, page_num
    
    # Strategy 1: Try exact phrase match (now secondary)
    for page_num in ranked_pages:
        page = doc[page_num - 1]
        exact_hits = page.search_for(anchor_text)
        if exact_hits:
            # Check for duplicates - skip if already marked
            for rect in exact_hits:
                mark_key = (page_num, round(rect.y0))
                if skip_duplicates and mark_key in placed_marks:
                    logger.debug(f"    [exact-skip] Rect at y={rect.y0:.1f} already marked, trying next...")
                    continue
                
                # If this is numeric evidence, refocus rect on the number itself
                if num:
                    refined_rect = find_number_rect_in_text(page, rect, num)
                    if refined_rect and refined_rect.x0 > rect.x0 + 5:
                        # Use refined rect if it's meaningfully to the right
                        logger.debug(f"    [refined] Using number location instead of text label\"")
                        rect = refined_rect
                
                logger.debug(f"    [exact] '{evidence_preview}' → EXACT PHRASE on page {page_num}")
                chosen = _expand_rect_to_line(page, rect) if expand_to_line else rect
                redirected = _redirect_if_header_like(page, chosen, expand_to_line)
                if redirected is not None:
                    return redirected, page_num
                return chosen, page_num
        
        # Strategy 2: Try cleaned phrase
        clean_phrase = clean_anchor_text(anchor_text)
        if clean_phrase and clean_phrase != anchor_text:
            clean_hits = page.search_for(clean_phrase)
            if clean_hits:
                for rect in clean_hits:
                    mark_key = (page_num, round(rect.y0))
                    if skip_duplicates and mark_key in placed_marks:
                        logger.debug(f"    [clean-skip] Rect at y={rect.y0:.1f} already marked, trying next...")
                        continue
                    # If this is numeric evidence, refocus rect on the number itself
                    if num:
                        refined_rect = find_number_rect_in_text(page, rect, num)
                        if refined_rect:
                            rect = refined_rect
                    logger.debug(f"    [clean] '{evidence_preview}' → CLEAN PHRASE on page {page_num}")
                    chosen = _expand_rect_to_line(page, rect) if expand_to_line else rect
                    redirected = _redirect_if_header_like(page, chosen, expand_to_line)
                    if redirected is not None:
                        return redirected, page_num
                    return chosen, page_num
        
        # Strategy 3: HYBRID - Number + Context words together (fallback position)
        if num:
            # Extract key content words (skip small words, numbers, special chars)
            context_words = _build_context_words(anchor_text, max_words=6)
            
            logger.debug(f"    → Trying hybrid: num={num}, context={context_words[:3]}")
            
            # Try hybrid matching: number + context
            hybrid_rect = find_number_with_context(page, num, context_words)
            if hybrid_rect:
                mark_key = (page_num, round(hybrid_rect.y0))
                if skip_duplicates and mark_key in placed_marks:
                    logger.debug(f"    [hybrid-skip] Rect at y={hybrid_rect.y0:.1f} already marked, trying next...")
                else:
                    logger.debug(f"    [hybrid] '{evidence_preview}' → NUMBER+CONTEXT match on page {page_num}")
                    chosen = _expand_rect_to_line(page, hybrid_rect) if expand_to_line else hybrid_rect
                    redirected = _redirect_if_header_like(page, chosen, expand_to_line)
                    if redirected is not None:
                        return redirected, page_num
                    return chosen, page_num
        
        # Strategy 4: Word-by-word clustering (words on same line)
        words = anchor_text.split()[:CONFIG['max_anchor_words']]
        rects_by_word = []
        
        for word in words:
            if len(word) > 2:  # Skip small words
                hits = find_text_rects_partial(page, word, full_match=False)
                if hits:
                    rects_by_word.append(hits)
        
        # Check if we found enough words and they're on same line
        if rects_by_word and len(rects_by_word) >= max(2, len(words) - 2):
            first_matches = [rects[0] for rects in rects_by_word if rects]
            if first_matches:
                first_matches.sort(key=lambda r: r.x0)
                y_vals = [r.y0 for r in first_matches]
                
                # Check if on same line (tight tolerance)
                if max(y_vals) - min(y_vals) <= CONFIG['y_tolerance']:
                    combined = first_matches[0]
                    for r in first_matches[1:]:
                        combined = combined | r
                    mark_key = (page_num, round(combined.y0))
                    if skip_duplicates and mark_key in placed_marks:
                        logger.debug(f"    [cluster-skip] Rect at y={combined.y0:.1f} already marked, trying next...")
                    else:
                        logger.debug(f"    [cluster] '{evidence_preview}' → WORD CLUSTER on page {page_num}")
                        chosen = _expand_rect_to_line(page, combined) if expand_to_line else combined
                        redirected = _redirect_if_header_like(page, chosen, expand_to_line)
                        if redirected is not None:
                            return redirected, page_num
                        return chosen, page_num
            
            # Keep first match as fallback
            if not best_rect:
                best_rect, best_page = first_matches[0], page_num
                logger.debug(f"    [fallback] Keeping word cluster as fallback on page {page_num}")

        # Strategy 4.5: Line-level fuzzy match (handles line breaks / punctuation differences)
        best_line = _find_best_line_match(page, anchor_text, required_number=num)
        if best_line:
            mark_key = (page_num, round(best_line.y0))
            if skip_duplicates and mark_key in placed_marks:
                logger.debug(f"    [line-fuzzy-skip] Rect at y={best_line.y0:.1f} already marked, trying next...")
            else:
                logger.debug(f"    [line-fuzzy] '{evidence_preview}' → BEST LINE TOKEN MATCH on page {page_num}")
                chosen = _expand_rect_to_line(page, best_line) if expand_to_line else best_line
                redirected = _redirect_if_header_like(page, chosen, expand_to_line)
                if redirected is not None:
                    return redirected, page_num
                return chosen, page_num
        
        # Strategy 5: LAST RESORT - number alone (without context validation)
        if num and not best_rect:
            num_hits = _search_number_variations(page, num)
            if num_hits:
                for rect in num_hits:
                    mark_key = (page_num, round(rect.y0))
                    if skip_duplicates and mark_key in placed_marks:
                        logger.debug(f"    [number-only-skip] Rect at y={rect.y0:.1f} already marked, trying next...")
                        continue
                    logger.debug(f"    [number-only] '{evidence_preview}' → NUMBER ALONE on page {page_num} (no context validation)")
                    best_rect, best_page = (_expand_rect_to_line(page, rect) if expand_to_line else rect), page_num
                    break
    
    if best_rect and best_page != -1:
        # Check fallback rect isn't already marked
        mark_key = (best_page, round(best_rect.y0))
        if not (skip_duplicates and mark_key in placed_marks):
            logger.debug(f"    [fallback-used] Anchor matched via fallback on page {best_page}")
            return best_rect, best_page
    
    logger.debug(f"    [FAILED] Could not match: '{evidence_preview}'")
    return None, None


def add_popup_for_comment(doc, comment, allowed_pages, placed_marks=None):
    if not comment or '→' not in comment:
        logger.debug(f"  No arrow in comment: '{comment[:40]}...'")
        return False

    # Skip total score popup
    if 'TOTAL SCORE' in comment.upper() or re.search(r'\d+\.\d+/\d+', comment):
        logger.debug(f"  Skipping total score comment")
        return False

    anchor_part = comment.split('→')[0].strip().strip('"\'')
    if not anchor_part or len(anchor_part) < 3:
        logger.debug(f"  Invalid anchor part: '{anchor_part}'")
        return False
    
    logger.debug(f"  [comment] Looking for anchor: '{anchor_part[:50]}...'" )
    rect, page_num = None, -1
    for candidate in _build_anchor_variations(anchor_part):
        rect, page_num = resolve_anchor_rect(
            doc, candidate, allowed_pages,
            placed_marks=placed_marks,
            skip_duplicates=False,  # Don't skip for comments, they can share locations
            page_token_sets=None,
        )
        if rect and page_num != -1:
            break
    if not rect or page_num == -1:
        logger.debug(f"  [comment] ✗ Anchor NOT found for: '{anchor_part[:50]}...'")
        return False

    page = doc[page_num - 1]
    x = page.rect.width - CONFIG['comment_offset']
    y = rect.y0 + CONFIG['criterion_score_offset_y']
    
    try:
        annot = page.add_text_annot((x, y), "", icon="Note")
        annot.set_colors(stroke=CONFIG['comment_color'])
        annot.set_opacity(0.85)
        annot.set_info(content=comment.strip(), title="Feedback")
        annot.update()
        logger.debug(f"  [comment] ✓ Comment popup added at page {page_num}, y={y:.1f}")
        return True
    except Exception as e:
        logger.debug(f"  [comment] ✗ Error adding annotation: {e}")
        return False


def place_score_near_anchor(
    doc,
    anchor_text,
    score_text,
    allowed_pages,
    placed_lines_per_page,
    placed_marks,
    unplaced_items,
    page_token_sets: Optional[dict[int, set[str]]] = None,
):
    # Keep content after ellipses: many evidence strings use "..." as a visual separator,
    # and truncating would drop the actual numeric target.
    evidence_clean = (anchor_text or "").replace("…", " ")
    evidence_clean = evidence_clean.replace("...", " ")
    evidence_clean = evidence_clean.replace('|', ' ')
    evidence_clean = re.sub(r"\s+", " ", evidence_clean).strip()
    if not evidence_clean:
        unplaced_items.append((score_text, "[empty]"))
        logger.debug(f"    [placement] ✗ Empty evidence for score {score_text}")
        return False
    
    logger.debug(f"    [placement] Searching for anchor: '{evidence_clean[:60]}'...")

    rect, page_num = None, -1
    used_anchor = evidence_clean
    for candidate in _build_anchor_variations(evidence_clean):
        used_anchor = candidate
        rect, page_num = resolve_anchor_rect(
            doc,
            candidate,
            allowed_pages,
            placed_marks=placed_marks,
            skip_duplicates=True,
            # Always resolve to a tight rect so score placement isn't forced to the left margin.
            expand_to_line=False,
            page_token_sets=page_token_sets,
        )
        if rect and page_num != -1:
            break
    if rect and page_num != -1:
        page = doc[page_num - 1]
        page_idx = page_num - 1

        # Last-chance protection: if we somehow resolved to a header/heading line, redirect now.
        line_rect = _expand_rect_to_line(page, rect)
        redirected = _redirect_if_header_like(page, rect, expand_to_line=False)
        if redirected is not None:
            rect = redirected
            line_rect = _expand_rect_to_line(page, rect)

        # Always underline the full point content (row), but keep underline y on the matched line.
        underline_rect = _expand_rect_to_row(page, rect)
        underline_line = line_rect
        
        # Place score text baseline near bottom of the matched value.
        score_font = CONFIG['criterion_score_fontsize']
        score_y = max(min(rect.y1 - 2, page.rect.height - 10), 10)

        # Track placed score boxes to avoid overlaps on dense lines.
        nearby_boxes: list[fitz.Rect] = placed_lines_per_page.get(page_idx, [])

        # Place score near the credited value on the SAME line (not above).
        # Prefer just to the right of the matched value; fall back to its x0.
        score_x = rect.x1 + 3
        if score_x > page.rect.width - 70:
            score_x = min(max(rect.x0, 50), page.rect.width - 70)

        def _score_box(x: float, y: float) -> fitz.Rect:
            # Approximate bounding box for the inserted score text.
            return fitz.Rect(x, y - (score_font + 1), x + 42, y + 3)

        placed_box = _score_box(max(score_x, 50), score_y)
        if any(placed_box.intersects(b) for b in nearby_boxes):
            # Try a few alternative X positions before moving Y.
            candidates_x = [
                max(rect.x0 - 34, 50),
                min(rect.x1 + 18, page.rect.width - 70),
            ]
            found = False
            for cx in candidates_x:
                cb = _score_box(cx, score_y)
                if not any(cb.intersects(b) for b in nearby_boxes):
                    score_x = cx
                    placed_box = cb
                    found = True
                    break

            if not found:
                # As a last resort, nudge down slightly (still close to the target line).
                for _ in range(5):
                    score_y = min(score_y + (score_font + 2), page.rect.height - 10)
                    cb = _score_box(max(score_x, 50), score_y)
                    if not any(cb.intersects(b) for b in nearby_boxes):
                        placed_box = cb
                        break

        # If we'd paste the mark on top of existing text, nudge it 10pt above.
        # (Only do this when overlapping PDF text, and don't introduce score-score collisions.)
        if _box_overlaps_page_text(page, placed_box):
            shifted_y = max(score_y - 10, 10)
            shifted_box = _score_box(max(score_x, 50), shifted_y)
            if not any(shifted_box.intersects(b) for b in nearby_boxes):
                score_y = shifted_y
                placed_box = shifted_box
                logger.debug(f"    [placement] Nudge up 10pt to avoid text overlap (y={score_y:.1f})")
            else:
                logger.debug("    [placement] Text overlap detected but nudge would collide; keeping y")
        
        # Draw underline
        underline_start = max(underline_rect.x0, 10)
        underline_end = min(underline_rect.x1, page.rect.width - 50)
        underline_y = underline_line.y1 + 2
        page.draw_line(
            (underline_start, underline_y),
            (underline_end, underline_y),
            color=CONFIG['underline_color'],
            width=0.8
        )
        
        page.insert_text(
            (max(score_x, 50), score_y),
            score_text,
            fontsize=CONFIG['criterion_score_fontsize'],
            color=CONFIG['criterion_score_color']
        )
        
        placed_lines_per_page.setdefault(page_idx, []).append(placed_box)
        # Track globally to avoid duplicate marking of the same line position
        placed_marks.add((page_num, round(rect.y0)))
        logger.debug(
            f"    [placement] ✓ Score {score_text} placed at page {page_num} y={score_y:.1f}, x={max(score_x, 50):.1f} (anchor='{used_anchor[:45]}...')"
        )
        return True
    else:
        unplaced_items.append((score_text, evidence_clean[:50]))
        logger.debug(f"    [placement] ✗ Failed: {score_text} - anchor not found for '{evidence_clean[:50]}'")
        return False


def add_main_score(doc, q_num, score_text, allowed_pages):
    if not q_num:
        logger.warning("No question number for main score placement")
        return False
    
    q_str = str(q_num).strip()
    strategies = [
        (q_str, "exact Q match"),                  # "2" or "Q2"
        (f"Q{q_str}", "Q prefix"),                 # "Q2"
        (f"Question {q_str}", "Question prefix"),  # "Question 2"
    ]
    
    for page_num in allowed_pages:
        page = doc[page_num - 1]
        
        for search_text, strategy_name in strategies:
            instances = page.search_for(search_text)
            if instances:
                heading_rect = instances[0]
                
                # ✓ NEW: Validate that there's substantive content BELOW the heading
                # Check for text/numbers in the region below heading (y > heading.y1)
                has_substantive_content = False
                min_y_below = heading_rect.y1 + 5
                max_y_below = min(heading_rect.y1 + 80, page.rect.height - 20)
                
                for word_obj in page.get_text("words"):
                    word_text = word_obj[4].strip()
                    word_y = word_obj[1]
                    
                    # Text below heading, not just whitespace or single chars
                    if (word_y >= min_y_below and word_y <= max_y_below and 
                        len(word_text) > 1 and not word_text.isspace() and 
                        not re.match(r'^[^a-zA-Z0-9]*$', word_text)):
                        has_substantive_content = True
                        logger.debug(f"      [main-score] Found substantive content below heading: '{word_text}'")
                        break
                
                if not has_substantive_content:
                    logger.debug(f"      [main-score] No substantive content below '{search_text}' heading - SKIPPING")
                    continue
                
                # Place to the left and above the heading (if content validated)
                x = heading_rect.x0 - CONFIG['main_score_offset_x']
                y = heading_rect.y0 + CONFIG['main_score_offset_y'] - 10
                
                page.insert_text(
                    (x, y),
                    score_text,
                    fontsize=CONFIG['main_score_fontsize'],
                    color=CONFIG['main_score_color']
                )
                logger.info(f"✓ Main score '{score_text}' placed (strategy: {strategy_name}) on page {page_num} - content validated")
                return True
    
    logger.warning(f"✗ Main score not placed - Q{q_str} heading not found or no substantive content")
    return False


# ==================== MAIN ANNOTATION FUNCTION ====================

def annotate_pdf(
    input_pdf_path: str,
    output_dir: str,
    student_name: str,
    grades_id: Optional[str] = None,
    grades_doc: Optional[dict] = None,
    student_pages: Optional[List[int]] = None
) -> Tuple[bool, str]:
    try:
        student_key = student_name.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = os.path.join(output_dir, student_key, f"{student_key}_annotated_{timestamp}.pdf")
        mapping_json = os.path.join(output_dir, student_key, f"{student_key}_mapping_{timestamp}.json")
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        # Fetch grading document (or accept a direct object for grading-on-hold workflows)
        if grades_doc is None:
            if not grades_id:
                logger.error("No grades provided (need grades_id or grades_doc)")
                return False, ""
            grades_coll = get_collection("student_grades")
            grades_doc = grades_coll.find_one({"_id": ObjectId(grades_id)})
            if not grades_doc:
                logger.error(f"No grades for _id={grades_id}")
                return False, ""

        total_breakdown = len(grades_doc.get('breakdown', []) or [])
        displayed_count = len([it for it in (grades_doc.get('breakdown', []) or []) if float(it.get('marks_awarded', 0) or 0) > 0])
        logger.info(
            f"Annotating {student_name} Q{grades_doc.get('question_number', '?')} "
            f"({displayed_count} displayed / {total_breakdown} total criteria)"
        )

        doc = fitz.open(input_pdf_path)
        allowed_pages = student_pages or list(range(1, len(doc) + 1))

        # Precompute per-page token sets to bias anchors to the correct page (esp. late-page headings/tables).
        page_token_sets: dict[int, set[str]] = {}
        for p in allowed_pages:
            try:
                page_text = doc[p - 1].get_text("text")
            except Exception:
                page_text = ""
            page_token_sets[p] = set(_tokenize(page_text))

        placed_lines_per_page = {i - 1: [] for i in allowed_pages}
        placed_marks = set()  # Global tracking to avoid duplicate marks at same position
        unplaced_items = []
        annotation_mapping = {
            'total_score_placed': False,
            'criterion_scores_placed': 0,
            # Updated below once we filter out zero-mark criteria
            'total_criteria': len(grades_doc.get('breakdown', [])),
            'total_breakdown': len(grades_doc.get('breakdown', []) or []),
            'comments_placed': 0,
            'unplaced_items': [],
            'allowed_pages': allowed_pages
        }

        # Main total score
        main_score_text = f"{grades_doc['total_marks_awarded']:.1f}/{grades_doc['total_max_possible']}"
        if add_main_score(doc, str(grades_doc.get('question_number', '')), main_score_text, allowed_pages):
            annotation_mapping['total_score_placed'] = True

        # Per-criterion - only display/underline awarded marks (skip 0s)
        breakdown = grades_doc.get('breakdown', [])
        displayed_breakdown = [
            it for it in breakdown
            if float(it.get('marks_awarded', 0) or 0) > 0
        ]
        annotation_mapping['total_criteria'] = len(displayed_breakdown)
        criteria_count = 0
        for idx, item in enumerate(displayed_breakdown, 1):
            marks = float(item.get('marks_awarded', 0))
            evidence = (item.get('evidence', '') or '').strip()
            criterion_name = item.get('criterion', '').strip()
            
            anchor = evidence
            if not anchor or anchor.startswith('[no parsed evidence]') or 'LLM awarded marks' in anchor:
                anchor = criterion_name

            # Guard against non-informative anchors (e.g., a standalone year like "20X3").
            # These tend to match many places and lead to wrong/unstable annotations.
            anchor_norm = _normalize_text_for_match(anchor)
            anchor_key = re.sub(r"[^a-z0-9]+", "", anchor_norm)
            if (
                len(anchor_norm) < 6
                or re.fullmatch(r"20x\d", anchor_key)
                or re.fullmatch(r"20\d{2}", anchor_key)
            ):
                if criterion_name and len(_normalize_text_for_match(criterion_name)) >= 6:
                    anchor = criterion_name

            # Place only items with usable anchor
            if anchor and len(anchor) > 5:
                logger.debug(f"  Criterion {idx}: {criterion_name} ({marks}pts)")
                logger.debug(f"    Anchor: {anchor[:80]}...")
                success = place_score_near_anchor(
                    doc, anchor, f"{marks}",
                    allowed_pages, placed_lines_per_page, placed_marks, unplaced_items,
                    page_token_sets=page_token_sets,
                )
                if success:
                    annotation_mapping['criterion_scores_placed'] += 1
                criteria_count += 1
        
        logger.info(f"✓ Placed {annotation_mapping['criterion_scores_placed']} of {criteria_count} criteria with evidence")
        
        # Fallback 1: Place unplaced high-value items (>0.5 marks) in margins
        if unplaced_items:
            high_value_unplaced = [
                (score, evidence) for score, evidence in unplaced_items 
                if isinstance(score, str) and float(score) > 0.5
            ]
            
            if high_value_unplaced:
                logger.warning(f"Fallback: placing {len(high_value_unplaced)} high-value items in margin")
                fallback_page = allowed_pages[0]
                page = doc[fallback_page - 1]
                y_pos = 80
                for score, evidence in high_value_unplaced[:5]:
                    page.insert_text(
                        (page.rect.width - 280, y_pos),
                        f"{score}pt: {evidence[:35]}",
                        fontsize=8,
                        color=(0.8, 0.4, 0)
                    )
                    y_pos += 18

        # Comments with deduplication support
        all_comments = grades_doc.get('comments', [])
        logger.info(f"Processing {len(all_comments)} comments for feedback...")
        
        comments_placed = 0
        comments_failed = []
        for idx, comment in enumerate(all_comments, 1):
            if not comment or not isinstance(comment, str):
                logger.debug(f"  Comment {idx}: INVALID (empty or non-string)")
                continue
            
            logger.debug(f"  Comment {idx}/{len(all_comments)}: {comment[:70]}...")
            
            # Extract anchor (text before →)
            if '→' not in comment:
                logger.debug(f"    ✗ No arrow (→) - skipping")
                comments_failed.append(("No arrow", comment[:40]))
                continue
            
            anchor_part = comment.split('→')[0].strip()
            if not anchor_part:
                logger.debug(f"    ✗ Empty anchor - skipping")
                comments_failed.append(("Empty anchor", comment[:40]))
                continue
            
            # Try to add comment popup
            try:
                success = add_popup_for_comment(doc, comment.strip(), allowed_pages, placed_marks=placed_marks)
                if success:
                    comments_placed += 1
                    logger.debug(f"    ✓ Comment placed")
                else:
                    logger.debug(f"    ✗ Anchor not found for: '{anchor_part[:50]}'")
                    comments_failed.append(("Anchor not found", anchor_part[:40]))
            except Exception as e:
                logger.debug(f"    ✗ Error: {e}")
                comments_failed.append(("Exception", str(e)[:40]))
        
        annotation_mapping['comments_placed'] = comments_placed
        logger.info(f"✓ Comments: {comments_placed}/{len(all_comments)} placed")

        doc.save(output_pdf, garbage=4, deflate=True, clean=True)
        doc.close()

        # Diagnostics JSON
        annotation_mapping['unplaced_items'] = unplaced_items[:10]
        with open(mapping_json, 'w') as f:
            json.dump(annotation_mapping, f, indent=2, default=str)

        success_rate = annotation_mapping['criterion_scores_placed'] / max(annotation_mapping['total_criteria'], 1)
        logger.info(f"Done: {success_rate:.0%} success → {output_pdf}")
        return True, output_pdf

    except Exception as e:
        logger.error(f"Annotation failed for {student_name}: {e}", exc_info=True)
        return False, ""