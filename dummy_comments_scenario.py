import fitz
import os
import re
import json
from datetime import datetime
from typing import List, Optional, Tuple
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
    
    # Find all number occurrences
    clean_num = number_str.replace(',', '').replace(' ', '')
    
    number_variations = [
        clean_num,
        f"{clean_num[:-3]},{clean_num[-3:]}",
        f"{clean_num[:-3]} {clean_num[-3:]}",
    ]
    
    num_rects = []
    for variation in number_variations:
        hits = page.search_for(variation)
        if hits:
            num_rects.extend(hits)
    
    if not num_rects:
        logger.debug(f"  [hybrid] Number not found: {number_str}")
        return None
    
    logger.debug(f"  [hybrid] Found {len(num_rects)} occurrence(s) of {number_str}, checking context...")
    
    # For each number occurrence, check if context words are nearby
    for idx, num_rect in enumerate(num_rects):
        # Get all words on the same line as this number (within y_tolerance)
        line_words = []
        for word_obj in page.get_text("words"):
            word_rect = fitz.Rect(word_obj[:4])
            # Same line check
            if abs(word_rect.y0 - num_rect.y0) <= CONFIG['y_tolerance']:
                line_words.append(word_obj[4].lower())
        
        # Check if any context words appear on this line
        line_text = " ".join(line_words).lower()
        context_match_count = 0
        matched_contexts = []
        
        for ctx_word in context_words:
            ctx_lower = ctx_word.lower()
            if any(ctx_lower in w or w in ctx_lower for w in line_words):
                context_match_count += 1
                matched_contexts.append(ctx_word)
        
        # If we found at least 1 context word (or no context specified), this is a valid match
        if context_match_count >= min(1, len(context_words)):
            logger.debug(f"  [hybrid] ✓ Occurrence #{idx+1}: Found {len(matched_contexts)} context word(s): {matched_contexts}")
            return num_rect
        else:
            logger.debug(f"  [hybrid] ✗ Occurrence #{idx+1}: No context match (line: {line_words[:5]}...)")
    
    logger.debug(f"  [hybrid] Number found but no matching context")
    return None


def extract_number_from_text(text: str) -> Optional[str]:
    # Find all numbers
    numbers = re.findall(r'[\d,]+(?:\.\d+)?', text)
    if not numbers:
        return None
    
    # Return the last/largest number (accounting result)
    if len(numbers) > 1:
        return numbers[-1].replace(',', '').replace(' ', '')
    return numbers[0].replace(',', '').replace(' ', '') if numbers else None


def is_on_same_line(r1, r2):
    return abs(r1.y0 - r2.y0) <= CONFIG['y_tolerance']


def find_number_rect_in_text(page, text_rect, number_str):
    """
    Try to find the rect of the specific number within found text.
    When text rect finds \"Consideration (375,000*32):\" at x=80, 
    this finds the number \"12,000,000\" at x=301 on the same line.
    """
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


def resolve_anchor_rect(doc, anchor_text, allowed_pages, placed_marks=None, skip_duplicates=True):
    if not anchor_text or not isinstance(anchor_text, str):
        return None, None
    
    if placed_marks is None:
        placed_marks = set()
    
    evidence_preview = anchor_text[:50] if len(anchor_text) > 50 else anchor_text
    best_rect, best_page = None, -1
    
    for page_num in allowed_pages:
        page = doc[page_num - 1]
        
        # Strategy 1: Try exact phrase match
        exact_hits = page.search_for(anchor_text)
        if exact_hits:
            # Check for duplicates - skip if already marked
            for rect in exact_hits:
                mark_key = (page_num, round(rect.y0))
                if skip_duplicates and mark_key in placed_marks:
                    logger.debug(f"    [exact-skip] Rect at y={rect.y0:.1f} already marked, trying next...")
                    continue
                
                # If this is numeric evidence, refocus rect on the number itself
                num = extract_number_from_text(anchor_text)
                if num:
                    refined_rect = find_number_rect_in_text(page, rect, num)
                    if refined_rect and refined_rect.x0 > rect.x0 + 5:
                        # Use refined rect if it's meaningfully to the right
                        logger.debug(f"    [refined] Using number location instead of text label\"")
                        rect = refined_rect
                
                logger.debug(f"    [exact] '{evidence_preview}' → EXACT PHRASE on page {page_num}")
                return rect, page_num
        
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
                    num = extract_number_from_text(anchor_text)
                    if num:
                        refined_rect = find_number_rect_in_text(page, rect, num)
                        if refined_rect:
                            rect = refined_rect
                    logger.debug(f"    [clean] '{evidence_preview}' → CLEAN PHRASE on page {page_num}")
                    return rect, page_num
        
        # Strategy 3: HYBRID - Number + Context words together
        num = extract_number_from_text(anchor_text)
        if num:
            # Extract key content words (skip small words, numbers, special chars)
            context_words = [
                w for w in anchor_text.split() 
                if len(w) > 2 and not re.match(r'^[\d,().=-]+$', w) and len(w) < 15
            ]
            # Remove duplicates while preserving order
            context_words = list(dict.fromkeys(context_words))[:5]  # Keep top 5
            
            logger.debug(f"    → Trying hybrid: num={num}, context={context_words[:3]}")
            
            # Try hybrid matching: number + context
            hybrid_rect = find_number_with_context(page, num, context_words)
            if hybrid_rect:
                mark_key = (page_num, round(hybrid_rect.y0))
                if skip_duplicates and mark_key in placed_marks:
                    logger.debug(f"    [hybrid-skip] Rect at y={hybrid_rect.y0:.1f} already marked, trying next...")
                else:
                    logger.debug(f"    [hybrid] '{evidence_preview}' → NUMBER+CONTEXT match on page {page_num}")
                    return hybrid_rect, page_num
        
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
                        return combined, page_num
            
            # Keep first match as fallback
            if not best_rect:
                best_rect, best_page = first_matches[0], page_num
                logger.debug(f"    [fallback] Keeping word cluster as fallback on page {page_num}")
        
        # Strategy 5: LAST RESORT - number alone (without context validation)
        if num and not best_rect:
            num_hits = page.search_for(num)
            if num_hits:
                for rect in num_hits:
                    mark_key = (page_num, round(rect.y0))
                    if skip_duplicates and mark_key in placed_marks:
                        logger.debug(f"    [number-only-skip] Rect at y={rect.y0:.1f} already marked, trying next...")
                        continue
                    logger.debug(f"    [number-only] '{evidence_preview}' → NUMBER ALONE on page {page_num} (no context validation)")
                    best_rect, best_page = rect, page_num
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
    """Add comment popup at anchor location with deduplication support."""
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
    
    logger.debug(f"  [comment] Looking for anchor: '{anchor_part[:50]}...'")
    rect, page_num = resolve_anchor_rect(
        doc, anchor_part, allowed_pages, 
        placed_marks=placed_marks, 
        skip_duplicates=False  # Don't skip for comments, they can share locations
    )
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


def place_score_near_anchor(doc, anchor_text, score_text, allowed_pages, placed_lines_per_page, placed_marks, unplaced_items):
    evidence_clean = re.sub(r'\.\.\..*', '', anchor_text).strip()
    if not evidence_clean:
        unplaced_items.append((score_text, "[empty]"))
        logger.debug(f"    [placement] ✗ Empty evidence for score {score_text}")
        return False
    
    logger.debug(f"    [placement] Searching for anchor: '{evidence_clean[:60]}'...")
    rect, page_num = resolve_anchor_rect(doc, evidence_clean, allowed_pages, placed_marks=placed_marks, skip_duplicates=True)
    if rect and page_num != -1:
        page = doc[page_num - 1]
        page_idx = page_num - 1
        
        final_y = rect.y0  # Track the actual y-position where we'll place the text
        nearby_ys = placed_lines_per_page.get(page_idx, set())
        
        # Check for collision with existing annotations on same line
        collision = any(abs(final_y - p) <= CONFIG['y_tolerance'] for p in nearby_ys)
        if collision:
            logger.debug(f"    [placement] Collision detected at y={final_y:.1f}, staggering...")
            # Find nearest available y-coordinate (stagger down)
            offset = CONFIG['y_tolerance'] * 1.5  # Increased offset for better spacing
            attempts = 0
            while collision and attempts < 8:  # More attempts for better placement
                final_y += offset
                collision = any(abs(final_y - p) <= CONFIG['y_tolerance'] for p in nearby_ys)
                attempts += 1
            
            if collision:
                # Ultimate fallback: place below in margin with larger offset
                final_y = max(nearby_ys) + CONFIG['y_tolerance'] * 3 if nearby_ys else rect.y0 + 50
                logger.debug(f"    [placement] Using margin fallback at y={final_y:.1f}")
        
        score_x = rect.x0 + CONFIG['criterion_score_offset_x']
        
        # Draw underline
        underline_start = rect.x0
        underline_end = min(rect.x1 + 15, page.rect.width - 50)
        page.draw_line(
            (underline_start, rect.y1 + 2),
            (underline_end, rect.y1 + 2),
            color=CONFIG['underline_color'],
            width=1.5
        )
        
        # Place score text at the determined position
        score_y = final_y + CONFIG['criterion_score_offset_y']
        page.insert_text(
            (max(score_x, 50), score_y),
            score_text,
            fontsize=CONFIG['criterion_score_fontsize'],
            color=CONFIG['criterion_score_color']
        )
        
        # Track the actual y-position where we placed the text
        placed_lines_per_page.setdefault(page_idx, set()).add(final_y)
        # Track globally to avoid duplicate marking of the same position
        placed_marks.add((page_num, round(final_y)))
        logger.debug(f"    [placement] ✓ Score {score_text} placed at page {page_num} y={final_y:.1f}, x={max(score_x, 50):.1f}")
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
                rect = instances[0]
                # Place to the left and above the heading
                x = rect.x0 - CONFIG['main_score_offset_x']
                y = rect.y0 + CONFIG['main_score_offset_y'] - 10
                
                page.insert_text(
                    (x, y),
                    score_text,
                    fontsize=CONFIG['main_score_fontsize'],
                    color=CONFIG['main_score_color']
                )
                logger.info(f"✓ Main score '{score_text}' placed (strategy: {strategy_name}) on page {page_num}")
                return True
    
    logger.warning(f"✗ Main score not placed - Q{q_str} heading not found")
    return False


# ==================== MAIN ANNOTATION FUNCTION ====================

def annotate_pdf(
    input_pdf_path: str,
    output_dir: str,
    student_name: str,
    grades_id: str,
    student_pages: Optional[List[int]] = None
) -> Tuple[bool, str]:
    try:
        student_key = student_name.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = os.path.join(output_dir, student_key, f"{student_key}_annotated_{timestamp}.pdf")
        mapping_json = os.path.join(output_dir, student_key, f"{student_key}_mapping_{timestamp}.json")
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        # Fetch grading document
        grades_coll = get_collection("student_grades")
        grades_doc = grades_coll.find_one({"_id": ObjectId(grades_id)})
        if not grades_doc:
            logger.error(f"No grades for _id={grades_id}")
            return False, ""

        logger.info(f"Annotating {student_name} Q{grades_doc.get('question_number', '?')} ({len(grades_doc.get('breakdown', []))} criteria)")

        doc = fitz.open(input_pdf_path)
        allowed_pages = student_pages or list(range(1, len(doc) + 1))

        placed_lines_per_page = {i - 1: set() for i in allowed_pages}
        placed_marks = set()  # Global tracking to avoid duplicate marks at same position
        unplaced_items = []
        annotation_mapping = {
            'total_score_placed': False,
            'criterion_scores_placed': 0,
            'total_criteria': len(grades_doc.get('breakdown', [])),
            'comments_placed': 0,
            'unplaced_items': [],
            'allowed_pages': allowed_pages
        }

        # Main total score
        main_score_text = f"{grades_doc['total_marks_awarded']:.1f}/{grades_doc['total_max_possible']}"
        if add_main_score(doc, str(grades_doc.get('question_number', '')), main_score_text, allowed_pages):
            annotation_mapping['total_score_placed'] = True

        # Per-criterion - try to place all marks (including 0)
        breakdown = grades_doc.get('breakdown', [])
        criteria_count = 0
        for idx, item in enumerate(breakdown, 1):
            marks = float(item.get('marks_awarded', 0))
            evidence = item.get('evidence', '').strip()
            criterion_name = item.get('criterion', '').strip()
            
            # Try to place all items with evidence, even if marks=0
            if evidence and len(evidence) > 5:
                logger.debug(f"  Criterion {idx}: {criterion_name} ({marks}pts)")
                logger.debug(f"    Evidence: {evidence[:80]}...")
                success = place_score_near_anchor(
                    doc, evidence, f"{marks}",
                    allowed_pages, placed_lines_per_page, placed_marks, unplaced_items
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