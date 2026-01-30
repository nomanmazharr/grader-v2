import ast
import pandas as pd
import fitz
import os
import re
from datetime import datetime

# ==================== CONFIG ====================
CONFIG = {
    'main_score_offset_x': 20,
    'main_score_offset_y': -8,  # increased upward offset to avoid overlap
    'main_score_fontsize': 14,
    'main_score_color': (0, 0, 1),

    'criterion_score_offset_x': -15,
    'criterion_score_offset_y': 2,
    'criterion_score_fontsize': 10,
    'criterion_score_color': (0, 0.3, 0),

    'underline_color': (0, 0.7, 0),
    'comment_color': (1, 0, 0),
    'comment_offset': 30,
    'y_tolerance': 2,
}

# ==================== HELPERS ====================

def clean_anchor_text(text: str, max_words=4):
    if not text or pd.isna(text):
        return None
    text = re.sub(r'\[\.\.\.\]', ' ', text).strip()
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    if len(words) < 3:
        return text if words else None
    best_chunk, best_score = None, 0
    for i in range(len(words) - max_words + 1):
        chunk = " ".join(words[i:i+max_words])
        score = (
            chunk.count(',') + chunk.count('£') + chunk.count('$') + chunk.count('%')
            + sum(1 for w in chunk.split() if w.isdigit() or w[0].isupper())
        )
        if score > best_score:
            best_score, best_chunk = score, chunk
    return best_chunk or " ".join(words[:max_words])

def find_text_rects_partial(page, search_text, full_match=False):
    if not search_text:
        return []
    if full_match:
        return page.search_for(search_text) or []
    search_lower = search_text.lower()
    matches = []
    for w in page.get_text("words"):
        word = w[4].lower()
        if search_lower in word or word in search_lower:
            matches.append(fitz.Rect(w[:4]))
    return matches

def extract_number_from_text(text: str):
    m = re.search(r'([£$]?[\d,]+\.?\d*)', text)
    return m.group(0).replace('£', '').replace('$', '').strip() if m else None

def is_on_same_line(r1, r2):
    return abs(r1.y0 - r2.y0) <= CONFIG['y_tolerance']

# ==================== SHARED ANCHOR RESOLVER ====================

def resolve_anchor_rect(doc, anchor_text, allowed_pages):
    anchor_text = clean_anchor_text(anchor_text)
    if not anchor_text:
        return None, None

    best_rect, best_page = None, -1

    for page_num in allowed_pages:
        page = doc[page_num - 1]

        # full match
        inst = page.search_for(anchor_text)
        if inst:
            return inst[0], page_num

        # words on same line
        words = anchor_text.split()[:4]
        rects = []
        for w in words:
            hits = find_text_rects_partial(page, w)
            if hits:
                rects.append(hits[0])

        if len(rects) == len(words) and all(is_on_same_line(rects[0], r) for r in rects):
            combined = rects[0]
            for r in rects[1:]:
                combined |= r
            best_rect, best_page = combined, page_num

        # number fallback
        num = extract_number_from_text(anchor_text)
        if num:
            hits = find_text_rects_partial(page, num)
            if hits:
                best_rect, best_page = hits[0], page_num

    return best_rect, best_page if best_page != -1 else (None, None)

# ==================== MODIFIED FUNCTIONS ====================

def add_popup_for_comment(doc, comment, allowed_pages):
    if '→' not in comment:
        return False

    # Skip total score popup
    if 'TOTAL SCORE' in comment.upper() or re.search(r'\d+\.\d+/\d+', comment):
        return False

    anchor_part = comment.split('→')[0].strip().strip('"\'')
    rect, page_num = resolve_anchor_rect(doc, anchor_part, allowed_pages)
    if not rect:
        print(f"  Anchor NOT found: '{anchor_part}'")
        return False

    page = doc[page_num - 1]
    x = page.rect.width - CONFIG['comment_offset']
    y = rect.y0 + CONFIG['criterion_score_offset_y']
    annot = page.add_text_annot((x, y), "", icon="Note")
    annot.set_colors(stroke=CONFIG['comment_color'])
    annot.set_opacity(0.85)
    annot.set_info(content=comment.strip(), title="Feedback")
    annot.update()
    return True

def place_score_near_anchor(doc, anchor_text, score_text, allowed_pages, placed_lines_per_page):
    # Clean evidence to part before "..."
    evidence_clean = re.sub(r'\.\.\..*', '', anchor_text).strip()

    rect, page_num = resolve_anchor_rect(doc, evidence_clean, allowed_pages)
    if not rect:
        print(f"  No location found for evidence: '{evidence_clean[:50]}...' - marks {score_text} not placed")
        return False

    page_idx = page_num - 1
    line_y = rect.y0
    if page_idx in placed_lines_per_page and any(abs(line_y - p) <= CONFIG['y_tolerance'] for p in placed_lines_per_page[page_idx]):
        print(f"  Skipped duplicate on line y={line_y:.1f} page {page_num} for marks {score_text}")
        return False

    # Underline the matched evidence
    page = doc[page_num - 1]
    page.draw_line(
        (rect.x0, rect.y1 + 2),
        (rect.x1, rect.y1 + 2),
        color=CONFIG['underline_color'],
        width=1.5
    )

    # Place score
    page.insert_text(
        (rect.x0 + CONFIG['criterion_score_offset_x'],
         rect.y0 + CONFIG['criterion_score_offset_y']),
        score_text,
        fontsize=CONFIG['criterion_score_fontsize'],
        color=CONFIG['criterion_score_color']
    )

    # Logging
    print(f"  Awarded {score_text} near evidence '{evidence_clean[:50]}...' on page {page_num} (y={line_y:.1f})")

    placed_lines_per_page.setdefault(page_idx, set()).add(line_y)
    return True

def add_main_score(doc, q_num, score_text, allowed_pages):
    placed = False
    for page_num in allowed_pages:
        page = doc[page_num - 1]
        instances = page.search_for(q_num)  # exact heading match
        if instances:
            rect = instances[0]
            # Place above the heading with reliable offset
            x = rect.x0 - CONFIG['main_score_offset_x']
            y = rect.y0 + CONFIG['main_score_offset_y'] - 20  # extra upward to avoid overlap
            page.insert_text(
                (x, y),
                score_text,
                fontsize=CONFIG['main_score_fontsize'],
                color=CONFIG['main_score_color']
            )
            print(f"  Main score '{score_text}' placed near '{q_num}' on page {page_num} (y={rect.y0 + CONFIG['main_score_offset_y'] - 20:.1f})")
            placed = True
            break

    if not placed:
        print(f"  Main heading NOT found on student pages: '{q_num}' - main score not placed")

    return placed

# ==================== MAIN FUNCTION ====================

def annotate_pdf(input_pdf_path, output_dir, student_name, grades_csv_path, student_pages=None):
    student_key = student_name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pdf = os.path.join(output_dir, student_key, f"{student_key}_annotated_{timestamp}.pdf")
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    df = pd.read_csv(grades_csv_path)
    doc = fitz.open(input_pdf_path)

    allowed_pages = student_pages or list(range(1, len(doc) + 1))

    # TOTAL SCORE - only place main score text (no popup)
    main = df[df['criterion'] == 'TOTAL SCORE'].iloc[0]
    main_score_text = f"{main['marks_awarded']}/{main['max_possible']}"
    add_main_score(doc, main['question_number'], main_score_text, allowed_pages)

    # PER CRITERION
    placed_lines_per_page = {i - 1: set() for i in allowed_pages}
    for _, row in df.iterrows():
        if row['criterion'] == 'TOTAL SCORE' or pd.isna(row['evidence']):
            continue
        if row['marks_awarded'] > 0:
            place_score_near_anchor(
                doc,
                row['evidence'],
                str(row['marks_awarded']),
                allowed_pages,
                placed_lines_per_page
            )

    # COMMENTS
    comments = re.split(r';\s*(?=[^"]*(?:[^"]*")*[^"]*$)', str(main.get('comments_summary', '')))
    for c in comments:
        if '→' in c:
            add_popup_for_comment(doc, c.strip().strip('"'), allowed_pages)

    doc.save(output_pdf, garbage=4, deflate=True, clean=True)
    doc.close()
    return True, output_pdf