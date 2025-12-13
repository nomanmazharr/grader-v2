import ast
import pandas as pd
import fitz
import os
import re
from datetime import datetime

# ==================== CONFIG ====================
CONFIG = {
    'score_offset_x': 30,
    'score_offset_y': 4,
    'score_fontsize': 12,
    'score_color': (0, 0, 1),
    'underline_color': (0, 0.7, 0),
    'comment_color': (1, 0, 0),
}

# ==================== HELPERS (WITH DEBUG PRINTS ONLY) ====================

def extract_number_after_equal(phrase: str):
    match = re.search(r'=([^=]+?)(?:\s|$)', phrase)
    if match:
        return match.group(1).strip()
    nums = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', phrase.replace(' ', ''))
    return nums[-1] if nums else None

def clean_anchor(comment_line: str):
    if '→' not in comment_line:
        return None
    text = comment_line.split('→')[0].strip().strip('"\'')
    num = extract_number_after_equal(text)
    return num if num else " ".join(text.split()[:8])

# Global sets to track completed items across the entire PDF
UNDERLINED_GLOBAL = set()  # Tracks unique correct phrases (keys)
SCORED_HEADINGS = set()    # Tracks unique headings scored
PLACED_FEEDBACK_GLOBAL = set()  # Tracks unique anchors for feedback

def add_popup_near_text(doc, anchor_text, comment):
    if not anchor_text or len(anchor_text) < 2:
        print(f"Skipping empty anchor")
        return False
    
    placed = False
    for page_num in range(len(doc)):
        page = doc[page_num]
        instances = page.search_for(anchor_text)
        if instances:
            rect = instances[0]  # First instance only
            point = fitz.Point(rect.x1 + 6, rect.y0 + 2)
            annot = page.add_text_annot(point, "", icon="Note")
            annot.set_colors(stroke=CONFIG['comment_color'])
            annot.set_opacity(0.9)
            annot.set_info(content=comment.strip(), title="Feedback")
            annot.update()
            print(f"Feedback placed near → '{anchor_text}' on page {page_num + 1}")
            placed = True
            break  # Stop after placing on first found page
    
    if not placed:
        print(f"Anchor NOT found anywhere → '{anchor_text}' → comment NOT placed")
    
    return placed

def underline_correct_phrase(doc, phrase):
    global UNDERLINED_GLOBAL
    
    key = re.sub(r'[^\w]', '', phrase.lower())
    if key in UNDERLINED_GLOBAL:
        print(f"Already underlined → '{phrase}' → skipping")
        return False
    
    found_and_underlined = False
    for page_num in range(len(doc)):
        page = doc[page_num]
        words = page.get_text("words")  # For token matching and coords
        
        target_rect = None
        underline_only_number = False
        
        # Step 1: Try full phrase match
        instances = page.search_for(phrase)
        if instances:
            target_rect = instances[0]
            print(f"       Underlined full phrase → '{phrase}' on page {page_num + 1}")
        
        # Step 2: If not found, split into text and numeric parts
        if not target_rect:
            num = extract_number_after_equal(phrase)
            if num:
                text_part = phrase.replace(num, '').strip()  # Remove number to get text
                if text_part:  # If there's text before number
                    # Search for text_part and num separately
                    text_instances = page.search_for(text_part)
                    num_instances = page.search_for(num)
                    
                    if text_instances and num_instances:
                        text_rect = text_instances[0]
                        num_rect = num_instances[0]
                        
                        # Check if on same line (similar y coordinates, allow ~5pt tolerance)
                        if abs(text_rect.y1 - num_rect.y1) <= 5:
                            target_rect = num_rect
                            underline_only_number = True
                            print(f"       Underlined number '{num}' (same line as text '{text_part}') on page {page_num + 1}")
                        else:
                            print(f"       Text and number found but not on same line for '{phrase}' on page {page_num + 1}")
                
                # Step 3: If still no target from split, or no num, try text_part only (or full if no num)
                if not target_rect:
                    search_text = text_part if 'text_part' in locals() else phrase
                    text_instances = page.search_for(search_text)
                    if text_instances:
                        target_rect = text_instances[0]
                        print(f"       Underlined text only → '{search_text}' from '{phrase}' on page {page_num + 1}")
        
        # Step 4: Fallback token matching for whole phrase if still not found
        if not target_rect:
            tokens = re.split(r'\s+', phrase)
            if len(tokens) >= 2:
                for i in range(len(words) - len(tokens) + 1):
                    match = True
                    rects = []
                    y1 = None
                    for j, token in enumerate(tokens):
                        word_text = words[i+j][4]
                        if not re.search(re.escape(token), word_text, re.I):
                            match = False
                            break
                        rects.append(fitz.Rect(words[i+j][:4]))
                        if y1 is None:
                            y1 = words[i+j][3]
                        elif abs(y1 - words[i+j][3]) > 5:
                            match = False
                            break
                    if match and rects:
                        # Combine all rects
                        combined = rects[0]
                        for r in rects[1:]:
                            combined = combined.include_rect(r)
                        target_rect = combined
                        print(f"       Underlined via token matching → '{phrase}' on page {page_num + 1}")
                        break
        
        # Final: Underline if found on this page
        if target_rect:
            # Try to find and underline ONLY the number within the matched area if applicable
            number_rect = None
            num = extract_number_after_equal(phrase)
            if num and not underline_only_number:
                num_in_region = page.search_for(num, clip=target_rect)
                if num_in_region:
                    number_rect = num_in_region[0]
                    print(f"       Underlined CORRECT NUMBER → '{num}' (proof of correct work) on page {page_num + 1}")
            
            underline_this = number_rect or target_rect
            
            page.draw_line(
                (underline_this.x0, underline_this.y1 + 2),
                (underline_this.x1, underline_this.y1 + 2),
                color=CONFIG['underline_color'],
                width=2.5
            )
            
            UNDERLINED_GLOBAL.add(key)
            found_and_underlined = True
            break  # Stop after underlining on first found page
    
    if not found_and_underlined:
        print(f"       NOT found anywhere → '{phrase}'")
    
    return found_and_underlined

def add_score_next_to_heading(doc, heading: str, score_text: str):
    global SCORED_HEADINGS
    
    search_text = heading.strip().rstrip(':').strip()
    if search_text in SCORED_HEADINGS:
        print(f"       Score already placed for '{search_text}' → skipping")
        return False
    
    placed = False
    for page_num in range(len(doc)):
        page = doc[page_num]
        instances = page.search_for(search_text)
        if instances:
            rect = instances[0]  # First instance
            x = rect.x0 - CONFIG['score_offset_x']
            y = rect.y0 + CONFIG['score_offset_y']
            if x + 100 > page.rect.width:
                x = page.rect.width - 100
            
            page.insert_text((x, y), score_text,
                             fontsize=CONFIG['score_fontsize'],
                             color=CONFIG['score_color'])
            
            SCORED_HEADINGS.add(search_text)
            print(f"       Score placed → '{search_text}' : {score_text} on page {page_num + 1}")
            placed = True
            break  # Stop after placing on first found page
    
    if not placed:
        print(f"       Heading NOT found anywhere → '{search_text}' → score NOT placed")
    
    return placed

# ==================== MAIN FUNCTION (OPTIMIZED FLOW: ITEM-BY-ITEM ACROSS ALL PAGES) ====================

def annotate_pdf(input_pdf_path, output_dir, student_name, grades_csv_path, student_pages=None):
    student_key = student_name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pdf = os.path.join(output_dir, student_key, f"{student_key}_annotated_{timestamp}.pdf")
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    print("\n" + "█" * 80)
    print(f"  STARTING ANNOTATION — {student_name.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 80)

    df = pd.read_csv(grades_csv_path)
    print(f"  Loaded {len(df)} question(s) from CSV")

    score_dict = {
        str(row['question_number']): f"{row['score']}/{row['total_marks']}"
        for _, row in df.iterrows()
    }

    # Collect all unique phrases and comments once
    correct_words_entries = df['correct_words'].dropna().tolist()
    all_phrases = []
    for entry in correct_words_entries:
        try:
            phrases = ast.literal_eval(entry)
            if isinstance(phrases, list):
                all_phrases.extend([p.strip() for p in phrases if p.strip()])
        except: pass
    all_phrases = list(set(all_phrases))  # Dedupe

    comment_entries = df['comment'].dropna().tolist()
    all_comments = []
    for entry in comment_entries:
        try:
            comments = ast.literal_eval(entry)
            if isinstance(comments, list):
                all_comments.extend(comments)
        except: pass

    print(f"  Scores to place     : {len(score_dict)}")
    print(f"  Unique correct phrases: {len(all_phrases)}")
    print(f"  Feedback comments   : {len(all_comments)}")

    doc = fitz.open(input_pdf_path)
    # If student_pages specified, limit to those (1-indexed to 0-indexed)
    if student_pages:
        print(f"\n  Limiting to student pages: {', '.join(map(str, student_pages))}\n")
    else:
        print(f"\n  Processing all pages\n")

    # Process scores item-by-item across pages
    scores_placed = 0
    for heading, score in score_dict.items():
        if add_score_next_to_heading(doc, heading, score):
            scores_placed += 1

    # Process underlines item-by-item across pages
    underlines_placed = 0
    for phrase in all_phrases:
        if underline_correct_phrase(doc, phrase):
            underlines_placed += 1

    # Process feedback comments item-by-item across pages
    feedbacks_placed = 0
    for line in all_comments:
        if '→' not in line: continue
        anchor = clean_anchor(line)
        if not anchor:
            print(f"       Skipping comment with no valid anchor: {line}")
            continue
        if anchor in PLACED_FEEDBACK_GLOBAL:
            print(f"       Feedback already placed for anchor '{anchor}' → skipping")
            continue
        feedback = line.split('→', 1)[1].strip()
        if add_popup_near_text(doc, anchor, feedback):
            PLACED_FEEDBACK_GLOBAL.add(anchor)
            feedbacks_placed += 1

    doc.save(output_pdf, garbage=4, deflate=True, clean=True)
    doc.close()

    print("█" * 80)
    print(f"  SUCCESS! Saved:")
    print(f"  → {output_pdf}")
    print(f"  Scores placed: {scores_placed}")
    print(f"  Underlines placed: {underlines_placed}")
    print(f"  Feedbacks placed: {feedbacks_placed}")
    print(f"  Finished: {datetime.now().strftime('%H:%M:%S')}")
    print("█" * 80 + "\n")

    return True, output_pdf
