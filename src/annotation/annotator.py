# ==================== MAIN ANNOTATION ORCHESTRATION ====================
#
# annotate_pdf() is the public entry point.  It:
#   1. Loads the grading document and student PDF.
#   2. Detects per-page Y boundaries for the target question.
#   3. In holistic mode:  underlines + tick-marks each correct point,
#                         then places sub-question scores near student labels.
#   4. In standard mode: places criterion scores near evidence anchors.
#   5. Places feedback comment popups.
#   6. Saves the annotated PDF.
#
# Bug fix included:
#   FIX-3: underline is drawn at tick_rect (key_phrase / evidence),
#           guaranteeing tick mark and underline are always co-located on
#           the correct student answer — never on a separate line.

import json
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import fitz
from bson import ObjectId

from logging_config import logger
from database.mongodb import get_collection

from .annotator_ocr import _init_ocr_cache, _page_search, _page_text, _page_words
from .annotator_text import (
    _normalize_text_for_match, _strip_llm_artifacts,
    _tokenize, _line_key, _build_anchor_variations, _build_candidate_fragments,
)
from .annotator_rect import _draw_underline_for_rect
from .annotator_match import resolve_anchor_rect, _rank_pages_for_anchor
from .annotator_draw import (
    _safe_float, _fmt_mark_value, _place_score_label,
    _place_ticks, place_score_near_anchor, add_main_score, add_popup_for_comment,
    place_not_required_marker, compute_subq_y_bounds, _strip_subq_prefix,
)


# ── Main annotation function ───────────────────────────────────────────────────

def annotate_pdf(
    input_pdf_path: str,
    output_dir: str,
    student_name: str,
    grades_id: Optional[str] = None,
    grades_doc: Optional[dict] = None,
    student_pages: Optional[List[int]] = None,
) -> Tuple[bool, str]:
    """Annotate a student PDF with scores, underlines, tick marks, and feedback.

    Returns (success, output_pdf_path).
    """
    try:
        student_key = student_name.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = os.path.join(
            output_dir, student_key, f"{student_key}_annotated_{timestamp}.pdf"
        )
        mapping_json = os.path.join(
            output_dir, student_key, f"{student_key}_mapping_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        # ── Load grading document ──────────────────────────────────────────────
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
        displayed_count = len(
            [it for it in (grades_doc.get('breakdown', []) or [])
             if float(it.get('marks_awarded', 0) or 0) > 0]
        )
        logger.info(
            f"Annotating {student_name} Q{grades_doc.get('question_number', '?')} "
            f"({displayed_count} displayed / {total_breakdown} total criteria)"
        )

        doc = fitz.open(input_pdf_path)
        allowed_pages = student_pages or list(range(1, len(doc) + 1))

        # OCR must be initialised BEFORE boundary detection so _page_search
        # can read text from scanned pages when locating question headings.
        _init_ocr_cache(doc, allowed_pages)

        # ── Fetch student assignment doc ───────────────────────────────────────
        page_token_sets: dict[int, set[str]] = {}
        student_page_texts: dict[int, str] = {}
        student_question_heading: Optional[str] = None
        try:
            student_answer_id = grades_doc.get('student_answer_id')
            if student_answer_id:
                s_coll = get_collection("student_assignments")
                s_doc = s_coll.find_one({"_id": ObjectId(student_answer_id)})
                if s_doc:
                    student_question_heading = s_doc.get("question_heading_text") or None
                    if isinstance(s_doc.get("page_texts"), list):
                        for item in s_doc.get("page_texts") or []:
                            try:
                                pp = int(item.get("page"))
                                tt = str(item.get("text") or "")
                            except Exception:
                                continue
                            if pp in allowed_pages and tt:
                                student_page_texts[pp] = tt
        except Exception:
            student_page_texts = {}

        # ── Detect per-page Y boundaries for the target question ───────────────
        # Prevents annotations from bleeding into adjacent questions on shared pages.
        q_num = str(grades_doc.get('question_number', '')).strip()
        min_y_per_page: dict[int, float] = {}
        max_y_per_page: dict[int, float] = {}

        if q_num:
            q_heading_patterns: list[str] = []
            if student_question_heading:
                q_heading_patterns.append(student_question_heading)
            q_heading_patterns += [
                f"ANSWER {q_num}", f"Answer {q_num}",
                f"ANSWER: {q_num}", f"Answer: {q_num}",
                f"Q-{q_num.zfill(2)}", f"Q-{q_num}",
                f"Q.{q_num}", f"Q{q_num} ",
                f"Question {q_num}",
            ]

            for page_num in allowed_pages:
                page = doc[page_num - 1]

                # Find this question's heading → min_y
                for pattern in q_heading_patterns:
                    instances = _page_search(page, pattern)
                    for inst in (instances or []):
                        if inst.x0 < 250:
                            min_y_per_page[page_num] = inst.y0 - 5
                            logger.debug(
                                f"Q{q_num} heading found on page {page_num} at y={inst.y0:.0f}"
                            )
                            break
                    if page_num in min_y_per_page:
                        break

                # Find next question's heading → max_y
                # STRICT criteria to avoid false positives from inline text:
                #   • x0 < 100  — heading must be at the far left margin
                #   • y0 must be genuinely below the current question's start
                # We deliberately exclude the ambiguous "Question N" template
                # because students write it inline ("As in Question 3…").
                # We also only use the student_prefix template when the prefix
                # is specific enough (len ≥ 2) — a bare prefix like "" would
                # generate template "3" matching every digit on the page.
                current_min = min_y_per_page.get(page_num, 0)
                best_next_y = float('inf')

                try:
                    q_int = int(q_num)
                    neighbor_nums = [n for n in range(1, 11) if n != q_int]
                except ValueError:
                    neighbor_nums = []

                student_prefix: Optional[str] = None
                student_suffix: Optional[str] = None
                if student_question_heading and q_num in student_question_heading:
                    idx = student_question_heading.find(q_num)
                    student_prefix = student_question_heading[:idx]
                    student_suffix = student_question_heading[idx + len(q_num):]

                for n in neighbor_nums:
                    neighbour_templates = [
                        f"ANSWER {n}", f"Answer {n}",
                        f"ANSWER: {n}", f"Answer: {n}",
                        f"Q-{str(n).zfill(2)}", f"Q.{n}", f"Q{n} ",
                    ]
                    # Only include the student-format template when the prefix is
                    # non-trivially specific (prevents bare-digit templates like "3").
                    if student_prefix and len(student_prefix.strip()) >= 2:
                        neighbour_templates.insert(
                            0, f"{student_prefix}{n}{student_suffix}"
                        )
                    for tmpl in neighbour_templates:
                        hits = _page_search(page, tmpl)
                        for hit in (hits or []):
                            # x0 < 100: only accept hits at the very left margin
                            if hit.x0 < 100 and hit.y0 > current_min + 30:
                                if hit.y0 < best_next_y:
                                    best_next_y = hit.y0
                                    logger.debug(
                                        f"  Neighbor Q{n} '{tmpl}' found on page {page_num}"
                                        f" at y={hit.y0:.0f} x={hit.x0:.0f}"
                                    )

                if best_next_y < float('inf'):
                    max_y_per_page[page_num] = best_next_y - 5
                    logger.debug(
                        f"  Max Y for Q{q_num} on page {page_num}: {best_next_y:.0f}"
                    )

        # ── Build page token sets (for page ranking) ───────────────────────────
        for p in allowed_pages:
            page_text = student_page_texts.get(p, "")
            if not page_text:
                try:
                    page_text = _page_text(doc[p - 1])
                except Exception:
                    page_text = ""
            page_token_sets[p] = set(_tokenize(page_text or ""))

        # ── Shared annotation state ────────────────────────────────────────────
        comment_page_y: dict[int, float] = {}
        comment_used_y: dict[int, list[float]] = {}
        ocr_textpages: dict[int, object] = {}

        placed_lines_per_page = {i - 1: [] for i in allowed_pages}
        placed_marks: set = set()
        line_score_accumulator: dict = {}
        unplaced_items: list = []

        annotation_mapping = {
            'total_score_placed': False,
            'criterion_scores_placed': 0,
            'total_criteria': len(grades_doc.get('breakdown', [])),
            'total_breakdown': len(grades_doc.get('breakdown', []) or []),
            'comments_placed': 0,
            'unplaced_items': [],
            'allowed_pages': allowed_pages,
        }

        # ── Total score label ──────────────────────────────────────────────────
        main_score_text = (
            f"{_fmt_mark_value(grades_doc['total_marks_awarded'])}/"
            f"{_fmt_mark_value(grades_doc['total_max_possible'])}"
        )
        if add_main_score(
            doc, str(grades_doc.get('question_number', '')),
            main_score_text, allowed_pages,
            student_heading_text=student_question_heading,
        ):
            annotation_mapping['total_score_placed'] = True

        # ── Per-criterion annotation ───────────────────────────────────────────
        breakdown = grades_doc.get('breakdown', [])
        is_holistic = grades_doc.get('holistic_grading', False)
        displayed_breakdown = [
            it for it in breakdown
            if float(it.get('marks_awarded', 0) or 0) > 0
            or "Marks given above" in str(it.get('reason', '') or '')
            or "Marks given below" in str(it.get('reason', '') or '')
        ]
        annotation_mapping['total_criteria'] = len(displayed_breakdown)
        criteria_count = 0

        def _is_informative_anchor(text: str) -> bool:
            """Return True when *text* is distinctive enough to anchor to."""
            norm = _normalize_text_for_match(text)
            if len(norm) < 6:
                return False
            key = re.sub(r"[^a-z0-9]+", "", norm)
            if re.fullmatch(r"20x\d", key) or re.fullmatch(r"20\d{2}", key):
                return False
            if re.search(r"\d+\s*/\s*\d+", text):
                return True
            if re.search(r"\d", text) and len(norm) >= 4:
                return True
            return len(_tokenize(text)) >= 2

        # ══════════════════════════════════════════════════════════════════════
        # HOLISTIC GRADING MODE
        # Each breakdown item = one sub-question.
        # _correct_points_with_marks = [{text, key_phrase, marks}, ...]
        # Strategy:
        #   Step 1 — underline each correct point AND tick-mark it, co-located.
        #   Step 2 — place sub-question score near the student's sub-q label.
        # ══════════════════════════════════════════════════════════════════════
        if is_holistic:
            logger.info(
                f"Holistic annotation mode: {len(displayed_breakdown)} "
                f"sub-question(s) with marks"
            )

            for idx, item in enumerate(displayed_breakdown, 1):
                marks = float(item.get('marks_awarded', 0))
                max_marks = float(item.get('max_possible', 0) or 0)
                sub_q = item.get('_sub_question', '')
                student_label = item.get('_student_label', '')

                points_with_marks = item.get('_correct_points_with_marks', [])

                # Fallback: build from flat evidence with even mark distribution
                if not points_with_marks:
                    raw_evidence = item.get('evidence_list')
                    if isinstance(raw_evidence, list):
                        ev_texts = [
                            str(x).strip() for x in raw_evidence
                            if x is not None and str(x).strip()
                        ]
                    else:
                        evidence = (item.get('evidence', '') or '').strip()
                        ev_texts = [
                            p.strip() for p in re.split(r"\s*;\s*", evidence)
                            if p and p.strip()
                        ]
                    if ev_texts:
                        per_point = (
                            round(marks / len(ev_texts) / 0.5) * 0.5
                            if ev_texts else 0.5
                        )
                        per_point = max(per_point, 0.5)
                        points_with_marks = [
                            {"text": t, "marks": per_point} for t in ev_texts
                        ]

                # ── Step 1: Underline + tick each correct point ────────────────
                # Two-pass approach:
                #   Pass A — resolve every correct point to a PDF rect.
                #   Pass B — draw 1 underline + 1 tick per found evidence point.
                #            One correct_point = one model-answer match = one tick.
                logger.info(
                    f"    Processing {len(points_with_marks)} correct points "
                    f"for sub-question {sub_q}"
                )

                # Pass A: resolve all evidence positions.
                # Strategy: try key_phrase FIRST (it's the precise tick target),
                # only fall back to full text if key_phrase search fails.
                #
                # Duplicate handling: dedup is PER-PHRASE, not per-line.
                #   • "qualified opinion" appearing twice → walk to next occurrence.
                #   • "Cost overruns" + "result in delays" on the same line → both
                #     should tick (different phrases, separate dedup namespaces).
                # placed_per_phrase maps phrase → set of line_keys already ticked
                # for THAT phrase. placed_tick_keys is a global safety-net for
                # exact (page, x, y) collisions across all points.
                found_evidence: list[tuple] = []  # (page_obj, tick_rect, pt_marks)
                placed_per_phrase: dict[str, set] = {}  # phrase → line_keys used
                placed_tick_keys: set[tuple] = set()  # global exact-position dedup
                for pt in points_with_marks:
                    pt_text = pt.get("text", "").strip()
                    pt_marks = float(pt.get("marks", 0.5) or 0.5)
                    key_phrase = pt.get("key_phrase", "").strip()
                    if not pt_text:
                        continue
                    if not _is_informative_anchor(pt_text):
                        logger.info(f"    Skipped (not informative): '{pt_text[:60]}'")
                        continue

                    evidence_clean = _strip_llm_artifacts(
                        pt_text.replace("\u2026", " ").replace("...", " ").replace('|', ' ')
                    )
                    if not evidence_clean:
                        continue

                    tick_rect = None
                    found_page_num = -1
                    ev_page = None

                    # Strategy 1: Search for key_phrase directly in the PDF.
                    # Use per-phrase dedup so different phrases on the same line
                    # don't block each other, but a repeated phrase walks past
                    # already-ticked occurrences.
                    if key_phrase and len(key_phrase) >= 3:
                        phrase_key = key_phrase.lower().strip()
                        phrase_marks = placed_per_phrase.setdefault(phrase_key, set())
                        for candidate in _build_anchor_variations(key_phrase):
                            tick_rect, found_page_num = resolve_anchor_rect(
                                doc, candidate, allowed_pages,
                                placed_marks=phrase_marks,
                                skip_duplicates=True,
                                expand_to_line=False,
                                page_token_sets=page_token_sets,
                                redirect_headings=False,
                                use_number_first=False,
                                min_y_per_page=min_y_per_page,
                                max_y_per_page=max_y_per_page,
                            )
                            if tick_rect and found_page_num > 0:
                                logger.info(f"      Tick via key_phrase: '{key_phrase}'")
                                break

                    # Strategy 2: Fall back to full text search if key_phrase failed.
                    # Per-phrase dedup keyed on the evidence text.
                    if not tick_rect or found_page_num <= 0:
                        text_key = pt_text.lower().strip()
                        text_marks = placed_per_phrase.setdefault(text_key, set())
                        candidate_fragments = _build_candidate_fragments(evidence_clean)
                        for fragment in candidate_fragments:
                            for candidate in _build_anchor_variations(fragment):
                                tick_rect, found_page_num = resolve_anchor_rect(
                                    doc, candidate, allowed_pages,
                                    placed_marks=text_marks,
                                    skip_duplicates=True,
                                    expand_to_line=False,
                                    page_token_sets=page_token_sets,
                                    redirect_headings=False,
                                    use_number_first=False,
                                    min_y_per_page=min_y_per_page,
                                    max_y_per_page=max_y_per_page,
                                )
                                if tick_rect and found_page_num > 0:
                                    break
                            if tick_rect and found_page_num > 0:
                                break

                    if tick_rect and found_page_num > 0:
                        # Refine tick_rect to the exact key_phrase position.
                        # When the full-text fallback returned a line-level rect,
                        # this narrows it to the specific key_phrase within that line
                        # so the tick + underline land on the right words.
                        if key_phrase and len(key_phrase) >= 3:
                            _refine_page = doc[found_page_num - 1]
                            _clip = fitz.Rect(
                                0, max(tick_rect.y0 - 3, 0),
                                _refine_page.rect.width,
                                min(tick_rect.y1 + 3, _refine_page.rect.height),
                            )
                            kp_words = key_phrase.split()
                            for _end in range(len(kp_words), max(1, len(kp_words) - 2) - 1, -1):
                                _sub = " ".join(kp_words[:_end])
                                if len(_sub) < 3:
                                    continue
                                _kp_hits = _page_search(_refine_page, _sub, clip=_clip)
                                if _kp_hits:
                                    tick_rect = _kp_hits[0]
                                    break

                        # Safety-net dedup for exact (page, x, y) collisions across
                        # all phrases. resolve_anchor_rect already walks past
                        # already-ticked occurrences of the same phrase via
                        # placed_per_phrase; this guards against pathological
                        # cross-phrase exact-rect collisions.
                        tick_key = (found_page_num, round(tick_rect.y0, 0), round(tick_rect.x0, 0))
                        if tick_key in placed_tick_keys:
                            logger.info(f"    ⊘ Duplicate tick position, skipping: '{key_phrase or pt_text[:40]}'")
                            continue
                        placed_tick_keys.add(tick_key)

                        # Record this line under the phrase that matched, so a
                        # repeat of the SAME phrase walks to the next occurrence.
                        line_mark = _line_key(found_page_num, tick_rect.y0)
                        if key_phrase and len(key_phrase) >= 3:
                            placed_per_phrase.setdefault(
                                key_phrase.lower().strip(), set()
                            ).add(line_mark)
                        else:
                            placed_per_phrase.setdefault(
                                pt_text.lower().strip(), set()
                            ).add(line_mark)

                        ev_page = doc[found_page_num - 1]
                        found_evidence.append((ev_page, tick_rect, pt_marks))
                        logger.info(f"    ✓ Found: '{key_phrase or pt_text[:60]}'")
                    else:
                        logger.info(f"    ✗ Not found in PDF: '{key_phrase or pt_text[:60]}'")

                # Pass B: draw underlines + place exactly 1 tick per found evidence point.
                # One correct_point = one matched model-answer point = one tick mark.
                # Track how many ticks have already been placed at each rect position
                # so that same-line fallback ticks are offset rather than stacked.
                if found_evidence:
                    tick_offset_per_rect: dict[tuple, int] = {}  # rect_key → ticks_placed
                    for ev_idx, (ev_page, ev_rect, ev_marks) in enumerate(found_evidence):
                        _draw_underline_for_rect(ev_page, ev_rect, phrase_only=True)
                        rect_key = (round(ev_rect.x0), round(ev_rect.y0))
                        offset = tick_offset_per_rect.get(rect_key, 0)
                        # Shift the rect right by (tick_width+gap) × offset so stacked
                        # ticks from the grade.py fallback split appear side by side.
                        tick_w = 10  # approx tick_width + gap from _place_ticks
                        shifted_rect = fitz.Rect(
                            ev_rect.x0 + offset * tick_w,
                            ev_rect.y0,
                            ev_rect.x1 + offset * tick_w,
                            ev_rect.y1,
                        )
                        _place_ticks(ev_page, shifted_rect, 1)
                        tick_offset_per_rect[rect_key] = offset + 1
                        logger.info(
                            f"    ✓ Underlined + 1 tick "
                            f"(evidence {ev_idx + 1}/{len(found_evidence)})"
                        )

                    logger.info(
                        f"    Sub-Q {sub_q}: {len(found_evidence)} tick(s) placed "
                        f"({len(found_evidence)}/{len(points_with_marks)} "
                        f"evidence points found)"
                    )

                # ── Step 1b: Mark off-topic content as "Not required" ──────────
                # No marks affected — purely instructional feedback for the student.
                # Uses the same anchor-resolution flow but with its own per-sub-q
                # line tracker so a "Not required" marker doesn't collide with ticks.
                not_required_points = item.get('_not_required_points', []) or []
                if not_required_points:
                    logger.info(
                        f"    Processing {len(not_required_points)} 'Not required' "
                        f"point(s) for sub-question {sub_q}"
                    )
                    nr_local_marks: set = set()
                    for nr in not_required_points:
                        nr_text = str(nr.get("text", "")).strip()
                        nr_kp = str(nr.get("key_phrase", "")).strip()
                        nr_reason = str(nr.get("reason", "")).strip()
                        if not nr_text:
                            continue

                        nr_clean = _strip_llm_artifacts(
                            nr_text.replace("…", " ").replace("...", " ").replace('|', ' ')
                        )
                        if not nr_clean:
                            continue

                        nr_rect = None
                        nr_page_num = -1

                        # Strategy 1: key_phrase match (precise).
                        if nr_kp and len(nr_kp) >= 3:
                            for candidate in _build_anchor_variations(nr_kp):
                                nr_rect, nr_page_num = resolve_anchor_rect(
                                    doc, candidate, allowed_pages,
                                    placed_marks=nr_local_marks,
                                    skip_duplicates=True,
                                    expand_to_line=True,  # full line for strikethrough
                                    page_token_sets=page_token_sets,
                                    redirect_headings=False,
                                    use_number_first=False,
                                    min_y_per_page=min_y_per_page,
                                    max_y_per_page=max_y_per_page,
                                )
                                if nr_rect and nr_page_num > 0:
                                    break

                        # Strategy 2: full text fallback.
                        if not nr_rect or nr_page_num <= 0:
                            for fragment in _build_candidate_fragments(nr_clean):
                                for candidate in _build_anchor_variations(fragment):
                                    nr_rect, nr_page_num = resolve_anchor_rect(
                                        doc, candidate, allowed_pages,
                                        placed_marks=nr_local_marks,
                                        skip_duplicates=True,
                                        expand_to_line=True,
                                        page_token_sets=page_token_sets,
                                        redirect_headings=False,
                                        use_number_first=False,
                                        min_y_per_page=min_y_per_page,
                                        max_y_per_page=max_y_per_page,
                                    )
                                    if nr_rect and nr_page_num > 0:
                                        break
                                if nr_rect and nr_page_num > 0:
                                    break

                        if nr_rect and nr_page_num > 0:
                            nr_page = doc[nr_page_num - 1]
                            place_not_required_marker(nr_page, nr_rect, nr_reason)
                            nr_local_marks.add(_line_key(nr_page_num, nr_rect.y0))
                            logger.info(f"    ⚑ Not required: '{nr_kp or nr_text[:60]}'")
                        else:
                            logger.info(f"    ✗ NR not found in PDF: '{nr_kp or nr_text[:60]}'")

                # ── Step 2: Place sub-question score near student's label ───────
                # FIX-1 (via draw_underline=False): heading/label lines are never
                # underlined — only the score mark is placed beside them.
                # ORDER: prefer the sub-question NUMBER lookup ("4.1", "4.2", …)
                # over the student_label, because student_label is often a
                # generic single character ("A", "B", "a)") that matches dozens
                # of false positions in the PDF — including the page header.
                score_text = (
                    f"{_fmt_mark_value(marks)}/{_fmt_mark_value(max_marks)}"
                )
                score_placed = False

                if sub_q:
                    for pattern in [
                        sub_q, f"({sub_q})", f"{sub_q})",
                        f"{sub_q}.", f"{sub_q}:",
                    ]:
                        score_placed = place_score_near_anchor(
                            doc, pattern, score_text,
                            allowed_pages, placed_lines_per_page, placed_marks,
                            unplaced_items,
                            page_token_sets=page_token_sets,
                            line_score_accumulator=None,
                            min_y_per_page=min_y_per_page,
                            max_y_per_page=max_y_per_page,
                            draw_underline=False,  # FIX-1
                        )
                        if score_placed:
                            break

                # Fallback to student_label ONLY when it is specific enough.
                # A label without a digit is too ambiguous: "A", "a", "(a)",
                # "(b)", "(i)", "(ii)" all match dozens of arbitrary positions
                # in the PDF (every parenthesised letter, every standalone
                # capital, etc.) and put the score in the wrong region.
                # Requiring a digit keeps "4.1", "1.1", "a.1" but rejects all
                # the letter-only labels that have caused mis-placement.
                def _label_is_specific(label: str) -> bool:
                    s = (label or "").strip()
                    if len(s) < 3:
                        return False
                    return bool(re.search(r"\d", s))

                if not score_placed and _label_is_specific(student_label):
                    score_placed = place_score_near_anchor(
                        doc, student_label.strip(), score_text,
                        allowed_pages, placed_lines_per_page, placed_marks,
                        unplaced_items,
                        page_token_sets=page_token_sets,
                        line_score_accumulator=None,
                        min_y_per_page=min_y_per_page,
                        max_y_per_page=max_y_per_page,
                        draw_underline=False,  # FIX-1: do not underline the sub-q heading
                    )

                # Fallback: anchor to first evidence fragment
                if not score_placed and points_with_marks:
                    for pt in points_with_marks[:2]:
                        pt_text = pt.get("text", "").strip()
                        if pt_text and _is_informative_anchor(pt_text):
                            score_placed = place_score_near_anchor(
                                doc, pt_text, score_text,
                                allowed_pages, placed_lines_per_page, placed_marks,
                                unplaced_items,
                                page_token_sets=page_token_sets,
                                line_score_accumulator=None,
                                min_y_per_page=min_y_per_page,
                                max_y_per_page=max_y_per_page,
                                draw_underline=False,  # FIX-1: evidence already underlined above
                            )
                            if score_placed:
                                break

                if score_placed:
                    annotation_mapping['criterion_scores_placed'] += 1
                    logger.info(f"  Sub-question {sub_q}: {score_text} placed")
                else:
                    logger.debug(f"  Sub-question {sub_q}: {score_text} NOT placed")
                criteria_count += 1

            logger.info(
                f"✓ Placed {annotation_mapping['criterion_scores_placed']} of "
                f"{criteria_count} sub-question scores"
            )

        # ══════════════════════════════════════════════════════════════════════
        # STANDARD PER-CRITERION ANNOTATION MODE
        # ══════════════════════════════════════════════════════════════════════
        else:
            for idx, item in enumerate(displayed_breakdown, 1):
                marks = float(item.get('marks_awarded', 0))
                raw_evidence = item.get('evidence_list')
                if isinstance(raw_evidence, list):
                    evidence_candidates = [
                        str(x).strip() for x in raw_evidence
                        if x is not None and str(x).strip()
                    ]
                else:
                    evidence = (item.get('evidence', '') or '').strip()
                    evidence_candidates = [
                        p.strip() for p in re.split(r"\s*;\s*", evidence)
                        if p and p.strip()
                    ]
                criterion_name = item.get('criterion', '').strip()

                from .annotator_rect import _is_heading_like
                criterion_anchor = ""
                if (
                    criterion_name
                    and not _is_heading_like(criterion_name)
                    and len(_normalize_text_for_match(criterion_name)) >= 6
                ):
                    criterion_anchor = criterion_name

                anchor_candidates: list[str] = []
                for ev in evidence_candidates[:3]:
                    if _is_informative_anchor(ev):
                        anchor_candidates.append(ev)
                if criterion_anchor and _is_informative_anchor(criterion_anchor):
                    anchor_candidates.append(criterion_anchor)

                _item_reason = str(item.get('reason', '') or '')
                is_of = item.get('is_of_mark', False) or bool(
                    re.match(r'^OF[\s\-–]', _item_reason, re.IGNORECASE)
                )
                if marks == 0 and "Marks given above" in _item_reason:
                    score_label = "Marks given above"
                elif marks == 0 and "Marks given below" in _item_reason:
                    score_label = "Marks given below"
                elif is_of:
                    score_label = f"OF {_fmt_mark_value(marks)}"
                else:
                    score_label = _fmt_mark_value(marks)

                success = False
                used_anchor = ""
                for anchor in anchor_candidates[:3]:
                    used_anchor = anchor
                    logger.debug(f"  Criterion {idx}: {criterion_name} ({marks}pts)")
                    logger.debug(f"    Anchor: {anchor[:80]}...")
                    success = place_score_near_anchor(
                        doc, anchor, score_label,
                        allowed_pages, placed_lines_per_page, placed_marks,
                        unplaced_items,
                        page_token_sets=page_token_sets,
                        line_score_accumulator=line_score_accumulator,
                        min_y_per_page=min_y_per_page,
                        max_y_per_page=max_y_per_page,
                        draw_underline=True,  # Evidence lines should be underlined
                    )
                    if success:
                        break

                if success:
                    annotation_mapping['criterion_scores_placed'] += 1
                else:
                    if anchor_candidates:
                        logger.debug(
                            f"    [placement] ✗ All anchors failed. "
                            f"Last tried: '{used_anchor[:60]}'"
                        )
                criteria_count += 1

            # Place one combined score label per resolved line
            for entry in line_score_accumulator.values():
                try:
                    entry_page_num = int(entry.get("page_num"))
                    entry_page_idx = int(entry.get("page_idx"))
                    entry_rect = entry.get("rect")
                    marks_list = entry.get("marks") or []
                    total = sum(float(m) for m in marks_list)
                except Exception:
                    continue
                if not entry_rect or entry_page_num <= 0:
                    continue
                entry_page = doc[entry_page_num - 1]
                _place_score_label(
                    entry_page, entry_rect, entry_page_idx,
                    placed_lines_per_page, _fmt_mark_value(total),
                )

            logger.info(
                f"✓ Placed {annotation_mapping['criterion_scores_placed']} of "
                f"{criteria_count} criteria with evidence"
            )

            # ── Standard mode: mark off-topic content as "Not required" ──────
            # No marks affected — purely instructional feedback for the student.
            # Mirrors the holistic-mode Step 1b but reads from the doc-level
            # not_required_points list (numerical questions don't have sub-Qs).
            nr_points = grades_doc.get('not_required_points', []) or []
            if nr_points:
                logger.info(
                    f"Processing {len(nr_points)} 'Not required' point(s) for numerical grading"
                )
                nr_local_marks: set = set()
                for nr in nr_points:
                    if not isinstance(nr, dict):
                        continue
                    nr_text = str(nr.get("text", "")).strip()
                    nr_kp = str(nr.get("key_phrase", "")).strip()
                    nr_reason = str(nr.get("reason", "")).strip()
                    if not nr_text:
                        continue

                    nr_clean = _strip_llm_artifacts(
                        nr_text.replace("…", " ").replace("...", " ").replace('|', ' ')
                    )
                    if not nr_clean:
                        continue

                    nr_rect = None
                    nr_page_num = -1

                    # Strategy 1: key_phrase match (precise).
                    if nr_kp and len(nr_kp) >= 3:
                        for candidate in _build_anchor_variations(nr_kp):
                            nr_rect, nr_page_num = resolve_anchor_rect(
                                doc, candidate, allowed_pages,
                                placed_marks=nr_local_marks,
                                skip_duplicates=True,
                                expand_to_line=True,
                                page_token_sets=page_token_sets,
                                redirect_headings=False,
                                use_number_first=False,
                                min_y_per_page=min_y_per_page,
                                max_y_per_page=max_y_per_page,
                            )
                            if nr_rect and nr_page_num > 0:
                                break

                    # Strategy 2: full text fallback.
                    if not nr_rect or nr_page_num <= 0:
                        for fragment in _build_candidate_fragments(nr_clean):
                            for candidate in _build_anchor_variations(fragment):
                                nr_rect, nr_page_num = resolve_anchor_rect(
                                    doc, candidate, allowed_pages,
                                    placed_marks=nr_local_marks,
                                    skip_duplicates=True,
                                    expand_to_line=True,
                                    page_token_sets=page_token_sets,
                                    redirect_headings=False,
                                    use_number_first=False,
                                    min_y_per_page=min_y_per_page,
                                    max_y_per_page=max_y_per_page,
                                )
                                if nr_rect and nr_page_num > 0:
                                    break
                            if nr_rect and nr_page_num > 0:
                                break

                    if nr_rect and nr_page_num > 0:
                        nr_page = doc[nr_page_num - 1]
                        place_not_required_marker(nr_page, nr_rect, nr_reason)
                        nr_local_marks.add(_line_key(nr_page_num, nr_rect.y0))
                        logger.info(f"  ⚑ Not required: '{nr_kp or nr_text[:60]}'")
                    else:
                        logger.info(f"  ✗ NR not found in PDF: '{nr_kp or nr_text[:60]}'")

        # ── Fallback: margin notes for unplaced high-value items ──────────────
        def _label_numeric_val(s: str) -> float:
            """Parse numeric value from label that may have 'OF ' prefix."""
            s = s.strip()
            if s.upper().startswith("OF "):
                s = s[3:].strip()
            return _safe_float(s)

        if unplaced_items:
            high_value = [
                (score, ev) for score, ev in unplaced_items
                if isinstance(score, str) and score.strip()
                and _label_numeric_val(score) >= 0.25
            ]
            if high_value:
                logger.warning(
                    f"Fallback: placing {len(high_value)} high-value items in margin"
                )
                fallback_page_obj = doc[allowed_pages[0] - 1]
                y_pos = 80
                for score, evidence in high_value[:5]:
                    fallback_page_obj.insert_text(
                        (fallback_page_obj.rect.width - 280, y_pos),
                        f"Marks given below: {score}pt",
                        fontsize=8,
                        color=(0.8, 0.4, 0),
                    )
                    y_pos += 14
                    fallback_page_obj.insert_text(
                        (fallback_page_obj.rect.width - 280, y_pos),
                        f"  ({evidence[:30]})",
                        fontsize=7,
                        color=(0.8, 0.4, 0),
                    )
                    y_pos += 16

        # ── Feedback comments ──────────────────────────────────────────────────
        all_comments = grades_doc.get('comments', [])
        logger.info(f"Processing {len(all_comments)} comments for feedback...")

        # Pre-compute sub-question Y bounds for every [<sub_question>] tag found
        # in the comments. The grading prompt prepends a tag like "[4.1]" or
        # "[4.3 Payroll Threats]" so the popup lands inside the correct sub-region.
        sub_ids_in_comments: list[str] = []
        seen_ids: set[str] = set()
        for c in all_comments:
            if not isinstance(c, str):
                continue
            sid, _ = _strip_subq_prefix(c)
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                sub_ids_in_comments.append(sid)
        subq_y_bounds = compute_subq_y_bounds(doc, allowed_pages, sub_ids_in_comments)
        if sub_ids_in_comments:
            pages_with_bounds = sum(1 for sid in sub_ids_in_comments if subq_y_bounds.get(sid))
            logger.info(
                f"Sub-question Y bounds resolved: {pages_with_bounds}/{len(sub_ids_in_comments)} "
                f"sub_ids found in PDF — comments will be constrained to their sub-question region"
            )

        comments_placed = 0
        unplaced_comments: list[str] = []
        for idx, comment in enumerate(all_comments, 1):
            if not comment or not isinstance(comment, str):
                logger.debug(f"  Comment {idx}: INVALID (empty or non-string)")
                continue

            logger.debug(f"  Comment {idx}/{len(all_comments)}: {comment[:70]}...")

            try:
                if add_popup_for_comment(
                    doc,
                    comment.strip(),
                    allowed_pages,
                    placed_marks=placed_marks,
                    page_token_sets=page_token_sets,
                    comment_page_y=comment_page_y,
                    comment_used_y=comment_used_y,
                    ocr_textpages=ocr_textpages,
                    placed_lines_per_page=placed_lines_per_page,
                    min_y_per_page=min_y_per_page,
                    max_y_per_page=max_y_per_page,
                    subq_y_bounds=subq_y_bounds,
                ):
                    comments_placed += 1
                    logger.debug("    ✓ Comment placed")
                else:
                    unplaced_comments.append(comment.strip())
                    logger.warning(
                        f"  ✗ Comment {idx} NOT PLACED on PDF "
                        f"(anchor split or match failed): {comment[:80]!r}"
                    )
            except Exception as e:
                unplaced_comments.append(comment.strip())
                logger.warning(f"  ✗ Comment {idx} error during placement: {e}")

        annotation_mapping['comments_placed'] = comments_placed
        annotation_mapping['unplaced_comments'] = unplaced_comments
        if unplaced_comments:
            logger.warning(
                f"✗ {len(unplaced_comments)} comment(s) failed to place on PDF "
                f"(see 'unplaced_comments' in annotation mapping JSON)"
            )
        logger.info(f"✓ Comments: {comments_placed}/{len(all_comments)} placed")

        # ── Save ───────────────────────────────────────────────────────────────
        doc.save(output_pdf, garbage=4, deflate=True, clean=True)
        doc.close()

        annotation_mapping['unplaced_items'] = unplaced_items[:10]
        with open(mapping_json, 'w') as f:
            json.dump(annotation_mapping, f, indent=2, default=str)

        success_rate = (
            annotation_mapping['criterion_scores_placed']
            / max(annotation_mapping['total_criteria'], 1)
        )
        logger.info(f"Done: {success_rate:.0%} success → {output_pdf}")
        return True, output_pdf

    except Exception as e:
        logger.error(f"Annotation failed for {student_name}: {e}", exc_info=True)
        return False, ""
