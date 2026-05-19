import json
import re
import os
import ast
import math
import traceback
from datetime import datetime
from typing import Optional, Any, Tuple
from bson import ObjectId
from pydantic import BaseModel, Field
from prompts.grading_prompts import grade_prompt, holistic_grade_prompt
from llm_setup import llm_grader
from logging_config import logger
from schemas.student_grades import StudentGradeDocument
from database.mongodb import get_collection
from errors import GradingError, classify_error


class NotRequiredPoint(BaseModel):
    text: str = Field(..., description="Verbatim line/sentence from student that is off-topic / not required")
    key_phrase: str = Field("", description="3-6 word verbatim anchor within text where the 'Not required' marker is placed")
    reason: str = Field("", description="One-sentence reason this content is off-topic or not asked for")


class CorrectPoint(BaseModel):
    text: str = Field(..., description="Verbatim line/sentence from student that earned marks")
    marks: float = Field(..., ge=0, description="Marks this specific point earned (0.5 increments)")
    key_phrase: str = Field("", description="2-5 word core concept within text where tick mark is placed")


class LLMGradingBreakdownItem(BaseModel):
    criterion: str = Field(..., description="Exact criterion description from model marking_criteria")
    marks_awarded: float = Field(..., ge=0, description="Marks awarded for this criterion")
    max_possible: float = Field(..., ge=0, description="Maximum marks possible for this criterion")
    reason: str = Field("", description="Brief reason for award")
    evidence: list[str] = Field(default_factory=list, description="1-3 verbatim quotes from student answer")
    comments_summary: Optional[str] = Field("", description="Optional short note")


class LLMGradingItem(BaseModel):
    question_number: str = Field(..., description="Question number being graded")
    score: float = Field(..., ge=0, description="Total marks awarded")
    total_marks: float = Field(..., ge=0, description="Maximum marks for this question")
    comments: list[str] = Field(default_factory=list, description="Feedback comments")
    correct_words: list[str] = Field(default_factory=list, description="Verbatim correct phrases")
    breakdown: list[LLMGradingBreakdownItem] = Field(default_factory=list, description="Per-criterion breakdown")
    not_required_points: list[NotRequiredPoint] = Field(
        default_factory=list,
        description="Off-topic / irrelevant content flagged for the student (no marks impact)",
    )


class LLMGradingResponse(BaseModel):
    grades: list[LLMGradingItem] = Field(..., min_length=1, description="Grades array")


# ── Holistic grading models (no-criteria theoretical questions) ──


class HolisticSubQuestionGrade(BaseModel):
    sub_question: str = Field(..., description="Sub-question identifier from model answer, e.g. 'a', '1.1', '(i)'")
    student_label: str = Field("", description="How the student labeled this part, e.g. 'Q1(a)', 'a)', 'Part a'")
    marks_awarded: float = Field(..., ge=0, description="Marks awarded for this sub-question")
    max_marks: float = Field(..., ge=0, description="Maximum marks for this sub-question")
    reason: str = Field("", description="Brief explanation of why marks were awarded/not awarded")
    correct_points: list[CorrectPoint] = Field(default_factory=list, description="Correct points with per-point marks")
    not_required_points: list[NotRequiredPoint] = Field(default_factory=list, description="Off-topic / irrelevant content that earns no marks but is flagged for the student")


class HolisticGradingResponse(BaseModel):
    question_number: str = Field(..., description="Main question number")
    score: float = Field(..., ge=0, description="Total marks awarded")
    total_marks: float = Field(..., ge=0, description="Maximum marks for the entire question")
    sub_grades: list[HolisticSubQuestionGrade] = Field(default_factory=list, description="Per-sub-question grades")
    comments: list[str] = Field(default_factory=list, description="Feedback comments")


class StudentGrader:

    COLLECTION_NAME = "student_grades"

    def __init__(
        self,
        student_name: str,
        question_number: str,
        questions_id: Optional[str],
        model_answers_id: Optional[str],
        student_answers_id: str,
        question_type: str = "numerical",
    ):
        self.student_name = student_name
        self.question_number = question_number
        self.questions_id = questions_id
        self.model_answers_id = model_answers_id
        self.student_answers_id = student_answers_id
        self.question_type = question_type  # "numerical" or "theoretical"

        # Flag: True when marking criteria were synthesized from answer text
        # (no formal rubric provided). Used to relax strict guardrails.
        self._criteria_were_synthesized: bool = False

        # Flag: True when no marking criteria exist and we use holistic grading
        # (compare full answers) instead of per-criterion grading.
        self._holistic_grading: bool = False
        # Cached holistic sub-question structure for the grading prompt.
        # Each entry: {"sub_question": str, "answer": str, "max_marks": float}
        self._holistic_sub_questions: list[dict] = []

        # Cache of the exact student text passed to the grader in the most recent run.
        # Used for post-grading guardrails (e.g., verify evidence quotes actually exist).
        self._student_text_last_run: str = ""

        # Cache of the question/model payload used in the most recent run.
        # Used to detect "tainted" evidence that is copied from the question/markscheme
        # (e.g., section headings) rather than the student's own work.
        self._question_text_last_run: str = ""
        self._model_text_last_run: str = ""

        # Optional debug trace of raw LLM outputs / parsing errors.
        # Enabled via DEBUG_SAVE_LLM_OUTPUT=1.
        self._llm_debug_trace: list[dict[str, Any]] = []

        # Cache of the rubric criteria (descriptions) used for the most recent run.
        # Used to (a) prevent the LLM from inventing criteria and (b) enforce max_possible.
        self._allowed_criteria_last_run: set[str] = set()
        self._criterion_max_map_last_run: dict[str, float] = {}
        self._criterion_category_map_last_run: dict[str, str] = {}
        # Criteria that require the exact expected number (no OF bypass allowed).
        # Populated from rubric fields with exact_match=True.
        self._exact_match_criteria_last_run: set[str] = set()
        # Cache rubric order (description → position) for post-processing heuristics.
        self._rubric_criteria_order_last_run: list[str] = []
        self._rubric_position_last_run: dict[str, int] = {}

        self.grades_coll = get_collection(self.COLLECTION_NAME)

        # Grading chain.
        # Prefer strict structured-output when the provider supports it.
        # Some providers/models (e.g., Grok / some Anthropic setups) may return
        # non-conforming JSON; we fall back to text parsing + one repair pass.
        self.grade_chain_structured = None
        try:
            structured_grader = llm_grader.with_structured_output(LLMGradingResponse)
            self.grade_chain_structured = grade_prompt | structured_grader
        except Exception:
            self.grade_chain_structured = None

        self.grade_chain_text = grade_prompt | llm_grader

        # Holistic grading chains (used when no marking criteria exist).
        self.holistic_chain_structured = None
        try:
            holistic_structured = llm_grader.with_structured_output(HolisticGradingResponse)
            self.holistic_chain_structured = holistic_grade_prompt | holistic_structured
        except Exception:
            self.holistic_chain_structured = None
        self.holistic_chain_text = holistic_grade_prompt | llm_grader

    @staticmethod
    def _extract_structured_args_from_message(output: Any) -> Optional[dict]:
        # Newer LangChain: tool_calls is a list of dict-like objects containing `args`.
        try:
            tool_calls = getattr(output, "tool_calls", None)
            if isinstance(tool_calls, list) and tool_calls:
                first = tool_calls[0]
                if isinstance(first, dict):
                    args = first.get("args")
                else:
                    args = getattr(first, "args", None)

                if isinstance(args, dict):
                    return args
                if isinstance(args, str) and args.strip():
                    return json.loads(args)
        except Exception:
            pass

        # Older/alternate: tool calls inside additional_kwargs
        try:
            additional = getattr(output, "additional_kwargs", None) or {}
            if isinstance(additional, dict):
                tc = additional.get("tool_calls")
                if isinstance(tc, list) and tc:
                    func = tc[0].get("function") if isinstance(tc[0], dict) else None
                    if isinstance(func, dict):
                        arguments = func.get("arguments")
                        if isinstance(arguments, str) and arguments.strip():
                            return json.loads(arguments)

                fc = additional.get("function_call")
                if isinstance(fc, dict):
                    arguments = fc.get("arguments")
                    if isinstance(arguments, str) and arguments.strip():
                        return json.loads(arguments)
        except Exception:
            pass

        return None

    @staticmethod
    def _extract_json_from_text(raw: str) -> str:

        if not isinstance(raw, str):
            raw = str(raw)

        cleaned = raw.strip()
        # Remove BOM / zero-width chars that can break json.loads at column 1
        cleaned = cleaned.lstrip("\ufeff\u200b\u200c\u200d")
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            return ""

        # Try to isolate the first JSON object/array.
        first_curly = cleaned.find("{")
        first_square = cleaned.find("[")

        starts = [i for i in (first_curly, first_square) if i != -1]
        if not starts:
            return cleaned

        start = min(starts)

        last_curly = cleaned.rfind("}")
        last_square = cleaned.rfind("]")
        ends = [i for i in (last_curly, last_square) if i != -1]
        end = max(ends) + 1 if ends else len(cleaned)

        return cleaned[start:end].strip()

    def _extract_question_max_marks(self, questions_data: dict, main_grade: Optional[dict] = None) -> float:
        """Extract the total maximum marks for the question being graded.

        Priority:
        1. Matching sub-question marks — find the question matching self.question_number
           and use its marks (from the "marks" field, or from trailing "(N)" in content).
           This handles papers where total_marks is the whole-paper total (e.g. 54)
           but each question has its own marks (e.g. 12).
        2. Document-level total_marks — only if there's a single question or no sub-questions.
        3. The LLM grader's own total_marks report in main_grade.
        4. Sum of individual sub-question marks as a last resort.
        """
        def _parse_nums(text: Any) -> list[float]:
            if text is None:
                return []
            return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(text))]

        def _best_from_nums(nums: list[float]) -> Optional[float]:
            pos = [n for n in nums if n > 0]
            return max(pos) if pos else None

        def _extract_trailing_marks(text: str) -> Optional[float]:
            """Extract marks from the end of question content, e.g. '...(12)' or '...(12 marks)'."""
            if not text:
                return None
            # Match trailing parenthesized number, optionally followed by "marks"
            m = re.search(r"\((\d+(?:\.\d+)?)\s*(?:[Mm]arks?)?\)\s*$", text.strip())
            if m:
                return float(m.group(1))
            return None

        # 0. Document-level total_marks — try first since the document is already
        # scoped to a single question and its total_marks is the authoritative total.
        if isinstance(questions_data, dict):
            q_total_raw = questions_data.get("total_marks")
            if q_total_raw is not None:
                q_text = str(q_total_raw)
                for pattern in (
                    r"maximum\s*marks?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                    r"max(?:imum)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                ):
                    m = re.search(pattern, q_text, flags=re.IGNORECASE)
                    if m:
                        logger.info(f"Using document-level total_marks for Q{self.question_number}: {m.group(1)}")
                        return float(m.group(1))
                v = _best_from_nums(_parse_nums(q_text))
                if v:
                    logger.info(f"Using document-level total_marks for Q{self.question_number}: {v}")
                    return v

        # 1. Try to find marks for the specific question being graded.
        if isinstance(questions_data, dict):
            questions_list = questions_data.get("questions")
            if isinstance(questions_list, list) and len(questions_list) > 0:
                q_digit = "".join(re.findall(r"\d+", str(self.question_number)))

                for q in questions_list:
                    if not isinstance(q, dict):
                        continue
                    q_num = str(q.get("question_number", ""))
                    q_num_digit = "".join(re.findall(r"\d+", q_num))

                    if not q_digit or not q_num_digit:
                        continue
                    if q_num_digit != q_digit and not q_num_digit.startswith(q_digit) and not q_digit.startswith(q_num_digit):
                        continue

                    # Found matching question.
                    # Priority: LLM-computed total_marks on the question item (most reliable).
                    llm_total = q.get("total_marks")
                    if llm_total is not None:
                        try:
                            v = float(llm_total)
                            if v > 0:
                                logger.info(f"Using LLM total_marks for Q{self.question_number}: {v}")
                                return v
                        except (TypeError, ValueError):
                            pass

                    # Fallback: recursively sum sub_questions marks —
                    # handles old extractions that pre-date the total_marks field.
                    def _sum_sq_marks(sq_list: list) -> float:
                        """Recursively sum leaf-level marks across all sub_questions."""
                        total = 0.0
                        for sq in sq_list:
                            if not isinstance(sq, dict):
                                continue
                            nested = sq.get("sub_questions")
                            if nested:
                                # Has deeper nesting — recurse instead of reading this level
                                child_total = _sum_sq_marks(nested)
                                if child_total > 0:
                                    total += child_total
                                    continue
                            # Leaf node — read marks directly
                            sq_v = None
                            for key in ("marks", "maximum_marks", "max_marks", "total_marks"):
                                raw = sq.get(key)
                                if raw is not None:
                                    sq_v = _best_from_nums(_parse_nums(raw))
                                    if sq_v:
                                        break
                            if sq_v is None:
                                sq_content = sq.get("content", "")
                                if isinstance(sq_content, str):
                                    sq_v = _extract_trailing_marks(sq_content)
                            if sq_v:
                                total += sq_v
                        return total

                    sub_questions = q.get("sub_questions") or []
                    if sub_questions:
                        sq_total = _sum_sq_marks(sub_questions)
                        if sq_total > 0:
                            logger.info(f"Using sum of sub-question marks for Q{self.question_number}: {sq_total}")
                            return sq_total

                    # No sub_questions — use the question-level marks field directly
                    for key in ("marks", "maximum_marks", "max_marks", "total_marks"):
                        raw = q.get(key)
                        if raw is not None:
                            v = _best_from_nums(_parse_nums(raw))
                            if v:
                                logger.info(f"Using sub-question marks for Q{self.question_number}: {v} (from '{key}': '{raw}')")
                                return v

                    # Fallback: extract trailing marks from the question content text
                    content = q.get("content", "")
                    if isinstance(content, str):
                        v = _extract_trailing_marks(content)
                        if v:
                            logger.info(f"Using trailing marks from question content for Q{self.question_number}: {v}")
                            return v

        # 2. (Skipped — document-level total_marks already handled in step 0.)

        # 3. LLM-reported total from main_grade.
        if main_grade and isinstance(main_grade, dict):
            for key in ("total_marks", "maximum_marks", "total_marks_available"):
                v = _best_from_nums(_parse_nums(main_grade.get(key)))
                if v:
                    return v

        # 4. Sum individual sub-question marks as a fallback.
        if isinstance(questions_data, dict):
            questions_list = questions_data.get("questions")
            if isinstance(questions_list, list):
                total = 0.0
                for q in questions_list:
                    if not isinstance(q, dict):
                        continue
                    for key in ("marks", "maximum_marks", "max_marks", "total_marks"):
                        nums = _parse_nums(q.get(key))
                        if nums:
                            total += max(n for n in nums if n > 0)
                            break
                if total > 0:
                    return total

        # 5. Last resort: document-level total_marks even for multi-question papers
        if isinstance(questions_data, dict):
            q_total_raw = questions_data.get("total_marks")
            if q_total_raw is not None:
                v = _best_from_nums(_parse_nums(str(q_total_raw)))
                if v:
                    logger.warning(f"Using paper-level total marks as fallback: {v} (may include marks for other questions)")
                    return v

        return 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Criteria synthesis — used when model answers have no marking_criteria
    # but contain inline marks in the answer text (e.g. "SL (3 Marks)")
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_inline_section_marks(answer_text: str) -> list[tuple[str, float, str]]:
        """Parse sections with inline marks from model answer text.

        Looks for patterns like:
          "SL (3 Marks)"  /  "Issue 1 - Peak Estate (4 Marks)"  /  "Required adjustment: (1 Marks)"
        at the start of sections in the answer text.

        Returns list of (section_title, marks, section_body) tuples.
        """
        if not answer_text or not isinstance(answer_text, str):
            return []

        # Pattern: a heading/label followed by (N Marks) or (N marks) or (N Mark)
        # Also handles: "SL (3 Marks)\n..." and "Issue 1 - Peak Estate (4 Marks)\n..."
        section_pattern = re.compile(
            r"^(.+?)\s*\((\d+(?:\.\d+)?)\s*[Mm]arks?\)",
            re.MULTILINE,
        )

        matches = list(section_pattern.finditer(answer_text))
        if not matches:
            return []

        sections: list[tuple[str, float, str]] = []
        for i, m in enumerate(matches):
            title = m.group(1).strip().rstrip("-–—:").strip()
            marks = float(m.group(2))
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(answer_text)
            body = answer_text[body_start:body_end].strip()
            if marks > 0 and body:
                sections.append((title, marks, body))

        return sections

    @staticmethod
    def _split_answer_into_points(section_body: str) -> list[str]:
        """Split a section of model answer text into individual marking points.

        Splits on:
        - Bullet points (-, •, *)
        - Numbered points (1., 2.), (i), (ii))
        - Lines starting with ":" after a keyword
        - Paragraph breaks (double newline)
        - Sentence boundaries for long non-bulleted paragraphs

        Returns list of non-empty point strings.
        """
        if not section_body or not isinstance(section_body, str):
            return []

        lines = section_body.strip().split("\n")
        points: list[str] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Paragraph break — flush current
                if current:
                    points.append(" ".join(current))
                    current = []
                continue

            # Detect bullet / numbered list starts
            is_new_point = bool(re.match(
                r"^(?:[-•*]|\d+[.):]|\([a-z]\)|\([ivx]+\))\s",
                stripped,
                re.IGNORECASE,
            ))

            if is_new_point and current:
                points.append(" ".join(current))
                current = []

            # Clean bullet/number prefix
            cleaned = re.sub(r"^(?:[-•*]|\d+[.):]|\([a-z]\)|\([ivx]+\))\s*", "", stripped).strip()
            if cleaned:
                current.append(cleaned)

        if current:
            points.append(" ".join(current))

        # Filter out very short/meaningless points
        points = [p for p in points if len(p) >= 10]

        # Post-process: split long non-bulleted paragraphs into sentences.
        # This handles model answers where distinct marking points are written
        # as continuous prose rather than bullet lists.
        expanded: list[str] = []
        for p in points:
            if len(p) > 150:
                # Split on sentence boundaries (period followed by space and capital letter,
                # or period followed by newline)
                sentences = re.split(r"(?<=\.)\s+(?=[A-Z])", p)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]
                if len(sentences) > 1:
                    expanded.extend(sentences)
                else:
                    expanded.append(p)
            else:
                expanded.append(p)

        return expanded

    def _synthesize_criteria_from_answer(self, answers: list[dict]) -> list[dict[str, Any]]:
        """Generate marking criteria from model answer text when none are provided.

        When the model answer has no marking_criteria but embeds section marks
        inline (e.g. "SL (3 Marks)"), this method:
        1. Parses out each section and its total marks
        2. Splits each section into individual marking points
        3. Distributes the section marks evenly across its points
        4. Returns a list of synthesized criteria dicts
        """
        synthesized: list[dict[str, Any]] = []

        for answer in answers:
            if not isinstance(answer, dict):
                continue
            answer_text = answer.get("answer")
            if not isinstance(answer_text, str) or not answer_text.strip():
                continue

            sections = self._parse_inline_section_marks(answer_text)

            if sections:
                for title, section_marks, body in sections:
                    points = self._split_answer_into_points(body)
                    if not points:
                        # Can't split — use the whole section as one criterion
                        synthesized.append({
                            "marks": section_marks,
                            "description": f"{title}: {body[:200]}",
                        })
                        continue

                    # Distribute marks across points so the total equals section_marks.
                    # Strategy: assign a base of 0.25 per point, then distribute remaining
                    # marks as bonus 0.25 increments to earlier (more important) points.
                    n = len(points)
                    base = 0.25
                    total_at_base = base * n

                    if total_at_base >= section_marks:
                        # More points than marks allow at 0.25 each — only keep enough points
                        max_points = int(section_marks / base)
                        points = points[:max_points] if max_points > 0 else points[:1]
                        n = len(points)
                        total_at_base = base * n

                    # Remaining marks to distribute as bonus 0.25 increments
                    remaining_marks = section_marks - total_at_base
                    bonus_slots = int(round(remaining_marks / 0.25))

                    for j, point in enumerate(points):
                        mark = base
                        if j < bonus_slots:
                            mark += 0.25
                        mark = round(mark / 0.25) * 0.25
                        synthesized.append({
                            "marks": mark,
                            "description": point,
                        })
            else:
                # No inline marks found — try to use question-level marks
                # and split the entire answer into points
                points = self._split_answer_into_points(answer_text)
                if not points:
                    continue

                # Try to get total marks from answer text ending pattern like "(12)" or "12 marks"
                total_marks_match = re.search(
                    r"\((\d+(?:\.\d+)?)\s*(?:[Mm]arks?)?\)\s*$", answer_text.strip()
                )
                if total_marks_match:
                    total = float(total_marks_match.group(1))
                else:
                    # Fallback: can't determine marks, assign equal weight placeholder
                    # These will be scaled in _flatten_model_answers when we know total marks
                    total = float(len(points))  # 1 mark per point as placeholder

                n = len(points)
                base = 0.25
                total_at_base = base * n

                if total_at_base >= total:
                    max_points = int(total / base)
                    points = points[:max_points] if max_points > 0 else points[:1]
                    n = len(points)
                    total_at_base = base * n

                remaining_marks = total - total_at_base
                bonus_slots = int(round(remaining_marks / 0.25))

                for j, point in enumerate(points):
                    mark = base
                    if j < bonus_slots:
                        mark += 0.25
                    mark = round(mark / 0.25) * 0.25
                    synthesized.append({
                        "marks": mark,
                        "description": point,
                    })

        if synthesized:
            logger.info(
                f"Synthesized {len(synthesized)} criteria from model answer text "
                f"(total marks: {sum(c['marks'] for c in synthesized):.1f})"
            )

        return synthesized

    @staticmethod
    def _answer_matches_question(answer_label: str, question_number: str) -> bool:
        """Check if a model answer's question_number matches the question being graded.

        Handles varied labelling conventions: "Ans.1", "Ans 1", "A1", "(a)", "1", "Q.1", etc.
        """
        if not answer_label or not question_number:
            return True  # If either is missing, don't filter

        def _extract_digits(s: str) -> str:
            return "".join(re.findall(r"\d+", s))

        a_digits = _extract_digits(answer_label)
        q_digits = _extract_digits(question_number)

        if not a_digits or not q_digits:
            return True  # Can't compare — don't filter

        # Match if the leading digit(s) agree (e.g. "Ans.1" vs "Q.1" → "1" == "1")
        return a_digits == q_digits or a_digits.startswith(q_digits) or q_digits.startswith(a_digits)

    def _flatten_model_answers(self, model_data: dict, questions_data: Optional[dict] = None) -> dict:
        if not isinstance(model_data, dict):
            return model_data

        answers = model_data.get("answers")
        if not isinstance(answers, list) or not answers:
            return model_data

        # ── Filter answers to only those matching the question being graded ──
        # Skip per-answer filtering when the document-level question_title already
        # matches the target question — all answers in the doc are sub-parts of it.
        doc_title = str(model_data.get("question_title", ""))
        doc_matches_target = self._answer_matches_question(doc_title, self.question_number)

        if len(answers) > 1 and not doc_matches_target:
            filtered = [
                a for a in answers
                if isinstance(a, dict) and self._answer_matches_question(
                    str(a.get("question_number", "")), self.question_number
                )
            ]
            if filtered:
                answers = filtered
                logger.info(
                    f"Filtered model answers to {len(answers)} matching Q{self.question_number} "
                    f"(from {len(model_data['answers'])} total)"
                )
        else:
            logger.info(
                f"Filtered model answers to {len(answers)} matching Q{self.question_number} "
                f"(from {len(model_data['answers'])} total)"
            )

        combined_criteria: list[dict[str, Any]] = []
        combined_answer_parts: list[str] = []

        def _norm_desc_key(s: str) -> str:
            if not s or not isinstance(s, str):
                return ""
            s = s.replace("\u00a0", " ")
            s = s.replace("–", "-").replace("—", "-")
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s

        def _dedup_criteria(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Deduplicate repeated criteria descriptions.

            Some marking guides repeat identical descriptions across nested parts; asking the
            LLM to grade duplicates harms stability and can double-count.
            """
            out: list[dict[str, Any]] = []
            key_to_index: dict[str, int] = {}

            for it in criteria:
                if not isinstance(it, dict):
                    continue
                desc = str(it.get("description", "") or "").strip()
                if not desc:
                    continue
                key = _norm_desc_key(desc)
                if not key:
                    continue

                marks = it.get("marks")
                marks_num: Optional[float] = None
                if isinstance(marks, (int, float)):
                    marks_num = float(marks)

                if key in key_to_index:
                    existing = out[key_to_index[key]]
                    ex_marks = existing.get("marks")
                    ex_marks_num: Optional[float] = float(ex_marks) if isinstance(ex_marks, (int, float)) else None
                    if marks_num is not None and (ex_marks_num is None or marks_num > ex_marks_num):
                        existing["marks"] = marks_num
                    continue

                _entry: dict[str, Any] = {
                    "marks": marks_num if marks_num is not None else marks,
                    "description": desc,
                }
                if it.get("category"):
                    _entry["category"] = it["category"]
                if it.get("exact_match"):
                    _entry["exact_match"] = it["exact_match"]
                out.append(_entry)
                key_to_index[key] = len(out) - 1

            return out

        def _drop_section_heading_criteria(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Drop broad section-level criteria when granular micro-criteria exist.

            Many marking guides include a broad, multi-mark "do the whole section" criterion
            alongside numerous micro-criteria. Keeping both encourages double-counting and
            causes PDF marks to anchor to headings.

            This is intentionally heuristic and conservative.
            """
            if not isinstance(criteria, list) or not criteria:
                return criteria

            enabled = os.getenv("DROP_SECTION_HEADING_CRITERIA", "1").strip().lower() not in {"0", "false", "no", "n"}
            if not enabled:
                return criteria

            stop = {
                "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "as", "is", "are",
                "was", "were", "be", "been", "being", "with", "from", "by", "this", "that", "these",
                "those", "must", "should", "would", "will", "student", "marks", "mark",
            }
            verbs = (
                "prepare",
                "calculate",
                "compute",
                "present",
                "explain",
                "discuss",
                "show",
                "derive",
                "evaluate",
            )

            def _token_set(desc: str) -> set[str]:
                d = _norm_desc_key(desc)
                if not d:
                    return set()
                toks = [w for w in re.findall(r"[a-z]{4,}", d) if w not in stop]
                return set(toks)

            # Identify "broad" candidates.
            broad_idxs: list[int] = []
            broad_startswith_verb: set[int] = set()
            token_sets: list[set[str]] = []
            for i, it in enumerate(criteria):
                desc = str((it or {}).get("description", "") or "").strip()
                marks = (it or {}).get("marks")
                marks_num: Optional[float] = float(marks) if isinstance(marks, (int, float)) else None
                toks = _token_set(desc)
                token_sets.append(toks)

                if not desc or marks_num is None:
                    continue

                dnorm = _norm_desc_key(desc)
                starts = any(dnorm.startswith(v + " ") for v in verbs)
                looks_broad = (marks_num >= 2.0) and (len(desc) >= 70 or starts)

                # Also flag short non-numeric "title" criteria with marks >= 1.0
                # e.g., "Electrostatic spraying room" (2/2 = 1.0). These are section
                # headings whose marks should be covered by their sub-criteria.
                desc_words = dnorm.split()
                if (
                    not looks_broad
                    and marks_num >= 1.0
                    and len(desc_words) <= 5
                    and not re.search(r"\d", dnorm)
                    and (not desc_words or desc_words[0] not in ("dr", "cr"))
                ):
                    looks_broad = True

                if looks_broad:
                    broad_idxs.append(i)
                    if starts:
                        broad_startswith_verb.add(i)

            if not broad_idxs:
                return criteria

            # Decide which broad criteria have enough overlapping micro-criteria to justify dropping.
            drop: set[int] = set()
            for i in broad_idxs:
                toks_i = token_sets[i]
                if not toks_i:
                    continue
                needed_overlaps = 1 if i in broad_startswith_verb else 3
                overlap_count = 0
                overlapping_marks_sum = 0.0
                broad_marks = float((criteria[i] or {}).get("marks", 0) or 0) if isinstance((criteria[i] or {}).get("marks"), (int, float)) else 0.0

                # Window-based micro-coverage heuristic (generic, order-aware):
                # Many mark schemes include a broad verb-led criterion (e.g. "Calculate EPS" 4 marks)
                # followed immediately by numerous micro-criteria totalling those marks. These micro
                # criteria often share few keywords with the broad sentence (lots of numeric-only lines),
                # so token overlap alone can fail. If nearby micro-criteria cover most of the marks,
                # drop the broad criterion to prevent double-counting and misleading annotations.
                if i in broad_startswith_verb and broad_marks >= 2.0:
                    window = 18
                    nearby_sum = 0.0
                    nearby_count = 0
                    for j in range(max(0, i - window), min(len(criteria), i + window + 1)):
                        if j == i:
                            continue
                        mj = (criteria[j] or {}).get("marks")
                        if not isinstance(mj, (int, float)):
                            continue
                        mj = float(mj)
                        if mj <= 0 or mj > 1.0:
                            continue
                        nearby_sum += mj
                        nearby_count += 1
                    if nearby_count >= 4 and nearby_sum >= broad_marks * 0.75:
                        drop.add(i)
                        continue

                for j, it in enumerate(criteria):
                    if i == j:
                        continue
                    desc_j = str((it or {}).get("description", "") or "").strip()
                    if not desc_j:
                        continue
                    marks_j = (it or {}).get("marks")
                    marks_j_num: Optional[float] = float(marks_j) if isinstance(marks_j, (int, float)) else None

                    toks_j = token_sets[j]
                    if len(toks_i & toks_j) < 2:
                        continue

                    # Count overlaps primarily against smaller/micro criteria.
                    if (marks_j_num is not None and marks_j_num <= 1.0) or len(desc_j) < 70:
                        overlap_count += 1
                        if marks_j_num is not None:
                            overlapping_marks_sum += marks_j_num
                        if overlap_count >= needed_overlaps:
                            break

                # Only drop if overlapping micro-criteria can cover at least half the
                # broad criterion's marks.  This prevents dropping a "Prepare SOCIE"
                # criterion worth 4 marks when the only overlap is a single vague
                # sub-criterion — keeping it ensures the table actually gets graded.
                if overlap_count >= needed_overlaps and overlapping_marks_sum >= broad_marks * 0.5:
                    drop.add(i)

            if not drop:
                return criteria

            filtered = [it for k, it in enumerate(criteria) if k not in drop]
            logger.info(f"Dropped {len(drop)} broad section-heading criteria to prevent double counting.")
            return filtered

        def _drop_commentary_criteria(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Drop non-marking commentary mistakenly extracted as criteria.

            Some PDFs embed solution commentary inside marking criteria blocks (eg "Tutorial note",
            "Proof of adjustment", or narrative observations like "appears to have been correctly
            dealt with"). These are not independent marking points and inflate scores.

            Heuristic + conservative: only drop when the phrase strongly signals commentary.
            """
            if not isinstance(criteria, list) or not criteria:
                return criteria

            enabled = os.getenv("DROP_COMMENTARY_CRITERIA", "1").strip().lower() not in {"0", "false", "no", "n"}
            if not enabled:
                return criteria

            commentary_phrases = (
                "tutorial note",
                "proof of adjustment",
                "appears to have been correctly dealt with",
                "this appears to have been correctly dealt with",
                "alternative assumptions",
                "alternative assumption",
            )

            out: list[dict[str, Any]] = []
            dropped = 0
            for it in criteria:
                if not isinstance(it, dict):
                    continue
                desc = str(it.get("description", "") or "").strip()
                if not desc:
                    continue
                dnorm = _norm_desc_key(desc)
                if any(p in dnorm for p in commentary_phrases):
                    dropped += 1
                    continue
                out.append(it)

            if dropped:
                logger.info(f"Dropped {dropped} commentary criteria (non-marking text extracted as criteria).")
            return out


        def normalize_marks_value(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)

            text = str(value).strip()

            # Mixed number like "3 1/2"
            mixed = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", text)
            if mixed:
                whole = float(mixed.group(1))
                num = float(mixed.group(2))
                den = float(mixed.group(3))
                if den != 0:
                    return whole + (num / den)
                return None

            # Fraction like 1/2, 3/12
            frac = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
            if frac:
                num = float(frac.group(1))
                den = float(frac.group(2))
                if den != 0:
                    return num / den
                return None

            # Fraction at start of a longer marking note, e.g. "1/2 mk each max 4"
            frac_prefix = re.match(r"^(\d+)\s*/\s*(\d+)", text)
            if frac_prefix:
                num = float(frac_prefix.group(1))
                den = float(frac_prefix.group(2))
                if den != 0:
                    return num / den
                return None

            # Plain numeric string
            num_match = re.fullmatch(r"\d+(?:\.\d+)?", text)
            if num_match:
                return float(text)

            # Keep as string for patterns we can't safely normalize
            return value

        def collect_numeric_marks(criteria_items: Any, out: list[float]) -> None:
            if not isinstance(criteria_items, list):
                return
            for it in criteria_items:
                if not isinstance(it, dict):
                    continue
                m = normalize_marks_value(it.get("marks"))
                if isinstance(m, (int, float)):
                    out.append(float(m))
                sub = it.get("sub_criteria")
                if isinstance(sub, list) and sub:
                    collect_numeric_marks(sub, out)

        def iter_answer_nodes(node: Any):
            """Yield answer/sub_answer nodes recursively."""
            if not isinstance(node, dict):
                return
            yield node
            for child in (node.get("sub_answers") or []):
                if isinstance(child, dict):
                    yield from iter_answer_nodes(child)

        # If we have granular micro-criteria, drop broad criteria (e.g. 26 marks learning outcomes)
        # to avoid polluting granular grading.
        all_marks: list[float] = []
        for ans in answers:
            for node in iter_answer_nodes(ans):
                collect_numeric_marks(node.get("marking_criteria"), all_marks)

        has_micro_criteria = any(m < 5 for m in all_marks)

        def flatten_criteria_items(
            criteria_items: Any,
            parent_description: Optional[str] = None,
            sibling_count: int = 0,
        ) -> None:
            if not isinstance(criteria_items, list):
                return

            for criteria_item in criteria_items:
                if not isinstance(criteria_item, dict):
                    continue

                description = str(criteria_item.get("description", "")).strip()
                raw_marks = criteria_item.get("marks")
                marks = normalize_marks_value(raw_marks)
                sub_criteria = criteria_item.get("sub_criteria")

                # Skip entire group when the parent is a known junk OCR/handwriting artefact.
                # Sub-criteria under junk parents (e.g. "Handwritten annotation" → "New York Wheels")
                # are section headings misidentified as criteria, not real marking points.
                _junk_parent_labels = {
                    "handwritten annotation", "annotation", "hr", "told - land.", "told - land",
                    "tutor note", "tutorial note", "marking guide", "mark scheme",
                    "scenario", "memorandum", "memo",
                }
                if description.strip().lower() in _junk_parent_labels and isinstance(sub_criteria, list) and sub_criteria:
                    continue

                # Expand compound criteria like "1/2 mk each max 4" where description
                # contains multiple semicolon/newline-separated line-items.
                if isinstance(raw_marks, str) and re.search(r"\bmk\s*each\b", raw_marks, flags=re.IGNORECASE):
                    per_mark = normalize_marks_value(raw_marks)
                    if isinstance(per_mark, (int, float)) and description:
                        # Extract optional "max N" cap from marks string.
                        max_match = re.search(r"\bmax\s+(\d+(?:\.\d+)?)", raw_marks, flags=re.IGNORECASE)
                        max_total = float(max_match.group(1)) if max_match else None

                        parts: list[str] = []
                        if ";" in description or "\n" in description:
                            parts = [p.strip() for p in re.split(r";|\n", description) if p and p.strip()]
                        else:
                            # Try to split number-heavy descriptions (e.g. SOCIE closing balances:
                            # "80,000 48,000 85,453 1,920 0 0 215,373") into individual number criteria.
                            nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,}", description)
                            if len(nums) >= 3:
                                parts = nums

                        if parts:
                            count = 0
                            for p in parts:
                                if max_total and count * float(per_mark) >= max_total:
                                    break
                                if not self._is_valid_criterion(p):
                                    # Pure numbers are always valid criteria in "mk each" context.
                                    if not re.search(r"\d", p):
                                        continue
                                combined_criteria.append({
                                    "marks": float(per_mark),
                                    "description": p,
                                })
                                count += 1
                            continue
                        elif max_total:
                            # Can't split, but we have a max cap — create one criterion
                            # worth the full max so the LLM can grade holistically.
                            # Only add if the description is meaningful (skip junk like
                            # "handwritten note" which the LLM cannot grade against).
                            if self._is_valid_criterion(description):
                                combined_criteria.append({
                                    "marks": float(max_total),
                                    "description": description,
                                })
                            continue

                # Skip non-numeric marks for grading (e.g., handwritten notes like "HR").
                # These should not be part of numeric breakdown scoring.
                if marks is not None and not isinstance(marks, (int, float)):
                    continue

                # When micro-criteria exist elsewhere, skip broad criteria (>=5 marks).
                if has_micro_criteria and isinstance(marks, (int, float)) and marks >= 5:
                    continue

                if isinstance(sub_criteria, list) and sub_criteria:
                    parent_desc = description if description else (parent_description or "")
                    # Don't propagate junk parent labels from OCR/handwriting artefacts.
                    if not self._is_valid_criterion(parent_desc):
                        parent_desc = ""

                    flattened_subs: list[dict[str, Any]] = []
                    for sub in sub_criteria:
                        if not isinstance(sub, dict):
                            continue

                        sub_desc = str(sub.get("description", "")).strip()
                        if parent_desc and sub_desc:
                            combined_desc = f"{parent_desc} — {sub_desc}"
                        else:
                            combined_desc = sub_desc or parent_desc

                        sub_marks = normalize_marks_value(sub.get("marks"))
                        if sub_marks is None:
                            continue
                        if not self._is_valid_criterion(combined_desc):
                            continue

                        flattened_subs.append({
                            "marks": sub_marks,
                            "description": combined_desc,
                        })

                    if flattened_subs:
                        combined_criteria.extend(flattened_subs)
                        continue

                    # Sub_criteria exist but none produced usable criteria (e.g., all had
                    # marks=None or failed validation).  Keep the parent as a leaf criterion
                    # if it is itself valid and has marks, so the LLM can still grade against
                    # it.  Example: "Prepare a revised SOCIE" (4 marks) whose only sub-criterion
                    # was a junk handwritten annotation — dropping the parent would lose all
                    # marks for that section.
                    if description and isinstance(marks, (int, float)) and marks > 0:
                        if self._is_valid_criterion(description):
                            combined_criteria.append({
                                "marks": marks,
                                "description": description,
                            })
                    continue

                # Leaf criterion
                if parent_description and description:
                    description = f"{parent_description} — {description}"
                elif parent_description and not description:
                    description = parent_description

                # If marks are missing and we have other criteria, skip to avoid generic/unweighted points.
                if marks is None and sibling_count > 1:
                    continue

                # Drop invalid/non-descriptive criteria to avoid polluting the grader.
                if not self._is_valid_criterion(description):
                    continue

                # Drop short non-numeric titles with high marks — these are section headings
                # (e.g., "Electrostatic spraying room" 2/2=1.0) whose marks overlap with sub-criteria.
                if isinstance(marks, (int, float)) and marks >= 1.0:
                    desc_words_flat = description.lower().split()
                    desc_stripped = re.sub(r"^\(\d+\)\s*", "", description.lower()).strip()
                    desc_stripped = re.sub(r"^\d+[.)]\s*", "", desc_stripped).strip()
                    if (
                        len(desc_stripped.split()) <= 5
                        and not re.search(r"\d", desc_stripped)
                        and (not desc_words_flat or desc_words_flat[0] not in ("dr", "cr"))
                    ):
                        logger.debug(f"Skipping section heading criterion: '{description}' ({marks} marks)")
                        continue

                _cat = str(criteria_item.get("category", "") or "").strip()
                _crit_entry: dict[str, Any] = {"marks": marks, "description": description}
                if _cat:
                    _crit_entry["category"] = _cat
                combined_criteria.append(_crit_entry)

        def _collect_answer_text(node: dict, parts_list: list) -> None:
            """Recursively collect non-empty answer text from a model-answer node and its sub_answers."""
            if not isinstance(node, dict):
                return
            text = node.get("answer")
            label = str(node.get("question_number", "")).strip()
            if isinstance(text, str) and text.strip():
                parts_list.append(f"Part {label}\n{text.strip()}" if label else text.strip())
            for child in (node.get("sub_answers") or []):
                _collect_answer_text(child, parts_list)

        for answer in answers:
            if not isinstance(answer, dict):
                continue

            part_label = str(answer.get("question_number", "")).strip()
            answer_text = answer.get("answer")
            if isinstance(answer_text, str) and answer_text.strip():
                if part_label:
                    combined_answer_parts.append(f"Part {part_label}\n{answer_text.strip()}")
                else:
                    combined_answer_parts.append(answer_text.strip())

            # Also collect text from sub_answers (hierarchical structure for theoretical papers)
            for child in (answer.get("sub_answers") or []):
                _collect_answer_text(child, combined_answer_parts)

            criteria = answer.get("marking_criteria")
            sibling_count = len(criteria) if isinstance(criteria, list) else 0
            flatten_criteria_items(criteria, sibling_count=sibling_count)

            # IMPORTANT: include nested sub_answer criteria (these contain most micro-marking points)
            for node in (answer.get("sub_answers") or []):
                for sub_node in iter_answer_nodes(node):
                    sub_criteria = sub_node.get("marking_criteria")
                    sub_sibling_count = len(sub_criteria) if isinstance(sub_criteria, list) else 0
                    flatten_criteria_items(sub_criteria, sibling_count=sub_sibling_count)

        # Optional compaction (OFF by default): some marking guides create extremely granular criteria.
        # Use COMPACT_ABC_GRADING=1 to enable the older compact behaviour.
        if os.getenv("COMPACT_ABC_GRADING", "").strip() in {"1", "true", "yes"}:
            # (We intentionally keep compaction disabled by default to preserve granular marking.)
            pass

        if not combined_criteria and not combined_answer_parts:
            return model_data

        # ── Holistic grading: for theoretical questions OR when no criteria exist ──
        self._criteria_were_synthesized = False
        self._holistic_grading = False
        use_holistic = (
            (self.question_type == "theoretical" and combined_answer_parts)
            or (not combined_criteria and combined_answer_parts)
        )
        if use_holistic:
            reason = f"question_type='{self.question_type}'" if self.question_type == "theoretical" else "no marking_criteria found"
            logger.info(
                f"Switching to HOLISTIC grading mode ({reason}) — "
                f"full answer comparison instead of per-criterion"
            )
            self._holistic_grading = True

            # Build sub-question structure from model answer for holistic prompt.
            # Each answer entry with a distinct question_number becomes a sub-question.

            # Build a lookup of sub-question marks from the question paper (authoritative).
            # This covers both direct sub_questions and nested structures.
            _paper_sq_marks: dict[str, float] = {}
            if isinstance(questions_data, dict):
                def _collect_sq_marks(sq_list: list) -> None:
                    for sq in sq_list:
                        if not isinstance(sq, dict):
                            continue
                        sq_label = str(sq.get("question_number", "")).strip()
                        raw_marks = sq.get("marks")
                        if sq_label and raw_marks is not None:
                            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(raw_marks))]
                            if nums:
                                _paper_sq_marks[sq_label] = max(nums)
                        nested = sq.get("sub_questions")
                        if nested:
                            _collect_sq_marks(nested)

                for q in (questions_data.get("questions") or []):
                    if isinstance(q, dict):
                        _collect_sq_marks(q.get("sub_questions") or [])

            holistic_subs: list[dict] = []

            def _add_holistic_sub(node: dict, parent_section: Optional[str] = None, section_cap: Optional[float] = None) -> None:
                """Recursively add leaf sub-questions to holistic_subs.

                For hierarchical model answers (theoretical papers), parent sections
                have answer='' and carry their sub-sections in sub_answers.  We recurse
                until we reach leaf nodes (non-empty answer text) and add those.
                The parent's subsection_max is forwarded as section_cap so the prompt
                can enforce cross-sub-question caps.
                """
                if not isinstance(node, dict):
                    return
                sq_num = str(node.get("question_number", "")).strip()
                sq_answer = str(node.get("answer", "") or "").strip()
                sub_list = node.get("sub_answers") or []

                if sub_list:
                    # Parent section: pass its subsection_max down as the cap for children
                    raw_cap = node.get("subsection_max")
                    child_cap = float(raw_cap) if raw_cap is not None else section_cap
                    for child in sub_list:
                        _add_holistic_sub(child, parent_section=sq_num, section_cap=child_cap)
                    return

                if not (sq_num and sq_answer):
                    return

                # Leaf node — resolve max_marks.
                # Priority: question-paper marks > maximum_marks / subsection_max > total_marks_available
                sq_marks = _paper_sq_marks.get(sq_num, 0.0)
                if not sq_marks:
                    for marks_key in ("maximum_marks", "subsection_max", "marks"):
                        raw = node.get(marks_key)
                        if raw is not None:
                            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(raw))]
                            if nums:
                                sq_marks = max(nums)
                                break
                if not sq_marks:
                    # Last resort: total_marks_available (may exceed section cap, but better than 0)
                    raw = node.get("total_marks_available")
                    if raw is not None:
                        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(raw))]
                        if nums:
                            sq_marks = max(nums)

                entry: dict = {
                    "sub_question": sq_num,
                    "answer": sq_answer,
                    "max_marks": sq_marks,
                    "marking_criteria": node.get("marking_criteria") or [],
                }
                rule = node.get("marking_rule")
                if rule:
                    entry["marking_rule"] = rule
                if parent_section is not None:
                    entry["parent_section"] = parent_section
                if section_cap is not None:
                    entry["section_cap"] = section_cap
                holistic_subs.append(entry)

            for answer in answers:
                _add_holistic_sub(answer)

            # If only one sub-question or all share the same number, treat as single block
            if len(holistic_subs) <= 1:
                # Single block: use the main question number
                full_answer = "\n\n".join(combined_answer_parts)
                total_marks = holistic_subs[0]["max_marks"] if holistic_subs else 0.0
                self._holistic_sub_questions = [{
                    "sub_question": self.question_number,
                    "answer": full_answer,
                    "max_marks": total_marks,
                }]
            else:
                self._holistic_sub_questions = holistic_subs

            logger.info(
                f"Holistic grading: {len(self._holistic_sub_questions)} sub-question(s) detected"
            )

            # For holistic mode, we still build a unified answer for the prompt
            # but WITHOUT marking_criteria — the LLM will compare holistically.
            unified_answer = {
                "question_number": self.question_number,
                "answer": "\n\n".join(combined_answer_parts),
                "marking_criteria": [],  # Empty — holistic mode
                "sub_questions": self._holistic_sub_questions,
            }

            flattened = dict(model_data)
            flattened["answers"] = [unified_answer]
            logger.info(
                f"Holistic model data prepared → {len(self._holistic_sub_questions)} sub-questions"
            )
            return flattened

        # ── Criteria exist: standard per-criterion path ──

        # Deduplicate criteria to stabilize marking and prevent repeated grading.
        combined_criteria = _dedup_criteria(combined_criteria)

        # Drop broad section headings when micro-criteria exist.
        combined_criteria = _drop_section_heading_criteria(combined_criteria)

        # Drop non-marking commentary accidentally extracted as criteria.
        combined_criteria = _drop_commentary_criteria(combined_criteria)

        unified_answer = {
            "question_number": self.question_number,
            "answer": "\n\n".join(combined_answer_parts),
            "marking_criteria": combined_criteria,
        }

        flattened = dict(model_data)
        flattened["answers"] = [unified_answer]

        logger.info(
            f"Unified model answers for grading → {len(combined_criteria)} criteria across {len(answers)} top-level answers"
        )
        return flattened

    def _cache_rubric_criteria(self, model_data: dict) -> None:
        """Cache allowed criteria and their max marks for strict post-processing.

        We enforce rubric max marks at the post-processing stage and ignore any LLM-supplied
        max_possible values to prevent inflation or hallucinated criteria.
        """
        allowed: set[str] = set()
        max_map: dict[str, float] = {}
        cat_map: dict[str, str] = {}
        exact_match: set[str] = set()
        ordered: list[str] = []
        pos_map: dict[str, int] = {}

        try:
            answers = (model_data or {}).get("answers")
            if not isinstance(answers, list) or not answers:
                self._allowed_criteria_last_run = set()
                self._criterion_max_map_last_run = {}
                self._criterion_category_map_last_run = {}
                self._exact_match_criteria_last_run = set()
                self._rubric_criteria_order_last_run = []
                self._rubric_position_last_run = {}
                return

            # Unified grading payload uses a single answer node.
            criteria = (answers[0] or {}).get("marking_criteria")
            if not isinstance(criteria, list):
                self._allowed_criteria_last_run = set()
                self._criterion_max_map_last_run = {}
                self._criterion_category_map_last_run = {}
                self._exact_match_criteria_last_run = set()
                self._rubric_criteria_order_last_run = []
                self._rubric_position_last_run = {}
                return

            for it in criteria:
                if not isinstance(it, dict):
                    continue
                desc = str(it.get("description", "") or "").strip()
                marks = it.get("marks")

                if not desc:
                    continue
                if not self._is_valid_criterion(desc):
                    continue

                # Only numeric marks are scoreable.
                if not isinstance(marks, (int, float)):
                    continue
                max_marks = float(marks)
                if max_marks < 0:
                    continue

                allowed.add(desc)
                if desc not in pos_map:
                    pos_map[desc] = len(ordered)
                    ordered.append(desc)
                # In case of duplicates, keep the maximum.
                prev = max_map.get(desc)
                if prev is None or max_marks > prev:
                    max_map[desc] = max_marks
                # Cache category from rubric (LLM output never includes this field).
                cat_val = str(it.get("category", "") or "").strip().lower()
                if cat_val and desc not in cat_map:
                    cat_map[desc] = cat_val
                # Cache exact_match flag — disables OF bypass for this criterion.
                if it.get("exact_match"):
                    exact_match.add(desc)

        finally:
            self._allowed_criteria_last_run = allowed
            self._criterion_max_map_last_run = max_map
            self._criterion_category_map_last_run = cat_map
            self._exact_match_criteria_last_run = exact_match
            self._rubric_criteria_order_last_run = ordered
            self._rubric_position_last_run = pos_map

    def _fetch_doc(self, collection_name: str, doc_id: str) -> Optional[dict[str, Any]]:
        """Fetch document by _id."""
        try:
            coll = get_collection(collection_name)
            doc = coll.find_one({"_id": ObjectId(doc_id)})
            if not doc:
                logger.warning(f"No document in {collection_name} for _id={doc_id}")
            return doc
        except Exception as e:
            logger.error(f"Failed to fetch {collection_name} {doc_id}: {e}", exc_info=True)
            return None

    def _clean_for_llm(self, doc: Optional[dict], allowed_keys: list[str]) -> dict:
        if not doc:
            return {}
        return {k: v for k, v in doc.items() if k in allowed_keys}

    def _is_valid_criterion(self, criterion: str) -> bool:
        if not isinstance(criterion, str):
            return False

        clean = criterion.strip().lower()
        if not clean:
            return False

        # Reject pure marking notation patterns
        # Examples: "1/2", "1/4", "2/2", "3/2", "1/2 mk each max 4", "3 1/2", "2 1/2"
        if re.match(r'^\d+/\d+(\s+(mk|marks?).*)?$', clean):
            return False

        # Reject "N 1/2" style mark notations (e.g. "3 1/2" = 3.5 marks)
        if re.match(r'^\d+\s+\d+/\d+\s*$', clean):
            return False

        # Reject if it's only marking notation variations
        if re.match(r'^\s*\d+\s*/\s*\d+\s*(mk|marks)?\s*(each|per)?\s*(max\s*\d+)?\s*$', clean):
            return False

        # Reject handwritten annotation / OCR artefact labels from the marking scheme PDF.
        # These are not real criteria — they are section headings or PDF annotation remnants.
        _junk_labels = {
            "handwritten annotation", "annotation", "hr", "told - land.", "told - land",
            "tutor note", "tutorial note", "marking guide", "mark scheme",
            "scenario", "memorandum", "memo",
        }
        if clean in _junk_labels:
            return False

        words = clean.split()

        # Strip numbered section prefixes like "(4)", "(1)", "1.", "2)" before heading checks.
        # These are section labels, not meaningful numeric content.
        heading_clean = re.sub(r"^\(\d+\)\s*", "", clean)
        heading_clean = re.sub(r"^\d+[.)]\s*", "", heading_clean)
        heading_words = heading_clean.split() if heading_clean else words
        had_number_prefix = (heading_clean != clean)

        # Reject numbered section headings like "(4) Electrostatic spraying room".
        # After stripping the number prefix, if the remaining text is short and non-numeric,
        # it's a section heading — not a grading criterion.
        if had_number_prefix and len(heading_words) <= 5 and not re.search(r"\d", heading_clean):
            if not heading_words or heading_words[0] not in {"dr", "cr"}:
                return False

        # Reject standalone topic/section headings that are just company or section names.
        # These appear when the PDF extractor picks up section titles as criteria.
        # Only reject if the text has no numbers, no Dr/Cr prefix, and looks like a plain heading.
        if len(heading_words) <= 4 and not re.search(r"\d", heading_clean) and heading_words[0] not in {"dr", "cr"}:
            _heading_indicators = {
                "revised", "consolidated", "statement", "changes", "equity",
                "prepare", "calculate", "determine", "explain",
            }
            if any(w in _heading_indicators for w in heading_words):
                return False

        # Journal entry line items are often short but meaningful (e.g., "Dr NCI", "Cr Disposal of subsidiary").
        # Accept common debit/credit prefixes.
        if len(words) >= 2 and words[0] in {"dr", "cr"}:
            return True

        # Accept longer, clearly descriptive criteria
        if len(words) >= 3:
            return True

        # Accept short criteria when they look like genuine line items / calculations
        # (numbers, currency, brackets, etc.)
        if re.search(r"\d", clean):
            return True
        if "(" in clean or ")" in clean:
            return True
        if any(tok in clean for tok in ("gbp", "usd", "eur", "percent")):
            return True

        # Allow-list for common short financial reporting labels
        short_allowlist = {
            "goodwill",
            "reserves",
            "depreciation",
            "impairment",
            "revaluation",
            "nci",
            "oci",
            "eps",
            "investment",
            "associate",
            "subsidiary",
            "sale proceeds",
            "fair value",
            "net assets",
            "share capital",
            "share premium",
            "retained earnings",
            "exchange gain",
            "revaluation gain",
            "revaluation loss",
        }
        if clean in short_allowlist:
            return True

        return False

    def _load_clean_data(self) -> Tuple[dict, dict, dict]:
        q_doc = self._fetch_doc("pac_questions", self.questions_id) if self.questions_id else {}
        m_doc = self._fetch_doc("model_answers", self.model_answers_id) if self.model_answers_id else {}
        s_doc = self._fetch_doc("student_assignments", self.student_answers_id)

        if not s_doc:
            raise GradingError(f"No student answer found for _id={self.student_answers_id}")

        # Only these fields go to LLM — metadata is completely excluded
        q_clean = self._clean_for_llm(q_doc, ["question_title", "description", "total_marks", "questions"])
        m_clean = self._clean_for_llm(m_doc, ["question_title", "description", "total_marks", "answers"])
        s_clean = self._clean_for_llm(s_doc, ["question", "sub_parts"])

        # Grade holistically by combining all sub-answers/criteria into one payload.
        m_clean = self._flatten_model_answers(m_clean, q_clean)

        # Cache rubric criteria for strict post-processing.
        self._cache_rubric_criteria(m_clean)

        return q_clean, m_clean, s_clean

    def _normalize_floating_letter_labels(self, student_data: dict) -> dict:
        """Combine bare letter-labels ('a)', 'b)', '(i)') with their inferred
        numeric parent, in-place on a copy of student_data.

        Some extractions (especially older ones, or when the student omits the
        '4.1' heading because it's pre-printed on the question paper) emit
        sub_parts as 'a)', 'b)', '4.2', '4.3', '4.4' — losing the '4.1'
        parent. The grader then says "Student did not attempt 4.1" even though
        the content is there.

        Inference: if a letter-label appears BEFORE any numeric sub-label, its
        parent is "{main_q}.1". Letter-labels appearing between numeric labels
        inherit the MOST-RECENT prior numeric as their parent. Combined label
        becomes e.g. "4.1(a)" preserving the student's original casing.
        """
        if not isinstance(student_data, dict):
            return student_data
        sub_parts = student_data.get("sub_parts")
        if not isinstance(sub_parts, list) or not sub_parts:
            return student_data

        main_q = str(student_data.get("question", "")).strip() or str(self.question_number)
        if not main_q:
            return student_data

        # Detect: do any bare letter-labels appear BEFORE the first numeric label?
        letter_re = re.compile(r"^[\(\[]?([A-Za-z]+|[ivxIVX]+)[\)\]\.]?$")
        numeric_re = re.compile(r"^\d+(?:\.\d+)*[)\.]?$")

        has_leading_letters = False
        for sp in sub_parts:
            if not isinstance(sp, dict):
                continue
            lab = str(sp.get("question_number", "")).strip()
            if numeric_re.match(lab):
                break
            if letter_re.match(lab):
                has_leading_letters = True
                break

        # Initial inferred parent if there are leading letter-labels with no
        # numeric predecessor: "{main_q}.1".
        current_parent: Optional[str] = f"{main_q}.1" if has_leading_letters else None

        new_sub_parts: list[dict] = []
        renamed_log: list[str] = []
        for sp in sub_parts:
            if not isinstance(sp, dict):
                new_sub_parts.append(sp)
                continue
            lab = str(sp.get("question_number", "")).strip()

            if numeric_re.match(lab):
                # Top-level numeric: update parent context for following letters.
                current_parent = lab.rstrip(")").rstrip(".")
                new_sub_parts.append(sp)
                continue

            m = letter_re.match(lab)
            if m and current_parent:
                letter = m.group(1)
                # Preserve original brackets/casing minimally — combine as parent(letter).
                combined = f"{current_parent}({letter})"
                renamed_log.append(f"{lab!r}→{combined!r}")
                new_sp = dict(sp)
                new_sp["question_number"] = combined
                new_sub_parts.append(new_sp)
                continue

            # Anything else (e.g. already-combined "4.1(a)", or unusual label)
            # passes through unchanged.
            new_sub_parts.append(sp)

        if renamed_log:
            logger.info(
                f"Normalized floating letter-labels in student sub_parts: "
                f"{'; '.join(renamed_log)}"
            )

        out = dict(student_data)
        out["sub_parts"] = new_sub_parts
        return out

    def _format_student_for_prompt(self, student_data: dict) -> str:
        """Flatten student assignment JSON into readable text for the grader.

        Passing a raw dict into the prompt makes it harder for the LLM to reliably
        locate table rows and journal lines.
        """
        if not isinstance(student_data, dict):
            return str(student_data)

        # Normalize floating letter-labels BEFORE flattening into prompt text.
        # This is the fallback for students extracted before the parent-preserving
        # extraction prompt was deployed.
        student_data = self._normalize_floating_letter_labels(student_data)

        def _extract_question_id(label: str) -> Optional[str]:
            """Extract the canonical question number from a sub-part label.

            Handles formats like: "Q-01", "Q.1", "1", "1.1", "1-", "1)", "(a)",
            "Issue-01 Peak State" (not a question-level label).
            Returns the number as a string with leading zeros stripped, or None
            if this doesn't look like a question-level label.
            """
            if not label:
                return None
            lab = label.strip()

            # Skip sub-issue labels like "Issue-01 Peak State" — these are
            # sub-sections within a question, not question-level identifiers.
            if re.match(r"(?:issue|part|section|topic)\s*[-:]?\s*\d", lab, re.IGNORECASE):
                return None

            # "N-" / "N- Topic name" style labels (e.g. "1-", "2- Tech limited:")
            # are scenario/sub-part labels within a question, NOT question IDs.
            # Returning None lets all such sub_parts pass through the filter.
            if re.match(r"^\d+\s*-", lab):
                return None

            # Extract question number from patterns like Q-01, Q.1, Q1, 1), 1.1)
            m = re.match(
                r"^(?:Q(?:uestion)?\.?\s*[-:]?\s*)?(\d+)",
                lab,
                re.IGNORECASE,
            )
            if m:
                return str(int(m.group(1)))  # strip leading zeros: "01" -> "1"

            return None

        target_qid = _extract_question_id(str(self.question_number))
        # Also try plain digit extraction as fallback for target
        if not target_qid:
            digits = re.findall(r"\d+", str(self.question_number))
            target_qid = str(int(digits[0])) if digits else None

        parts: list[str] = []
        q = str(student_data.get("question", "")).strip()
        if q:
            parts.append(f"Question: {q}")

        sub_parts = student_data.get("sub_parts")
        if isinstance(sub_parts, list) and sub_parts:
            # Check if any sub_part has a question-level label matching the target
            any_matches = False
            if target_qid:
                for sp in sub_parts:
                    if not isinstance(sp, dict):
                        continue
                    lab = str(sp.get("question_number", "")).strip()
                    sp_qid = _extract_question_id(lab)
                    if sp_qid == target_qid:
                        any_matches = True
                        break

            in_relevant_block = False
            for sp in sub_parts:
                if not isinstance(sp, dict):
                    continue
                sp_no = str(sp.get("question_number", "")).strip() or q or "(unknown)"

                if any_matches and target_qid:
                    sp_qid = _extract_question_id(sp_no)
                    if sp_qid == target_qid:
                        in_relevant_block = True
                    elif sp_qid is not None and sp_qid != target_qid:
                        # Different question — stop including
                        in_relevant_block = False

                    # Sub-issue labels (Issue-01, etc.) are children of whatever
                    # question block we're currently in. Include them only if
                    # we're in a relevant block.
                    if sp_qid is None:
                        # This is a sub-issue or unlabeled part
                        if not in_relevant_block:
                            continue
                    elif not in_relevant_block:
                        continue

                ans = sp.get("answer")
                ans = ans if isinstance(ans, str) else str(ans or "")
                ans = ans.strip()
                if not ans:
                    continue
                # Preserve tables by keeping answers verbatim.
                parts.append(f"\n--- {sp_no} ---\n{ans}")

        if not parts:
            return json.dumps(student_data, ensure_ascii=False)

        return "\n".join(parts).strip()

    @staticmethod
    def _numbers_in_text(s: str) -> list[str]:
        if not s or not isinstance(s, str):
            return []
        # Pull out sequences that look like accounting numbers.
        return re.findall(r"\d[\d,]*(?:\.\d+)?", s)

    @staticmethod
    def _contains_number_variant(haystack: str, needle: str) -> bool:
        if not haystack or not needle:
            return False
        h = haystack.replace(",", "").replace(" ", "")
        n = needle.replace(",", "").replace(" ", "")
        if n in h:
            return True

        # Expand shorthand magnitudes: "7.2" from "GBP7.2m" may need to match "7200000".
        # Try common multiplier suffixes on the raw needle.
        for suffix, mult in [("m", 1_000_000), ("k", 1_000)]:
            expanded_needle = n + suffix
            parsed = StudentGrader._parse_number_token(expanded_needle)
            if parsed is not None and parsed > 100:
                int_val = int(round(parsed))
                if str(int_val) in h:
                    return True

        return False

    @staticmethod
    def _parse_number_token(token: str) -> Optional[float]:
        """Parse a single numeric token used in marking criteria.

        Supports common forms:
        - 375,000
        - 7.2m (million)
        - 480k (thousand)
        - 50p (pence -> 0.50)
        - 25% (percent -> 0.25)
        - 9/12 (fraction)
        """
        if not token or not isinstance(token, str):
            return None

        t = token.strip().lower()
        t = t.strip("()")
        t = t.replace("£", "").replace("$", "")
        # Handle currency prefixes like "usd3" as well as separate tokens like "USD 3".
        t = re.sub(r"^(gbp|usd|eur)", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\b(gbp|usd|eur)\b", " ", t, flags=re.IGNORECASE)

        # Word magnitudes (common in marking criteria): "3 million" / "8 thousand"
        mult_word = 1.0
        if re.search(r"\bmillion\b", t):
            mult_word = 1_000_000.0
            t = re.sub(r"\bmillion\b", " ", t)
        if re.search(r"\bthousand\b", t):
            mult_word = 1_000.0
            t = re.sub(r"\bthousand\b", " ", t)

        # Word "percent" / "per cent" → treat as % suffix
        t = re.sub(r"\bper\s*cent\b", "%", t)
        t = re.sub(r"\bpercent\b", "%", t)

        t = re.sub(r"\s+", " ", t).strip()
        t = t.replace(",", "").replace(" ", "")

        if not t:
            return None

        # Fraction like 9/12
        frac = re.fullmatch(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", t)
        if frac:
            try:
                num = float(frac.group(1))
                den = float(frac.group(2))
                if den == 0:
                    return None
                return num / den
            except Exception:
                return None

        # Percent
        if t.endswith("%"):
            try:
                return float(t[:-1]) / 100.0
            except Exception:
                return None

        # Pence
        if t.endswith("p") and re.fullmatch(r"\d+(?:\.\d+)?p", t):
            try:
                return float(t[:-1]) / 100.0
            except Exception:
                return None

        # Million / thousand suffixes
        mult = 1.0
        if t.endswith("m") and re.fullmatch(r"\d+(?:\.\d+)?m", t):
            mult = 1_000_000.0
            t = t[:-1]
        elif t.endswith("k") and re.fullmatch(r"\d+(?:\.\d+)?k", t):
            mult = 1_000.0
            t = t[:-1]

        # Plain float/int
        try:
            return float(t) * mult * mult_word
        except Exception:
            return None

    @staticmethod
    def _compute_simple_calc_from_criterion(criterion: str) -> Optional[float]:
        """Compute expected result for simple bracketed calculations.

        Examples:
        - "Cost of investment (375,000 x GBP32)" -> 12000000
        - "Share capital (500,000 x 50p)" -> 250000

        Only handles simple x/* and / expressions inside parentheses. Returns None if
        expression is composite/ambiguous.
        """
        if not criterion or not isinstance(criterion, str):
            return None

        m = re.search(r"\(([^)]*)\)", criterion)
        if not m:
            return None

        expr = m.group(1)
        expr_low = expr.lower()

        # Skip composite expressions; these often list components where evidence may show
        # only a final figure and we can't safely derive a single expected value.
        if "+" in expr_low or "-" in expr_low:
            return None

        # Normalize symbols
        expr_low = expr_low.replace("×", "x")
        expr_low = re.sub(r"\s+", " ", expr_low).strip()

        # Tokenize on operators while keeping them
        parts = re.split(r"\s*(x|\*|/)\s*", expr_low)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) < 3:
            return None

        # Expression must alternate: number, op, number, op, number ...
        # Validate quick
        if parts[1] not in {"x", "*", "/"}:
            return None

        try:
            acc = self_first = StudentGrader._parse_number_token(parts[0])
            if acc is None:
                return None
            i = 1
            while i < len(parts) - 1:
                op = parts[i]
                rhs = StudentGrader._parse_number_token(parts[i + 1])
                if rhs is None:
                    return None
                if op in {"x", "*"}:
                    acc = acc * rhs
                elif op == "/":
                    if rhs == 0:
                        return None
                    acc = acc / rhs
                else:
                    return None
                i += 2
            return acc
        except Exception:
            return None

    @staticmethod
    def _format_expected_number_variants(value: float) -> list[str]:
        """Return a small set of string variants for matching expected values in evidence."""
        try:
            v = float(value)
        except Exception:
            return []

        # If it's very close to an integer, treat it as one.
        if abs(v - round(v)) < 1e-6:
            iv = int(round(v))
            variants = [str(iv)]

            # Also provide compact million/thousand forms commonly used in workings.
            if iv % 1_000_000 == 0:
                m = iv // 1_000_000
                variants.extend([f"{m}m", f"{m} million"])
            elif iv % 1_000 == 0 and iv >= 10_000:
                k = iv // 1_000
                variants.append(f"{k}k")

            # Dedup
            out: list[str] = []
            seen: set[str] = set()
            for s in variants:
                key = s.replace(" ", "").lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
            return out

        # Otherwise keep a couple of sensible formats.
        return [
            (f"{v:.2f}").rstrip("0").rstrip("."),
            (f"{v:.4f}").rstrip("0").rstrip("."),
        ]

    @staticmethod
    def _requires_strict_number_match(criterion: str) -> bool:
        """Heuristic: only enforce strict numeric matching for 'atomic' numeric criteria.

        We avoid enforcing for narrative criteria that merely contain dates/percentages.
        """
        if not isinstance(criterion, str):
            return False

        c = criterion.strip()
        c_low = c.lower()

        # Short currency conversion/FX line-items are typically numeric and should match.
        if c_low.startswith(("usd", "gbp", "eur")):
            nums = StudentGrader._numbers_in_text(c)
            return len(nums) <= 3 and len(c) <= 80

        # Simple bracketed calculations like (375,000 x GBP32).
        # Only enforce on criteria where the calc IS the main content (short to medium length),
        # not long narrative criteria that happen to mention a calc in passing.
        m = re.search(r"\(([^)]*)\)", c)
        if m and len(c) > 100:
            # For long criteria, only apply if the bracket is near the end (the "answer" portion)
            # and the text before the bracket is short.
            bracket_start = m.start()
            text_before = c[:bracket_start].strip()
            if len(text_before) > 60:
                return False
        if m:
            inside = m.group(1).lower()
            if not re.search(r"\d", inside):
                return False
            # Skip composite expressions (they often list components but evidence may show only the final figure)
            if "+" in inside or "-" in inside:
                return False
            if "x" in inside or "*" in inside or "/" in inside:
                return True

        return False



    def _sanitize_holistic_comments(
        self, comments: list, student_text: str
    ) -> list[str]:
        """Pre-flight check on comment anchors before they hit the annotator.

        The annotator dumps any comment whose anchor it cannot find into
        `unanchored_comments` — invisible to the student. Two LLM failure modes
        cause this:
          1. Anchor too long (6+ words) — spans PDF lines, exact match fails.
          2. Hallucinated anchor — not a verbatim substring of the student text.

        This pass trims oversize anchors to a 3-5 word window and verifies
        each anchor is actually present in the student text. Comments that
        can't be salvaged are dropped with a debug log.
        """
        if not isinstance(comments, list) or not comments:
            return []

        # Normalised version of student text for substring search — tolerant of
        # whitespace differences but preserves typos/casing the LLM should copy.
        st = student_text or self._student_text_last_run or ""
        st_norm = re.sub(r"\s+", " ", st)
        st_norm_lower = st_norm.lower()

        def _anchor_present(anchor: str) -> bool:
            if not anchor:
                return False
            a_norm = re.sub(r"\s+", " ", anchor).strip()
            return a_norm.lower() in st_norm_lower

        prefix_re = re.compile(r"^\s*\[([^\]]+)\]\s*")
        out: list[str] = []
        dropped = 0
        trimmed = 0

        for c in comments:
            if not isinstance(c, str) or "→" not in c:
                # Wrong format — pass through; annotator will skip it itself.
                if isinstance(c, str) and c.strip():
                    out.append(c)
                continue

            prefix_match = prefix_re.match(c)
            prefix = prefix_match.group(0) if prefix_match else ""
            body = c[len(prefix):] if prefix else c

            left, right = body.split("→", 1)
            anchor = left.strip().strip('"\'')
            feedback = right.strip()

            if not anchor or not feedback:
                dropped += 1
                logger.debug(f"  Dropping malformed comment: {c[:80]!r}")
                continue

            # 1. Verify the anchor is actually in the student text. If the LLM
            #    hallucinated something the student didn't write, the annotator
            #    will silently fail — drop the comment now.
            if not _anchor_present(anchor):
                # Last-chance salvage: try shorter prefixes (3 words, 4 words).
                a_words = anchor.split()
                salvaged: Optional[str] = None
                for n in (3, 4, 5):
                    if n < len(a_words):
                        candidate = " ".join(a_words[:n])
                        if _anchor_present(candidate):
                            salvaged = candidate
                            break
                if not salvaged:
                    dropped += 1
                    logger.debug(
                        f"  Dropping comment with hallucinated/unverifiable anchor: "
                        f"{anchor!r}"
                    )
                    continue
                anchor = salvaged
                trimmed += 1

            # 2. Trim oversize anchors to a 3-5 word window. Search for a 4-word
            #    or 5-word sub-window that appears verbatim in the student text;
            #    prefer the first such window so the popup lands near the start
            #    of the issue.
            a_words = anchor.split()
            if len(a_words) > 5:
                shorter: Optional[str] = None
                for size in (5, 4, 3):
                    for start in range(0, len(a_words) - size + 1):
                        cand = " ".join(a_words[start:start + size])
                        if _anchor_present(cand):
                            shorter = cand
                            break
                    if shorter:
                        break
                if shorter:
                    anchor = shorter
                    trimmed += 1
                else:
                    # Fall back to first 4 words even if not perfect match.
                    anchor = " ".join(a_words[:4])
                    trimmed += 1

            out.append(f"{prefix}{anchor} → {feedback}")

        if dropped or trimmed:
            logger.info(
                f"Comment sanitization: trimmed {trimmed} anchor(s), "
                f"dropped {dropped} comment(s) with un-locatable anchors"
            )
        return out

    # Stop-words that should never be the FIRST or LAST word of a key_phrase
    # (they make the underline bleed onto a stray article/preposition).
    _KP_STOPWORDS = frozenset({
        "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "by",
        "and", "or", "but", "so", "as", "is", "are", "was", "were", "be",
        "this", "that", "these", "those", "it", "its", "their",
    })

    @classmethod
    def _trim_stopword_edges(cls, key_phrase: str, sentence: str) -> str:
        """Strip leading/trailing stop-words from key_phrase so the underline
        doesn't bleed onto a stray "a" / "the" / "to" / "of" at the edges.

        Example: "Griffins goals aligning to closely to" → "Griffins goals aligning to closely"
        Example: "unable to identify issues in the" → "unable to identify issues"
        Example: "to keep Nicola on as engagement partner" → "keep Nicola on as engagement partner"
        """
        if not key_phrase:
            return key_phrase
        tokens = key_phrase.split()

        def _is_stop(tok: str) -> bool:
            return re.sub(r"[^\w]+", "", tok).lower() in cls._KP_STOPWORDS

        while tokens and _is_stop(tokens[-1]):
            tokens.pop()
        while tokens and _is_stop(tokens[0]):
            tokens.pop(0)
        return " ".join(tokens) if tokens else key_phrase

    @staticmethod
    def _expand_short_key_phrase(short_kp: str, sentence: str) -> str:
        """Grow a too-short key_phrase (1-3 words) into a 4-6 word window of
        surrounding context from the parent sentence.

        Used to avoid placing ticks on bare fragments like "reviewing payroll"
        or "to Yeti's" — those visually land on stray articles in the rendered
        PDF. We find the short phrase inside the sentence and pad outward
        until the slice is 4-6 words, preferring left-padding (subject context)
        over right-padding when the short phrase already contains the verb.
        """
        if not short_kp or not sentence:
            return short_kp
        kp_tokens = short_kp.split()
        if len(kp_tokens) >= 4:
            return short_kp

        sent_tokens = sentence.split()
        if len(sent_tokens) < 4:
            return short_kp  # whole sentence is too short to grow into

        # Locate the short phrase inside the sentence (case-insensitive, tolerant
        # of trailing/leading punctuation differences like "yrs" vs "yrs.").
        def _norm(s: str) -> str:
            return re.sub(r"[^\w]+", "", s).lower()

        kp_norm = _norm(short_kp)
        sent_norm = [_norm(t) for t in sent_tokens]
        for start in range(len(sent_tokens) - len(kp_tokens) + 1):
            window_norm = "".join(sent_norm[start:start + len(kp_tokens)])
            if window_norm == kp_norm:
                # Found placement. Pad to 4-6 words.
                target = min(6, max(4, len(kp_tokens) + 2))
                # Prefer extending leftward first (gives context/subject).
                lo, hi = start, start + len(kp_tokens)
                while (hi - lo) < target and (lo > 0 or hi < len(sent_tokens)):
                    if lo > 0 and (hi - lo) < target:
                        lo -= 1
                    elif hi < len(sent_tokens):
                        hi += 1
                    else:
                        break
                return " ".join(sent_tokens[lo:hi])

        # Phrase not found by full-match — fall back to any 4-5 word window
        # of the sentence that contains at least one substantive keyword from
        # the original short phrase.
        kp_word_set = {_norm(w) for w in kp_tokens if len(w) >= 3}
        for size in (5, 4):
            for start in range(max(0, len(sent_tokens) - size + 1)):
                window = sent_tokens[start:start + size]
                if len(window) < 4:
                    continue
                window_norm_set = {_norm(w) for w in window}
                if kp_word_set & window_norm_set:
                    return " ".join(window)

        return short_kp  # give up; caller will drop it

    # ── Coverage audit helpers (option B post-processing) ──────────────────

    @staticmethod
    def _audit_split_sentences(text: str) -> list[str]:
        """Split student text into sentence-like chunks for keyword scanning."""
        if not text:
            return []
        chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [c.strip() for c in chunks if c and c.strip()]

    @staticmethod
    def _audit_stem_match(kw_token: str, txt_tokens: set) -> bool:
        """True if any text token shares a stem prefix with kw_token.

        Catches morphological variants like physical/physically, decide/decision,
        consent/consenting, separated/separation. Uses a 4–5-char prefix as the stem.
        """
        kw = kw_token.lower()
        if len(kw) < 4:
            return kw in txt_tokens
        stem = kw[:5] if len(kw) >= 6 else kw[:4]
        for t in txt_tokens:
            if len(t) < 3:
                continue
            if t.startswith(stem) or kw.startswith(t[:max(4, min(len(t), 5))]):
                return True
        return False

    @staticmethod
    def _audit_keyword_in_text(keyword: str, text: str) -> bool:
        """Check if a keyword (or close morphological/partial variant) appears in text.

        Multi-word keywords: pass if the literal phrase appears, OR if at least
        half of the substantive (≥4-char) tokens have a stem match in text.
        Single-word keywords: pass on stem-prefix match (handles plural/-ed/-ing/etc.).
        """
        if not keyword or not text:
            return False
        kw_low = keyword.lower().strip()
        txt_low = text.lower()
        txt_tokens = set(re.findall(r"[a-z]+", txt_low))

        # Multi-word / separator-containing keyword.
        if any(c in kw_low for c in (" ", "/", "-")):
            if kw_low in txt_low:
                return True
            kw_tokens = [t for t in re.findall(r"[a-z]+", kw_low) if len(t) >= 4]
            if not kw_tokens:
                return False
            matches = sum(
                1 for kt in kw_tokens if StudentGrader._audit_stem_match(kt, txt_tokens)
            )
            return matches >= max(1, (len(kw_tokens) + 1) // 2)

        # Single-word keyword.
        if len(kw_low) < 3:
            return False
        return StudentGrader._audit_stem_match(kw_low, txt_tokens)

    def _audit_score_criterion(self, criterion: dict, sentence: str) -> float:
        """0..1 score for how strongly a sentence supports a criterion."""
        keywords = [k for k in (criterion.get("keywords") or []) if isinstance(k, str)]
        if not keywords:
            desc = str(criterion.get("description", "") or "")
            keywords = [w for w in re.findall(r"\w+", desc) if len(w) > 4][:6]
            if not keywords:
                return 0.0
        matched = sum(1 for kw in keywords if self._audit_keyword_in_text(kw, sentence))
        return matched / float(len(keywords))

    def _audit_best_sentence(
        self, criterion: dict, sentences: list[str], threshold: float = 0.34
    ) -> Optional[Tuple[str, float]]:
        best, best_score = None, 0.0
        for s in sentences:
            score = self._audit_score_criterion(criterion, s)
            if score > best_score:
                best, best_score = s, score
        if best is not None and best_score >= threshold:
            return best, best_score
        return None

    def _audit_existing_ticks_for_criterion(
        self, criterion: dict, existing_pts: list[dict]
    ) -> int:
        """Count how many existing correct_points already credit this criterion
        (by keyword presence in the tick's text or key_phrase)."""
        keywords = [k for k in (criterion.get("keywords") or []) if isinstance(k, str)]
        if not keywords:
            return 0
        count = 0
        for pt in existing_pts:
            blob = f"{pt.get('text', '')} {pt.get('key_phrase', '')}"
            if any(self._audit_keyword_in_text(kw, blob) for kw in keywords):
                count += 1
        return count

    def _audit_pick_anchor(
        self, sentence: str, keywords: list[str], used_phrases: list[str]
    ) -> Optional[str]:
        """Pick a 4-6 word slice of sentence containing a keyword, with no word
        overlap against used_phrases. Falls back to any non-overlapping window.

        Minimum 4 words: a 1-3 word slice places the tick on an ambiguous
        fragment that visually looks like a tick on a stray article in the PDF.
        """
        if not sentence:
            return None
        used_words: set = set()
        for up in used_phrases:
            used_words.update(re.findall(r"\w+", (up or "").lower()))

        tokens = sentence.split()
        n = len(tokens)
        if n < 4:
            return None

        kw_positions: list[int] = []
        for kw in keywords or []:
            kw_low = kw.lower().strip()
            if not kw_low:
                continue
            kw_tokens = [t for t in re.findall(r"[a-z]+", kw_low) if len(t) >= 3]
            for i, tok in enumerate(tokens):
                tok_clean = re.sub(r"[^a-z]+", "", tok.lower())
                if not tok_clean:
                    continue
                if any(
                    self._audit_stem_match(kt, {tok_clean}) for kt in kw_tokens
                ):
                    kw_positions.append(i)

        for pos in kw_positions:
            for size in (5, 4, 6):
                for start in range(max(0, pos - size + 1), min(n - size + 1, pos + 1) + 1):
                    if start < 0 or start + size > n:
                        continue
                    window = tokens[start:start + size]
                    if len(window) < 4:
                        continue
                    win_words = set(re.findall(r"\w+", " ".join(window).lower()))
                    if not (win_words & used_words):
                        return " ".join(window)

        for size in (5, 4):
            for start in range(max(0, n - size + 1)):
                window = tokens[start:start + size]
                if len(window) < 4:
                    continue
                win_words = set(re.findall(r"\w+", " ".join(window).lower()))
                if not (win_words & used_words):
                    return " ".join(window)
        return None

    def _audit_holistic_coverage(
        self, breakdown: list, student_text: str
    ) -> list:
        """For each sub-question, ensure rubric criteria with sufficient student
        text support have their full mark value's worth of ticks. Adds ticks for
        under-credited criteria, anchored at non-overlapping key_phrases inside
        the best-matching student sentence. Caps at each sub-question's max_marks.

        Guardrails:
        • Dual threshold — easier to AUGMENT criteria the LLM already credited
          (existing_for_crit > 0) than to introduce NEW credit (existing == 0).
        • Per-sentence global cap — any single student sentence can earn at most
          PER_TEXT_TICK_CAP ticks across all leaves (prevents one sentence from
          being credited for every shared-keyword criterion in the rubric).
        """
        if os.getenv("AUDIT_HOLISTIC_COVERAGE", "1").strip().lower() in {"0", "false", "no"}:
            return breakdown
        if not getattr(self, "_holistic_grading", False):
            return breakdown
        if not (self._holistic_sub_questions and student_text):
            return breakdown

        AUGMENT_THRESHOLD = float(os.getenv("AUDIT_AUGMENT_THRESHOLD", "0.34"))
        NEW_THRESHOLD = float(os.getenv("AUDIT_NEW_THRESHOLD", "0.55"))
        PER_TEXT_TICK_CAP = int(os.getenv("AUDIT_PER_TEXT_TICK_CAP", "4"))
        # When the LLM has already credited a parent section to ≥SECTION_TRUST_RATIO
        # of its section_cap, skip the audit ENTIRELY for that section's leaves
        # (no new credit AND no augmenting). The LLM's coverage call is final.
        # Default 0.75 — at ≥75% of the cap, trust the LLM.
        SECTION_TRUST_RATIO = float(os.getenv("AUDIT_SECTION_TRUST_RATIO", "0.75"))

        sq_meta: dict = {}
        for sq in self._holistic_sub_questions:
            label = str(sq.get("sub_question", "")).strip()
            sq_meta[label] = {
                "criteria": sq.get("marking_criteria") or [],
                "max_marks": float(sq.get("max_marks") or 0),
                "parent_section": sq.get("parent_section"),
                "section_cap": sq.get("section_cap"),
            }

        sentences = self._audit_split_sentences(student_text)

        # Seed per-sentence tick counter with all existing LLM ticks across the
        # whole question so audit additions stay under the cap globally.
        sentence_ticks: dict[str, int] = {}
        for item in breakdown:
            for pt in (item.get("_correct_points_with_marks", []) or []):
                txt = pt.get("text", "")
                if txt:
                    sentence_ticks[txt] = sentence_ticks.get(txt, 0) + 1

        # Pre-compute LLM-awarded marks per parent_section. If the LLM has
        # already credited a section close to its cap, we should not add any
        # NEW criteria there (only augment existing partial credit). This
        # prevents the audit from over-shooting on tightly-capped sections
        # like 4.2 (cap=4, where rubric criteria across leaves are similar).
        section_llm_marks: dict[str, float] = {}
        section_caps: dict[str, float] = {}
        for item in breakdown:
            sq = str(item.get("_sub_question", "")).strip()
            meta = sq_meta.get(sq) or {}
            parent = meta.get("parent_section")
            cap = meta.get("section_cap")
            if parent and cap is not None:
                section_llm_marks[parent] = (
                    section_llm_marks.get(parent, 0.0)
                    + float(item.get("marks_awarded", 0) or 0)
                )
                section_caps[parent] = float(cap)

        section_trusted: set[str] = set()
        for parent, llm_total in section_llm_marks.items():
            cap = section_caps.get(parent, 0.0)
            if cap > 0 and llm_total >= cap * SECTION_TRUST_RATIO:
                section_trusted.add(parent)
                logger.info(
                    f"Audit: section {parent} LLM gave {llm_total}/{cap} "
                    f"(≥{SECTION_TRUST_RATIO:.0%}) — skipping audit entirely"
                )

        # Running marks per parent_section so audit additions stop at the cap.
        section_running_marks: dict[str, float] = dict(section_llm_marks)

        audit_log: list[str] = []

        for item in breakdown:
            sq = str(item.get("_sub_question", "")).strip()
            meta = sq_meta.get(sq)
            if not meta or not meta["criteria"]:
                continue

            existing_pts = list(item.get("_correct_points_with_marks", []) or [])
            added = 0
            parent_section = meta.get("parent_section")
            section_cap_val = meta.get("section_cap")
            is_trusted = parent_section in section_trusted

            # Trusted-section short-circuit: when the LLM has already credited
            # this parent section close to its cap (≥ SECTION_TRUST_RATIO), skip
            # the audit entirely for this leaf — no new credit AND no augmenting
            # of partial credits. The LLM's coverage call is treated as final.
            # Without this, partial-credit augmentation can still push the
            # section's total to the cap when teacher would have left it lower.
            if is_trusted:
                continue

            for crit in meta["criteria"]:
                crit_marks = float(crit.get("marks") or 1)
                expected_ticks = int(round(crit_marks * 2))
                existing_for_crit = self._audit_existing_ticks_for_criterion(crit, existing_pts)
                if existing_for_crit >= expected_ticks:
                    continue

                threshold = AUGMENT_THRESHOLD if existing_for_crit > 0 else NEW_THRESHOLD
                match = self._audit_best_sentence(crit, sentences, threshold=threshold)
                if not match:
                    continue
                best_sent, _score = match
                keywords = [k for k in (crit.get("keywords") or []) if isinstance(k, str)]
                used_phrases_in_sent = [
                    pt.get("key_phrase", "") for pt in existing_pts
                    if pt.get("text", "") == best_sent
                ]

                need = expected_ticks - existing_for_crit
                for _ in range(need):
                    # Stop if the parent section_cap is already saturated.
                    if (
                        parent_section
                        and section_cap_val is not None
                        and section_running_marks.get(parent_section, 0.0) >= float(section_cap_val)
                    ):
                        break
                    # Per-sentence global cap — protects against over-crediting
                    # the same student sentence under multiple shared-keyword criteria.
                    if sentence_ticks.get(best_sent, 0) >= PER_TEXT_TICK_CAP:
                        break
                    anchor = self._audit_pick_anchor(best_sent, keywords, used_phrases_in_sent)
                    if not anchor:
                        break
                    existing_pts.append({
                        "text": best_sent,
                        "marks": 0.5,
                        "key_phrase": anchor,
                    })
                    used_phrases_in_sent.append(anchor)
                    sentence_ticks[best_sent] = sentence_ticks.get(best_sent, 0) + 1
                    if parent_section:
                        section_running_marks[parent_section] = (
                            section_running_marks.get(parent_section, 0.0) + 0.5
                        )
                    added += 1

            sub_max = meta["max_marks"]
            if sub_max > 0:
                cap_count = int(round(sub_max / 0.5))
                if len(existing_pts) > cap_count:
                    existing_pts = existing_pts[:cap_count]

            item["_correct_points_with_marks"] = existing_pts
            item["marks_awarded"] = len(existing_pts) * 0.5
            item["evidence"] = list(dict.fromkeys(pt["text"] for pt in existing_pts))

            if added > 0:
                audit_log.append(
                    f"{sq}: +{added} ticks (now {len(existing_pts)} = {item['marks_awarded']} marks)"
                )

        if audit_log:
            logger.info(f"Coverage audit added ticks → {'; '.join(audit_log)}")
        return breakdown

    def _aggregate_holistic_breakdown(self, breakdown: list) -> list:
        """Aggregate per-criterion holistic breakdown into parent-section level entries.

        For theoretical papers, the LLM grades fine-grained sub-questions like
        "4.1(a) Consequences", "4.1(b) Recommendations", etc.  The MongoDB
        breakdown should show one entry per top-level section (4.1, 4.2, …)
        with aggregated marks, while keeping all tick annotation data merged.
        """
        from collections import OrderedDict

        # Build lookup: sub_question_label → {parent_section, section_cap}
        sq_meta: dict = {}
        for sq in (getattr(self, "_holistic_sub_questions", None) or []):
            label = str(sq.get("sub_question", "")).strip()
            sq_meta[label] = {
                "parent_section": sq.get("parent_section"),
                "section_cap": sq.get("section_cap"),
            }

        groups: "OrderedDict[str, list]" = OrderedDict()
        group_cap: dict = {}

        for item in breakdown:
            sub_q = str(item.get("_sub_question", "")).strip()
            meta = sq_meta.get(sub_q, {})
            parent = meta.get("parent_section")
            section_cap = meta.get("section_cap")

            group_key = parent if parent else sub_q
            if group_key not in groups:
                groups[group_key] = []
                group_cap[group_key] = section_cap if (parent and section_cap is not None) else item.get("max_possible", 0)
            groups[group_key].append(item)

        # If every item is its own group there is nothing to aggregate.
        if len(groups) == len(breakdown):
            return breakdown

        aggregated = []
        for group_key, items in groups.items():
            total_awarded = sum(float(i.get("marks_awarded", 0) or 0) for i in items)
            max_possible = float(group_cap.get(group_key) or 0)
            total_awarded = min(total_awarded, max_possible)
            total_awarded = round(total_awarded / 0.5) * 0.5

            # Merge evidence (deduplicated, order preserved).
            seen_ev: set = set()
            all_evidence = []
            for item in items:
                for ev in (item.get("evidence") or []):
                    if ev not in seen_ev:
                        all_evidence.append(ev)
                        seen_ev.add(ev)

            # Merge tick-annotation points.
            all_correct_points = []
            for item in items:
                all_correct_points.extend(item.get("_correct_points_with_marks") or [])
            target_count = int(round(total_awarded / 0.5))
            if len(all_correct_points) > target_count:
                all_correct_points = all_correct_points[:target_count]

            # Merge not-required points (deduplicated).
            seen_nr: set = set()
            all_nr: list = []
            for item in items:
                for nr in (item.get("_not_required_points") or []):
                    key = nr.get("text", "")
                    if key not in seen_nr:
                        all_nr.append(nr)
                        seen_nr.add(key)

            reasons = [i.get("reason", "").strip() for i in items if i.get("reason", "").strip()]
            combined_reason = "; ".join(reasons)

            student_label = next(
                (i.get("_student_label", "") for i in items if i.get("_student_label", "")), ""
            )

            aggregated.append({
                "criterion": f"Sub-question {group_key}",
                "marks_awarded": total_awarded,
                "max_possible": max_possible,
                "reason": combined_reason,
                "evidence": all_evidence,
                "comments_summary": "",
                "_sub_question": group_key,
                "_student_label": student_label,
                "_correct_points_with_marks": all_correct_points,
                "_not_required_points": all_nr,
            })

        return aggregated

    def _run_holistic_grading(self, student_data: dict, model_data: dict, questions_data: dict) -> dict:
        """Execute holistic grading: compare full answers without per-criterion breakdown.

        Returns a dict in the same shape as the standard grading response so that
        _build_grade_doc() can process it uniformly via a conversion step.
        """
        student_text = self._format_student_for_prompt(student_data)
        self._student_text_last_run = student_text or ""

        compact_json_kwargs = {"ensure_ascii": False, "separators": (",", ":")}
        try:
            self._question_text_last_run = json.dumps(questions_data, **compact_json_kwargs)
        except Exception:
            self._question_text_last_run = str(questions_data)
        try:
            self._model_text_last_run = json.dumps(model_data, **compact_json_kwargs)
        except Exception:
            self._model_text_last_run = str(model_data)

        payload = {
            "model_data": json.dumps(model_data, **compact_json_kwargs),
            "chunks": student_text,
            "questions": json.dumps(questions_data, **compact_json_kwargs),
        }

        debug_enabled = os.getenv("DEBUG_SAVE_LLM_OUTPUT", "").strip().lower() in {"1", "true", "yes", "y"}

        def _coerce_holistic(result: Any) -> dict:
            if isinstance(result, BaseModel):
                return result.model_dump()
            if isinstance(result, dict):
                return result
            structured_args = self._extract_structured_args_from_message(result)
            if isinstance(structured_args, dict):
                return structured_args
            content = getattr(result, "content", None)
            if content is None:
                content = str(result)
            raw = str(content)
            json_text = self._extract_json_from_text(raw)
            if not json_text:
                raise GradingError("Empty holistic grading output")
            return json.loads(json_text)

        # Attempt 1: structured output
        holistic_parsed = None
        try:
            if self.holistic_chain_structured is not None:
                output = self.holistic_chain_structured.invoke(payload)
                holistic_parsed = _coerce_holistic(output)
                validated = HolisticGradingResponse(**holistic_parsed)
                holistic_parsed = validated.model_dump()
                logger.info(f"Holistic grading complete → {self.student_name} (Q{self.question_number}) [structured]")
        except Exception as e:
            logger.warning(f"Holistic structured grading failed; falling back to text: {e}")
            holistic_parsed = None

        # Degenerate-case detection: some providers/structured-output paths return
        # the top-level score but drop the nested sub_grades array entirely. Treat
        # that as a structured-output failure and fall through to the text path,
        # which carries the same content as raw JSON in the message body.
        if (
            holistic_parsed is not None
            and float(holistic_parsed.get("score", 0) or 0) > 0
            and not (holistic_parsed.get("sub_grades") or [])
        ):
            logger.warning(
                "Holistic structured output reported "
                f"score={holistic_parsed.get('score')} but returned 0 sub_grades — "
                "treating as parse failure and retrying via text path"
            )
            holistic_parsed = None

        # Attempt 2: text output + parse
        if holistic_parsed is None:
            try:
                output = self.holistic_chain_text.invoke(payload)
                holistic_parsed = _coerce_holistic(output)
                validated = HolisticGradingResponse(**holistic_parsed)
                holistic_parsed = validated.model_dump()
                # Same degenerate-case guard for the text path.
                if (
                    float(holistic_parsed.get("score", 0) or 0) > 0
                    and not (holistic_parsed.get("sub_grades") or [])
                ):
                    raise GradingError(
                        f"Text path returned score={holistic_parsed.get('score')} with 0 sub_grades"
                    )
                logger.info(f"Holistic grading complete → {self.student_name} (Q{self.question_number}) [text]")
            except Exception as e:
                logger.warning(f"Holistic text grading failed; attempting repair: {e}")
                holistic_parsed = None

        # Attempt 3: repair
        if holistic_parsed is None:
            try:
                raw_content = getattr(output, "content", None) if "output" in locals() else None
                raw_content = raw_content if raw_content is not None else ""
                repair_prompt = (
                    "You MUST return ONLY valid JSON (no markdown, no commentary). "
                    "Fix the following output to match this exact schema:\n"
                    "{\n"
                    "  \"question_number\": string,\n"
                    "  \"score\": number,\n"
                    "  \"total_marks\": number,\n"
                    "  \"sub_grades\": [\n"
                    "    {\n"
                    "      \"sub_question\": string,\n"
                    "      \"student_label\": string,\n"
                    "      \"marks_awarded\": number,\n"
                    "      \"max_marks\": number,\n"
                    "      \"reason\": string,\n"
                    "      \"correct_points\": [{\"text\": string, \"marks\": number, \"key_phrase\": string}, ...]\n"
                    "    }\n"
                    "  ],\n"
                    "  \"comments\": [string, ...]\n"
                    "}\n\n"
                    "OUTPUT TO FIX:\n"
                    f"{raw_content}"
                )
                fixed = llm_grader.invoke(repair_prompt)
                fixed_content = getattr(fixed, "content", None) or str(fixed)
                fixed_json = self._extract_json_from_text(str(fixed_content))
                holistic_parsed = json.loads(fixed_json)
                validated = HolisticGradingResponse(**holistic_parsed)
                holistic_parsed = validated.model_dump()
                logger.info(f"Holistic grading complete → {self.student_name} (Q{self.question_number}) [repaired]")
            except Exception as e2:
                logger.error("Holistic grading chain failed", exc_info=True)
                raise GradingError("Holistic grading step failed") from e2

        # Convert holistic response to standard grades format for _build_grade_doc().
        # Each sub_grade becomes a breakdown item where:
        #   criterion = "Sub-question <sub_question>" (or just question number for single block)
        #   evidence_list = correct_points texts (for anchoring)
        #   _correct_points_with_marks = full objects with per-point marks (for tick annotation)
        #   marks_awarded / max_possible = sub-question marks
        breakdown = []
        for sg in holistic_parsed.get("sub_grades", []):
            sq_label = sg.get("sub_question", "")

            # Strip artificial chunk separator dashes so annotator finds real PDF text.
            # Chunks are formatted as "--- 4.1 ---\n<answer>" so the LLM returns
            # "--- 4.1 ---" verbatim. Strip to just "4.1" for PDF search.
            student_label = sg.get("student_label", "")
            student_label = re.sub(r'^[-\s]+|[-\s]+$', '', student_label).strip()

            criterion_text = f"Sub-question {sq_label}" if len(holistic_parsed.get("sub_grades", [])) > 1 else f"Question {self.question_number}"

            # Marks awarded for this sub-question — round to nearest 0.5.
            marks_awarded = round(float(sg.get("marks_awarded", 0) or 0) / 0.5) * 0.5

            # correct_points is now list of {"text": str, "marks": float, "key_phrase": str}
            raw_points = sg.get("correct_points", [])
            # Handle both formats: list of objects or list of strings (backward compat)
            evidence_texts = []
            points_with_marks = []
            for pt in raw_points:
                if isinstance(pt, dict):
                    text = str(pt.get("text", "")).strip()
                    # Every correct_point must be exactly 0.5 marks (1 tick).
                    # If the LLM outputs marks > 0.5, split into multiple 0.5 entries.
                    pt_marks = float(pt.get("marks", 0.5) or 0.5)
                    pt_marks = max(0.5, round(pt_marks / 0.5) * 0.5)
                    key_phrase = str(pt.get("key_phrase", "")).strip()
                    # Enforce 4-6 word window. Long phrases (>6 words) span PDF
                    # lines and can't be found; short phrases (<4 words) place
                    # the tick on an ambiguous fragment that visually looks like
                    # a tick on a stray article ("a", "the") in the rendered PDF.
                    kp_words = key_phrase.split()
                    if len(kp_words) > 6:
                        key_phrase = " ".join(kp_words[:6])
                    elif 0 < len(kp_words) < 4 and text:
                        # Try to grow the slice in-place by extending into the
                        # surrounding sentence words.
                        key_phrase = self._expand_short_key_phrase(key_phrase, text)
                        kp_words = key_phrase.split()
                        if len(kp_words) < 4:
                            # Couldn't grow to ≥4 words — drop this tick rather
                            # than place it on a misleading fragment.
                            logger.debug(
                                f"  Dropping tick with un-growable short key_phrase: "
                                f"{pt.get('key_phrase', '')!r} (text: {text[:60]!r})"
                            )
                            continue
                    # Trim trailing/leading stop-words so the underline doesn't
                    # extend onto a stray article/preposition ("to", "the", "a",
                    # "of", "in"). This is what creates the visual "tick on a"
                    # complaint — the rect ends on a stop word and the underline
                    # bleeds onto it. After trim, re-expand if we fell under 4.
                    key_phrase = self._trim_stopword_edges(key_phrase, text)
                    kp_words = key_phrase.split()
                    if len(kp_words) < 4 and text:
                        key_phrase = self._expand_short_key_phrase(key_phrase, text)
                        kp_words = key_phrase.split()
                        if len(kp_words) < 4:
                            logger.debug(
                                f"  Dropping tick after stop-word trim left short phrase: "
                                f"{pt.get('key_phrase', '')!r} (text: {text[:60]!r})"
                            )
                            continue
                    if text:
                        evidence_texts.append(text)
                        if pt_marks == 0.5:
                            points_with_marks.append({
                                "text": text,
                                "marks": 0.5,
                                "key_phrase": key_phrase,
                            })
                        else:
                            # Split into N × 0.5 entries. First entry keeps key_phrase;
                            # subsequent entries reuse the same text (annotator underlines
                            # the same line, placing additional ticks alongside).
                            n_ticks = int(round(pt_marks / 0.5))
                            for tick_i in range(n_ticks):
                                points_with_marks.append({
                                    "text": text,
                                    "marks": 0.5,
                                    "key_phrase": key_phrase if tick_i == 0 else "",
                                })
                elif isinstance(pt, str) and pt.strip():
                    evidence_texts.append(pt.strip())
                    points_with_marks.append({"text": pt.strip(), "marks": 0.5, "key_phrase": ""})

            # Strict consistency: marks_awarded MUST equal len(points_with_marks) × 0.5.
            # Each correct_point = 0.5 marks = 1 tick on the PDF, so the totals
            # rendered to the student cannot diverge from the visible tick count.
            if marks_awarded > 0:
                target_count = int(round(marks_awarded / 0.5))
                if len(points_with_marks) > target_count:
                    # Too many evidenced ticks — LLM reported lower marks; raise marks
                    # to match the evidence it produced (each tick is 0.5 of evidence).
                    marks_awarded = len(points_with_marks) * 0.5
                elif len(points_with_marks) < target_count:
                    # Fewer evidenced ticks than LLM-reported marks — trust the
                    # evidence: marks must equal the visible tick count × 0.5.
                    marks_awarded = len(points_with_marks) * 0.5
            elif points_with_marks:
                # LLM reported 0 marks but produced ticks — trust the ticks.
                marks_awarded = len(points_with_marks) * 0.5

            # Re-cap at this sub-question's own max_marks after the alignment.
            sq_max = float(sg.get("max_marks", 0) or 0)
            if sq_max > 0 and marks_awarded > sq_max:
                marks_awarded = sq_max
                target_count = int(round(marks_awarded / 0.5))
                if len(points_with_marks) > target_count:
                    points_with_marks = points_with_marks[:target_count]

            # Off-topic ("Not required") points — flagged by LLM, no marks.
            # Same key_phrase truncation rule as correct_points.
            raw_nr = sg.get("not_required_points", []) or []
            not_required_points: list[dict] = []
            for nr in raw_nr:
                if not isinstance(nr, dict):
                    continue
                nr_text = str(nr.get("text", "")).strip()
                if not nr_text:
                    continue
                nr_kp = str(nr.get("key_phrase", "")).strip()
                kp_words = nr_kp.split()
                if len(kp_words) > 6:
                    nr_kp = " ".join(kp_words[:6])
                not_required_points.append({
                    "text": nr_text,
                    "key_phrase": nr_kp,
                    "reason": str(nr.get("reason", "")).strip(),
                })

            breakdown.append({
                "criterion": criterion_text,
                "marks_awarded": marks_awarded,
                "max_possible": float(sg.get("max_marks", 0) or 0),
                "reason": sg.get("reason", ""),
                "evidence": evidence_texts,
                "comments_summary": "",
                # Extra fields for annotation
                "_sub_question": sq_label,
                "_student_label": student_label,
                "_correct_points_with_marks": points_with_marks,
                "_not_required_points": not_required_points,
            })

        # Guard: if the LLM reported a non-zero score but produced no sub_grades,
        # the structured output parsing silently dropped the breakdown. Fail loudly
        # so the text-based fallback / repair path can try instead of saving 0/20.
        llm_score_raw = float(holistic_parsed.get("score", 0) or 0)
        if llm_score_raw > 0 and not breakdown:
            raise GradingError(
                f"Holistic LLM reported score={llm_score_raw} but returned "
                f"0 sub_grades — structured output likely failed to parse"
            )

        # Coverage audit: add ticks for rubric criteria the LLM under-credited.
        # The audit becomes the authoritative source of marks for each sub-question;
        # the LLM's reported score is no longer used as an upper bound after this.
        breakdown = self._audit_holistic_coverage(breakdown, student_text)

        # Aggregate fine-grained sub-question entries into parent-section level
        # (e.g. "4.1(a) Consequences" + "4.1(b) Recommendations" + … → "4.1").
        breakdown = self._aggregate_holistic_breakdown(breakdown)

        # Recompute total score from audited+aggregated breakdown so the
        # student-facing total matches the tick counts shown in each section.
        total_max = float(holistic_parsed.get("total_marks", 0) or 0)
        audited_score = sum(float(b.get("marks_awarded", 0) or 0) for b in breakdown)
        if total_max > 0:
            audited_score = min(audited_score, total_max)
        audited_score = round(audited_score / 0.5) * 0.5

        comments = self._sanitize_holistic_comments(
            holistic_parsed.get("comments", []) or [],
            student_text,
        )

        converted = {
            "grades": [{
                "question_number": holistic_parsed.get("question_number", self.question_number),
                "score": audited_score,
                "total_marks": total_max,
                "comments": comments,
                "correct_words": [],
                "breakdown": breakdown,
            }]
        }

        logger.info(
            f"Holistic grading converted to standard format: "
            f"{len(breakdown)} section-level breakdown items, "
            f"score={audited_score}/{total_max} "
            f"(LLM-only score was {holistic_parsed.get('score', 0)})"
        )
        return converted

    def _run_grading(self, student_data: dict, model_data: dict, questions_data: dict) -> dict:
        """Execute grading chain with clean content (holistic evaluation against all criteria)."""

        # Route to holistic grading when no marking criteria exist.
        if self._holistic_grading:
            return self._run_holistic_grading(student_data, model_data, questions_data)

        student_text = self._format_student_for_prompt(student_data)
        self._student_text_last_run = student_text or ""
        # Use compact JSON to reduce prompt/token bloat.
        # (This can materially reduce latency/cost on large rubrics.)
        compact_json_kwargs = {"ensure_ascii": False, "separators": (",", ":")}

        # Store the raw payload blobs so downstream guardrails can detect evidence that
        # originates from the question / marking guide rather than student work.
        try:
            self._question_text_last_run = json.dumps(questions_data, **compact_json_kwargs)
        except Exception:
            self._question_text_last_run = str(questions_data)
        try:
            self._model_text_last_run = json.dumps(model_data, **compact_json_kwargs)
        except Exception:
            self._model_text_last_run = str(model_data)

        payload = {
            "model_data": json.dumps(model_data, **compact_json_kwargs),
            "chunks": student_text,
            "questions": json.dumps(questions_data, **compact_json_kwargs),
        }

        debug_enabled = os.getenv("DEBUG_SAVE_LLM_OUTPUT", "").strip().lower() in {"1", "true", "yes", "y"}
        debug_dir = os.getenv("DEBUG_LLM_OUTPUT_DIR", "logs").strip() or "logs"
        debug_max_chars = int(os.getenv("DEBUG_LLM_OUTPUT_MAX_CHARS", "20000"))

        def _truncate(s: str) -> str:
            if not isinstance(s, str):
                s = str(s)
            if debug_max_chars <= 0:
                return s
            return s[:debug_max_chars]

        def _capture_debug(
            attempt: str,
            stage: str,
            output_obj: Any = None,
            raw_text: Optional[str] = None,
            error: Optional[BaseException] = None,
        ) -> None:
            if not debug_enabled:
                return

            try:
                if raw_text is None and output_obj is not None:
                    raw_text = getattr(output_obj, "content", None)
                    if raw_text is None:
                        raw_text = str(output_obj)
                raw_text = _truncate(raw_text or "")

                record: dict[str, Any] = {
                    "ts": datetime.utcnow().isoformat(),
                    "attempt": attempt,
                    "stage": stage,
                    "provider": os.getenv("GRADING_PROVIDER") or os.getenv("LLM_PROVIDER") or "",
                    "model": os.getenv("LLM_GRADER_MODEL") or "",
                    "error": str(error) if error else "",
                    "traceback": traceback.format_exc() if error else "",
                    "raw": raw_text,
                }
                self._llm_debug_trace.append(record)

                os.makedirs(debug_dir, exist_ok=True)
                fname = f"llm_grading_debug_{self.student_name}_Q{self.question_number}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json"
                safe_fname = re.sub(r"[^a-zA-Z0-9._-]+", "_", fname)
                fpath = os.path.join(debug_dir, safe_fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved LLM debug trace → {fpath}")
            except Exception:
                # Never let debug capture break grading.
                pass

        def _norm_for_evidence_match(s: str) -> str:
            if not s or not isinstance(s, str):
                return ""
            s = s.replace("\u00a0", " ")
            s = s.replace("×", "x")
            s = s.replace("–", "-").replace("—", "-")
            s = re.sub(r"\s+", " ", s).strip().lower()
            # Remove most punctuation while keeping separators meaningful for ratios.
            s = re.sub(r"[^a-z0-9%/().,\- ]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _is_distinctive_short_evidence(ev_norm: str) -> bool:
            if not ev_norm or not isinstance(ev_norm, str):
                return False
            compact = ev_norm.replace(" ", "")
            if re.search(r"\d", ev_norm):
                return len(compact) >= 4
            if "/" in ev_norm or "%" in ev_norm:
                return len(compact) >= 3
            words = ev_norm.split()
            return len(words) <= 3 and any(len(w) >= 7 for w in words)

        def _evidence_present_in_student(evidence_items: list[str]) -> bool:
            student_blob = _norm_for_evidence_match(self._student_text_last_run)
            if not student_blob:
                return False
            for ev in evidence_items:
                if not ev or not isinstance(ev, str):
                    continue
                ev_norm = _norm_for_evidence_match(ev)
                if len(ev_norm) < 12 and not _is_distinctive_short_evidence(ev_norm):
                    continue

                # Fast path: full normalized substring.
                if ev_norm in student_blob:
                    return True

                # Sliding-window fallback: LLMs sometimes add/remove minor punctuation in
                # verbatim quotes.  A contiguous N-word window in the student text is
                # sufficient to confirm the quote is genuine.
                words = [w for w in ev_norm.split() if len(w) >= 3]
                if len(words) >= 10:
                    for i in range(0, min(len(words) - 9, 8)):
                        window = " ".join(words[i:i + 10])
                        if window in student_blob:
                            return True
                elif len(words) >= 5:
                    for i in range(0, min(len(words) - 4, 6)):
                        window = " ".join(words[i:i + 5])
                        if window in student_blob:
                            return True
            return False

        def _coerce_to_dict(result: Any) -> dict:
            if isinstance(result, BaseModel):
                return result.model_dump()
            if isinstance(result, dict):
                return result

            structured_args = self._extract_structured_args_from_message(result)
            if isinstance(structured_args, dict):
                return structured_args

            # Try content-based JSON parsing (common for non-tool providers)
            content = getattr(result, "content", None)
            if content is None:
                content = str(result)
            raw = str(content)
            json_text = self._extract_json_from_text(raw)
            if not json_text:
                raise GradingError("Empty grading output")
            try:
                return json.loads(json_text)
            except Exception as je:
                raise GradingError(f"Invalid JSON from grader: {je}") from je

        # Attempt 1: structured output if available
        try:
            if self.grade_chain_structured is not None:
                output = self.grade_chain_structured.invoke(payload)
                _capture_debug("structured", "received", output_obj=output)
                parsed = _coerce_to_dict(output)
                _capture_debug("structured", "parsed", raw_text=json.dumps(parsed, ensure_ascii=False)[:debug_max_chars] if debug_enabled else None)
                # Validate schema
                validated = LLMGradingResponse(**parsed)
                logger.info(f"Grading complete → {self.student_name} (Q{self.question_number}) [structured]")
                result = validated.model_dump()
                # Guardrail: evidence must be a verbatim quote that exists in the student text.
                try:
                    for g in result.get("grades", []) or []:
                        for bi in g.get("breakdown", []) or []:
                            if float(bi.get("marks_awarded", 0) or 0) <= 0:
                                continue
                            ev = bi.get("evidence", [])
                            ev_list = ev if isinstance(ev, list) else ([str(ev)] if ev else [])
                            if not _evidence_present_in_student([str(x) for x in ev_list if x is not None]):
                                bi["marks_awarded"] = 0.0
                                bi["reason"] = ("Marks revoked by guardrails: evidence not found verbatim in student answer")
                except Exception:
                    # Non-fatal: fallback guardrails will still run in _build_grade_doc.
                    pass
                return result
        except Exception as e:
            _capture_debug("structured", "error", output_obj=locals().get("output"), error=e)
            logger.warning(f"Structured grading failed; falling back to text parsing: {e}")

        # Attempt 2: text output + parse
        try:
            output = self.grade_chain_text.invoke(payload)
            _capture_debug("text", "received", output_obj=output)
            parsed = _coerce_to_dict(output)
            _capture_debug("text", "parsed", raw_text=json.dumps(parsed, ensure_ascii=False)[:debug_max_chars] if debug_enabled else None)
            validated = LLMGradingResponse(**parsed)
            logger.info(f"Grading complete → {self.student_name} (Q{self.question_number}) [text]")
            result = validated.model_dump()
            try:
                for g in result.get("grades", []) or []:
                    for bi in g.get("breakdown", []) or []:
                        if float(bi.get("marks_awarded", 0) or 0) <= 0:
                            continue
                        ev = bi.get("evidence", [])
                        ev_list = ev if isinstance(ev, list) else ([str(ev)] if ev else [])
                        if not _evidence_present_in_student([str(x) for x in ev_list if x is not None]):
                            bi["marks_awarded"] = 0.0
                            bi["reason"] = ("Marks revoked by guardrails: evidence not found verbatim in student answer")
            except Exception:
                pass
            return result
        except Exception as e:
            _capture_debug("text", "error", output_obj=locals().get("output"), error=e)
            logger.warning(f"Text grading parse failed; attempting one JSON repair pass: {e}")

        # Attempt 3: repair by asking the same model to output strict JSON only
        try:
            # Get the raw text from the previous output if possible
            raw_content = getattr(output, "content", None) if "output" in locals() else None
            raw_content = raw_content if raw_content is not None else str(e)

            repair_prompt = (
                "You MUST return ONLY valid JSON (no markdown, no commentary). "
                "Fix the following output to match this exact schema:\n"
                "{\n"
                "  \"grades\": [\n"
                "    {\n"
                "      \"question_number\": string,\n"
                "      \"score\": number,\n"
                "      \"total_marks\": number,\n"
                "      \"comments\": [string, ...],\n"
                "      \"correct_words\": [string, ...],\n"
                "      \"breakdown\": [\n"
                "        {\n"
                "          \"criterion\": string,\n"
                "          \"marks_awarded\": number,\n"
                "          \"max_possible\": number,\n"
                "          \"reason\": string,\n"
                "          \"evidence\": [string, ...],\n"
                "          \"comments_summary\": string\n"
                "        }\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "OUTPUT TO FIX:\n"
                f"{raw_content}"
            )

            fixed = llm_grader.invoke(repair_prompt)
            _capture_debug("repair", "received", output_obj=fixed)
            fixed_content = getattr(fixed, "content", None)
            fixed_content = fixed_content if fixed_content is not None else str(fixed)
            fixed_json = self._extract_json_from_text(str(fixed_content))
            parsed = json.loads(fixed_json)
            _capture_debug("repair", "parsed", raw_text=json.dumps(parsed, ensure_ascii=False)[:debug_max_chars] if debug_enabled else None)
            validated = LLMGradingResponse(**parsed)
            logger.info(f"Grading complete → {self.student_name} (Q{self.question_number}) [repaired]")
            result = validated.model_dump()
            try:
                for g in result.get("grades", []) or []:
                    for bi in g.get("breakdown", []) or []:
                        if float(bi.get("marks_awarded", 0) or 0) <= 0:
                            continue
                        ev = bi.get("evidence", [])
                        ev_list = ev if isinstance(ev, list) else ([str(ev)] if ev else [])
                        if not _evidence_present_in_student([str(x) for x in ev_list if x is not None]):
                            bi["marks_awarded"] = 0.0
                            bi["reason"] = ("Marks revoked by guardrails: evidence not found verbatim in student answer")
            except Exception:
                pass
            return result
        except Exception as e2:
            _capture_debug("repair", "error", output_obj=locals().get("fixed"), error=e2)
            logger.error("Grading chain failed", exc_info=True)
            raise GradingError("Grading step failed") from e2

    def _build_grade_doc(self, parsed_grades: dict, questions_data: dict) -> dict:
        now_iso = datetime.utcnow().isoformat()

        # Conservative heuristic partial credit is OFF by default.
        # Enable explicitly with ENABLE_PARTIAL_CREDIT=1/true/yes.
        partial_credit_enabled = os.getenv("ENABLE_PARTIAL_CREDIT", "0").strip().lower() in {"1", "true", "yes", "y"}

        allowed_criteria = self._allowed_criteria_last_run or set()
        rubric_max_map = self._criterion_max_map_last_run or {}

        _STOPWORDS = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "for",
            "on",
            "at",
            "as",
            "is",
            "are",
            "be",
            "been",
            "being",
            "with",
            "from",
            "by",
            "this",
            "that",
            "these",
            "those",
            "should",
            "would",
            "will",
            "must",
            "therefore",
            "figure",
            "million",
            "thousand",
            "gbp",
        }

        def _partial_overlap_ok(criterion_text: str, evid_items: list[str]) -> bool:
            """Heuristic for partial credit.

            Returns True when evidence suggests the student is addressing the criterion but
            the response is incomplete/incorrect. Conservative: requires multiple keyword/number overlaps.
            """
            if not criterion_text or not evid_items:
                return False
            crit = _norm_for_evidence_match(criterion_text)
            ev_blob = _norm_for_evidence_match(" ".join(evid_items))
            if not crit or not ev_blob:
                return False

            # Numbers in criterion (e.g., 300,000 / 9/12) are strong signals.
            crit_nums = self._numbers_in_text(criterion_text)
            num_hits = 0
            for n in crit_nums:
                if self._contains_number_variant(ev_blob, n):
                    num_hits += 1

            # Keyword overlap: long-ish content words.
            crit_words = [w for w in re.findall(r"[a-z]{4,}", crit) if w not in _STOPWORDS]
            # Emphasize accounting nouns that often matter for journals.
            boosted = {"goodwill", "disposal", "associate", "subsidiary", "investment", "nci", "retained", "earnings", "assets", "surplus", "revaluation", "oci", "impairment", "profit", "loss"}
            hits = 0
            for w in crit_words:
                if w in ev_blob:
                    hits += 1
            boosted_hits = sum(1 for w in boosted if w in crit and w in ev_blob)

            # Require either:
            # - strong keyword overlap, OR
            # - some keyword overlap + at least one number match.
            return (hits + boosted_hits) >= 2 or ((hits + boosted_hits) >= 1 and num_hits >= 1)

        def _quantize_to_quarter(mark: float) -> float:
            try:
                m = float(mark)
            except Exception:
                return 0.0
            return round(m / 0.25) * 0.25

        def _extract_expected_gbp_amount(criterion_text: str) -> Optional[float]:
            """Extract a single expected GBP amount from a criterion.

            Handles patterns like:
            - "GBP5.4 million"
            - "GBP630,000"
            Returns a float in absolute GBP units.

            Intentionally conservative: returns only the first clear GBP amount.
            """
            if not criterion_text or not isinstance(criterion_text, str):
                return None

            # Prefer amounts explicitly prefixed by GBP.
            m = re.search(
                r"\bgbp\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|m|thousand|k)?\b",
                criterion_text,
                flags=re.IGNORECASE,
            )
            if not m:
                return None

            raw = m.group(1)
            suffix = (m.group(2) or "").lower().strip()
            try:
                val = float(raw.replace(",", ""))
            except Exception:
                return None

            if suffix in ("million", "m"):
                val *= 1_000_000.0
            elif suffix in ("thousand", "k"):
                val *= 1_000.0
            return val

        def _safe_eval_arithmetic(expr: str) -> Optional[float]:
            """Safely evaluate simple arithmetic expressions.

            Allowed: + - * / parentheses, numeric literals, unary +/-.
            Also supports suffixes like 7.2m / 300k and percent literals like 25%.
            """
            if not expr or not isinstance(expr, str):
                return None

            s = expr.strip().lower()
            if len(s) > 120:
                return None

            # Normalize common tokens
            s = s.replace(",", "")
            s = s.replace("×", "*").replace("x", "*")

            # Convert percents: 25% => (25/100)
            s = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", s)

            # Convert k/m suffixes: 7.2m => (7.2*1000000)
            s = re.sub(r"(\d+(?:\.\d+)?)\s*m\b", r"(\1*1000000)", s)
            s = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", r"(\1*1000)", s)

            # Strip currency symbols/words
            s = re.sub(r"[£$]", "", s)
            s = re.sub(r"\bgbp\b", "", s)
            s = re.sub(r"\s+", "", s)

            # Some extracted snippets include extra outer parentheses or a trailing ')'
            # (e.g., when pulled from inside a larger expression like 12750000+(7200000*9/12)).
            # Stripping outer parens makes parsing robust while keeping inner grouping.
            s = s.strip("()")

            # Must contain at least one operator.
            if not re.search(r"[+\-*/]", s):
                return None

            try:
                tree = ast.parse(s, mode="eval")
            except Exception:
                return None

            allowed_nodes = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.USub,
                ast.UAdd,
                ast.Constant,
                ast.Load,
            )

            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    return None

            def _eval(n: ast.AST) -> float:
                if isinstance(n, ast.Expression):
                    return _eval(n.body)
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                    return float(n.value)
                if isinstance(n, ast.UnaryOp):
                    v = _eval(n.operand)
                    if isinstance(n.op, ast.USub):
                        return -v
                    if isinstance(n.op, ast.UAdd):
                        return v
                    raise ValueError("bad unary")
                if isinstance(n, ast.BinOp):
                    left = _eval(n.left)
                    right = _eval(n.right)
                    if isinstance(n.op, ast.Add):
                        return left + right
                    if isinstance(n.op, ast.Sub):
                        return left - right
                    if isinstance(n.op, ast.Mult):
                        return left * right
                    if isinstance(n.op, ast.Div):
                        return left / right
                    raise ValueError("bad binop")
                raise ValueError("bad node")

            try:
                out = float(_eval(tree))
                if not math.isfinite(out):
                    return None
                return out
            except Exception:
                return None

        def _award_if_calc_matches_expected(
            criterion_text: str,
            evid_items: list[str],
            max_possible: float,
        ) -> tuple[bool, str]:
            """Return (award?, reason_suffix) if evidence contains calc matching expected GBP amount."""
            expected = _extract_expected_gbp_amount(criterion_text)
            if expected is None:
                return False, ""

            ev_blob = " ".join([e for e in evid_items if isinstance(e, str)])
            if not ev_blob:
                return False, ""

            # Context guard: a criterion that explicitly names profit/contribution/revenue/income
            # as the concept being measured requires the evidence to also contain that vocabulary.
            # Without this, a calculation like "7,200,000*9/12" in the student's net-assets working
            # table would be incorrectly credited for a criterion about profit contribution, purely
            # because the arithmetic result equals the expected GBP amount.
            _INCOME_TERMS_RE = re.compile(
                r"\b(profit|contribution|revenue|income|earning)\b", re.IGNORECASE
            )
            if _INCOME_TERMS_RE.search(criterion_text) and not _INCOME_TERMS_RE.search(ev_blob):
                return False, ""

            # Find candidate expressions like 7200000*9/12 or (7200000*9/12)
            expr_re = re.compile(
                r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?(?:\s*[+\-*/x×]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?\s*\)?)+",
                flags=re.IGNORECASE,
            )

            # Also extract pure multiplication/division sub-expressions, which may be nested
            # inside a larger '+' working line (e.g., 12750000+(7200000*9/12)).
            muldiv_re = re.compile(
                r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?(?:\s*[*/x×]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?\s*\)?)+",
                flags=re.IGNORECASE,
            )

            candidates = [m.group(0) for m in expr_re.finditer(ev_blob)]
            candidates.extend([m.group(0) for m in muldiv_re.finditer(ev_blob)])
            # Also pull sub-expressions after '=' which often contain the calc.
            if "=" in ev_blob:
                rhs = ev_blob.split("=", 1)[1]
                candidates.extend([m.group(0) for m in expr_re.finditer(rhs)])
                candidates.extend([m.group(0) for m in muldiv_re.finditer(rhs)])

            # De-dup while preserving order.
            seen = set()
            uniq: list[str] = []
            for c in candidates:
                cc = c.strip()
                if not cc:
                    continue
                key = re.sub(r"\s+", "", cc.lower())
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(cc)

            # Evaluate candidates and check for match.
            for expr in uniq[:12]:
                val = _safe_eval_arithmetic(expr)
                if val is None:
                    continue
                if math.isclose(val, expected, rel_tol=1e-6, abs_tol=2.0):
                    return True, f"Awarded by numeric-calc check (evidence calc {expr.strip()} = {expected:,.0f} GBP)."

            return False, ""

        def _evidence_has_calc_result(ev_blob: str, expected_value: float) -> bool:
            """Return True if evidence contains an arithmetic expression that evaluates to expected_value."""
            if expected_value is None or not isinstance(expected_value, (int, float)):
                return False
            if not ev_blob or not isinstance(ev_blob, str):
                return False

            expr_re = re.compile(
                r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?(?:\s*[+\-*/x×]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?\s*\)?)+",
                flags=re.IGNORECASE,
            )
            muldiv_re = re.compile(
                r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?(?:\s*[*/x×]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?\s*\)?)+",
                flags=re.IGNORECASE,
            )
            candidates = [m.group(0) for m in expr_re.finditer(ev_blob)]
            candidates.extend([m.group(0) for m in muldiv_re.finditer(ev_blob)])
            if "=" in ev_blob:
                rhs = ev_blob.split("=", 1)[1]
                candidates.extend([m.group(0) for m in expr_re.finditer(rhs)])
                candidates.extend([m.group(0) for m in muldiv_re.finditer(rhs)])

            seen = set()
            for expr in candidates:
                key = re.sub(r"\s+", "", (expr or "").lower())
                if not key or key in seen:
                    continue
                seen.add(key)
                val = _safe_eval_arithmetic(expr)
                if val is None:
                    continue
                if math.isclose(float(val), float(expected_value), rel_tol=1e-6, abs_tol=2.0):
                    return True
            return False

        def _find_calc_snippet_in_student(criterion_text: str) -> Optional[str]:
            """If criterion contains a simple x/* and / calc in parentheses, try to find the same calc in the student text.

            Returns a short expression snippet (e.g., "7200000*9/12") if found.
            """
            if not isinstance(criterion_text, str) or not criterion_text:
                return None

            m = re.search(r"\(([^)]*)\)", criterion_text)
            if not m:
                return None

            expr = m.group(1)
            expr_low = expr.lower().replace("×", "x")
            if "+" in expr_low or "-" in expr_low:
                return None

            # Tokenize similarly to _compute_simple_calc_from_criterion
            parts = re.split(r"\s*(x|\*|/)\s*", expr_low)
            parts = [p.strip() for p in parts if p and p.strip()]
            if len(parts) < 3:
                return None

            # Build a regex that matches the numeric expression in student text.
            # Convert operands to canonical numeric strings where possible (e.g., 7.2 million -> 7200000).
            def _int_with_commas_pattern(ival: str) -> str:
                """Match either the raw digits or a correctly comma-grouped variant.

                Example: 7200000 matches "7200000" or "7,200,000" (allowing optional spaces around commas).
                """
                if not ival or not ival.isdigit():
                    return re.escape(ival or "")
                if len(ival) <= 3:
                    return re.escape(ival)

                first_len = len(ival) % 3
                if first_len == 0:
                    first_len = 3
                groups = [ival[:first_len]]
                for j in range(first_len, len(ival), 3):
                    groups.append(ival[j:j + 3])
                grouped = r"\s*,\s*".join(re.escape(g) for g in groups)
                return rf"(?:{re.escape(ival)}|{grouped})"

            pattern_parts: list[str] = []
            i = 0
            while i < len(parts):
                token = parts[i]
                if token in {"x", "*", "/"}:
                    if token == "/":
                        pattern_parts.append(r"\s*/\s*")
                    else:
                        pattern_parts.append(r"\s*[x\*]\s*")
                    i += 1
                    continue

                # Operand
                tok_compact = re.sub(r"\s+", "", token.lower())
                # Special-case fractions (e.g., 9/12) so we can match either the literal
                # fraction form or its decimal equivalent in student workings.
                if re.fullmatch(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?", tok_compact):
                    try:
                        num_s, den_s = tok_compact.split("/", 1)
                    except Exception:
                        return None
                    frac_pat = rf"{re.escape(num_s)}\s*/\s*{re.escape(den_s)}"
                    val = StudentGrader._parse_number_token(token)
                    if val is not None:
                        # Keep a conservative decimal string (avoid scientific notation).
                        dec = (f"{float(val):.6f}").rstrip("0").rstrip(".")
                        if dec:
                            pattern_parts.append(rf"(?:{frac_pat}|{re.escape(dec)})")
                        else:
                            pattern_parts.append(frac_pat)
                    else:
                        pattern_parts.append(frac_pat)
                    i += 1
                    continue

                val = StudentGrader._parse_number_token(token)
                if val is None:
                    return None

                # Prefer integer representation when close.
                if abs(val - round(val)) < 1e-6:
                    ival = str(int(round(val)))
                    pattern_parts.append(_int_with_commas_pattern(ival))
                else:
                    # Keep a conservative float token
                    sval = str(val)
                    pattern_parts.append(re.escape(sval))
                i += 1

            if not pattern_parts:
                return None

            regex = re.compile("".join(pattern_parts), flags=re.IGNORECASE)
            hay = getattr(self, "_student_text_last_run", "") or ""
            m2 = regex.search(hay)
            if not m2:
                # Try a more permissive match on the normalized blob
                m3 = regex.search(student_blob_norm)
                if not m3:
                    return None
                snippet = m3.group(0)
            else:
                snippet = m2.group(0)

            snippet = (snippet or "").strip()
            snippet = snippet.replace(" ", "")
            snippet = snippet.replace("×", "*").replace("x", "*")
            # Keep snippet short.
            return snippet[:60] if snippet else None

        def _norm_for_evidence_match(s: str) -> str:
            if not s or not isinstance(s, str):
                return ""
            s = s.replace("\u00a0", " ")
            s = s.replace("×", "x")
            s = s.replace("–", "-").replace("—", "-")
            s = re.sub(r"\s+", " ", s).strip().lower()
            s = re.sub(r"[^a-z0-9%/().,\- ]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        student_blob_norm = _norm_for_evidence_match(getattr(self, "_student_text_last_run", "") or "")

        # Normalized question/model blobs for detecting evidence that is copied from the question
        # text or marking guide (e.g., headings). If evidence only appears in these sources,
        # we treat it as "tainted" and revoke marks.
        question_blob_norm = _norm_for_evidence_match(getattr(self, "_question_text_last_run", "") or "")
        model_blob_norm = _norm_for_evidence_match(getattr(self, "_model_text_last_run", "") or "")
        reference_blob_norm = " ".join([t for t in (question_blob_norm, model_blob_norm) if t])

        def _evidence_has_untainted_snippet(evidence_items: list[str]) -> bool:
            """Return True if at least one evidence snippet is present in student text AND not present in reference text."""
            if not student_blob_norm:
                return False
            for ev in evidence_items:
                if not ev or not isinstance(ev, str):
                    continue
                ev_norm = _norm_for_evidence_match(ev)
                if len(ev_norm) < 12 and not _is_distinctive_short_evidence(ev_norm):
                    continue
                if ev_norm not in student_blob_norm:
                    continue
                if reference_blob_norm and ev_norm in reference_blob_norm:
                    # Evidence also appears in question/markscheme.
                    # Only flag as "tainted" when the snippet is pure text (no numbers).
                    # Evidence containing a meaningful number (3+ digits) almost certainly
                    # represents the student's own calculation or stated value — the fact that
                    # the same number appears in the model answer just means the student got it right.
                    # Pure-text phrases (e.g. section headings like "statement of financial position")
                    # with 2+ content words ARE tainted — they were likely extracted from the PDF template.
                    has_meaningful_number = bool(re.search(r"\d{3,}|\d+\.\d{2,}", ev_norm))
                    if not has_meaningful_number:
                        alpha_words = re.findall(r"[a-z]{4,}", ev_norm)
                        if len(alpha_words) >= 2:
                            continue
                return True
            return False

        def _is_distinctive_short_evidence(ev_norm: str) -> bool:
            if not ev_norm or not isinstance(ev_norm, str):
                return False
            compact = ev_norm.replace(" ", "")
            if re.search(r"\d", ev_norm):
                return len(compact) >= 4
            if "/" in ev_norm or "%" in ev_norm:
                return len(compact) >= 3
            words = ev_norm.split()
            return len(words) <= 3 and any(len(w) >= 7 for w in words)

        def _criterion_evidence_alignment_ok(criterion_text: str, evid_items: list[str]) -> bool:
            """Heuristic guardrail: ensure awarded evidence actually aligns to the criterion.

            This is NOT a correctness check vs the model answer.
            It prevents awarding marks when the evidence only weakly relates to a long narrative criterion
            (common failure mode: giving partial credit for mentioning a vague related term).

            Keep it conservative and only apply to longer/descriptive criteria.
            """
            if not isinstance(criterion_text, str) or not evid_items:
                return True

            crit_norm = _norm_for_evidence_match(criterion_text)
            if not crit_norm:
                return True

            # Only enforce this alignment guardrail for narrative criteria.
            # Numeric/method criteria (with GBP amounts, large calculations) are better handled by the
            # other guardrails (evidence-present, taint check, strict-number match for full marks).
            # Enforcing alignment here tends to incorrectly revoke legitimate own-figure work.
            # HOWEVER: criteria whose only digits are dates or small ordinals (e.g. "31 May 20X4",
            # "within 30 days") are still narrative criteria — keep alignment enforcement for those.
            if re.search(r"\d", crit_norm):
                # Has a large number (3+ digits) or explicit GBP/currency marker → numeric criterion.
                has_large_num = bool(re.search(r"\b\d{3,}\b", crit_norm))
                has_currency = bool(re.search(r"[£$]|\bgbp\b", crit_norm, re.IGNORECASE))
                if has_large_num or has_currency:
                    return True
                # Only small numbers (≤ 2 digits) present — treat as narrative (date-qualified).
                # Fall through to apply alignment check.
            if crit_norm.strip().startswith("dr ") or crit_norm.strip().startswith("cr "):
                return True

            # Do not enforce on short labels/headings.
            if len(crit_norm) < 50 and len(crit_norm.split()) < 8:
                return True

            ev_blob = _norm_for_evidence_match(" ".join([e for e in evid_items if isinstance(e, str)]))
            if not ev_blob:
                return False

            # Numbers/ratios in evidence are strong anchors.
            has_number = bool(re.search(r"\d", ev_blob))
            has_ratio = bool(re.search(r"\d+\s*/\s*\d+", ev_blob))

            stop = {
                "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "as", "is", "are",
                "was", "were", "be", "been", "being", "with", "from", "by", "this", "that", "these",
                "those", "should", "would", "will", "must", "therefore", "account", "accounts", "year",
                "statement", "financial", "reporting", "treatment",
            }

            crit_words = [w for w in re.findall(r"[a-z]{4,}", crit_norm) if w not in stop]
            if not crit_words:
                return True

            ev_words = set(re.findall(r"[a-z]{4,}", ev_blob))

            # If evidence is predominantly numeric (no/one content word) and contains meaningful
            # numbers (3+ digits), trust the LLM's criterion-evidence pairing.
            # Numeric evidence like "25%*(12,750,000-2,750,000) 2,500,000.00" is inherently
            # specific — the earlier guards already confirmed it exists in the student text
            # and is not tainted from the question/markscheme.
            if len(ev_words) <= 1 and bool(re.search(r"\d{3,}", ev_blob)):
                return True

            # Evidence with few content words (2-3) can pass IF a specific number from the
            # criterion also appears in the evidence. This handles accounting synonym issues
            # (e.g., "staff expense 480,000" for criterion about "GBP480,000 charged to P/L")
            # without being overly permissive.
            if 2 <= len(ev_words) <= 3 and bool(re.search(r"\d{3,}", ev_blob)):
                crit_nums_raw = re.findall(r"\d[\d,]*\.?\d*", crit_norm)
                ev_blob_compact = ev_blob.replace(",", "").replace(" ", "")
                for cn in crit_nums_raw:
                    cn_stripped = cn.replace(",", "")
                    if len(cn_stripped) >= 3 and cn_stripped in ev_blob_compact:
                        return True

            # Accounting domain synonym groups for alignment matching.
            # Words in the same group are treated as equivalent when computing
            # overlap, so "FX gain" matches "exchange movement", "close" matches
            # "closing", "debited" matches "charged", etc.
            _SYNONYM_GROUPS = [
                frozenset({"exchange", "forex", "currency", "translation", "retranslation"}),
                frozenset({"movement", "gain", "loss", "difference", "change"}),
                frozenset({"charged", "debited", "expensed", "recognised", "recognized", "impact"}),
                frozenset({"closing", "close", "closed", "yearend"}),
                frozenset({"revaluation", "revalued", "remeasured", "remeasurement", "restated"}),
                frozenset({"consolidation", "consolidated", "consolidate"}),
                frozenset({"profit", "income", "earnings"}),
                frozenset({"comprehensive", "reserve", "surplus"}),
                # Personnel synonyms: "Four cyclists departed / 8 riders" ↔ "8 staff members"
                frozenset({"cyclist", "rider", "employee", "staff", "member", "player", "worker"}),
                # Associate / equity method synonyms
                frozenset({"associate", "equity", "accounted"}),
            ]
            _WORD_TO_GROUP: dict[str, int] = {}
            for _gi, _grp in enumerate(_SYNONYM_GROUPS):
                for _sw in _grp:
                    _WORD_TO_GROUP[_sw] = _gi
                    # Also map common plural form so "cyclists" matches "cyclist", etc.
                    if not _sw.endswith("s"):
                        _WORD_TO_GROUP[_sw + "s"] = _gi

            # Compute overlap: exact word matches first.
            exact_overlap = {w for w in crit_words if w in ev_words}
            overlap = len(exact_overlap)

            # Add synonym-based overlap: for each unmatched criterion word that
            # belongs to a synonym group, check if any evidence word is in the
            # same group.  Each group contributes at most one additional point.
            matched_groups: set[int] = set()
            for w in crit_words:
                if w in exact_overlap:
                    continue
                g = _WORD_TO_GROUP.get(w)
                if g is None or g in matched_groups:
                    continue
                if any(_WORD_TO_GROUP.get(ew) == g for ew in ev_words):
                    overlap += 1
                    matched_groups.add(g)

            # Require at least 2 overlapping content words, OR 1 overlap plus a strong anchor.
            if overlap >= 2:
                return True
            if overlap >= 1 and (has_number or has_ratio):
                return True
            # A single long domain-specific word (≥7 chars) is a strong enough anchor on its own.
            # e.g. "associate" in both criterion and evidence passes without a numeric anchor.
            long_exact = {w for w in exact_overlap if len(w) >= 7}
            if long_exact:
                return True
            return False

        def _find_journal_line_in_student(criterion_text: str) -> Optional[str]:
            """Try to recover a verbatim journal/workings line from the student text.

            Generic recovery for criteria that are journal-line-shaped (Dr/Cr) and include a numeric amount.
            Returns a short snippet from the student's submitted text.
            """
            if not isinstance(criterion_text, str) or not criterion_text:
                return None

            crit = criterion_text.strip()
            crit_low = crit.lower().strip()
            if not (crit_low.startswith("dr ") or crit_low.startswith("cr ")):
                return None

            crit_nums = self._numbers_in_text(crit)
            if not crit_nums:
                return None

            # Extract a couple of account-ish tokens to increase precision.
            crit_words = [w for w in re.findall(r"[a-z]{4,}", crit_low) if w not in {"dr", "cr", "gbp", "usd", "eur", "million", "thousand"}]
            crit_words = crit_words[:4]

            hay = getattr(self, "_student_text_last_run", "") or ""
            if not isinstance(hay, str) or not hay.strip():
                return None

            # Work line-by-line to preserve journals/tables.
            lines = [ln.strip() for ln in hay.splitlines() if ln and ln.strip()]

            # Precompute normalized versions for matching.
            for ln in lines:
                ln_low = ln.lower()
                if not ("dr" in ln_low.split() or "cr" in ln_low.split() or ln_low.startswith("dr ") or ln_low.startswith("cr ")):
                    continue

                # Must contain correct direction.
                if crit_low.startswith("dr ") and not (ln_low.startswith("dr ") or "dr" in ln_low.split()):
                    continue
                if crit_low.startswith("cr ") and not (ln_low.startswith("cr ") or "cr" in ln_low.split()):
                    continue

                # Must contain at least one of the criterion numbers.
                if not any(self._contains_number_variant(ln, n) for n in crit_nums):
                    continue

                # Must contain at least one criterion content word (if any) to avoid matching random journal lines.
                if crit_words and not any(w in ln_low for w in crit_words):
                    continue

                # Return verbatim line, truncated.
                return ln[:160]

            return None

        def _evidence_present(evidence_items: list[str]) -> bool:
            if not student_blob_norm:
                return False
            for ev in evidence_items:
                if not ev or not isinstance(ev, str):
                    continue
                ev_norm = _norm_for_evidence_match(ev)
                if len(ev_norm) < 12 and not _is_distinctive_short_evidence(ev_norm):
                    continue
                if ev_norm in student_blob_norm:
                    return True
                # Sliding-window fallback: LLMs sometimes add/remove minor punctuation in
                # verbatim quotes.  A contiguous N-word window in the student text is
                # sufficient to confirm the quote is genuine.
                words = [w for w in ev_norm.split() if len(w) >= 3]
                if len(words) >= 10:
                    for i in range(0, min(len(words) - 9, 8)):
                        window = " ".join(words[i:i + 10])
                        if window in student_blob_norm:
                            return True
                elif len(words) >= 5:
                    for i in range(0, min(len(words) - 4, 6)):
                        window = " ".join(words[i:i + 5])
                        if window in student_blob_norm:
                            return True
            return False

        grades = parsed_grades.get("grades", [])
        if not grades:
            raise GradingError("No grades returned from LLM")

        main_grade = grades[0]  # assuming single main grade object
        all_comments = main_grade.get("comments", []) if isinstance(main_grade.get("comments", []), list) else []

        def _is_annotation_comment(s: str) -> bool:
            if not s or not isinstance(s, str):
                return False
            if "\n" in s or "\r" in s:
                return False
            if "→" in s:
                left, right = s.split("→", 1)
            elif "->" in s:
                left, right = s.split("->", 1)
            else:
                return False
            left = (left or "").strip()
            right = (right or "").strip()
            if not left or not right:
                return False
            # Expect two short sentences after the arrow; require at least 2 periods.
            if right.count(".") < 2:
                return False
            return True

        annotation_comments: list[str] = []
        unanchored_comments: list[str] = []
        for c in all_comments:
            try:
                (annotation_comments if _is_annotation_comment(c) else unanchored_comments).append(str(c))
            except Exception:
                continue

        normalized_breakdown = []
        evidence_warnings = []
        sum_awarded_calc = 0.0  # debug only

        for item in main_grade.get("breakdown", []) or []:
            criterion = item.get("criterion", "Unknown")
            criterion = str(criterion or "").strip()

            # Strictly require that the criterion exists in the rubric we provided.
            # This prevents the LLM from inventing criteria or grading headings/commentary.
            # When criteria were synthesized from answer text, relax this check since the
            # LLM may reasonably rephrase the auto-generated criterion descriptions.
            # For holistic grading, skip rubric validation entirely — breakdown items are
            # sub-questions, not rubric criteria.
            if not criterion:
                continue
            if allowed_criteria and criterion not in allowed_criteria and not self._criteria_were_synthesized and not self._holistic_grading:
                logger.debug(f"Skipping out-of-rubric criterion: '{criterion}'")
                continue

            if not self._holistic_grading and not self._is_valid_criterion(criterion):
                logger.debug(f"Skipping invalid criterion: '{criterion}'")
                continue

            original_marks_awarded = float(item.get("marks_awarded", 0) or 0)

            # Enforce max_possible from the rubric (ignore LLM-supplied max_possible).
            # If the rubric doesn't have a numeric max for this criterion, treat it as non-scoreable.
            # For holistic grading, use the authoritative max_marks from _holistic_sub_questions
            # (sourced from the question paper), falling back to the LLM's value only if not found.
            # Note: model answer sub-criteria marks (e.g. 0.5/point) are still used by the LLM to
            # score individual points — the cap only limits the final marks_awarded total.
            rubric_max = rubric_max_map.get(criterion)
            if self._holistic_grading:
                sq_label = item.get("_sub_question", "")
                authoritative_max = 0.0
                for hq in (self._holistic_sub_questions or []):
                    if str(hq.get("sub_question", "")).strip() == str(sq_label).strip():
                        authoritative_max = float(hq.get("max_marks", 0) or 0)
                        break
                if authoritative_max > 0:
                    max_possible = authoritative_max
                else:
                    llm_max = float(item.get("max_possible", 0) or 0)
                    max_possible = llm_max if llm_max > 0 else original_marks_awarded
            elif rubric_max is None and self._criteria_were_synthesized:
                # For synthesized criteria, the LLM may rephrase — trust the LLM's max_possible
                llm_max = float(item.get("max_possible", 0) or 0)
                max_possible = llm_max if llm_max > 0 else original_marks_awarded
            else:
                max_possible = float(rubric_max) if isinstance(rubric_max, (int, float)) else 0.0
            # Clamp marks_awarded: never exceed max_possible and quantize to 0.25 steps.
            marks_awarded = min(original_marks_awarded, max_possible)
            marks_awarded = round(marks_awarded / 0.25) * 0.25

            raw_evidence = item.get("evidence", [])
            evid_list = []

            # Super-robust parsing
            if isinstance(raw_evidence, str):
                cleaned = raw_evidence.replace('\\n', '\n').replace('\\r', '').replace('\\t', ' ').strip()
                if cleaned:
                    # Do NOT split on '|' because that destroys table rows.
                    evid_list = [s.strip() for s in re.split(r'\n|;', cleaned) if s.strip()]
            elif isinstance(raw_evidence, list):
                for ev in raw_evidence:
                    if ev is None:
                        continue
                    if isinstance(ev, str):
                        cleaned_ev = ev.replace('\\n', '\n').strip()
                        if cleaned_ev:
                            evid_list.append(cleaned_ev)
                    else:
                        cleaned_ev = str(ev).strip()
                        if cleaned_ev:
                            evid_list.append(cleaned_ev)
            else:
                # fallback for odd types
                cleaned = str(raw_evidence).strip()
                if cleaned:
                    evid_list = [cleaned]

            # Evidence expansion for tables: keep original evidence, but also add
            # OCR-friendly variants when evidence contains pipe-separated columns.
            # This helps PDF annotation anchor marks within tables.
            try:
                expanded: list[str] = []
                for ev in list(evid_list or []):
                    if not isinstance(ev, str) or "|" not in ev:
                        continue
                    cells = [c.strip() for c in ev.split("|")]
                    cells = [c for c in cells if c]
                    if len(cells) < 2:
                        continue

                    # Prefer a compact "label value" variant when last cell looks numeric.
                    last = cells[-1]
                    if re.search(r"\d", last):
                        compact = f"{cells[0]} {last}".strip()
                        if compact and compact not in evid_list and compact not in expanded:
                            expanded.append(compact)

                    joined = " ".join(cells).strip()
                    if joined and joined not in evid_list and joined not in expanded:
                        expanded.append(joined)

                if expanded:
                    evid_list = expanded + evid_list
            except Exception:
                pass

            # Enforce evidence: if we can't parse evidence, we cannot justify awarding marks.
            # This prevents incorrect awards when the model "guesses".
            # For holistic grading, still require evidence but don't revoke — it's possible
            # the LLM awarded marks for overall understanding without pinpointing exact lines.
            if marks_awarded > 0 and not evid_list and not self._holistic_grading:
                evidence_warnings.append(f"Marks revoked (missing evidence): {criterion}")
                marks_awarded = 0.0

            # Stronger guardrail: evidence must actually exist in the student answer text.
            # This prevents marks being awarded when the LLM fabricates an evidence quote.
            # SKIP for holistic grading — the LLM's holistic comparison is trusted.
            if marks_awarded > 0 and evid_list and student_blob_norm and not self._holistic_grading:
                if not _evidence_present(evid_list):
                    evidence_warnings.append(f"Marks revoked (evidence not found in student answer): {criterion}")
                    marks_awarded = 0.0

            # Strongest guardrail: do not award marks based on evidence copied from the question/rubric.
            # This prevents awarding marks from section headings that appear in the PDF but contain no student work.
            # SKIP for synthesized criteria and holistic grading — theoretical answers naturally share
            # terminology with the model answer.
            if marks_awarded > 0 and evid_list and student_blob_norm and not self._criteria_were_synthesized and not self._holistic_grading:
                if reference_blob_norm and not _evidence_has_untainted_snippet(evid_list):
                    evidence_warnings.append(f"Marks revoked (evidence appears in question/markscheme, not student work): {criterion}")
                    marks_awarded = 0.0

            # Income/profit context revocation: a criterion that explicitly names profit,
            # contribution, revenue, income, or earnings as the concept measured requires the
            # evidence to also contain that vocabulary.  Without this, a calculation like
            # "7,200,000*9/12" in the student's net-assets working table would remain credited
            # for a profit-contribution criterion after the LLM incorrectly awarded marks.
            # This is intentionally narrow (only income-context terms) to avoid revoking
            # legitimate marks for criteria that describe profit without using that exact word.
            # Limit to micro-criteria (≤ 1 mark); section-total criteria with large marks
            # are less likely to be wrongly awarded and should not be revoked this way.
            # SKIP for journal and calc criteria — they have dedicated direction/number guards.
            # Those criteria contain "profit or loss" as an ACCOUNT NAME not a concept check,
            # so the income guard fires spuriously (e.g. "Dr Profit or loss 167" → evidence
            # shows "Dr Revaluation Loss (PL)" which is equivalent but lacks the word "profit").
            # Look up category from rubric cache (LLM output never includes category).
            _income_cat = self._criterion_category_map_last_run.get(
                criterion, str(item.get("category", "") or "")
            ).lower()
            _skip_income_guard = _income_cat in ("journal", "calculation", "calc")
            if (
                not _skip_income_guard
                and marks_awarded > 0
                and evid_list
                and isinstance(criterion, str)
                and float(max_possible) <= 1.0
                and not self._holistic_grading
            ):
                _INCOME_REVOKE_RE = re.compile(
                    r"\b(profit|contribution|revenue|income|earning)\b", re.IGNORECASE
                )
                if _INCOME_REVOKE_RE.search(criterion):
                    _ev_income_ctx = " ".join(evid_list)
                    # Accept P&L, P/L, PL, loss, OCI, exchange/FX reserve, comprehensive,
                    # NCI (non-controlling interest share of profit), EPS (earnings per share),
                    # and plural forms (profits, earnings) as valid income-context evidence.
                    _INCOME_EVID_RE = re.compile(
                        r"\b(profits?|contribution|revenue|income|earnings?|loss|oci|"
                        r"exchange|reserve|comprehensive|nci|eps)\b"
                        r"|p[&/]l|\bpl\b|\bfx\b",
                        re.IGNORECASE,
                    )
                    if not _INCOME_EVID_RE.search(_ev_income_ctx):
                        evidence_warnings.append(
                            f"Marks revoked (profit-criterion evidence lacks profit context): {criterion}"
                        )
                        marks_awarded = 0.0

            # Alignment guardrail: for longer narrative criteria, require evidence to actually align.
            # Prevents awarding marks for vague mentions (e.g., saying "exchange difference" but not stating the required treatment).
            # Skip for large-number / calculation criteria; those are handled by numeric guardrails.
            # Also skip for Dr/Cr journal criteria, synthesized criteria, and holistic grading.
            # BUT apply even when criterion has small numbers (dates, ordinals) — those are still narrative.
            def _criterion_has_large_number(crit: str) -> bool:
                c = _norm_for_evidence_match(crit)
                return bool(re.search(r"\b\d{3,}\b", c)) or bool(
                    re.search(r"[£$]|\bgbp\b", c, re.IGNORECASE)
                )

            if (
                marks_awarded > 0
                and evid_list
                and student_blob_norm
                and isinstance(criterion, str)
                and criterion.strip()
                and not _criterion_has_large_number(criterion)
                and not criterion.strip().lower().startswith(("dr ", "cr "))
                and float(max_possible) < 1.0
                and not self._criteria_were_synthesized
                and not self._holistic_grading
            ):
                if not _criterion_evidence_alignment_ok(str(criterion), evid_list):
                    evidence_warnings.append(f"Marks revoked (weak evidence alignment to criterion): {criterion}")
                    marks_awarded = 0.0

            # Negation/contradiction guard: if a criterion states something "is required"
            # or "must" occur, and the evidence explicitly states the opposite (e.g., "no
            # impairment review", "not required"), revoke marks.
            # This catches the case where the alignment guardrail was bypassed (e.g., because
            # the criterion contains a date like "31 May 20X4") but the evidence contradicts
            # the criterion's core assertion.
            if marks_awarded > 0 and evid_list and isinstance(criterion, str) and not self._holistic_grading:
                _crit_lower = criterion.lower()
                if re.search(r"\brequired\b|\bnecessary\b|\bmust\b", _crit_lower):
                    _NEG_STOP = {
                        "required", "necessary", "must", "should", "would", "which", "this",
                        "that", "also", "have", "been", "will", "need", "needs", "being",
                        "review", "report", "audit", "before", "after",
                    }
                    _crit_key = [
                        w for w in re.findall(r"[a-z]{4,}", _crit_lower) if w not in _NEG_STOP
                    ]
                    _ev_combined = " ".join(evid_list).lower()
                    for _kw in _crit_key[:6]:
                        # Match "no <optional words> <keyword-prefix>" — covers "no impairments"
                        # when keyword is "impairment" and similar plural/suffix variations.
                        _kw_prefix = _kw[:min(len(_kw), 7)]
                        if re.search(rf"\bno\s+(?:\w+\s+){{0,2}}{re.escape(_kw_prefix)}", _ev_combined) or \
                           re.search(rf"\bnot\s+(?:\w+\s+){{0,2}}{re.escape(_kw_prefix)}", _ev_combined):
                            evidence_warnings.append(
                                f"Marks revoked (evidence contradicts required criterion): {criterion}"
                            )
                            marks_awarded = 0.0
                            break

            # Evidence recovery for numeric-calc criteria (SKIP for holistic grading):
            # Sometimes the LLM attaches unhelpful evidence (e.g., just "Profit after tax 7,200,000")
            # even though the student has the exact calculation elsewhere (e.g., "7200000*9/12").
            # If marks are currently 0 and we can find the calc in the student's text, use it as evidence
            # and award full marks.
            if marks_awarded == 0 and max_possible > 0 and student_blob_norm and not self._holistic_grading:
                expected_gbp = _extract_expected_gbp_amount(str(criterion))
                expected_calc = self._compute_simple_calc_from_criterion(str(criterion))
                if expected_gbp is not None and expected_calc is not None:
                    if math.isclose(float(expected_gbp), float(expected_calc), rel_tol=1e-6, abs_tol=2.0):
                        snippet = _find_calc_snippet_in_student(str(criterion))
                        # Context guard: if the criterion is about profit/contribution/income,
                        # only accept the recovered snippet when the evidence or snippet itself
                        # also contains profit-context vocabulary.  A "7,200,000*9/12" pattern
                        # found in a net-assets table must not be credited for a profit criterion.
                        if snippet:
                            _INCOME_TERMS_RE = re.compile(
                                r"\b(profit|contribution|revenue|income|earning)\b", re.IGNORECASE
                            )
                            if _INCOME_TERMS_RE.search(str(criterion)):
                                ev_ctx = _norm_for_evidence_match(
                                    " ".join((evid_list or []) + [snippet])
                                )
                                if not _INCOME_TERMS_RE.search(ev_ctx):
                                    snippet = None
                        if snippet:
                            # Attach recovered evidence and award.
                            if snippet not in evid_list:
                                evid_list = [snippet] + evid_list
                            marks_awarded = float(max_possible)
                            existing = (item.get("reason", "") or "").strip()
                            add = f"Awarded by calc-evidence recovery (found '{snippet}' in student answer)."
                            item["reason"] = (f"{existing} {add}".strip() if existing else add)

            # Deterministic credit for numeric-result criteria (SKIP for holistic grading):
            # If the criterion states an expected GBP amount and the evidence contains a clear calculation
            # that evaluates to that amount, award FULL marks (even if the LLM awarded 0 or partial).
            # Kept narrow + only when evidence exists in the student answer to avoid false positives.
            if marks_awarded < max_possible and max_possible > 0 and evid_list and student_blob_norm and not self._holistic_grading:
                if _evidence_present(evid_list):
                    award, suffix = _award_if_calc_matches_expected(str(criterion), evid_list, max_possible)
                    if award:
                        marks_awarded = float(max_possible)
                        existing = (item.get("reason", "") or "").strip()
                        saved_reason = existing
                        if saved_reason:
                            saved_reason = f"{saved_reason} {suffix}".strip()
                        else:
                            saved_reason = suffix
                        item["reason"] = saved_reason

            # Optional partial credit (disabled by default).
            # This is intended for cases where the student is clearly addressing the criterion
            # but the full correct treatment isn't provided.
            if (
                partial_credit_enabled
                and marks_awarded == 0
                and max_possible >= 0.5
                and evid_list
                and student_blob_norm
                and _evidence_present(evid_list)
            ):
                if _partial_overlap_ok(str(criterion), evid_list):
                    partial = _quantize_to_quarter(max_possible / 2)
                    partial = max(0.25, min(float(max_possible), float(partial)))
                    # Only award partial if it is strictly less than full marks.
                    if 0 < partial < float(max_possible):
                        marks_awarded = float(partial)
                        existing = (item.get("reason", "") or "").strip()
                        add = f"Partial credit awarded by heuristic ({partial}/{max_possible})."
                        item["reason"] = (f"{existing} {add}".strip() if existing else add)

            # Lightweight numeric consistency guard:
            # If the criterion is an atomic numeric criterion, require evidence to contain those numbers too.
            # This is intentionally narrow to avoid revoking narrative marks that mention dates/percentages.
            # Skip this guard when the LLM already gave partial credit — it's signalling an "own figure"
            # scenario where the student used the correct method but got a different number.
            if marks_awarded > 0 and marks_awarded >= max_possible:
                if self._requires_strict_number_match(str(criterion)):
                    crit_nums = self._numbers_in_text(str(criterion))
                    if crit_nums:
                        ev_blob = " ".join(evid_list)
                        # Accept either:
                        # 1) all literal numbers in the criterion appear in evidence, OR
                        # 2) evidence contains the derived result of a simple bracketed calc
                        literal_ok = all(self._contains_number_variant(ev_blob, n) for n in crit_nums)
                        expected_val = self._compute_simple_calc_from_criterion(str(criterion))
                        expected_ok = False
                        if expected_val is not None:
                            for variant in self._format_expected_number_variants(expected_val):
                                if self._contains_number_variant(ev_blob, variant):
                                    expected_ok = True
                                    break
                            # If the student shows the working (e.g., 7200000*9/12) but not the final
                            # number, accept it if the calculation evaluates to the expected value.
                            if not expected_ok and _evidence_has_calc_result(ev_blob, expected_val):
                                expected_ok = True

                        # Also accept: if the criterion explicitly states "= X" (the direct answer),
                        # check that stated answer against evidence. This handles criteria where
                        # _compute_simple_calc returns None or a unit-mismatch value (e.g., £k vs £).
                        # Example: "NCI column = 1,350 (25% × £5,400k × 9/12)" — the "= 1,350" is
                        # the canonical answer; evidence "1350" should pass even if the bracketed
                        # calc computes to a different unit scale.
                        if not (literal_ok or expected_ok):
                            _eq_match = re.search(r"=\s*\(?([\d,]+)\)?", str(criterion))
                            if _eq_match:
                                _stated_ans = _eq_match.group(1)
                                if self._contains_number_variant(ev_blob, _stated_ans):
                                    expected_ok = True

                        # If LLM explicitly identified this as own-figure (OF), skip number
                        # mismatch revocation. In UK professional exams, OF for CALC criteria
                        # awards FULL marks — the student is not penalised twice for one wrong
                        # input. The numbers in evidence will differ from the criterion by design.
                        _reason_text_nm = str(item.get("reason", "")).lower()
                        _is_of_nm = bool(re.search(
                            r"\bof\b|\bown.?figure\b|\bown.?fig\b", _reason_text_nm
                        ))
                        # exact_match criteria (e.g. SOCIE financial-statement rows) must show
                        # the rubric's expected number — OF bypass is not allowed because the
                        # amount itself is the assessable element, not a downstream carry-forward.
                        if _is_of_nm and criterion in self._exact_match_criteria_last_run:
                            _is_of_nm = False

                        if not (literal_ok or expected_ok) and not _is_of_nm:
                            evidence_warnings.append(f"Marks revoked (number mismatch): {criterion}")
                            marks_awarded = 0.0

            # Journal-direction guard: if criterion starts with Dr/Cr, evidence must include that direction.
            if marks_awarded > 0 and isinstance(criterion, str):
                crit_clean = criterion.strip().lower()
                if crit_clean.startswith("dr ") or crit_clean.startswith("cr "):
                    ev_blob = " ".join(evid_list).lower()
                    required = "dr" if crit_clean.startswith("dr ") else "cr"
                    # Use word-boundary regex so "Dr.", "DR", "dr" all match
                    if not re.search(rf"\b{re.escape(required)}\b", ev_blob):
                        evidence_warnings.append(f"Marks revoked (missing {required.upper()} in evidence): {criterion}")
                        marks_awarded = 0.0

                    # Also require the journal amount to be present when the criterion includes a number.
                    # BUT: if the LLM already gave partial credit (marks < max), it likely recognised
                    # an "own figure" scenario (correct journal structure, wrong amount). Don't override that.
                    # If the LLM gave FULL marks but the amount is wrong, award 50% OF instead of
                    # revoking to 0 — the student demonstrated correct journal structure (OF mark).
                    crit_nums = self._numbers_in_text(criterion)
                    if marks_awarded > 0 and marks_awarded >= max_possible and crit_nums:
                        if not any(self._contains_number_variant(" ".join(evid_list), n) for n in crit_nums):
                            _ev_lower = " ".join(evid_list).lower()
                            _crit_lower2 = str(criterion).strip().lower()
                            _direction_present = (
                                (_crit_lower2.startswith("dr ") and re.search(r"\bdr\b", _ev_lower)) or
                                (_crit_lower2.startswith("cr ") and re.search(r"\bcr\b", _ev_lower))
                            )
                            if _direction_present:
                                _of_val = max(0.25, round(float(max_possible) * 0.5 / 0.25) * 0.25)
                                _of_val = min(_of_val, float(max_possible) - 0.25)
                                marks_awarded = _of_val
                                evidence_warnings.append(
                                    f"OF mark (correct journal direction, own figure amount): {criterion}"
                                )
                            else:
                                evidence_warnings.append(
                                    f"Marks revoked (missing journal amount in evidence): {criterion}"
                                )
                                marks_awarded = 0.0

            # Journal-line recovery: if a Dr/Cr criterion got 0 but a matching journal line exists in the student text,
            # attach that verbatim line as evidence and award full marks.
            if marks_awarded == 0 and max_possible > 0 and student_blob_norm and isinstance(criterion, str):
                crit_clean = criterion.strip().lower()
                if crit_clean.startswith("dr ") or crit_clean.startswith("cr "):
                    snippet = _find_journal_line_in_student(criterion)
                    if snippet:
                        evid_list = [snippet] + (evid_list or [])
                        marks_awarded = float(max_possible)
                        existing = (item.get("reason", "") or "").strip()
                        add = "Awarded by journal-line recovery (matched Dr/Cr + amount in student answer)."
                        item["reason"] = (f"{existing} {add}".strip() if existing else add)

            # If we revoked marks in post-processing, ensure the saved reason doesn't contradict the score.
            saved_reason = item.get("reason", "")
            if original_marks_awarded > 0 and marks_awarded == 0:
                if saved_reason:
                    saved_reason = f"Marks revoked by guardrails: {saved_reason}"
                else:
                    saved_reason = "Marks revoked by guardrails"

            # Deduplicate evidence_list: the LLM sometimes produces both a plain-text
            # and a pipe-table version of the same quote.  Keep the more descriptive
            # form (longer after normalisation) and drop near-duplicates.
            if evid_list:
                seen_ev_norm: dict[str, str] = {}
                deduped_ev: list[str] = []
                for _ev in evid_list:
                    if not isinstance(_ev, str):
                        continue
                    _ev_norm = re.sub(r"[|\s]+", " ", _ev).strip().lower()
                    if not _ev_norm:
                        continue
                    if _ev_norm not in seen_ev_norm:
                        seen_ev_norm[_ev_norm] = _ev
                        deduped_ev.append(_ev)
                evid_list = deduped_ev

            sum_awarded_calc += marks_awarded

            # Detect "own figure" (OF) scenarios: LLM gave partial credit on a
            # calc/journal criterion, or journal guard downgraded to partial.
            # This flag is surfaced in annotations so students see "OF" clearly.
            # Use rubric category cache — LLM output items never include category.
            _item_category = self._criterion_category_map_last_run.get(
                criterion, str(item.get("category", "") or "")
            ).lower()
            _is_of_mark = (
                0 < float(marks_awarded) < float(max_possible)
                and _item_category in ("calculation", "journal", "calc")
                and not self._holistic_grading
            )

            bd_item = {
                "criterion": criterion,
                "marks_awarded": marks_awarded,
                "max_possible": max_possible,
                "reason": saved_reason,
                "evidence": "; ".join(evid_list) if evid_list else "",
                "evidence_list": evid_list if evid_list else [],
                "comments_summary": item.get("comments_summary", ""),
                "is_of_mark": _is_of_mark,
            }
            # For holistic grading, preserve sub-question metadata for the annotator.
            if self._holistic_grading:
                bd_item["_sub_question"] = item.get("_sub_question", "")
                bd_item["_student_label"] = item.get("_student_label", "")
                bd_item["_correct_points_with_marks"] = item.get("_correct_points_with_marks", [])
                bd_item["_not_required_points"] = item.get("_not_required_points", [])
            normalized_breakdown.append(bd_item)

        # "Marks given above / below" guard — SKIP for holistic grading.
        # When two criteria in the same grading run were both awarded marks and
        # their evidence strings are identical (after whitespace normalisation),
        # keep only the one with the higher max_possible and zero out the other.
        # This mirrors the teacher's annotation rule: marks are awarded once, at
        # the location where the actual working appears; subsequent references to
        # the same result (e.g., restating a goodwill figure in narrative or in
        # a journal after calculating it in a working) do NOT earn extra marks.
        # Exception: a journal entry criterion citing the same number as a calc
        # criterion is a DIFFERENT skill and keeps its marks — we only zero out
        # when the criteria descriptions themselves overlap significantly.
        if normalized_breakdown and not self._holistic_grading:
            try:
                _ev_index: dict[str, list[int]] = {}
                for _idx, _bd in enumerate(normalized_breakdown):
                    _ev_raw = _bd.get("evidence", "") or ""
                    _awarded = float(_bd.get("marks_awarded", 0) or 0)
                    if not _ev_raw or _awarded <= 0:
                        continue
                    _ev_key = re.sub(r"[\s;|]+", " ", _ev_raw).strip().lower()
                    if len(_ev_key) < 12:
                        continue
                    _ev_index.setdefault(_ev_key, []).append(_idx)
                for _ev_key, _idxs in _ev_index.items():
                    if len(_idxs) < 2:
                        continue
                    # Sort: keep the one with highest max_possible (most specific criterion).
                    _idxs_sorted = sorted(
                        _idxs,
                        key=lambda i: (
                            float(normalized_breakdown[i].get("max_possible", 0) or 0),
                            float(normalized_breakdown[i].get("marks_awarded", 0) or 0),
                        ),
                        reverse=True,
                    )
                    _keeper_crit = str(normalized_breakdown[_idxs_sorted[0]].get("criterion", "") or "")
                    for _dup_idx in _idxs_sorted[1:]:
                        _dup_crit = str(normalized_breakdown[_dup_idx].get("criterion", "") or "").lower()
                        _keep_crit_lower = _keeper_crit.lower()
                        # Only zero out when the duplicate criterion describes the SAME concept
                        # (shares 3+ significant words with the keeper), not a different skill.
                        _dup_words = set(re.findall(r"[a-z]{4,}", _dup_crit))
                        _keep_words = set(re.findall(r"[a-z]{4,}", _keep_crit_lower))
                        _overlap = _dup_words & _keep_words
                        _stop = {"mark", "marks", "amount", "value", "total", "year", "each", "from", "with", "that", "this", "have", "been"}
                        _overlap -= _stop
                        if len(_overlap) >= 2:
                            _dup_marks = float(normalized_breakdown[_dup_idx].get("marks_awarded", 0) or 0)
                            if _dup_marks > 0:
                                normalized_breakdown[_dup_idx]["marks_awarded"] = 0.0
                                sum_awarded_calc -= _dup_marks
                                _prev_reason = (normalized_breakdown[_dup_idx].get("reason", "") or "").strip()
                                normalized_breakdown[_dup_idx]["reason"] = (
                                    f"Marks given above: same evidence already credited for "
                                    f"'{_keeper_crit[:60]}'. " + _prev_reason
                                ).strip()
            except Exception:
                pass  # Guard must never fail the grader

        # Broad-criterion gating (generic) — SKIP for holistic grading:
        # If a high-mark narrative criterion sits next to many micro-criteria (<= 0.5 each),
        # don't award the broad marks unless the student scores well on the micro-criteria.
        # This prevents over-awarding for broad statements when the detailed workings are wrong
        # (common in EPS / table-style sections) while still allowing broad criteria in sections
        # that have no micro breakdown.
        try:
            pos_map = self._rubric_position_last_run or {}
            if pos_map and normalized_breakdown and not self._holistic_grading:
                # position → indices (handle duplicates conservatively)
                pos_to_indices: dict[int, list[int]] = {}
                for idx, bi in enumerate(normalized_breakdown):
                    crit = bi.get("criterion")
                    if not isinstance(crit, str):
                        continue
                    p = pos_map.get(crit)
                    if isinstance(p, int):
                        pos_to_indices.setdefault(p, []).append(idx)

                def _is_broad_narrative(crit: str, maxp: float) -> bool:
                    if not isinstance(crit, str) or not crit.strip():
                        return False
                    if float(maxp) < 1.0:
                        return False
                    # Only gate medium-sized broad criteria (EPS-style, 1–2 marks).
                    # Do not gate large statement criteria (e.g., SOCIE totals 4 marks).
                    if float(maxp) > 2.0:
                        return False
                    c = crit.strip().lower()
                    if c.startswith("dr ") or c.startswith("cr "):
                        return False
                    if re.search(r"\d", crit):
                        return False
                    return True

                def _is_micro(maxp: float) -> bool:
                    try:
                        return 0 < float(maxp) <= 0.5
                    except Exception:
                        return False

                window = 18
                micro_success_threshold = float(os.getenv("BROAD_CRITERION_MICRO_SUCCESS_THRESHOLD", "0.8") or 0.8)
                for idx, bi in enumerate(normalized_breakdown):
                    marks = float(bi.get("marks_awarded") or 0.0)
                    maxp = float(bi.get("max_possible") or 0.0)
                    crit = bi.get("criterion")
                    if marks <= 0 or not isinstance(crit, str):
                        continue
                    if not _is_broad_narrative(crit, maxp):
                        continue

                    p = pos_map.get(crit)
                    if not isinstance(p, int):
                        continue

                    micro_max_sum = 0.0
                    micro_awarded_sum = 0.0
                    for q in range(p - window, p + window + 1):
                        if q == p:
                            continue
                        for j in pos_to_indices.get(q, []):
                            bj = normalized_breakdown[j]
                            maxj = float(bj.get("max_possible") or 0.0)
                            if not _is_micro(maxj):
                                continue
                            micro_max_sum += maxj
                            micro_awarded_sum += float(bj.get("marks_awarded") or 0.0)

                    # Only gate if there is meaningful micro coverage nearby.
                    if micro_max_sum <= 0:
                        continue
                    if micro_max_sum < maxp * 0.75:
                        continue

                    ratio = (micro_awarded_sum / micro_max_sum) if micro_max_sum > 0 else 0.0
                    if ratio < micro_success_threshold:
                        # Revoke broad marks (but do not change micro marks).
                        revoked = marks
                        bi["marks_awarded"] = 0.0
                        sum_awarded_calc -= revoked
                        existing = (bi.get("reason") or "").strip()
                        add = (
                            f"Marks revoked by guardrails: broad criterion gated by micro-criteria performance "
                            f"({micro_awarded_sum:.2f}/{micro_max_sum:.2f} nearby micro marks)."
                        )
                        bi["reason"] = f"{existing} {add}".strip() if existing else add
                        evidence_warnings.append(f"Marks revoked (broad criterion gated by micro performance): {crit}")
        except Exception:
            pass

        # Post-grading dedup: when one awarded criterion's description fully
        # contains the text of 2+ other awarded criteria, it is a duplicate
        # "paragraph total" annotation from the marking guide.  Zero it out
        # to prevent double-counting.  This does NOT remove criteria from the
        # LLM prompt (which causes instability), only revokes marks afterward.
        for i, bi in enumerate(normalized_breakdown):
            if bi["marks_awarded"] <= 0:
                continue
            crit_i = _norm_for_evidence_match(bi.get("criterion", ""))
            if not crit_i or len(crit_i) < 40:
                continue
            contained_count = 0
            for j, bj in enumerate(normalized_breakdown):
                if i == j or bj["marks_awarded"] <= 0:
                    continue
                crit_j = _norm_for_evidence_match(bj.get("criterion", ""))
                if not crit_j or len(crit_j) >= len(crit_i) * 0.85:
                    continue
                if crit_j in crit_i:
                    contained_count += 1
            if contained_count >= 2:
                revoked = bi["marks_awarded"]
                sum_awarded_calc -= revoked
                bi["marks_awarded"] = 0.0
                bi["reason"] = f"Marks revoked (duplicate superset criterion): {bi['reason']}"
                evidence_warnings.append(
                    f"Marks revoked (superset criterion duplicates {contained_count} sub-criteria): {bi['criterion'][:80]}"
                )
                logger.info(f"Superset dedup: revoked {revoked} from '{bi['criterion'][:60]}...'")

        # Evidence-reuse dedup (OPTIONAL): Some earlier versions revoked marks when the same
        # evidence snippet was used across multiple micro-criteria. This is too aggressive for
        # official marking guides: a single table row can legitimately earn multiple 0.25/0.5
        # marks (eg a number and its supporting rate/working).
        #
        # Default is OFF. Enable explicitly with DEDUP_EVIDENCE_REUSE=1.
        if os.getenv("DEDUP_EVIDENCE_REUSE", "0").strip().lower() in {"1", "true", "yes", "y"}:
            _evidence_owner: dict[str, int] = {}  # norm_snippet → index of first awardee
            # Sort by marks descending so the highest-value award "claims" the evidence.
            _award_order = sorted(
                range(len(normalized_breakdown)),
                key=lambda idx: normalized_breakdown[idx]["marks_awarded"],
                reverse=True,
            )
            for idx in _award_order:
                bi = normalized_breakdown[idx]
                if bi["marks_awarded"] <= 0:
                    continue
                evid_list = bi.get("evidence_list") or []
                for ev in evid_list:
                    if not isinstance(ev, str):
                        continue
                    ev_key = _norm_for_evidence_match(ev)
                    if len(ev_key) < 20:
                        continue
                    if ev_key in _evidence_owner:
                        first_idx = _evidence_owner[ev_key]
                        if first_idx != idx:
                            revoked = bi["marks_awarded"]
                            sum_awarded_calc -= revoked
                            bi["marks_awarded"] = 0.0
                            first_crit = normalized_breakdown[first_idx]["criterion"][:60]
                            bi["reason"] = (
                                f"Marks revoked (evidence already used for '{first_crit}'): "
                                f"{bi['reason']}"
                            )
                            evidence_warnings.append(
                                f"Marks revoked (duplicate evidence reuse): "
                                f"{bi['criterion'][:80]}"
                            )
                            logger.info(
                                f"Evidence dedup: revoked {revoked} from "
                                f"'{bi['criterion'][:60]}' (evidence claimed by "
                                f"'{first_crit}')"
                            )
                            break  # already revoked this criterion, move on
                    else:
                        _evidence_owner[ev_key] = idx

        total_max = self._extract_question_max_marks(questions_data, main_grade)

        # Use the post-processed breakdown sum (after any revocations) as the source of truth.
        # Round to nearest 0.5 (standard mathematical rounding, half-up) to match marking
        # convention (only .5 or whole marks allowed).  Previous ceiling rounding systematically
        # inflated scores; round-to-nearest is fairer.
        rounded_total = math.floor(sum_awarded_calc * 2 + 0.5) / 2
        if total_max > 0:
            rounded_total = min(rounded_total, total_max)

        # Debug logs
        logger.info(f"Raw LLM breakdown count: {len(main_grade.get('breakdown', []))}")
        logger.info(f"Saved breakdown count: {len(normalized_breakdown)}")
        logger.info(f"Calc sum (post-check): {sum_awarded_calc}")
        logger.info(f"Question max marks used: {total_max}")
        logger.info(f"Final saved total: {rounded_total}")

        # Capture top-level not_required_points from the numerical LLM output.
        # In holistic mode these are stored per-sub-question on the breakdown items
        # (under _not_required_points) and the top-level field stays empty.
        nr_points_top: list[dict] = []
        if not self._holistic_grading:
            raw_nr = main_grade.get("not_required_points", []) or []
            if isinstance(raw_nr, list):
                for nr in raw_nr:
                    if not isinstance(nr, dict):
                        continue
                    text = str(nr.get("text", "") or "").strip()
                    if not text:
                        continue
                    nr_points_top.append({
                        "text": text,
                        "key_phrase": str(nr.get("key_phrase", "") or "").strip(),
                        "reason": str(nr.get("reason", "") or "").strip(),
                    })

        doc = {
            "student_id": self.student_name,
            "question_number": self.question_number,
            "total_marks_awarded": rounded_total,
            "total_max_possible": total_max,
            "overall_reason": main_grade.get("reason", "Graded automatically"),
            "breakdown": normalized_breakdown,
            # Keep comments annotation-friendly; store guardrail and unanchored notes separately.
            "comments": annotation_comments,
            "guardrail_warnings": evidence_warnings,
            "unanchored_comments": unanchored_comments,
            "not_required_points": nr_points_top,
            "extracted_at": now_iso,
            "question_id": self.questions_id,
            "model_answer_id": self.model_answers_id,
            "student_answer_id": self.student_answers_id,
        }

        # Flag for the annotator: holistic grading uses sub-question level annotation.
        if self._holistic_grading:
            doc["holistic_grading"] = True

        if os.getenv("DEBUG_SAVE_LLM_OUTPUT", "").strip().lower() in {"1", "true", "yes", "y"}:
            doc["llm_debug"] = self._llm_debug_trace[-5:]
            doc["llm_grading_provider"] = os.getenv("GRADING_PROVIDER")
            doc["llm_grader_model"] = os.getenv("LLM_GRADER_MODEL")

        try:
            validated = StudentGradeDocument(**doc)
            return validated.model_dump(exclude_none=True, by_alias=True)
        except Exception as ve:
            logger.warning(f"Pydantic validation failed - saving raw: {ve}")
            return doc

    def _save(self, doc: dict) -> str:
        try:
            result = self.grades_coll.insert_one(doc)
            doc_id = str(result.inserted_id)
            logger.info(f"Grading results saved → _id = {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"MongoDB save failed: {e}", exc_info=True)
            raise GradingError("Failed to save grading result") from e

    def grade(self) -> Optional[str]:
        try:
            logger.info(f"Starting grading → {self.student_name} | Q{self.question_number}")

            q_clean, m_clean, s_clean = self._load_clean_data()
            grades_parsed = self._run_grading(s_clean, m_clean, q_clean)
            grade_doc = self._build_grade_doc(grades_parsed, q_clean)
            doc_id = self._save(grade_doc)

            return doc_id

        except GradingError as ge:
            logger.error(f"Grading pipeline failed: {ge}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected grading error: {e}", exc_info=True)
            return None


# Public API
def grade_student(
    student_name: str,
    question_number: str,
    questions_id: Optional[str],
    model_answers_id: Optional[str],
    student_answers_id: str,
    question_type: str = "numerical",
) -> Optional[str]:
    grader = StudentGrader(
        student_name=student_name,
        question_number=question_number,
        questions_id=questions_id,
        model_answers_id=model_answers_id,
        student_answers_id=student_answers_id,
        question_type=question_type,
    )
    return grader.grade()