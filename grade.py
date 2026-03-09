import json
import re
import os
from datetime import datetime
from typing import Optional, Any, Tuple
from bson import ObjectId
from pydantic import BaseModel, Field
from prompts.grading_prompts import grade_prompt
from llm_setup import llm, llm_grader
from logging_config import logger
from schemas.student_grades import StudentGradeDocument
from database.mongodb import get_collection


class GradingError(Exception):
    pass


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


class LLMGradingResponse(BaseModel):
    grades: list[LLMGradingItem] = Field(..., min_length=1, description="Grades array")


class StudentGrader:

    COLLECTION_NAME = "student_grades"

    def __init__(
        self,
        student_name: str,
        question_number: str,
        questions_id: Optional[str],
        model_answers_id: Optional[str],
        student_answers_id: str,
    ):
        self.student_name = student_name
        self.question_number = question_number
        self.questions_id = questions_id
        self.model_answers_id = model_answers_id
        self.student_answers_id = student_answers_id

        self.grades_coll = get_collection(self.COLLECTION_NAME)

        # Grading chain (mapping step removed for holistic evaluation)
        # Enforce strict JSON via structured output to prevent malformed responses.
        structured_grader = llm_grader.with_structured_output(LLMGradingResponse)
        self.grade_chain = grade_prompt | structured_grader

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
        """Extract maximum marks from question data, preferring explicit 'maximum' values."""
        candidates: list[float] = []

        if isinstance(questions_data, dict):
            q_total_raw = questions_data.get("total_marks")
            if q_total_raw is not None:
                q_text = str(q_total_raw)

                max_match = re.search(r"maximum\s*marks?\s*[:=]?\s*(\d+(?:\.\d+)?)", q_text, flags=re.IGNORECASE)
                if max_match:
                    candidates.append(float(max_match.group(1)))

                max_generic = re.search(r"max(?:imum)?\s*[:=]?\s*(\d+(?:\.\d+)?)", q_text, flags=re.IGNORECASE)
                if max_generic:
                    candidates.append(float(max_generic.group(1)))

                nums = re.findall(r"\d+(?:\.\d+)?", q_text)
                if nums:
                    candidates.extend(float(n) for n in nums)

            questions_list = questions_data.get("questions")
            if isinstance(questions_list, list):
                for q in questions_list:
                    if not isinstance(q, dict):
                        continue
                    for key in ("maximum_marks", "max_marks", "total_marks"):
                        value = q.get(key)
                        if value is None:
                            continue
                        extracted = re.findall(r"\d+(?:\.\d+)?", str(value))
                        candidates.extend(float(n) for n in extracted)

        if main_grade and isinstance(main_grade, dict):
            for key in ("total_marks", "maximum_marks", "total_marks_available"):
                value = main_grade.get(key)
                if value is None:
                    continue
                extracted = re.findall(r"\d+(?:\.\d+)?", str(value))
                candidates.extend(float(n) for n in extracted)

        if not candidates:
            return 0.0

        # Prefer the smallest positive candidate to avoid choosing "available marks" over "maximum marks".
        positive = [c for c in candidates if c > 0]
        return min(positive) if positive else 0.0

    def _flatten_model_answers(self, model_data: dict) -> dict:
        if not isinstance(model_data, dict):
            return model_data

        answers = model_data.get("answers")
        if not isinstance(answers, list) or not answers:
            return model_data

        combined_criteria: list[dict[str, Any]] = []
        combined_answer_parts: list[str] = []


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

                # Expand compound criteria like "1/2 mk each max 4" where description
                # contains multiple semicolon/newline-separated line-items.
                if isinstance(raw_marks, str) and re.search(r"\bmk\s*each\b", raw_marks, flags=re.IGNORECASE):
                    per_mark = normalize_marks_value(raw_marks)
                    if isinstance(per_mark, (int, float)) and description and (";" in description or "\n" in description):
                        parts = [p.strip() for p in re.split(r";|\n", description) if p and p.strip()]
                        if parts:
                            for p in parts:
                                if not self._is_valid_criterion(p):
                                    continue
                                combined_criteria.append({
                                    "marks": float(per_mark),
                                    "description": p,
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

                    # If sub_criteria exist but none are usable, treat this as a heading.
                    # When there are multiple sibling criteria, drop the heading to avoid polluting the grader.
                    if sibling_count > 1:
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

                combined_criteria.append({
                    "marks": marks,
                    "description": description,
                })

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
        # Examples: "1/2", "1/4", "2/2", "3/2", "1/2 mk each max 4"
        if re.match(r'^\d+/\d+(\s+(mk|marks?).*)?$', clean):
            return False

        # Reject if it's only marking notation variations
        if re.match(r'^\s*\d+\s*/\s*\d+\s*(mk|marks)?\s*(each|per)?\s*(max\s*\d+)?\s*$', clean):
            return False

        words = clean.split()

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
        m_clean = self._flatten_model_answers(m_clean)

        return q_clean, m_clean, s_clean



    def _run_grading(self, student_data: dict, model_data: dict, questions_data: dict) -> dict:
        """Execute grading chain with clean content (holistic evaluation against all criteria)."""
        try:
            output = self.grade_chain.invoke({
                "model_data": json.dumps(model_data),
                "chunks": student_data,
                "questions": json.dumps(questions_data),
            })

            if isinstance(output, BaseModel):
                parsed = output.model_dump()
            elif isinstance(output, dict):
                parsed = output
            else:
                # If a provider returns tool-call style output, try to extract the args.
                structured_args = self._extract_structured_args_from_message(output)
                if isinstance(structured_args, dict):
                    parsed = structured_args
                else:
                    raise GradingError(
                        "Grader did not return structured output. "
                        "Set GRADING_PROVIDER=openai and use a model that supports structured output."
                    )

            logger.info(f"Grading complete → {self.student_name} (Q{self.question_number})")
            return parsed

        except Exception as e:
            logger.error("Grading chain failed", exc_info=True)
            if isinstance(e, GradingError):
                raise
            raise GradingError("Grading step failed") from e

    def _build_grade_doc(self, parsed_grades: dict, questions_data: dict) -> dict:
        now_iso = datetime.utcnow().isoformat()

        grades = parsed_grades.get("grades", [])
        if not grades:
            raise GradingError("No grades returned from LLM")

        main_grade = grades[0]  # assuming single main grade object
        all_comments = main_grade.get("comments", []) if isinstance(main_grade.get("comments", []), list) else []

        normalized_breakdown = []
        evidence_warnings = []
        sum_awarded_calc = 0.0  # debug only

        for item in main_grade.get("breakdown", []) or []:
            criterion = item.get("criterion", "Unknown")

            if not self._is_valid_criterion(criterion):
                logger.debug(f"Skipping invalid criterion: '{criterion}'")
                continue
            
            marks_awarded = float(item.get("marks_awarded", 0))
            max_possible = float(item.get("max_possible", 0)) if item.get("max_possible") is not None else 0.0

            raw_evidence = item.get("evidence", [])
            evid_list = []

            # Super-robust parsing
            if isinstance(raw_evidence, str):
                cleaned = raw_evidence.replace('\\n', '\n').replace('\\r', '').replace('\\t', ' ').strip()
                if cleaned:
                    evid_list = [s.strip() for s in re.split(r'\n|;|\|', cleaned) if s.strip()]
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

            # No revocation - keep marks if LLM awarded them
            if marks_awarded > 0 and not evid_list:
                evid_list = ["[LLM awarded marks - evidence not parsed, check raw output]"]
                evidence_warnings.append(f"Evidence parsing issue but marks kept: {criterion}")

            sum_awarded_calc += marks_awarded

            normalized_breakdown.append({
                "criterion": criterion,
                "marks_awarded": marks_awarded,
                "max_possible": max_possible,
                "reason": item.get("reason", ""),
                "evidence": "; ".join(evid_list) if evid_list else "[no parsed evidence]",
                "comments_summary": item.get("comments_summary", "")
            })

        total_max = self._extract_question_max_marks(questions_data, main_grade)

        # Trust LLM score
        llm_score = float(main_grade.get("score", sum_awarded_calc))
        rounded_total = round(llm_score * 2) / 2
        if total_max > 0:
            rounded_total = min(rounded_total, total_max)

        # Debug logs
        logger.info(f"Raw LLM breakdown count: {len(main_grade.get('breakdown', []))}")
        logger.info(f"Saved breakdown count: {len(normalized_breakdown)}")
        logger.info(f"LLM score: {llm_score}")
        logger.info(f"Calc sum (debug): {sum_awarded_calc}")
        logger.info(f"Question max marks used: {total_max}")
        logger.info(f"Final saved total: {rounded_total}")

        doc = {
            "student_id": self.student_name,
            "question_number": self.question_number,
            "total_marks_awarded": rounded_total,
            "total_max_possible": total_max,
            "overall_reason": main_grade.get("reason", "Graded automatically"),
            "breakdown": normalized_breakdown,
            "comments": list(all_comments) + evidence_warnings,
            "extracted_at": now_iso,
            "question_id": self.questions_id,
            "model_answer_id": self.model_answers_id,
            "student_answer_id": self.student_answers_id,
        }

        try:
            validated = StudentGradeDocument(**doc)
            return validated.model_dump(exclude_none=True)
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
) -> Optional[str]:
    grader = StudentGrader(
        student_name=student_name,
        question_number=question_number,
        questions_id=questions_id,
        model_answers_id=model_answers_id,
        student_answers_id=student_answers_id,
    )
    return grader.grade()