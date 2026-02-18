import json
import re
from datetime import datetime
from typing import Optional, Any, Tuple
from bson import ObjectId
from prompts.grading_prompts import map_to_questions_prompt, grade_prompt
from llm_setup import llm, llm_grader
from logging_config import logger
from schemas.student_grades import StudentGradeDocument
from database.mongodb import get_collection


class GradingError(Exception):
    pass


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

        # Chains from your prompts file
        self.map_chain = map_to_questions_prompt | llm
        self.grade_chain = grade_prompt | llm_grader

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
        """Remove all metadata and return only content fields needed by LLM."""
        if not doc:
            return {}
        return {k: v for k, v in doc.items() if k in allowed_keys}

    def _load_clean_data(self) -> Tuple[dict, dict, dict]:
        """Load documents and extract only LLM-relevant content."""
        q_doc = self._fetch_doc("pac_questions", self.questions_id) if self.questions_id else {}
        m_doc = self._fetch_doc("model_answers", self.model_answers_id) if self.model_answers_id else {}
        s_doc = self._fetch_doc("student_assignments", self.student_answers_id)

        if not s_doc:
            raise GradingError(f"No student answer found for _id={self.student_answers_id}")

        # Only these fields go to LLM — metadata is completely excluded
        q_clean = self._clean_for_llm(q_doc, ["question_title", "description", "total_marks", "questions"])
        m_clean = self._clean_for_llm(m_doc, ["question_title", "description", "total_marks", "answers"])
        s_clean = self._clean_for_llm(s_doc, ["question", "sub_parts"])

        return q_clean, m_clean, s_clean

    def _run_mapping(self, student_data: dict, questions_data: dict) -> dict:
        """Execute mapping chain with clean content."""
        try:
            output = self.map_chain.invoke({
                "chunks": student_data,
                "questions": json.dumps(questions_data)
            })
            parsed = json.loads(output.content)
            logger.info(f"Mapping complete → {self.student_name} (Q{self.question_number})")
            return parsed
        except Exception as e:
            logger.error("Mapping failed", exc_info=True)
            raise GradingError("Mapping step failed") from e

    def _run_grading(self, mappings: dict, student_data: dict, model_data: dict, questions_data: dict) -> dict:
        """Execute grading chain with clean content."""
        try:
            output = self.grade_chain.invoke({
                "mappings": mappings,
                "model_data": json.dumps(model_data),
                "chunks": student_data,
                "questions": json.dumps(questions_data)
            })

            raw = output.content.strip()
            cleaned = re.sub(r'^\s*(```(?:json)?\s*\n?)?', '', raw)
            cleaned = re.sub(r'\s*(```)?\s*$', '', cleaned).strip()

            # Extract valid JSON block
            start = min(
                i for i, char in enumerate(cleaned) if char in "{["
            ) if any(char in cleaned for char in "{[") else len(cleaned)

            end = max(
                cleaned.rfind(char) for char in "}]"
            ) + 1 if any(char in cleaned for char in "}]") else len(cleaned)

            json_str = cleaned[start:end] if start < end else cleaned
            
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # Try with extra cleaning - remove problematic characters
                json_str = json_str.encode('utf-8', 'ignore').decode('utf-8')
                parsed = json.loads(json_str)

            logger.info(f"Grading complete → {self.student_name} (Q{self.question_number})")
            return parsed

        except json.JSONDecodeError as je:
            logger.error(f"Grading JSON parse failed at line {je.lineno} column {je.colno}", exc_info=True)
            logger.debug(f"Failed JSON string (first 500 chars): {json_str[:500] if 'json_str' in locals() else 'Unknown'}...")
            logger.debug(f"Full cleaned output: {cleaned if 'cleaned' in locals() else 'Unknown'}")
            raise GradingError("Invalid JSON from grader") from je
        except Exception as e:
            logger.error("Grading chain failed", exc_info=True)
            raise GradingError("Grading step failed") from e

    def _build_grade_doc(self, parsed_grades: dict) -> dict:
        now_iso = datetime.utcnow().isoformat()

        grades = parsed_grades.get("grades", [])
        if not grades:
            raise GradingError("No grades returned from LLM")

        main_grade = grades[0]  # assuming single main grade object

        # Round total to nearest 0.5 (17.75 → 18.0)
        raw_total = float(main_grade.get("score", 0))
        rounded_total = round(raw_total * 2) / 2

        # Extract comments from LLM output
        all_comments = main_grade.get("comments", [])  # list of strings

        doc = {
            "student_id": self.student_name,
            "question_number": self.question_number,
            "total_marks_awarded": rounded_total,
            "total_max_possible": float(main_grade.get("total_marks", 0)),
            "overall_reason": main_grade.get("reason", "Graded automatically"),
            "breakdown": [
                {
                    "criterion": item.get("criterion", "Unknown"),
                    "marks_awarded": float(item.get("marks_awarded", 0)),
                    "max_possible": float(item.get("max_possible", 0)),
                    "reason": item.get("reason", ""),
                    "evidence": "; ".join(item.get("evidence", [])) if item.get("evidence") else "",
                    "comments_summary": item.get("comments_summary", "")
                }
                for item in main_grade.get("breakdown", [])
            ],
            "comments": all_comments,  # ← only this field now
            "extracted_at": now_iso,
            "question_id": self.questions_id,
            "model_answer_id": self.model_answers_id,
            "student_answer_id": self.student_answers_id,
        }

        # Validate with updated Pydantic schema
        try:
            validated = StudentGradeDocument(**doc)
            return validated.model_dump(exclude_none=True)
        except Exception as ve:
            logger.warning(f"Grade validation failed – saving raw: {ve}")
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
            mapping = self._run_mapping(s_clean, q_clean)
            grades_parsed = self._run_grading(mapping, s_clean, m_clean, q_clean)
            grade_doc = self._build_grade_doc(grades_parsed)
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
    """
    Grade student using MongoDB IDs.
    Returns _id of saved grading document in 'student_grades' collection.
    """
    grader = StudentGrader(
        student_name=student_name,
        question_number=question_number,
        questions_id=questions_id,
        model_answers_id=model_answers_id,
        student_answers_id=student_answers_id,
    )
    return grader.grade()