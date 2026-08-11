"""Production grading pipeline.

Loads a pre-saved model answer from MongoDB, extracts the student PDF,
grades, and annotates. No question/rubric PDF extraction happens here.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from typing import Tuple, Optional, List, Literal
from datetime import datetime
import nest_asyncio

nest_asyncio.apply()

from bson import ObjectId

from logging_config import logger
from extraction.student_assignment_extraction import extract_assignment_pipeline
from grading.grade import grade_student
from grading.grade_csv import build_breakdown_csv
from annotation.annotator import annotate_pdf
from database.mongodb import get_collection
from errors import classify_error


def _build_grades_csv(grades_id: str) -> Optional[str]:
    """Fetch the saved grade doc and render its marks breakdown as CSV text.

    Best-effort: the CSV is a verification aid, so a failure here must never
    fail the grading run — it just yields no CSV.
    """
    try:
        grades_doc = get_collection("student_grades").find_one({"_id": ObjectId(grades_id)})
        if not grades_doc:
            logger.warning(f"[CSV] No grade doc for _id={grades_id}")
            return None
        return build_breakdown_csv(grades_doc)
    except Exception as e:
        logger.warning(f"[CSV] Failed to build breakdown CSV: {e}")
        return None


QuestionType = Literal["numerical", "theoretical"]


async def _extract_student_async(
    pdf_path: str,
    pages: List[int],
    student_name: str,
    question_num: str,
) -> Tuple[bool, Optional[str]]:
    loop = asyncio.get_running_loop()
    try:
        logger.info(f"[Student {student_name}] Extracting (pages {pages}, Q{question_num})")
        doc_id = await loop.run_in_executor(
            None,
            lambda: extract_assignment_pipeline(
                pdf_path=pdf_path,
                pages=pages,
                student_name=student_name,
                question_number=question_num,
            )
        )
        if not doc_id:
            logger.error(f"[Student {student_name}] Extraction returned no _id")
            return False, None
        logger.info(f"[Student {student_name}] Done → _id = {doc_id}")
        return True, doc_id
    except Exception as e:
        clean_msg, show_tb = classify_error(e)
        logger.error(f"[Student {student_name}] {clean_msg}", exc_info=show_tb)
        return False, None


async def grade_from_db_async(
    model_answers_id: str,
    student_pdf_path: str,
    student_pages: List[int],
    student_name: str,
    output_dir: str,
    question_num: str,
    question_type: str = "numerical",
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Grade a student PDF using a pre-saved model answer from MongoDB.

    Returns (success, message, annotated_pdf_path, grades_csv_text).
    The CSV is produced as soon as grading succeeds, so it is returned even
    when annotation later fails — that is precisely when it is most useful.
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info(f"GRADE FROM DB → {student_name} | Q{question_num} | type={question_type}")
    logger.info(f"  model_answers_id={model_answers_id}")
    logger.info("=" * 70)

    s_ok, student_answers_id = await _extract_student_async(
        student_pdf_path, student_pages, student_name, question_num
    )
    if not s_ok or not student_answers_id:
        return False, "Student answer extraction failed", None, None

    loop = asyncio.get_running_loop()
    try:
        grades_id = await loop.run_in_executor(
            None,
            lambda: grade_student(
                student_name=student_name,
                question_number=question_num,
                questions_id=None,
                model_answers_id=model_answers_id,
                student_answers_id=student_answers_id,
                question_type=question_type,
            )
        )
    except Exception as e:
        clean_msg, show_tb = classify_error(e)
        logger.error(f"[Grading] {clean_msg}", exc_info=show_tb)
        return False, clean_msg, None, None

    if not grades_id:
        return False, "Grading returned no result", None, None

    # Build the CSV now — independent of annotation, so it survives an
    # annotation failure below.
    grades_csv = _build_grades_csv(grades_id)

    try:
        annotation_ok, annotated_pdf = annotate_pdf(
            input_pdf_path=student_pdf_path,
            output_dir=output_dir,
            student_name=student_name,
            grades_id=grades_id,
            student_pages=student_pages,
        )
    except Exception as e:
        clean_msg, show_tb = classify_error(e)
        logger.error(f"[Annotation] {clean_msg}", exc_info=show_tb)
        return False, clean_msg, None, grades_csv

    duration = (datetime.now() - start_time).total_seconds()
    status = "SUCCESS" if annotation_ok else "PARTIAL (graded, annotation failed)"
    logger.info(f"GRADE FROM DB {status} → {student_name} Q{question_num} | {duration:.2f}s")
    logger.info("=" * 70 + "\n")

    msg = "Grading and annotation complete" if annotation_ok else "Annotation failed"
    return annotation_ok, msg, annotated_pdf, grades_csv


def grade_from_db(
    model_answers_id: str,
    student_pdf_path: str,
    student_pages: List[int],
    student_name: str,
    output_dir: str,
    question_num: str,
    question_type: str = "numerical",
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Sync entry point for the production grading pipeline."""
    return asyncio.run(
        grade_from_db_async(
            model_answers_id=model_answers_id,
            student_pdf_path=student_pdf_path,
            student_pages=student_pages,
            student_name=student_name,
            output_dir=output_dir,
            question_num=question_num,
            question_type=question_type,
        )
    )
