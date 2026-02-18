import asyncio
from typing import Tuple, Optional, List
from datetime import datetime
import nest_asyncio

nest_asyncio.apply()

from logging_config import logger
from question_pdf import extract_questions_pipeline
from rubric_extraction import extract_pdf_annotations_pipeline
from student_assignment_extraction import extract_assignment_pipeline
from grade import grade_student
from dummy_comments_scenario import annotate_pdf


# ── Async Extraction Wrappers ────────────────────────────────────────────────

async def extract_question_text_async(
    pdf_path: str,
    pages: List[int],
    question_num: str
) -> Tuple[bool, Optional[str]]:
    loop = asyncio.get_running_loop()
    try:
        logger.info(f"[Q{question_num}] Starting extraction (pages {pages})")

        doc_id = await loop.run_in_executor(
            None,
            lambda: extract_questions_pipeline(pdf_path=pdf_path, pages=pages)
        )

        if not doc_id:
            logger.error(f"[Q{question_num}] Extraction failed – no _id")
            return False, None

        logger.info(f"[Q{question_num}] Done → _id = {doc_id}")
        return True, doc_id

    except Exception as e:
        logger.error(f"[Q{question_num}] Failed: {str(e)}", exc_info=True)
        return False, None


async def extract_rubric_async(
    pdf_path: str,
    pages: List[int]
) -> Tuple[bool, Optional[str]]:
    loop = asyncio.get_running_loop()
    try:
        logger.info(f"[Rubric] Starting extraction (pages {pages})")

        doc_id = await loop.run_in_executor(
            None,
            lambda: extract_pdf_annotations_pipeline(pdf_path=pdf_path, pages=pages)
        )

        if not doc_id:
            logger.warning("[Rubric] No extraction – text fallback")
            return False, None

        logger.info(f"[Rubric] Done → _id = {doc_id}")
        return True, doc_id

    except Exception as e:
        logger.error(f"[Rubric] Failed: {str(e)}", exc_info=True)
        return False, None


async def extract_student_answers_async(
    pdf_path: str,
    pages: List[int],
    student_name: str
) -> Tuple[bool, Optional[str]]:
    loop = asyncio.get_running_loop()
    try:
        logger.info(f"[Student {student_name}] Starting extraction (pages {pages})")

        doc_id = await loop.run_in_executor(
            None,
            lambda: extract_assignment_pipeline(
                pdf_path=pdf_path,
                pages=pages,
                student_name=student_name
            )
        )

        if not doc_id:
            logger.error(f"[Student {student_name}] Extraction failed")
            return False, None

        logger.info(f"[Student {student_name}] Done → _id = {doc_id}")
        return True, doc_id

    except Exception as e:
        logger.error(f"[Student {student_name}] Failed: {str(e)}", exc_info=True)
        return False, None

def grade_student_wrapper(
    student_name: str,
    question_num: str,
    questions_id: str,
    model_answers_id: Optional[str],
    student_answers_id: str
) -> Tuple[bool, str, Optional[str]]:
    try:
        logger.info(f"Starting grading for {student_name} - Q{question_num}")

        grades_id = grade_student(
            student_name=student_name,
            question_number=question_num,
            questions_id=questions_id,
            model_answers_id=model_answers_id,
            student_answers_id=student_answers_id
        )

        if not grades_id:
            logger.error(f"Grading failed for {student_name} (Q{question_num})")
            return False, "Grading failed", None

        logger.info(f"Grading done → grades _id = {grades_id}")
        return True, "Grading successful", grades_id

    except Exception as e:
        logger.error(f"Grading error for {student_name} (Q{question_num}): {str(e)}", exc_info=True)
        return False, f"Grading error: {str(e)}", None


# ── Annotation Wrapper (sync) ────────────────────────────────────────────────

def annotate_student_wrapper(
    student_pdf_path: str,
    student_name: str,
    grades_id: str,
    student_pages: List[int],
    output_dir: str
) -> Tuple[bool, str, Optional[str]]:
    try:
        logger.info(f"Starting annotation for {student_name}")
        annotation_ok, annotated_pdf = annotate_pdf(
            input_pdf_path=student_pdf_path,
            output_dir=output_dir,
            student_name=student_name,
            grades_id=grades_id,
            student_pages=student_pages
        )

        if annotation_ok:
            logger.info(f"Annotation done → {annotated_pdf}")
            return True, "Annotation successful", annotated_pdf
        else:
            logger.error(f"Annotation failed for {student_name}")
            return False, "Annotation failed", None

    except Exception as e:
        logger.error(f"Annotation error for {student_name}: {str(e)}", exc_info=True)
        return False, f"Annotation error: {str(e)}", None

async def process_exam_async(
    question_pdf_path: str,
    question_pages: List[int],
    question_num: str,
    model_answer_pdf_path: str,
    answer_pages: List[int],
    student_pdf_path: str,
    student_pages: List[int],
    student_name: str,
    output_dir: str
) -> Tuple[bool, str, Optional[str], Optional[str]]:

    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info(f"ASYNC PIPELINE START → {student_name} | Q{question_num}")
    logger.info("=" * 70)

    # Parallel extractions
    q_task = extract_question_text_async(question_pdf_path, question_pages, question_num)
    r_task = extract_rubric_async(model_answer_pdf_path, answer_pages)
    s_task = extract_student_answers_async(student_pdf_path, student_pages, student_name)

    (q_ok, question_id), (r_ok, model_answers_id), (s_ok, student_answers_id) = await asyncio.gather(
        q_task, r_task, s_task, return_exceptions=True
    )

    # Handle any exceptions from parallel tasks
    if isinstance(q_ok, Exception):
        logger.error(f"Question extraction crashed: {q_ok}")
        q_ok, question_id = False, None
    if isinstance(r_ok, Exception):
        logger.error(f"Rubric extraction crashed: {r_ok}")
        r_ok, model_answers_id = False, None
    if isinstance(s_ok, Exception):
        logger.error(f"Student extraction crashed: {s_ok}")
        s_ok, student_answers_id = False, None

    if not q_ok:
        logger.error("Pipeline stopped: Question extraction failed")
        return False, "Question extraction failed", None, None

    if not r_ok:
        model_answers_id = None

    if not s_ok:
        logger.error(f"Pipeline stopped: Student extraction failed for {student_name}")
        return False, "Student extraction failed", None, None
    g_ok, g_message, grades_id = grade_student_wrapper(
        student_name=student_name,
        question_num=question_num,
        questions_id=question_id,
        model_answers_id=model_answers_id,
        student_answers_id=student_answers_id
    )

    if not g_ok:
        logger.error(f"Grading failed → {g_message}")
        return False, g_message, None, None
    a_ok, a_message, annotated_pdf = annotate_student_wrapper(
        student_pdf_path=student_pdf_path,
        student_name=student_name,
        grades_id=grades_id,
        student_pages=student_pages,
        output_dir=output_dir
    )

    status = "SUCCESS" if a_ok else "FAILED"
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"PIPELINE {status} → {student_name} (Q{question_num}) | {a_message} | took {duration:.2f}s")
    logger.info("=" * 70 + "\n")

    return a_ok, a_message, question_id, annotated_pdf


# Sync entry point
def process_exam(
    question_pdf_path: str,
    question_pages: List[int],
    question_num: str,
    model_answer_pdf_path: str,
    answer_pages: List[int],
    student_pdf_path: str,
    student_pages: List[int],
    student_name: str,
    output_dir: str
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Run the async pipeline synchronously."""
    return asyncio.run(
        process_exam_async(
            question_pdf_path=question_pdf_path,
            question_pages=question_pages,
            question_num=question_num,
            model_answer_pdf_path=model_answer_pdf_path,
            answer_pages=answer_pages,
            student_pdf_path=student_pdf_path,
            student_pages=student_pages,
            student_name=student_name,
            output_dir=output_dir
        )
    )


# if __name__ == "__main__":
#     logger.info("Async Exam Grading Pipeline")

#     process_exam(
#         question_pdf_path='dataset/ICAEW_CR_Tuition_Exam_Qs_2025.pdf',
#         question_pages=[2, 3, 4],
#         question_num='1',
#         model_answer_pdf_path='dataset/Bauhaus prepped answer.pdf',
#         answer_pages=[9, 10, 11, 12, 13, 14, 15],
#         student_pdf_path='dataset/Arend Schuiteman_612183_assignsubmission_file_CR_TuitionExam_Arend_Schuiteman.pdf',
#         student_pages=[1, 2, 3, 4, 5],
#         student_name='Arend_Schuiteman',
#         output_dir='annotations'
#     )
