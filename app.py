import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import re
import tempfile
import traceback
import streamlit as st

from main import grade_from_db
from database.question_loader import list_available_questions


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pages(page_str: str) -> list[int]:
    if not page_str.strip():
        return []
    pages = []
    for part in page_str.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            if a.strip().isdigit() and b.strip().isdigit():
                pages.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            pages.append(int(part))
    return pages


def _extract_question_number(title: str) -> str:
    """Extract leading number from title like '1 BAUHAUS PLC' → '1'."""
    match = re.match(r'^(\d+)', title.strip())
    return match.group(1) if match else "1"


def _save_file(uploaded_file, temp_dir: str) -> str:
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


@st.cache_data(ttl=60)
def _load_questions() -> list[dict]:
    return list_available_questions()


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Exam Grader", page_icon="📚", layout="wide")
    st.title("📚 Automated Exam Grader")

    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="exam_grader_")
    temp_dir = st.session_state.temp_dir

    # ── Load questions from MongoDB ──────────────────────────────────────────
    try:
        questions = _load_questions()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.stop()

    if not questions:
        st.warning("No questions found in the database. Use the extraction tool to add questions first.")
        st.stop()

    # ── Question selector ────────────────────────────────────────────────────
    st.header("Select Question")
    labels = [f"{q['question_title']}  ({q['total_marks']} marks)" for q in questions]
    idx = st.selectbox("Question", range(len(labels)), format_func=lambda i: labels[i])
    selected = questions[idx]

    if selected.get("description"):
        with st.expander("Question description"):
            st.write(selected["description"])

    st.markdown("---")

    # ── Student upload + config ──────────────────────────────────────────────
    st.header("Student Assignment")
    student_pdf = st.file_uploader("Upload Student PDF", type="pdf")

    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Student Name", value="Student")
        student_pages_str = st.text_input("Student Pages (e.g. 1-5 or 1,2,3)", value="1,2,3,4,5")
    with col2:
        output_dir = st.text_input("Output Directory", value="annotations")
        question_type = st.selectbox("Question Type", options=["numerical", "theoretical"], index=0) or "numerical"

    # ── Grade ────────────────────────────────────────────────────────────────
    if st.button("Grade Student", type="primary"):
        if not student_pdf:
            st.error("Please upload the student assignment PDF.")
            st.stop()

        student_pages = _parse_pages(student_pages_str)
        if not student_pages:
            st.error("Please specify valid student page numbers.")
            st.stop()

        model_answers_id = selected["_id"]
        question_num = _extract_question_number(selected["question_title"])

        os.makedirs(output_dir, exist_ok=True)
        student_path = _save_file(student_pdf, temp_dir)

        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Extracting student answers, grading and annotating...")
            progress.progress(20)

            ok, message, annotated_path = grade_from_db(
                model_answers_id=model_answers_id,
                student_pdf_path=student_path,
                student_pages=student_pages,
                student_name=student_name,
                output_dir=output_dir,
                question_num=question_num,
                question_type=question_type,
            )
            progress.progress(100)

            if ok:
                status.success("Grading complete!")
                st.success(message)
                if annotated_path and os.path.exists(annotated_path):
                    with open(annotated_path, "rb") as f:
                        st.download_button(
                            "Download Annotated PDF",
                            f.read(),
                            file_name=f"{student_name}_annotated.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.warning("Grading succeeded but annotated PDF not found.")
                    if annotated_path:
                        st.caption(f"Expected path: {annotated_path}")
            else:
                status.error(f"Failed: {message}")

        except Exception as e:
            status.error(f"Unexpected error: {e}")
            st.code(traceback.format_exc())
            progress.empty()


if __name__ == "__main__":
    main()
