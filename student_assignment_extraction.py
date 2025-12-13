import io
import fitz
from llm_setup import client
import json
from logging_config import logger
from datetime import datetime
from pathlib import Path


def _create_pdf_subset(pdf_path: str, pages: list[int]) -> io.BytesIO:
    """
    Extract specified pages from a PDF and return as in-memory BytesIO PDF.
    Ensures the buffer behaves like a real PDF file.
    """
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()

    for p in pages:
        if 1 <= p <= len(doc):
            new_doc.insert_pdf(doc, from_page=p-1, to_page=p-1)

    # convert to proper PDF bytes
    pdf_bytes = new_doc.tobytes()
    buf = io.BytesIO(pdf_bytes)
    buf.name = "subset.pdf"   # this is critical for OpenAI to detect it's a PDF
    buf.seek(0)

    doc.close()
    new_doc.close()
    return buf

def _upload_to_openai(pdf_buffer: io.BytesIO, filename: str = "subset.pdf"):
    """
    Upload an in-memory PDF to OpenAI and return file object.
    """
    file_obj = client.files.create(
        file=pdf_buffer,
        purpose="user_data"
    )
    logger.info(f"Uploaded → file_obj: {file_obj}")
    return file_obj

assignment_schema = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The main question number (e.g., '1' or '4')"
        },
        "sub_parts": {
            "type": "array",
            "description": "List of subsections with their content, only if subsections like 1.1, a), A) are present",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {
                        "type": "string",
                        "description": "The identifier of the subsection or scenario (e.g., '1.1' or 'a)')"
                    },
                    "answer": {
                        "type": "string",
                        "description": "content paragraphs from the student's answer for marking criteria"
                    }
                },
                "required": ["question_number", "answer"],
                "additionalProperties": False
            }
        }
    },
    "required": ["question", "sub_parts"],
    "additionalProperties": False

}

def _extract_assignment_with_vision(file_obj):
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_obj.id},
                    {"type": "input_text", "text": """You are an expert at extracting and structuring handwritten or typed student answers from exam answer sheet PDFs.

CRITICAL RULES:
- Extract ONLY the complete answer of the main question visible on the page.
- NEVER merge or pull content from any overlapping/neighbouring question.
- Identify the main question number from the first clear label (e.g., Q1, Q.1, 1., Question 1, etc.).

SUB-SECTION DETECTION (in priority order):
1. If the student explicitly wrote numbered or lettered sub-parts such as:
   1.1, 1.2, 1(a), 1(b), a), b), c), (i), (ii), (A), (B), A., B., etc.
   → treat each as a separate sub_part with that exact id.

2. If there are NO such numbered/lettered labels, but the student used clear **headings** (usually bold, underlined, capitalized, or on a new line) such as:
   Advantages, Disadvantages, Definition, Types of ROM, Working, Conclusion, etc.
   → treat each heading as a separate sub_part and use the heading text as the id.

3. If neither (1) nor (2) exists (plain continuous text, bullet points, or numbered list that is part of explanation)
   → treat the entire answer as ONE single sub_part with id equal to the main question number.

Additional rules:
- Preserve original line breaks with \n\n between paragraphs.
- Keep bullet points, numbering, diagrams descriptions exactly as written.
- Ignore page headers like “Word Processing”, “Continued…”, watermarks, etc.
- Do NOT invent headings or sub-parts that are not visibly separated by the student.

---

Return only valid JSON — no markdown, no commentary, no preamble.
"""}
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "universal_exam_assignment",
                "strict": True,
                "schema": assignment_schema  # this is your updated flexible schema
            }
        }
    )

    try:
        raw_json = response.output_text.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3].strip()
        data = json.loads(raw_json)
        logger.info("assignment successfully parsed from vision model")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from model response: {e}")
        logger.debug(f"Raw output: {raw_json[:500]}...")
        raise

def extract_assignment_pipeline(pdf_path: str, pages: list[int], output_dir= "questions_and_model_answers_json_and_scripts"):
    """
    Full end-to-end pipeline:
      1. Extract subset of pages (in memory)
      2. Upload to OpenAI
      3. Get annotations JSON
    """
    try:
        logger.info(f"Starting assignment extraction from pages {pages}")

        # 1. Create subset
        pdf_buffer = _create_pdf_subset(pdf_path, pages)

        # 2. Upload
        file_id = _upload_to_openai(pdf_buffer)

        # 3. Extract with vision LLM
        assignment_data = _extract_assignment_with_vision(file_id)

        # 4. Save to disk
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        assignment_dir = Path(output_dir) / "assignment"
        assignment_dir.mkdir(parents=True, exist_ok=True)

        output_path = assignment_dir / f"assignment_extracted_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(assignment_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Assignment saved → {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Assignment extraction pipeline failed: {e}")
        logger.debug(f"Traceback: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    pdf_path = "Nov_25_testing/dataset/ICAEW_CR_Tuition_Exam_Qs_2025.pdf"
    pages = [2,3,4]
    extract_assignment_pipeline(pdf_path, pages)