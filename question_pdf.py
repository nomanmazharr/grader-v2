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

question_schema = {

  "type": "object",
  "properties": {
    "question_title": {
      "type": "string",
      "description": "Main question title, e.g., 'Question 4'"
    },
    "description": {
      "type": ["string", "null"],
      "description": "Any introductory text, scenario, or description before the subquestions"
    },
    "total_marks": {
      "type": ["string", "null"],
      "description": "Total marks for the entire question if stated, captured exactly as written"
    },
    "questions": {
      "type": "array",
      "description": "List of subquestions (or single main question if no subs)",
      "items": {
        "type": "object",
        "properties": {
          "question_number": {
            "type": "string",
            "description": "Subquestion identifier like '1.1', 'a)', or heading if no explicit number"
          },
          "content": {
            "type": "string",
            "description": "Full original text of the subquestion, preserving exact wording, line breaks, bullet points, and formatting"
          },
          "marks": {
            "type": ["string", "null"],
            "description": "Marks allocation if mentioned, captured exactly as written, e.g., '5 marks' or '(6)'"
          },
          "sub_questions": {
            "type": ["array", "null"],
            "description": "Nested subquestions if present (e.g., (i), (ii))",
            "items": {
              "type": "object",
              "properties": {},
              "required": [],
              "additionalProperties": False
            }
          }
        },
        "required": ["question_number", "content", "marks", "sub_questions"],
        "additionalProperties": False
      }
    }
  },
  "required": ["question_title", "description", "total_marks", "questions"],
  "additionalProperties": False
}


def _extract_question_with_vision(file_obj):
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_obj.id},
                    {"type": "input_text", "text": """You are an expert at extracting exam questions from PDF question papers.

Focus strictly on Question that is present as a whole and extact all of its question content ,

Rules:
- Preserve **exact original wording**, line breaks, bullet points, and formatting.
- Include any introductory scenario/description in the `description` field.
- If there are no subquestions, create one SubQuestion with the main question number.
- Capture marks exactly as written (e.g., "(6 marks)", "Total: 20 marks").


---

Return only valid JSON — no markdown, no commentary, no preamble.

"""}
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "universal_exam_question",
                "strict": True,
                "schema": question_schema  # this is your updated flexible schema
            }
        }
        # service_tier="priority"
    )

    try:
        raw_json = response.output_text.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3].strip()
        data = json.loads(raw_json)
        logger.info("question successfully parsed from vision model")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from model response: {e}")
        logger.debug(f"Raw output: {raw_json[:500]}...")
        raise

def extract_questions_pipeline(pdf_path: str, pages: list[int], output_dir= "questions_and_model_answers_json_and_scripts"):
    """
    Full end-to-end pipeline:
      1. Extract subset of pages (in memory)
      2. Upload to OpenAI
      3. Get annotations JSON
    """
    try:
        logger.info(f"Starting question extraction from pages {pages}")

        # 1. Create subset
        pdf_buffer = _create_pdf_subset(pdf_path, pages)

        # 2. Upload
        file_id = _upload_to_openai(pdf_buffer)

        # 3. Extract with vision LLM
        question_data = _extract_question_with_vision(file_id)

        # 4. Save to disk
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        question_dir = Path(output_dir) / "question"
        question_dir.mkdir(parents=True, exist_ok=True)

        output_path = question_dir / f"question_extracted_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(question_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Question saved → {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Question extraction pipeline failed: {e}")
        logger.debug(f"Traceback: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    pdf_path = "Nov_25_testing/dataset/ICAEW_CR_Tuition_Exam_Qs_2025.pdf"
    pages = [2,3,4]
    extract_questions_pipeline(pdf_path, pages)