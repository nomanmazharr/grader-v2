import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pymongo.collection import Collection

from database.mongodb import get_collection
from llm_setup import client
from logging_config import logger
from schemas.student_assignment import StudentAssignmentDocument, OPENAI_STUDENT_SCHEMA
from utils.pdf_openai_utils import create_pdf_subset, upload_to_openai
from utils.db_utils import add_metadata, validate_and_prepare, save_to_mongodb


class StudentAssignmentExtractionError(Exception):
    pass


class StudentAssignmentExtractor:
    COLLECTION_NAME = "student_assignments"

    def __init__(self, pdf_path: str, pages: List[int], student_name: Optional[str] = None):
        self.pdf_path = pdf_path
        self.pages = pages
        self.student_name = student_name
        self.collection: Collection = get_collection(self.COLLECTION_NAME)

    def _extract_with_vision(self, file_id: str) -> dict:
        try:
            response = client.responses.create(
                model="gpt-5-mini-2025-08-07",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": file_id},
                            {
                                "type": "input_text",
                                "text": """You are an expert at extracting and structuring handwritten or typed student answers from PDFs.

CRITICAL RULES:
- Extract ONLY the complete answer of the main question visible on these pages.
- NEVER merge or pull content from any overlapping/neighbouring question.
- Identify the main question number from the first clear label (e.g., Q1, Q.1, 1., Question 1, etc.).

SUB-SECTION DETECTION (strict priority order – apply ONLY the first that matches):
1. If the student has explicitly written numbered or lettered sub-parts such as:
   1.1, 1.2, 1(a), 1(b), a), b), c), (i), (ii), (A), (B), A., B., (i), (ii), i), ii), etc.
   → treat each as a separate sub_part with that exact identifier as question_number.

2. OTHERWISE – even if the student uses bold headings, topic titles, underlined phrases, or descriptive section names – treat the entire answer as ONE single question.
   → Output exactly ONE sub_part.
   → Use the main question identifier (e.g., "Question 1" or "1") as the question_number.
   → Concatenate all content in order, preserving paragraphs and line breaks.

Additional rules:
- Preserve original line breaks with \\n\\n between paragraphs.
- Keep bullet points, numbering, tables, and diagram descriptions exactly as written.
- Ignore page headers, footers, “Continued…”, watermarks, candidate numbers, etc.
- NEVER invent or create sub-parts based on content headings or topic names.
- Do NOT treat descriptive headings as sub-question identifiers.

Return ONLY valid JSON — no markdown, no commentary, no preamble.
"""
                            }
                        ]
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "universal_exam_assignment",
                        "strict": True,
                        "schema": OPENAI_STUDENT_SCHEMA
                    }
                }
            )

            raw_json = response.output_text.strip()
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:-3].strip()

            data = json.loads(raw_json)
            logger.info("Student assignment successfully parsed from vision model")
            return data
        except Exception as e:
            logger.error(f"Student answer extraction failed: {e}", exc_info=True)
            raise StudentAssignmentExtractionError("Failed to extract student answers") from e

    def run(self) -> Optional[str]:
        try:
            logger.info(f"Extracting student answers from {Path(self.pdf_path).name} pages {self.pages}")

            pdf_buffer = create_pdf_subset(self.pdf_path, self.pages)
            file_id = upload_to_openai(pdf_buffer)
            raw_data = self._extract_with_vision(file_id)
            
            data_with_meta = add_metadata(raw_data, self.pdf_path, self.pages, student_name=self.student_name)
            to_save = validate_and_prepare(data_with_meta, StudentAssignmentDocument)
            doc_id = save_to_mongodb(self.collection, to_save, entity_type="student assignment")

            return doc_id
        except StudentAssignmentExtractionError as sae:
            logger.error(f"Student assignment extraction failed: {sae}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in student assignment extraction: {e}", exc_info=True)
            return None


# Compatibility wrapper (update your main.py calls to use this)
def extract_assignment_pipeline(
    pdf_path: str,
    pages: List[int],
    student_name: Optional[str] = None
) -> Optional[str]:
    extractor = StudentAssignmentExtractor(pdf_path, pages, student_name)
    return extractor.run()