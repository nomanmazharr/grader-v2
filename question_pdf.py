import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pymongo.collection import Collection

from database.mongodb import get_questions_collection
from llm_setup import client
from logging_config import logger
from schemas.question import OPENAI_QUESTION_SCHEMA, QuestionDocument
from utils.pdf_openai_utils import create_pdf_subset, upload_to_openai
from utils.db_utils import add_metadata, validate_and_prepare, save_to_mongodb


class QuestionExtractionError(Exception):
    pass


class QuestionExtractor:
    def __init__(self, pdf_path: str, pages: List[int]):
        self.pdf_path = pdf_path
        self.pages = pages
        self.collection: Collection = get_questions_collection()

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
                                "text": """You are an expert at extracting exam questions from PDF question papers.

Focus strictly on Question that is present as a whole and extact all of its question content ,

Rules:
- Preserve **exact original wording**, line breaks, bullet points, and formatting.
- Include any introductory scenario/description in the `description` field.
- If there are no subquestions, create one SubQuestion with the main question number.
- Capture marks exactly as written (e.g., "(6 marks)", "Total: 20 marks").


---

Return only valid JSON — no markdown, no commentary, no preamble.
"""
                            }
                        ]
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "universal_exam_question",
                        "strict": True,
                        "schema": OPENAI_QUESTION_SCHEMA
                    }
                }
            )

            raw_json = response.output_text.strip()
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:-3].strip()

            data = json.loads(raw_json)
            logger.info("Question successfully parsed from vision model")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed from OpenAI response: {e}", exc_info=True)
            raise QuestionExtractionError("Failed to parse JSON from OpenAI response") from e
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}", exc_info=True)
            raise QuestionExtractionError("Failed to extract question with OpenAI vision") from e

    def run(self) -> Optional[str]:
        try:
            logger.info(f"Extracting question from {Path(self.pdf_path).name} pages {self.pages}")

            pdf_buffer = create_pdf_subset(self.pdf_path, self.pages)
            file_id = upload_to_openai(pdf_buffer)
            question_data = self._extract_with_vision(file_id)
            
            question_data = add_metadata(question_data, self.pdf_path, self.pages)
            to_save = validate_and_prepare(question_data, QuestionDocument)
            doc_id = save_to_mongodb(self.collection, to_save, entity_type="question")

            return doc_id
        except QuestionExtractionError as qe:
            logger.error(f"Question extraction pipeline failed: {qe}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}", exc_info=True)
            return None


def extract_questions_pipeline(pdf_path: str, pages: List[int]) -> Optional[str]:
    extractor = QuestionExtractor(pdf_path, pages)
    return extractor.run()