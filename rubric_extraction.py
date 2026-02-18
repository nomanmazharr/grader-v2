import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json
import fitz
from bson import ObjectId
from pymongo.collection import Collection

from database.mongodb import get_collection  
from llm_setup import client
from logging_config import logger
from schemas.model_answers import OPENAI_MODEL_ANSWER_SCHEMA, ModelAnswerDocument
from utils.db_utils import add_metadata, validate_and_prepare, save_to_mongodb
from utils.pdf_openai_utils import create_pdf_subset, upload_to_openai


class ModelAnswerExtractionError(Exception):
    pass


class ModelAnswerExtractor:

    def __init__(self, pdf_path: str, pages: List[int]):
        self.pdf_path = pdf_path
        self.pages = pages
        self.collection: Collection = get_collection("model_answers")

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
                                "text": """You are an expert in extracting and structuring model answers and marking criteria from exam marking guides, including any handwritten annotations visible in the PDF.

Focus strictly on the Question that is present as a whole and extract all of its model answers, printed marking criteria, and handwritten annotations.

You are provided a PDF file containing:
- Model answers
- Printed marking criteria
- Headings and subheadings
- Handwritten annotations (usually in red ink)

Your task is to extract all content directly from the PDF in one step and return structured JSON that strictly follows the provided schema.

IMPORTANT GLOBAL RULES:
- Preserve wording EXACTLY as written. No rewriting, rephrasing, summarizing, or adding interpretations.
- NEVER omit marking criteria for any subsection if marking criteria text or annotations exist anywhere in the PDF.
- ALWAYS extract marking criteria for the main question AND for each subsection separately if present.
- Handwritten annotations (red ink) must be merged directly into the `marking_criteria` array as separate objects.
- If marking criteria or annotations apply to multiple subsections or the entire main question, you MUST replicate the entire block across all relevant parts.
- If unsure where a criteria block or annotation belongs → assign it to the main question AND to all subsections to avoid any loss.

---

EXTRACTION RULES

1. Identify Subsections Reliably
Create a subsection whenever explicit labels appear:
- Numeric: 1.1, 1.2, 1.3 …
- Alphabetic: a), b), c)
- Roman or nested: (i), (ii), (iii), A., B.

If no subsections → single entry with main question number.

2. Hierarchy Enforcement
Preserve full nesting:
Main question
→ Subquestion (1.1, 1.2, a), b))
  → Nested sub ((i), (ii))
    → Deeper nested
Use `sub_answers` only when real nesting exists.

3. Answer vs Marking Criteria Separation
- `answer` contains ONLY the model answer content.
- `marking_criteria` contains ONLY marking rules (printed text + red handwritten annotations).
- DO NOT include maximum_marks or total_marks_available inside marking_criteria.

4. Marks Extraction
- Extract maximum_marks from phrases like "Maximum marks", "Maximum full marks", "Maximum", "[6]", etc.
- Extract total_marks_available from "Total Possible Marks", "Marks Available", "Total Marks", etc.
- Never duplicate maximum/total marks in child levels — only once at the correct parent level.
- NEVER leave maximum_marks or total_marks_available null if present — use the table or text values.
- If a subsection has nested sub-subsections, maximum marks and total available marks appear ONLY once for the parent subsection.
- Use high-level table marks (e.g., 26 for (a)) for total_marks_available and maximum_marks fields — NOT in marking_criteria array.

5. Marking Criteria & Annotations Handling — STRUCTURED VERSION (REQUIRED)

For every question and subsection:
- Identify ALL individual marking points from printed criteria and handwritten red annotations.
- Focus on detailed red marks next to model answer lines — ignore high-level table marks (use those for total_marks_available/maximum_marks).
- Each separate red mark (e.g., ½, 1, ¼, "1 each", "max 3", "tick") next to a line becomes ONE object.
- Pair each mark with the EXACT nearest sentence or phrase in the model answer text.

Create one object per point:
{
  "marks": <number if possible (0.5, 1, 2, 0.25) — or original string like "1 each", "max 4", "OF", "tick">,
  "description": "<exact original wording from the model answer line the mark is next to — preserve 100%>"
}

Rules:
- Scan every page for red marks and pair with closest text line.
- Never merge points — one object per mark.
- Include printed detailed criteria (e.g., bullet points) as separate items if not already covered by red marks.
- Include general notes (e.g., "Tutorial note: ...", "Own figure rule") as separate items.
- If criteria apply to multiple parts → duplicate array.
- If no detailed criteria → empty array []

Generic examples of correct items (from red marks next to text):
- {"marks": 0.5, "description": "correct definition of key term"}
- {"marks": 1, "description": "explained process with example"}
- {"marks": 0.25, "description": "mentioned relevant factor"}
- {"marks": 2, "description": "fully labelled diagram provided"}
- {"marks": "1 each", "description": "for any valid point (max 3)"}
- {"marks": 0.5, "description": "correct formula stated"}
- {"marks": 0.5, "description": "unit included in answer"}
- {"marks": "tick", "description": "correct final calculation"}
- {"marks": "OF", "description": "own figure rule applies"}

Example full array:
[
  {"marks": 1, "description": "correct identification of main concept"},
  {"marks": 0.5, "description": "relevant principle applied"},
  {"marks": "1 each", "description": "for each valid example (max 2)"},
  {"marks": 2, "description": "accurate calculation shown"}
]

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
                        "name": "universal_exam_rubric",
                        "strict": True,
                        "schema": OPENAI_MODEL_ANSWER_SCHEMA
                    }
                }
            )

            raw_json = response.output_text.strip()
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:-3].strip()

            data = json.loads(raw_json)
            logger.info("Model answer + rubric successfully parsed")
            return data
        except Exception as e:
            logger.error(f"Model answer extraction failed: {e}", exc_info=True)
            raise ModelAnswerExtractionError("Failed to extract model answers") from e

    def run(self) -> Optional[str]:
        try:
            logger.info(f"Extracting model answers from {Path(self.pdf_path).name} pages {self.pages}")

            pdf_buffer = create_pdf_subset(self.pdf_path, self.pages)
            file_id = upload_to_openai(pdf_buffer)
            raw_data = self._extract_with_vision(file_id)
            
            data_with_meta = add_metadata(raw_data, self.pdf_path, self.pages)
            to_save = validate_and_prepare(data_with_meta, ModelAnswerDocument)
            doc_id = save_to_mongodb(self.collection, to_save, entity_type="model answer")

            return doc_id
        except ModelAnswerExtractionError as me:
            logger.error(f"Model answer extraction pipeline failed: {me}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in model answer extraction: {e}", exc_info=True)
            return None


def extract_pdf_annotations_pipeline(pdf_path: str, pages: List[int]) -> Optional[str]:
    extractor = ModelAnswerExtractor(pdf_path, pages)
    return extractor.run()