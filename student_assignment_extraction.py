import llm_setup

from pathlib import Path
from typing import List, Optional

from pymongo.collection import Collection

from database.mongodb import get_collection
from logging_config import logger
from providers.langchain_pdf_extractor import PDFExtractor, PDFExtractionError
from prompts.extraction_prompts import get_student_extraction_prompt
from schemas.student_assignment import StudentAssignmentDocument
from utils.db_utils import add_metadata, validate_and_prepare, save_to_mongodb


class StudentAssignmentExtractionError(Exception):
    pass

class StudentAssignmentExtractor:
    COLLECTION_NAME = "student_assignments"

    def __init__(self, pdf_path: str, pages: List[int], student_name: Optional[str] = None,
                 question_number: Optional[str] = None):
        self.pdf_path = pdf_path
        self.pages = pages
        self.student_name = student_name
        self.question_number = question_number
        self.collection: Collection = get_collection(self.COLLECTION_NAME)

    def _extract_with_vision(self) -> dict:
        try:
            prompt = get_student_extraction_prompt(self.question_number)
            extractor = PDFExtractor(
                self.pdf_path,
                self.pages,
                model_name=llm_setup.LLM_EXTRACTION_MODEL,
                render_dpi=llm_setup.LLM_PDF_RENDER_DPI,
            )
            data = extractor.extract(
                instruction_prompt=prompt,
                output_schema=StudentAssignmentDocument,
            )
            logger.info("Student assignment successfully extracted via LangChain")
            return data
        except PDFExtractionError as e:
            logger.error(f"Student extraction failed: {e}", exc_info=True)
            raise StudentAssignmentExtractionError("Student extraction failed") from e

    def run(self) -> Optional[str]:
        try:
            logger.info(f"Extracting student answers from {Path(self.pdf_path).name} pages {self.pages}")

            raw_data = self._extract_with_vision()
            
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
    student_name: Optional[str] = None,
    question_number: Optional[str] = None
) -> Optional[str]:
    extractor = StudentAssignmentExtractor(pdf_path, pages, student_name, question_number)
    return extractor.run()
