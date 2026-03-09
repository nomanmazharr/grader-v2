import llm_setup

from pathlib import Path
from typing import List, Optional

from pymongo.collection import Collection

from database.mongodb import get_questions_collection
from logging_config import logger
from providers.langchain_pdf_extractor import PDFExtractor, PDFExtractionError
from prompts.extraction_prompts import QUESTION_EXTRACTION_PROMPT
from schemas.question import QuestionDocument
from utils.db_utils import add_metadata, validate_and_prepare, save_to_mongodb


class QuestionExtractionError(Exception):
    pass

class QuestionExtractor:
    def __init__(self, pdf_path: str, pages: List[int]):
        self.pdf_path = pdf_path
        self.pages = pages
        self.collection: Collection = get_questions_collection()

    def _extract_with_vision(self) -> dict:
        try:
            extractor = PDFExtractor(
                self.pdf_path,
                self.pages,
                model_name=llm_setup.LLM_EXTRACTION_MODEL,
                render_dpi=llm_setup.LLM_PDF_RENDER_DPI,
            )
            data = extractor.extract(
                instruction_prompt=QUESTION_EXTRACTION_PROMPT,
                output_schema=QuestionDocument
            )
            logger.info("Question successfully extracted via LangChain")
            return data
        except PDFExtractionError as e:
            logger.error(f"Provider extraction failed: {e}", exc_info=True)
            raise QuestionExtractionError("Failed to extract question") from e


    def run(self) -> Optional[str]:
        try:
            logger.info(f"Extracting question from {Path(self.pdf_path).name} pages {self.pages}")

            question_data = self._extract_with_vision()
            
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