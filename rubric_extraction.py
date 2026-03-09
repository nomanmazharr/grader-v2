import llm_setup

from pathlib import Path
from typing import List, Optional
from pymongo.collection import Collection

from database.mongodb import get_collection  
from logging_config import logger
from providers.langchain_pdf_extractor import PDFExtractor, PDFExtractionError
from prompts.extraction_prompts import MODEL_ANSWER_EXTRACTION_PROMPT
from schemas.model_answers import ModelAnswerDocument
from utils.db_utils import add_metadata, validate_and_prepare, save_to_mongodb


class ModelAnswerExtractionError(Exception):
    pass

class ModelAnswerExtractor:

    def __init__(self, pdf_path: str, pages: List[int]):
        self.pdf_path = pdf_path
        self.pages = pages
        self.collection: Collection = get_collection("model_answers")

    def _extract_with_vision(self) -> dict:
        try:
            extractor = PDFExtractor(
                self.pdf_path,
                self.pages,
                model_name=llm_setup.LLM_EXTRACTION_MODEL,
                render_dpi=llm_setup.LLM_PDF_RENDER_DPI,
            )
            data = extractor.extract(
                instruction_prompt=MODEL_ANSWER_EXTRACTION_PROMPT,
                output_schema=ModelAnswerDocument
            )
            logger.info("Model answer + rubric successfully extracted via LangChain")
            return data
        except PDFExtractionError as e:
            logger.error(f"Provider extraction failed: {e}", exc_info=True)
            raise ModelAnswerExtractionError("Failed to extract model answers") from e

    def run(self) -> Optional[str]:
        try:
            logger.info(f"Extracting model answers from {Path(self.pdf_path).name} pages {self.pages}")

            raw_data = self._extract_with_vision()
            
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