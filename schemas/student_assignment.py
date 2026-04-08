from typing import List, Optional
from pydantic import BaseModel, Field


class SubPart(BaseModel):
    question_number: str = Field(..., description="Identifier e.g. '1.1', 'a)', '(i)'")
    answer: str = Field(..., description="Student's full answer content for this part")


class PageText(BaseModel):
    page: int = Field(..., description="1-based page number from the source PDF")
    text: str = Field(..., description="Extracted text content for that page")


class StudentAssignmentDocument(BaseModel):
    question: str = Field(..., description="Main question number/label")
    sub_parts: List[SubPart] = Field(..., description="Student answers per sub-part or single entry")

    # Exact verbatim heading/label the student wrote (e.g. "Quiestion 4", "Q4.", "Q 4")
    # Used by the annotator to anchor the main score and question boundaries even when the
    # student has a typo — the LLM extractor sees the PDF and captures this precisely.
    question_heading_text: Optional[str] = Field(
        default=None,
        description="Exact verbatim heading text the student wrote for this question (including typos)"
    )

    # Optional per-page extraction (helps downstream annotation on scanned PDFs)
    page_texts: Optional[List[PageText]] = Field(
        default=None,
        description="Optional list of extracted text per page (1-based page numbers)"
    )

    # Metadata
    pages: List[int] = Field(..., description="Extracted pages")
    extracted_at: str = Field(..., description="ISO timestamp")
    source_filename: str = Field(..., description="Original PDF filename")
    student_name: Optional[str] = Field(None, description="Student name if known")


OPENAI_STUDENT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The main question number (e.g., '1' or '4')"
        },
        "sub_parts": {
            "type": "array",
            "description": "All sub-parts or scenario entries within this question. Include EVERY sub-part the student wrote, across ALL provided pages. Sub-parts can be labelled as '1-', '2-', '3-', '1)', '2)', 'a)', 'b)', '1.1', '(i)', 'Case 1', 'Scenario 2', etc. A new sub-part label does NOT mean a new question — only a label containing the word 'Question' / 'Q' followed by a DIFFERENT number signals a new top-level question.",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {
                        "type": "string",
                        "description": "The exact label the student used for this sub-part (e.g., '1-', '2-', 'a)', '(ii)')"
                    },
                    "answer": {
                        "type": "string",
                        "description": "Full text of the student's answer for this sub-part, preserving all paragraphs and bullet points"
                    }
                },
                "required": ["question_number", "answer"],
                "additionalProperties": False
            }
        },
        "question_heading_text": {
            "type": ["string", "null"],
            "description": "Exact verbatim heading text the student wrote for this question (including typos, e.g. 'Quiestion 4'). Null if no visible heading."
        },
        "page_texts": {
            "type": ["array", "null"],
            "description": "Optional list of extracted text per page to support downstream annotation",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "number", "description": "1-based page number"},
                    "text": {"type": "string", "description": "Extracted text for that page"}
                },
                "required": ["page", "text"],
                "additionalProperties": False
            }
        }
    },
    "required": ["question", "sub_parts"],
    "additionalProperties": False

}