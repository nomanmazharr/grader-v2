from typing import List, Optional
from pydantic import BaseModel, Field


class SubPart(BaseModel):
    question_number: str = Field(..., description="Identifier e.g. '1.1', 'a)', '(i)'")
    answer: str = Field(..., description="Student's full answer content for this part")


class StudentAssignmentDocument(BaseModel):
    question: str = Field(..., description="Main question number/label")
    sub_parts: List[SubPart] = Field(..., description="Student answers per sub-part or single entry")

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