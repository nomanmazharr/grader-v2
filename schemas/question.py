from typing import List, Optional
from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    question_number: str = Field(..., description="e.g. '1.1', 'a)', '(i)'")
    content: str = Field(..., description="Full text of this sub-question")
    marks: Optional[str] = Field(None, description="Marks e.g. '4 marks', '(6)'")
    sub_questions: Optional[List["SubQuestion"]] = Field(None, description="Deeper nesting if needed")


class QuestionItem(BaseModel):
    question_number: str = Field(...)
    content: str = Field(...)
    marks: Optional[str] = Field(None)
    sub_questions: Optional[List[SubQuestion]] = Field(None)


class QuestionDocument(BaseModel):
    question_title: str = Field(...)
    description: Optional[str] = Field(None)
    total_marks: Optional[str] = Field(None)
    questions: List[QuestionItem]

    # Metadata fields we add ourselves
    pdf_path: Optional[str] = None
    pages: Optional[List[int]] = None
    extracted_at: Optional[str] = None
    source_filename: Optional[str] = None


# JSON Schema still used for OpenAI structured output
OPENAI_QUESTION_SCHEMA = {
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