from __future__ import annotations

from typing import List, Optional, Union
from pydantic import BaseModel, Field


class MarkingPoint(BaseModel):
    marks: Union[float, str, None] = Field(..., description="Mark value (number or string like '1 each', 'max 4', 'tick')")
    description: str = Field(..., description="Exact original text describing what earns the mark")
    sub_criteria: Optional[List["MarkingPoint"]] = Field(
        None,
        description=(
            "Optional nested criteria for this point. Use when the marking guide has a broad criterion (e.g. 4 marks) "
            "that is broken into multiple smaller markable sub-points."
        ),
    )


class AnswerItem(BaseModel):
    question_number: str = Field(..., description="Subquestion identifier e.g. '1.1', 'a)', '(i)'")
    answer: Optional[str] = Field(None, description="Model answer content – no marking criteria here")
    marking_criteria: Optional[List[MarkingPoint]] = Field(None, description="List of individual markable points")
    total_marks_available: Optional[str] = Field(None, description="Marks available for this part")
    maximum_marks: Optional[str] = Field(None, description="Maximum marks if explicitly stated")
    sub_answers: Optional[List["AnswerItem"]] = Field(None, description="Nested sub-parts")


class ModelAnswerDocument(BaseModel):
    question_title: str = Field(..., description="The question label only, e.g. 'Ans.5' or 'Question 5'. Do NOT include sub-section or topic names here.")
    description: Optional[str] = Field(None, description="Introductory context / assumptions")
    total_marks: Optional[str] = Field(None, description="Total marks for the entire question (sum of all sub-sections)")
    answers: List[AnswerItem] = Field(..., description="One entry per sub-section or topic heading with marks. If the question has multiple sub-sections (e.g. topic A (5 Marks), topic B (5 Marks)), each becomes a separate entry. Never merge sub-sections into one entry.")

    # Metadata
    pages: List[int] = Field(..., description="Extracted pages")
    extracted_at: str = Field(..., description="ISO timestamp")
    source_filename: str = Field(..., description="Original PDF filename")


OPENAI_MODEL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "question_title": {
            "type": "string",
            "description": "The question identifier only, e.g. 'Ans.5' or 'Question 5'. Do NOT embed sub-section or topic names here — those belong in the answers array."
        },
        "description": {
            "type": ["string", "null"],
            "description": "Introductory paragraph or assumptions if present"
        },
        "total_marks": {
            "type": ["string", "null"],
            "description": "Total marks for the main question"
        },
        "answers": {
            "type": "array",
            "description": "One entry per sub-section or topic heading with marks. If the question has sub-sections (e.g. 'Topic A (5 Marks)', 'Topic B (5 Marks)'), each is a separate entry. Always check ALL provided pages before finalising this array.",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {
                        "type": "string",
                        "description": "Subquestion number such as '4.1', '4.1(a)', etc. if question number explicitly present don't include heading into question number else if subsections are on base of heading only use heading in that case"
                    },
                    "answer": {
                        "type": ["string", "null"],
                        "description": "Model answer content for this question_number, never include marking criteria in answer"
                    },
                    "marking_criteria": {
                        "type": ["array", "null"],
                        "description": "List of individual markable points from printed and handwritten criteria",
                        "items": {
                            "type": "object",
                            "properties": {
                                "marks": {
                                    "type": ["number", "string"],
                                    "description": "Mark value: number (0.5, 1, 2, 1/2) or string like '1 each', 'max 4', 'OF', 'tick'"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Exact original text describing what earns the mark"
                                },
                                "sub_criteria": {
                                    "type": ["array", "null"],
                                    "description": "Optional nested sub-criteria for this criterion",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "marks": {"type": ["number", "string"]},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["marks", "description"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["marks", "description"],
                            "additionalProperties": False
                        }
                    },
                    "total_marks_available": {
                        "type": ["string", "null"],
                        "description": "Marks available for this specific part"
                    },
                    "maximum_marks": {
                        "type": ["string", "null"],
                        "description": "Maximum marks if explicitly mentioned"
                    },
                    "sub_answers": {
                        "type": ["array", "null"],
                        "description": "Nested subdivisions (e.g. (a), (b), (i), etc.)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question_number": {
                                    "type": "string",
                                    "description": "Sub-subquestion number, don't include heading if question number explicitly present for sub answer"
                                },
                                "answer": {
                                    "type": ["string", "null"],
                                    "description": "Model answer content, don't include marking criteria in answers"
                                },
                                "marking_criteria": {
                                    "type": ["array", "null"],
                                    "description": "List of individual markable points",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "marks": {"type": ["number", "string"]},
                                            "description": {"type": "string"},
                                            "sub_criteria": {
                                                "type": ["array", "null"],
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "marks": {"type": ["number", "string"]},
                                                        "description": {"type": "string"}
                                                    },
                                                    "required": ["marks", "description"],
                                                    "additionalProperties": False
                                                }
                                            }
                                        },
                                        "required": ["marks", "description"],
                                        "additionalProperties": False
                                    }
                                },
                                "total_marks_available": {
                                    "type": ["string", "null"],
                                    "description": "Marks available"
                                },
                                "maximum_marks": {
                                    "type": ["string", "null"],
                                    "description": "Maximum marks"
                                },
                                "sub_answers": {
                                    "type": ["array", "null"],
                                    "description": "Further nested subdivisions",
                                    "items": {
                                        "type": "object",
                                        "properties": {},
                                        "required": [],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["question_number", "answer", "marking_criteria", "total_marks_available", "maximum_marks", "sub_answers"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["question_number", "answer", "marking_criteria", "total_marks_available", "maximum_marks", "sub_answers"],
                "additionalProperties": False
            }
        }
    },
    "required": ["question_title", "answers", "description", "total_marks"],
    "additionalProperties": False
}
