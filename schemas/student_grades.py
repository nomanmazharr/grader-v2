from typing import List, Optional, Union
from pydantic import BaseModel, Field


class GradeBreakdownItem(BaseModel):
    criterion: str = Field(..., description="Marking criterion or point description")
    marks_awarded: Union[float, int] = Field(..., description="Marks given")
    max_possible: Union[float, int] = Field(..., description="Maximum marks for this item")
    reason: str = Field(..., description="Explanation for awarded marks")
    evidence: Optional[str] = Field(None, description="Relevant student text snippet")
    comments_summary: Optional[str] = Field(None, description="Grader comment for this specific item (if any)")


class StudentGradeDocument(BaseModel):
    student_id: str = Field(..., description="Student name or ID")
    question_number: str = Field(..., description="Question number (e.g. '1', '4.2')")
    total_marks_awarded: Union[float, int] = Field(..., description="Total marks awarded")
    total_max_possible: Union[float, int] = Field(..., description="Total possible marks")
    overall_reason: str = Field(..., description="Summary reason for total score")
    
    breakdown: List[GradeBreakdownItem] = Field(
        ..., description="Detailed per-criterion breakdown"
    )
    
    comments: List[str] = Field(
        default_factory=list,
        description="Structured list of LLM-generated feedback comments for the whole question"
    )

    # References & Metadata
    extracted_at: str = Field(..., description="ISO timestamp of grading")
    question_id: Optional[str] = Field(None, description="pac_questions _id")
    model_answer_id: Optional[str] = Field(None, description="model_answers _id")
    student_answer_id: Optional[str] = Field(None, description="student_assignments _id")