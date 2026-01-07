from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from typing import Dict
from langchain_core.prompts import PromptTemplate
import json
import fitz
import re
import pandas as pd
import os
import datetime
from llm_setup import llm, llm_grader
from logging_config import logger
from student_assignment_extraction import extract_assignment_pipeline


# Pydantic Schemas
class SubPart(BaseModel):
    question_number: str = Field(description="The identifier of the subsection or scenario (e.g., '1.1' or 'a)')")
    answer: str = Field(description="content paragraphs from the student's answer for marking criteria")

class QuestionExtraction(BaseModel):
    question: str = Field(description="The main question number (e.g., '1' or '4')")
    sub_parts: List[SubPart] = Field(description="List of subsections with their content, only if subsections like 1.1, a), A) are present")

class MappingItem(BaseModel):
    chunk_id: int = Field(..., description="Identifier of the student answer chunk.")
    mapped_question_number: str = Field(..., description="The matched question number, e.g., '1.1', or '0' if unmapped.")

class MappingList(BaseModel):
    mappings: List[MappingItem]

class GradingItem(BaseModel):
    question_number: str = Field(..., description="The number of the question/sub-question, e.g., '1.1'.")
    score: str = Field(..., description="Marks obtained by the student, e.g., '3'.")
    total_marks: str = Field(..., description="Total marks for the question, e.g., '5', from maximum_marks, only include integer value nothing else like marks and other words.")
    comment: List[str] = Field(..., description="Feedback comment for the student, Should be concise but covering what went wrong and to the point, should not exceed three lines")
    correct_lines: List[str] = Field(..., description="Exact lines from the student's answer that are deemed correct, should be exact matching with same wording and everything")
    correct_words: List[str] = Field(..., description="Exact words from the student's answer explaining why the lines are correct.")

class GradingList(BaseModel):
    grades: List[GradingItem]


def load_json_data(questions_path, model_answers_path):
    """Load questions and model answers from JSON files."""
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        with open(model_answers_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        logger.info(f"Loaded questions from {questions_path} and model answers from {model_answers_path}")
        return questions, model_data
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        raise

def save_json(data, filename, folder="test_assignments_and_mappings"):
    """Save Python dict or list to a JSON file."""
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f" Saved JSON to {file_path}")

def extract_page_text(pdf_path: str, page_num: int) -> str:
    """
    Extracts text from a specific page of the PDF using PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
        if page_num < 0 or page_num >= len(doc):
            return ""
        page = doc.load_page(page_num)
        text = page.get_text("text")
        doc.close()
        # Clean the text to remove headers and extra formatting
        text = re.sub(r"^\d+ /\d+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"Word Processing area.*?- use the shortcut keys to copy from the spreadsheet\s*", "", text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from page {page_num}: {e}")
        return ""

def extract_answers(pdf_path: str, question_num: str, page_nums: List[int]) -> Dict:
    """
    Extracts and processes the answer for a given question by:
    1. Extracting text from specified pages of the student's PDF.
    2. Sending it to the LLM chain for structured answer extraction.
    
    Args:
        pdf_path (str): Path to the student's PDF file.
        question_num (str): Question number (e.g., '1', '4', etc.).
        page_nums (List[int]): List of page numbers containing the answer.

    Returns:
        Dict: Parsed model output containing question and sub_parts, 
              or an error dictionary if extraction fails.
    """
    try:
        # --- Step 1: Extract and combine text from relevant pages ---
        texts = []
        for p in page_nums:
            text = extract_page_text(pdf_path, p - 1)  # Assuming extract_page_text handles 0-indexing
            if text:
                texts.append(f"--- Page {p} ---\n{text.strip()}")
        
        answer_text = "\n\n".join(texts)

        # --- Step 2: Handle case where no content is found ---
        if not answer_text.strip():
            return {"error": f"No content found for question {question_num} on pages {page_nums}"}

        # --- Step 3: Run the LLM chain for structured extraction ---
        response = chain_answer.invoke({
            "answer_text": answer_text,
            "question_num": question_num
        })

        # --- Step 4: Return structured output ---
        student_answer = response.model_dump()
        return student_answer

    except Exception as e:
        # --- Handle unexpected errors gracefully ---
        return {"error": f"Failed to extract or parse answer for question {question_num}: {str(e)}"}


map_to_questions_parser = PydanticOutputParser(pydantic_object=MappingList)

# Prompt template for answer extraction
prompt_template = """
You are an expert in extracting and structuring student answers from exam PDFs for marking.

Focus on question {question_num} and its parts.

Given the following student answer text from a PDF page(s):

{answer_text}

Instructions:
- Identify the main question number based on the content (e.g., starts with 1.1 for question 1).
- Only create separate sub_parts if explicit subsections are present (e.g., 1.1, 1.2, a), b), A), B)).
- If subsections are present (e.g., 1.1, 1.2 or a), b)), extract each subsection's content with its id and split into paragraphs if present with proper new lines characters.
- If no subsections are present (e.g., no 1.1, 1.2, a), b), A), B)), treat the entire content as a single sub_part with id equal to the question number and include all content as given in paras or as it is.
- Focus only on the answer content, ignoring headers like 'Word Processing area'.
- Do not add or change information; extract and structure what's present.
- Alwasy remeber that only create subsections if student has specified the subsections else keep the content as a single question answer.
- Output strictly in the specified JSON format.

{format_instructions}
"""

# Parser for the output
parser = PydanticOutputParser(pydantic_object=QuestionExtraction)

# Create the prompt with format instructions
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["answer_text", "question_num"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the LLM chain for answer extraction
chain_answer = prompt | llm | parser


map_to_questions_prompt = ChatPromptTemplate.from_template(
    """Student chunks: {chunks}\n
Questions: {questions}\n

Instructions:
- Map each student chunk to the question it most likely answers based on semantic meaning.
- Focus on the **intent** and **content** of the student chunk and question, not just exact wording.
- Each chunk must map to exactly one question number.
- If a chunk does not answer any question, assign it to '0'.
- Do NOT output explanations, schemas, or markdown. 
- Return ONLY valid JSON in the following format:

{{
  "mappings": [
    {{ "chunk_id": 1, "mapped_question_number": "1.1" }},
    {{ "chunk_id": 2, "mapped_question_number": "2.3" }},
    {{ "chunk_id": 3, "mapped_question_number": "0" }}
  ]
}}

Now produce the mappings:
"""
)


map_chain = map_to_questions_prompt | llm

grade_parser = PydanticOutputParser(pydantic_object=GradingList)

grade_prompt = ChatPromptTemplate.from_template(
    """
    You are a strict, objective examiner. You award marks only when the student's answer explicitly and fully conveys the exact meaning required by the model answer and marking criteria. When in doubt, always award 0. Never be lenient.

### Input
- Questions: {questions}
- Model answers & marking criteria: {model_data}
- Mappings: {mappings}
- Student answers: {chunks}

### Core Grading Principles
  - Award marks when the student clearly conveys the required meaning, even if wording differs from the model answer.
  - Rephrased answers are fully acceptable if they express identical or equivalent technical meaning.
  - Credit should be given for correct application of concepts, accurate calculations, and logical structure.
  - Do not require exact reproduction of model answer phrasing.
  - Keywords alone are insufficient, but correct use within proper context can contribute.
  - Only withhold marks if meaning is incomplete, incorrect, or absent.
  - When significantly uncertain whether a point is fully met, default to 0 — but lean toward credit where understanding is evident.

### Scoring Precision
- All scores in increments of 0.5 only (0, 0.5, 1, 1.5, etc.).
- Never exceed the maximum_marks.
- If maximum marks are not specified for individual questions or sub-questions, use the total marks stated in the question text not in the model answer but in the question and in that case treat all the questions as one.
- At all times, ensure that the total awarded marks do not exceed the total marks defined for the question.

### Structured Criteria Handling (Preferred Format)
 - If marking_criteria is an array of objects (each with 'marks' and 'description'):
   → Evaluate each item independently.
   → Award the specified marks if the student's answer clearly satisfies the description.
   → For "1 each" or "max X" items: count valid instances up to the limit.
   → Sum awarded marks at the end.

### Step-by-Step Grading Process (Follow This Order)
 1. Review all marking criteria items (array) or parse pipe-separated string if present.
 2. For each criterion, search the full student answer for evidence.
 3. Quote relevant student phrases that match the required meaning.
 4. Decide if the point is fully met, partially met (only if explicitly allowed), or not met.
 5. Award marks accordingly.
 6. Only after evaluating all points, calculate total score.
 7. Then generate feedback and JSON.

### Question Structure Rules (CRITICAL)
- Examine the "questions" input first to determine if the question is single or has formal sub-questions.
- If "questions" contains only one question object with no "sub_questions" array (or sub_questions is null) AND the total_marks applies to the whole question → this is a SINGLE holistic question.
- In this case: Output EXACTLY ONE object in the "grades" array.
  → Use "question_number" from the top-level "question" field in student answers that is chunks (e.g., "Question 1" or "1").
  → Grade the entire student answer (all sub_parts concatenated if present) against the full model answer and criteria.
  → total_marks = the overall marks for the question (28 in this case).
- Only output multiple objects in "grades" if questions explicitly defines separate sub-questions with their own maximum marks.

- Never split the grades array based solely on descriptive headings in the student's answer.

### Evidence (only when marks > 0)
- correct_words: List of verbatim phrases taken directly from the student's answer that directly and fully justified the awarded marks.
  • Each phrase must be an exact substring from the student (no rephrasing).
  • Length: 3-12 words per phrase.
  • Include every critical phrase that contributed to the score.
  • These will be highlighted for the student to show exactly what earned marks.
  • Order roughly as they appear in the answer.

### Feedback (only when score < maximum_marks)
- comments: An array of strings.
  • Provide one separate comment for each major sub-topic, issue, or error area.
  • Every comment must strictly follow this exact generic format (no exceptions):
    "<5-10-word verbatim quote from student> → <error description>. <actionable advice>."
  • The quote at the beginning must be copied character-for-character from the student's answer (unique but should be searchable in the pdf) so the marker instantly knows where to place the comment in the PDF.
  • After "→" write exactly two short sentences:
      - First sentence: Precisely state what is missing, incomplete, or incorrect.
      - Second sentence: One clear, concise, actionable piece of advice.
  • Never use bullet points, numbering, or extra text inside the comment string.
  • Never reveal any part of the model answer.

### Special Cases
- No relevant content → score 0.0, correct_words empty, comments: ["No relevant content provided."]

### Output Format (ONLY this valid JSON, nothing else)
{{
  "grades": [
    {{
      "question_number": "Exact value of the 'question' field if single question, or the sub-question identifier from questions if multiple.",
      "score": Integer or float value as score that student got,
      "total_marks": maximum marks available for the question, extract it if maximum marks given in the model answer against each question or sub question, else use total marks for that question from the question itself, don't include any keywords like marks,
      "comments": [
        "Exact student answer words. Sub-topic/error 1: Description of error. What was missing/incorrect. Actionable advice.",
        "Exact student answer words. Sub-topic/error 2: Description of error. What was missing/incorrect. Actionable advice.",
        "..."
      ],
      "correct_words": [
        "exact phrase from student that earned marks 1",
        "exact phrase from student that earned marks 2",
        "..."
      ]
    }}
  ]
}}

### Final Safeguards
- Never output anything except the exact JSON.
- Never reveal or paraphrase model answer content.
- Prioritize strictness and accuracy above all.
    """
)

grade_chain = grade_prompt | llm_grader

def grade_student(student_pdf_path, student_name, questions_path, model_answers_path, question_number, student_pages):
    """Grade a student's PDF and save results to CSV."""
    try:
        # student_pdf_path = os.path.join(input_dir, f"{student_name}.pdf")
        if not os.path.exists(student_pdf_path):
            logger.error(f"Student PDF not found: {student_pdf_path}")
            return None

        # Ensure grades directory exists
        grades_dir = os.path.join("student_assignment", "grades")
        os.makedirs(grades_dir, exist_ok=True)

        # Generate output CSV path with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(grades_dir, f"{student_name}_grades_{timestamp}.csv")

        questions, model_data = load_json_data(questions_path, model_answers_path)
   
        # student_chunks = extract_answers(student_pdf_path, question_number, student_pages)
        student_chunks = extract_assignment_pipeline(student_pdf_path, student_pages)
        # student_chunks = r'questions_and_model_answers_json_and_scripts/assignment/assignment_extracted_2025-12-23_13-58-44.json'
        with open(student_chunks, 'r', encoding='utf-8') as f:
            student_chunks = json.load(f)
        # save_json(student_chunks, f'{student_name}_data.json')
        logger.info(f"Loaded student assignment data for question number: {question_number}")
        logger.info(f"Student's Assignment: {student_chunks}")

        if not student_chunks:
            logger.error(f"No answers could be extracted for {student_name}. Skipping grading.")
            return None

        # Map to questions
        map_output = map_chain.invoke({
            "chunks": student_chunks,
            "questions": json.dumps(questions)
        })
        
        print(map_output)
        parsed_output = json.loads(map_output.content)
        save_json(parsed_output, f'{student_name}_mappings.json')
        logger.info(f"Mapped question number to the student assignments: {parsed_output}")
        # Now you can access the "mappings" list
        mappings = parsed_output["mappings"]
        
        logger.info(f"Starting grading for {student_name} for question number {question_number}")
        grade_output = grade_chain.invoke({
            "mappings": mappings,
            "model_data": json.dumps(model_data),
            "chunks": student_chunks,
            "questions": json.dumps(questions)
        })
        logger.info(f"Grading done saving data into csv for: {student_name}")
        # print(grade_output)
        raw_content = grade_output.content
        cleaned_json_str = re.sub(r"^```json\n|```$", "", raw_content.strip())
        parsed_output = json.loads(cleaned_json_str)
    
        results = []


        all_questions = []
        for q in model_data:
            # for q in model_data:
            if not isinstance(q, dict):
                continue
            
            all_questions.append({
                "question_number": q["question_number"],
                "maximum_marks": q.get("maximum_marks", "0")
            })
        # Process graded results
        for g in parsed_output['grades']:

            question_number = g["question_number"]
            student_chunks_dict = {
                sp["question_number"]: sp for sp in student_chunks.get("sub_parts", [])
            }

            # Then you can safely do:
            chunk_text = student_chunks_dict.get(question_number)
            snippet = (
                chunk_text["answer"].split("\n")[0][:30]
                if chunk_text and chunk_text.get("answer")
                else "No answer provided"
            )
            results.append({
                "student_id": student_name,
                "question_number": g["question_number"],
                "score": g["score"],
                "total_marks": g["total_marks"],
                "comment": g["comments"],
                # "correct_lines": g["correct_lines"],
                "correct_words": g["correct_words"],
                "student_answer_snippet": snippet
            })

        # Ensure all questions are covered
        graded_questions = {r["question_number"] for r in results}
        for q in all_questions:
            q_num = q["question_number"]
            if q_num not in graded_questions:
                results.append({
                    "student_id": student_name,
                    "question_number": q_num,
                    "score": "0",
                    "total_marks": q["maximum_marks"],
                    "comment": "No answer provided",
                    # "correct_lines": [],
                    "correct_words": [],
                    "student_answer_snippet": "No answer provided"
                })

        # Export to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Grading complete! CSV saved to {output_csv}")
        return output_csv
    except Exception as e:
        logger.error(f"Error during grading for {student_name}: {e}")
        return None