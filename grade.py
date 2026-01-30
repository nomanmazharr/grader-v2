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
    You are a strict but fair objective examiner. Award marks when the student's answer clearly conveys the required meaning from the model answer and marking criteria, even if partial understanding is evident.

### Input
- Questions: {questions}
- Model answers & marking criteria: {model_data}
- Mappings: {mappings}
- Student answers: {chunks}

### Core Grading Principles
  - Award marks when the student conveys the required meaning, even if wording differs.
  - Rephrased answers are acceptable if they express identical or equivalent technical meaning.
  - Credit correct application of concepts, accurate calculations, logical structure.
  - Do not require exact model answer phrasing.
  - Keywords alone are insufficient — must be in correct context.
  - Withhold full marks if meaning is incomplete or incorrect, but award partial if substantial understanding is shown.
  - When uncertain, default to partial credit if evidence indicates some grasp.

### Scoring Precision
- Use **exactly** the 'marks' values from each marking_criteria item.
- Never invent or force increments - follow the scale defined in model_data (0.25, 0.5, 1, etc.).
- Never exceed maximum_marks or any per-item 'marks' value.
- Prefer explicit 'total_marks', 'maximum_marks' or 'total_marks_available' from model_data if present.
- If absent, use total marks from question JSON and treat as one holistic question.
- Total awarded marks must never exceed the defined total.

### Structured Criteria Handling
- marking_criteria is an array of objects (each with 'marks' and 'description').
  → Evaluate **each individual item separately** — never combine, merge, or group multiple criteria items into one evaluation or one breakdown entry under any circumstances.
  → Award **exactly** the numeric 'marks' value from that specific item when fully satisfied.
  → If 'marks' is non-numeric ("N/A", "1 each", "½ each", "max X"):
     - "1 each" → award 1 per valid instance (up to any stated max)
     - "½ each" → award 0.5 per valid instance
     - "max X" → count up to X, award per-instance value
  → Keep every awarded point as a separate breakdown entry — do NOT reduce granularity.
  → Sum all awarded marks precisely.

### Step-by-Step Grading Process
1. Review marking_criteria array in model_data.
2. For each criterion, search full student answer for evidence.
3. Quote exact student phrases that match.
4. Decide: fully met → full marks from that item; partially met → half or quarter of that item's marks if criterion has sub-parts or student shows partial understanding; otherwise 0 or full.
5. Award marks strictly per criterion, following its 'marks' value.
6. After all criteria, calculate total score (sum of awarded).
7. Generate feedback and JSON.

### Question Structure Rules (CRITICAL)
- Examine the "questions" input first to determine if the question is single or has formal sub-questions.
- If "questions" contains only one question object with no "sub_questions" array (or sub_questions is null) AND the total_marks applies to the whole question → this is a SINGLE holistic question.
  → Output EXACTLY ONE object in the "grades" array.
  → Use "question_number" from the top-level "question" field in student answers that is chunks (e.g., "Question 1" or "1" or "Required").
  → Grade the entire student answer (all sub_parts concatenated if present) against the full model answer and criteria.
  → total_marks = the overall marks for the question (from question JSON or model_data top-level).
- Only output multiple objects in "grades" if questions explicitly defines separate sub-questions with their own maximum marks.
- Never split the grades array based solely on descriptive headings in the student's answer or internal model_data "answers" array.
- Do not create separate grade objects for internal sub-parts like "(a)", "(b)", "(c)" even if present in model_data — treat everything under the single top-level question.

### Evidence (only when marks > 0)
- correct_words: verbatim phrases from student that justified awarded marks.
  • Exact substring (no rephrasing).
  • 3–15 words per phrase.
  • Include every critical phrase.
  • Avoid duplicating same phrase across items.
  • Order roughly as they appear.
  • Never include any main headings in the evidence only those phrases for which marks were awarded.

### Detailed Breakdown (only for awarded points)
Include "breakdown" array only when score > 0.

Each object corresponds to **one single marking_criteria item** that was awarded marks > 0.

Rules:
- For every marking_criteria item where you award marks > 0, create **exactly one** breakdown entry.
- It is forbidden to combine two or more marking_criteria items into one breakdown entry under any circumstances.
- Use the **exact 'description'** from that marking_criteria item (or very close paraphrase) as the "criterion" title.
- "max_possible" must be **exactly** the numeric 'marks' value from that item.
- "marks_awarded" must not exceed the 'max_possible' for that specific item.
- SUM of all marks_awarded across the breakdown MUST EQUAL the "score".
- If score = 0.0 → omit "breakdown" or use empty array [].
- "evidence": array of 1-3 **exact verbatim substrings** from student answer that directly justified the awarded marks. Use the exact content as it appears in the student answer (including numbers with commas, £/$, %, proper nouns), even if spellings are wrong. Do not add ... or any other special symbols; use only words and numbers that appear as is. If no words are present, use the number alone if unique. Keep phrases short (4-10 words) and unique to identify the location for placement using fitz search (which looks for exact substrings in the PDF text layer). Note: This evidence will be used with fitz to search the PDF text layer for exact matches, so make it literal, unique, and findable.

### Feedback Output Requirements
comments: An array of strings.

Each string must represent one feedback comment and must follow all rules below without exception.

Content rules

- Provide one separate comment for each major sub-topic, issue, or error area.
- In addition to major issues, include comments where the student’s answer is incorrect or incomplete and could receive more marks.
- Do not include praise-only comments.

Mandatory format (strict)
Each comment string must follow this exact format:

"<5–10 word verbatim quote from student> → <error description>. <actionable advice>."

Quote rules:
- The quote must be copied character-for-character from the student’s answer.
- The quote must be unique, precise, and searchable in the PDF (used with fitz).
- Do not paraphrase, summarize, or modify the quote in any way.
- The quoted text must appear exactly as written in the student’s PDF.

Text after the arrow (→)

Write exactly two short sentences:
- Sentence 1: Clearly state what is missing, incorrect, or incomplete.
- Sentence 2: Provide one clear, concise, actionable improvement.

Do not include examples, explanations, or model-answer content.

Strict prohibitions
- Do not use bullet points, numbering, line breaks, or extra text inside a comment string.
- Do not reveal or reference any part of the model answer.
- Do not add introductions, conclusions, or explanations outside the array.
- Do not deviate from the required format under any circumstances.

### Special Cases
- No relevant content → score 0.0, correct_words empty, comments: ["No relevant content provided."]

### Output Format (ONLY this valid JSON)
{{
  "grades": [
    {{
      "question_number": "...",
      "score": number,
      "total_marks": number,
      "comments": ["quote → description. Advice.", "..."],
      "correct_words": ["phrase1", "..."],
      "breakdown": [
        {{"criterion": "exact description from criteria", "marks_awarded": 0.5, "max_possible": 0.5, "evidence": ["..."], "reason": "Fully correct"}},
        ...
      ]
    }}
  ]
}}

### Critical Output Rules
- Output ONLY the JSON — no text, no ```json, no explanations.
- No trailing commas.
- All strings properly closed.
- Prioritize accuracy, granularity of marking_criteria, balanced feedback, and single holistic output for combined questions.
    """
)
grade_chain = grade_prompt | llm_grader

def grade_student(student_pdf_path, student_name, questions_path, model_answers_path, question_number, student_pages):
    """Grade a student's PDF and save results to CSV with detailed breakdown."""
    try:
        if not os.path.exists(student_pdf_path):
            logger.error(f"Student PDF not found: {student_pdf_path}")
            return None

        grades_dir = os.path.join("student_assignment", "grades")
        os.makedirs(grades_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(grades_dir, f"{student_name}_grades_{timestamp}.csv")

        questions, model_data = load_json_data(questions_path, model_answers_path)

        # Log model_data structure for debugging
        logger.info(f"model_data type: {type(model_data)}")
        if isinstance(model_data, dict):
            logger.info(f"model_data keys: {list(model_data.keys())}")
        elif isinstance(model_data, list):
            logger.info(f"model_data length: {len(model_data)}, first item type: {type(model_data[0]) if model_data else 'empty'}")

        student_chunks_path = extract_assignment_pipeline(student_pdf_path, student_pages)
        with open(student_chunks_path, 'r', encoding='utf-8') as f:
            student_chunks = json.load(f)

        logger.info(f"Loaded student assignment data for question number: {question_number}")

        if not student_chunks:
            logger.error(f"No answers could be extracted for {student_name}. Skipping grading.")
            return None

        # Mapping step
        map_output = map_chain.invoke({
            "chunks": student_chunks,
            "questions": json.dumps(questions)
        })
        parsed_mapping = json.loads(map_output.content)
        mappings = parsed_mapping["mappings"]
        save_json(parsed_mapping, f'{student_name}_mappings.json')
        logger.info(f"Mapped question number to the student assignments")

        # Grading step
        grade_output = grade_chain.invoke({
            "mappings": mappings,
            "model_data": json.dumps(model_data),
            "chunks": student_chunks,
            "questions": json.dumps(questions)
        })
        logger.info(f"Grading done for: {student_name}")

        # Robust JSON cleaning
        raw_content = grade_output.content.strip()
        # Remove markdown fences and extra whitespace
        cleaned = re.sub(r'^\s*(```(?:json)?\s*\n?)?', '', raw_content)
        cleaned = re.sub(r'\s*(```)?\s*$', '', cleaned).strip()

        # Find actual JSON start/end
        start_idx = min(
            cleaned.find('{') if '{' in cleaned else len(cleaned),
            cleaned.find('[') if '[' in cleaned else len(cleaned)
        )
        if start_idx < len(cleaned):
            cleaned = cleaned[start_idx:]

        end_idx = max(
            cleaned.rfind('}') if '}' in cleaned else -1,
            cleaned.rfind(']') if ']' in cleaned else -1
        )
        if end_idx >= 0:
            cleaned = cleaned[:end_idx + 1]

        try:
            parsed_output = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Cleaned JSON preview (first 500 chars): {cleaned[:500]}...")
            # Optional: save failed output for manual check
            with open(f"debug_{student_name}_failed.json", "w", encoding="utf-8") as f:
                f.write(cleaned)
            raise

        results = []

        for grade in parsed_output.get('grades', []):
            q_num = grade.get("question_number", "Unknown")
            total_score = grade.get("score", 0.0)
            total_marks = grade.get("total_marks")  # fallback to common value

            # Student answer snippet
            student_answer_snippet = "No answer provided"
            for part in student_chunks.get("sub_parts", []):
                if part.get("question_number") == q_num:
                    ans = part.get("answer", "").strip()
                    if ans:
                        student_answer_snippet = ans.split("\n")[0][:60] + "..." if len(ans.split("\n")[0]) > 60 else ans.split("\n")[0]
                    break

            # Add TOTAL SCORE row
            results.append({
                "student_id": student_name,
                "question_number": q_num,
                "criterion": "TOTAL SCORE",
                "marks_awarded": total_score,
                "max_possible": total_marks,
                "evidence": "",
                "reason": f"Overall awarded {total_score}/{total_marks}",
                "comments_summary": "; ".join(grade.get("comments", [])) if grade.get("comments") else "",
                "student_answer_snippet": student_answer_snippet
            })

            # Detailed breakdown rows
            breakdown = grade.get("breakdown", [])
            for item in breakdown:
                results.append({
                    "student_id": student_name,
                    "question_number": q_num,
                    "criterion": item.get("criterion", "Unknown criterion"),
                    "marks_awarded": item.get("marks_awarded", 0),
                    "max_possible": item.get("max_possible", 0),
                    "evidence": "; ".join(item.get("evidence", [])) if item.get("evidence") else "",
                    "reason": item.get("reason", ""),
                    "comments_summary": "",
                    "student_answer_snippet": ""
                })

        # Handle missing questions (safe access to model_data structure)
        graded_q_nums = {r["question_number"] for r in results if r["criterion"] == "TOTAL SCORE"}

        question_list = []
        if isinstance(model_data, dict):
            question_list = model_data.get("answers", []) or model_data.get("answers_list", []) or []
        elif isinstance(model_data, list):
            question_list = model_data

        for q in question_list:
            if not isinstance(q, dict):
                continue
            qn = q.get("question_number")
            if qn and qn not in graded_q_nums:
                max_m = q.get("maximum_marks") or q.get("total_marks_available") or q.get("total_marks") or 28
                results.append({
                    "student_id": student_name,
                    "question_number": qn,
                    "criterion": "TOTAL SCORE",
                    "marks_awarded": 0,
                    "max_possible": max_m,
                    "evidence": "",
                    "reason": "No answer or no grading output",
                    "comments_summary": "No relevant content provided.",
                    "student_answer_snippet": "No answer provided"
                })

        # Build DataFrame
        df = pd.DataFrame(results)
        column_order = [
            "student_id", "question_number", "criterion", "marks_awarded",
            "max_possible", "reason", "evidence", "comments_summary",
            "student_answer_snippet"
        ]
        df = df[[c for c in column_order if c in df.columns]]

        df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"Grading complete! Detailed CSV saved to {output_csv}")

        return output_csv

    except Exception as e:
        logger.error(f"Error during grading for {student_name}: {e}", exc_info=True)
        return None