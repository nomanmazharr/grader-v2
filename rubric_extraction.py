import io
import fitz
from llm_setup import client
import json
from logging_config import logger
from datetime import datetime
from pathlib import Path

def _create_pdf_subset(pdf_path: str, pages: list[int]) -> io.BytesIO:
    """
    Extract specified pages from a PDF and return as in-memory BytesIO PDF.
    Ensures the buffer behaves like a real PDF file.
    """
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()

    for p in pages:
        if 1 <= p <= len(doc):
            new_doc.insert_pdf(doc, from_page=p-1, to_page=p-1)

    # convert to proper PDF bytes
    pdf_bytes = new_doc.tobytes()
    buf = io.BytesIO(pdf_bytes)
    buf.name = "subset.pdf"   # this is critical for OpenAI to detect it's a PDF
    buf.seek(0)

    doc.close()
    new_doc.close()
    return buf

def _upload_to_openai(pdf_buffer: io.BytesIO, filename: str = "subset.pdf"):
    """
    Upload an in-memory PDF to OpenAI and return file object.
    """
    file_obj = client.files.create(
        file=pdf_buffer,
        purpose="user_data"
    )
    logger.info(f"Uploaded → file_obj: {file_obj}")
    return file_obj

model_answer_schema_2 = {
    "type": "object",
    "properties": {
        "question_title": {
            "type": "string",
            "description": "Main question title, e.g., 'Question 4'"
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
            "description": "Model answer if no subsections are present for this question",
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
                                            "description": {"type": "string"}
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

def _extract_rubric_with_vision(file_obj):
    response = client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_obj.id},
                    {"type": "input_text", "text": """
You are an expert in extracting and structuring model answers and marking criteria from exam marking guides, including any handwritten annotations visible in the PDF.

Focus strictly on the Question that is present as a whole and extract all of its model answers, printed marking criteria, and handwritten annotations.

You are provided a PDF file containing:
- Model answers
- Printed marking criteria
- Headings and subheadings
- Handwritten annotations (usually in red ink)

Your task is to extract all content directly from the PDF in one step and return structured JSON that strictly follows the provided schema.

IMPORTANT GLOBAL RULES:
- Preserve wording EXACTLY as written. No rewriting, rephrasing, summarizing, or adding interpretations.
- NEVER omit marking criteria for any subsection if marking criteria text or annotations exist anywhere in the PDF.
- ALWAYS extract marking criteria for the main question AND for each subsection separately if present.
- Handwritten annotations (red ink) must be merged directly into the `marking_criteria` array as separate objects.
- If marking criteria or annotations apply to multiple subsections or the entire main question, you MUST replicate the entire block across all relevant parts.
- If unsure where a criteria block or annotation belongs → assign it to the main question AND to all subsections to avoid any loss.

---

EXTRACTION RULES

1. Identify Subsections Reliably
Create a subsection whenever explicit labels appear:
- Numeric: 1.1, 1.2, 1.3 …
- Alphabetic: a), b), c)
- Roman or nested: (i), (ii), (iii), A., B.

If no subsections → single entry with main question number.

2. Hierarchy Enforcement
Preserve full nesting:
Main question
→ Subquestion (1.1, 1.2, a), b))
  → Nested sub ((i), (ii))
    → Deeper nested
Use `sub_answers` only when real nesting exists.

3. Answer vs Marking Criteria Separation
- `answer` contains ONLY the model answer content.
- `marking_criteria` contains ONLY marking rules (printed text + red handwritten annotations).
- DO NOT include maximum_marks or total_marks_available inside marking_criteria.

4. Marks Extraction
- Extract maximum_marks from phrases like "Maximum marks", "Maximum full marks", "Maximum", "[6]", etc.
- Extract total_marks_available from "Total Possible Marks", "Marks Available", "Total Marks", etc.
- Never duplicate maximum/total marks in child levels — only once at the correct parent level.
- NEVER leave maximum_marks or total_marks_available null if present — use the table or text values.
- If a subsection has nested sub-subsections, maximum marks and total available marks appear ONLY once for the parent subsection.
- Use high-level table marks (e.g., 26 for (a)) for total_marks_available and maximum_marks fields — NOT in marking_criteria array.

5. Marking Criteria & Annotations Handling — STRUCTURED VERSION (REQUIRED)

For every question and subsection:
- Identify ALL individual marking points from printed criteria and handwritten red annotations.
- Focus on detailed red marks next to model answer lines — ignore high-level table marks (use those for total_marks_available/maximum_marks).
- Each separate red mark (e.g., ½, 1, ¼, "1 each", "max 3", "tick") next to a line becomes ONE object.
- Pair each mark with the EXACT nearest sentence or phrase in the model answer text.

Create one object per point:
{
  "marks": <number if possible (0.5, 1, 2, 0.25) — or original string like "1 each", "max 4", "OF", "tick">,
  "description": "<exact original wording from the model answer line the mark is next to — preserve 100%>"
}

Rules:
- Scan every page for red marks and pair with closest text line.
- Never merge points — one object per mark.
- Include printed detailed criteria (e.g., bullet points) as separate items if not already covered by red marks.
- Include general notes (e.g., "Tutorial note: ...", "Own figure rule") as separate items.
- If criteria apply to multiple parts → duplicate array.
- If no detailed criteria → empty array []

Generic examples of correct items (from red marks next to text):
- {"marks": 0.5, "description": "correct definition of key term"}
- {"marks": 1, "description": "explained process with example"}
- {"marks": 0.25, "description": "mentioned relevant factor"}
- {"marks": 2, "description": "fully labelled diagram provided"}
- {"marks": "1 each", "description": "for any valid point (max 3)"}
- {"marks": 0.5, "description": "correct formula stated"}
- {"marks": 0.5, "description": "unit included in answer"}
- {"marks": "tick", "description": "correct final calculation"}
- {"marks": "OF", "description": "own figure rule applies"}

Example full array:
[
  {"marks": 1, "description": "correct identification of main concept"},
  {"marks": 0.5, "description": "relevant principle applied"},
  {"marks": "1 each", "description": "for each valid example (max 2)"},
  {"marks": 2, "description": "accurate calculation shown"}
]

---

Return only valid JSON — no markdown, no commentary, no preamble.
"""
}
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "universal_exam_rubric",
                "strict": True,
                "schema": model_answer_schema_2  # this is your updated flexible schema
            }
        }
    )

    try:
        raw_json = response.output_text.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3].strip()
        data = json.loads(raw_json)
        logger.info("Rubric successfully parsed from vision model")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from model response: {e}")
        logger.debug(f"Raw output: {raw_json[:500]}...")
        raise

def extract_pdf_annotations_pipeline(pdf_path: str, pages: list[int], output_dir= "questions_and_model_answers_json_and_scripts"):
    """
    Full end-to-end pipeline:
      1. Extract subset of pages (in memory)
      2. Upload to OpenAI
      3. Get annotations JSON
    """
    try:
        logger.info(f"Starting rubric extraction from pages {pages}")

        # 1. Create subset
        pdf_buffer = _create_pdf_subset(pdf_path, pages)

        # 2. Upload
        file_id = _upload_to_openai(pdf_buffer)

        # 3. Extract with vision LLM
        rubric_data = _extract_rubric_with_vision(file_id)

        # 4. Save to disk
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rubric_dir = Path(output_dir) / "rubric"
        rubric_dir.mkdir(parents=True, exist_ok=True)

        output_path = rubric_dir / f"rubric_extracted_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rubric_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Rubric saved → {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Rubric extraction pipeline failed: {e}")
        logger.debug(f"Traceback: {e}", exc_info=True)
        return None
