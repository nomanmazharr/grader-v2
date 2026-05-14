STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE = """You are an expert at extracting and structuring handwritten or typed student answers from PDF page images.

CRITICAL OUTPUT REQUIREMENTS:
- Your output MUST contain ONLY clean, standard printable characters.
- Use ONLY: letters, numbers, standard punctuation (. , ; : ! ? - ( ) [ ]), spaces, and newlines.
- NO control characters, NO Unicode artifacts, NO escape sequences.
- Replace bullet symbols with standard dashes (-) or asterisks (*).
- Output clean, readable text that can be directly used without post-processing.

TARGET QUESTION: {question_number}
You MUST extract ONLY the student's answer for Question {question_number}.

EXTRACTION RULES:
- Key rule all the given pages belongs to the same question mainly focus on first and last page as they may contain different question, but check properly if question is different because mostly it's same question with different subpart.
- From the given pages only first and last may contain answers to MULTIPLE questions. You MUST identify and extract ONLY the content belonging to Question {question_number}.
- Look for question labels such as "Q-{question_number}", "Q.{question_number}", "Q{question_number}", "Question {question_number}", "{question_number}.", "{question_number}-", "{question_number})" or similar patterns to identify where Question {question_number} starts.
- STOP extracting ONLY when you see a label that contains the word "Question", "Q.", "Q-", or "Q " immediately followed by a number DIFFERENT from {question_number} (e.g. "Question 2", "Q.3", "Q-02"). That is the ONLY valid question boundary.
- SAME-PAGE BOUNDARY RULE (CRITICAL): A new question label (e.g. "Q2") may appear partway through a page. Any content that appears BEFORE that label on the SAME page still belongs to Question {question_number} and MUST be included. Extract ALL text on the page up to (but not including) the new question label, then stop. Never discard content just because the same page also contains a later question.
- SCENARIO NUMBERING IS NEVER A QUESTION BOUNDARY: Patterns such as "1-", "2-", "3-", "1.", "2.", "1)", "(1)", "Case 1", "Case 2", "Scenario 1", "Scenario 2" etc. are sub-parts of the current question. NEVER stop extraction or treat them as the start of a new top-level question.
- In auditing or case-study questions, students frequently label separate company scenarios as "1- Company Name" and "2- Another Company". These are ALWAYS sub-parts of the same question, never a new question.
- Content that appears BEFORE the start of Question {question_number} belongs to a different question — EXCLUDE it entirely.
- Content that appears AFTER Question {question_number} ends (i.e., after a new TOP-LEVEL question label) — EXCLUDE it entirely.
- Within Question {question_number}, extract ALL content including sub-sections, sub-headings, tables, and workings across ALL provided pages.
- MULTI-PAGE SUB-QUESTIONS: A sub-question (e.g. 4.1, 4.2) may start near the BOTTOM of one page and continue to the TOP of the NEXT page. In that case BOTH pages belong to that sub-question. Do NOT treat a page break as the end of a sub-question. Continue extracting the sub-question's content from the next page until you see a NEW sub-question label (e.g. 4.2) or a new top-level question label.
- If only the sub-question HEADING appears on one page and its full answer is on the following page, include ALL the answer content — the heading and the answer are one unit.
- If the pages only contain one question and it matches {question_number}, extract everything.
- If you cannot find Question {question_number} on the provided pages, return minimal output with empty answer content.
- NEVER merge or pull content from any other question.
- Ignore page headers, footers, "Continued...", watermarks, candidate numbers, etc.
- LAST-PAGE BOUNDARY CHECK: The last provided page may contain BOTH the end of Question {question_number} AND the start of a new question (e.g. "Question 2"). In that case extract ALL content on that page that belongs to Question {question_number} (everything before the new question label), then stop. Never discard content from the last page just because another question also starts on it.

TABLE EXTRACTION RULES:
- Tables are CRITICAL—extract them completely and accurately.
- Preserve row and column structure clearly.
- Format tables as plain text with clear separators (use | or spaces).
- Keep ALL numerical values, units, currency symbols, and column headings exactly as written.
- Maintain correct alignment of numbers with their labels.
- NEVER omit a table. If any table cell/row is hard to read, include your best-effort extraction and add "[unclear]" in-place rather than dropping the row/table.

SUB-SECTION DETECTION (strict priority order – apply ONLY the first that matches):

1. MULTIPLE SCENARIOS / CASES PATTERN (most common in auditing/accounting questions)
   If the answer contains clear numbered scenario blocks in any of these formats:
       • "1- ", "2- ", "3- ", ... (dash style)
       • "1. ", "2. ", "3. ", ... (dot style)
       • "1)", "2)", "(1)", "(2)", "Case 1", "Case 2", "Scenario 1", "Scenario 2"
   followed by a company name, topic, or descriptive title (e.g. "1- Saffron limited:-", "2- Tech limited:"),
   → treat EACH block as a separate sub_part.
   → The sub_question_number = the exact prefix the student wrote ("1-", "2-", "1.", "Case 1", etc.).
   → Extract the full content under each block (Materiality, Auditor action, Impact on report, etc.).

2. OTHER NUMBERED/LETTERED SUB-PARTS
   If the student has explicitly written sub-parts such as:
       1.1, 1.2, 1(a), 1(b), a), b), c), (i), (ii), (A), (B), A., B., i), ii), etc.
   → treat each as a separate sub_part with that exact identifier as question_number.

3. INTELLIGENT FALLBACK (use your reasoning)
   If the student used any other clear, consistent NUMBERED OR LETTERED labeling system (e.g. "Part A", "Part B", "Step 1", "Step 2", "W1", "W2") that logically divides the answer into separate parts, treat those as sub_parts.
   CRITICAL: Purely descriptive topic headings such as "Correct financial reporting treatment of X", "Discussion of Y", "Analysis of Z" etc. — with NO numeric or alphabetic label — do NOT qualify as sub-parts under this rule, no matter how consistently repeated. These are section headings within a single answer.
   Only use this fallback when the labeling is numbered/lettered AND obvious and consistent across the pages.

4. OTHERWISE (final fallback)
   If none of the above patterns are present, treat the entire answer as ONE single question.
   → Output exactly ONE sub_part.
   → Use the main question identifier (e.g. "Question 1" or "1") as the question_number.
   → The answer field MUST contain ALL content for this question from ALL provided pages, concatenated in order — including every topic section, every working, every journal, every table. Do NOT stop at the first section heading. Continue until you reach the end of the question (a new top-level question label) or the end of the provided pages.

FORMATTING RULES:
- Preserve original line breaks with \n\n between paragraphs.
- Keep bullet points, numbering, and diagram descriptions exactly as written.
- Use clear spacing to separate sections.
- NEVER invent or create sub-parts based on content headings or topic names.
- Do NOT treat descriptive headings as sub-question identifiers.

QUESTION HEADING TEXT (CRITICAL FOR ANNOTATION):
- Populate a field `question_heading_text` with the EXACT verbatim text the student wrote as the heading/label for this question.
- Copy it character-for-character, including any typos, spacing, or punctuation (e.g. "Quiestion 4", "Q4.", "QUESTION  4", "Q 4)").
- If the student wrote no visible heading, set `question_heading_text` to null.

OPTIONAL PAGE TEXTS (IMPORTANT FOR ANNOTATION):
- Populate a field `page_texts` as an array with one entry per provided page:
    - `page`: the 1-based page number as given in the input
    - `text`: ALL visible text on that page, verbatim — even if the page belongs to a different question, still copy its full raw text here.
- `page_texts` must have one entry for EVERY page provided.

Return ONLY valid JSON with clean text — no markdown, no commentary, no preamble.
"""


def get_student_extraction_prompt(question_number: str = None) -> str:
    """Return the student extraction prompt, optionally scoped to a specific question number."""
    if question_number:
        return STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE.format(
            question_number=question_number
        )
    return STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE.format(
        question_number="(unknown)"
    )


# Backward compatibility: default prompt without question filtering
STUDENT_ASSIGNMENT_EXTRACTION_PROMPT = STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE.format(
    question_number="(the main question on these pages)"
)
