# STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE = """You are an expert at extracting and structuring handwritten or typed student answers from PDF page images.

# CRITICAL OUTPUT REQUIREMENTS:
# - Your output MUST contain ONLY clean, standard printable characters.
# - Use ONLY: letters, numbers, standard punctuation (. , ; : ! ? - ( ) [ ]), spaces, and newlines.
# - NO control characters, NO Unicode artifacts, NO escape sequences.
# - Replace bullet symbols with standard dashes (-) or asterisks (*).
# - Output clean, readable text that can be directly used without post-processing.

# TARGET QUESTION: {question_number}
# You MUST extract ONLY the student's answer for Question {question_number}.

# EXTRACTION RULES:
# - The provided pages may contain answers to MULTIPLE questions. You MUST identify and extract ONLY the content belonging to Question {question_number}.
# - Look for question labels such as "Q-{question_number}", "Q.{question_number}", "Q{question_number}", "Question {question_number}", "{question_number}.", "{question_number}-", "{question_number})" or similar patterns to identify where Question {question_number} starts.
# - STOP extracting ONLY when you see a label that contains the word "Question", "Q.", "Q-", or "Q " immediately followed by a number DIFFERENT from {question_number} (e.g. "Question 2", "Q.3", "Q-02"). That is the ONLY valid question boundary.
# - A bare number with any suffix — "2-", "2.", "2)", "(2)", "Case 2", "Scenario 2" — is NEVER a question boundary. It is mostly a sub-part of the current question. For better see top level student way of using question patterns as that will help to identify it's sub part or next question. Keep extracting and add it as a new entry in sub_parts.
# - Content that appears BEFORE the start of Question {question_number} belongs to a different question — EXCLUDE it entirely.
# - Content that appears AFTER Question {question_number} ends (i.e., after a new TOP-LEVEL question label) — EXCLUDE it entirely.
# - Within Question {question_number}, extract ALL content including sub-sections, sub-headings, tables, and workings across ALL provided pages.
# - If the pages only contain one question and it matches {question_number}, extract everything.
# - If you cannot find Question {question_number} on the provided pages, return minimal output with empty answer content.
# - NEVER merge or pull content from any other question.
# - Ignore page headers, footers, "Continued...", watermarks, candidate numbers, etc.

# TABLE EXTRACTION RULES:
# - Tables are CRITICAL—extract them completely and accurately.
# - Preserve row and column structure clearly.
# - Format tables as plain text with clear separators (use | or spaces).
# - Keep ALL numerical values, units, currency symbols, and column headings exactly as written.
# - Maintain correct alignment of numbers with their labels.
# - If a calculation table has multiple columns (description | amount), preserve that structure.
# - NEVER omit a table. If any table cell/row is hard to read, include your best-effort extraction and add "[unclear]" in-place rather than dropping the row/table.

# SUB-SECTION DETECTION (strict priority order – apply ONLY the first that matches):
# 1. If the student has explicitly written numbered or lettered sub-parts such as:
#     1.1, 1.2, 1(a), 1(b), a), b), c), (i), (ii), (A), (B), A., B., (i), (ii), i), ii), etc.
#     → treat each as a separate sub_part with that exact identifier as question_number.

# 2. OTHERWISE – even if the student uses bold headings, topic titles, underlined phrases, or descriptive section names – treat the entire answer as ONE single question.
#     → Output exactly ONE sub_part.
#     → Use the main question identifier (e.g., "Question 1" or "1") as the question_number.
#     → Concatenate all content in order, preserving paragraphs and line breaks.

# FORMATTING RULES:
# - Preserve original line breaks with \n\n between paragraphs.
# - Keep bullet points, numbering, and diagram descriptions exactly as written.
# - Use clear spacing to separate sections.
# - NEVER invent or create sub-parts based on content headings or topic names.
# - Do NOT treat descriptive headings as sub-question identifiers.

# QUESTION HEADING TEXT (CRITICAL FOR ANNOTATION):
# - Populate a field `question_heading_text` with the EXACT verbatim text the student wrote as the heading/label for this question.
# - Copy it character-for-character, including any typos, spacing, or punctuation (e.g. "Quiestion 4", "Q4.", "QUESTION  4", "Q 4)").
# - If the student wrote no visible heading, set `question_heading_text` to null.
# - This is used by the annotation engine to locate the heading in the PDF — accuracy is critical.

# OPTIONAL PAGE TEXTS (IMPORTANT FOR ANNOTATION):
# - Populate a field `page_texts` as an array with one entry per provided page:
#     - `page`: the 1-based page number as given in the input
#     - `text`: ALL visible text on that page, verbatim — even if the page belongs to a different question, still copy its full raw text here (it is used for PDF annotation positioning)
# - `page_texts` must have one entry for EVERY page provided — never omit a page or leave `text` as an empty string unless the page is genuinely blank.
# - Do not invent pages; include ONLY the provided pages.

# Return ONLY valid JSON with clean text — no markdown, no commentary, no preamble.
# """

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
- The provided pages may contain answers to MULTIPLE questions. You MUST identify and extract ONLY the content belonging to Question {question_number}.
- Look for question labels such as "Q-{question_number}", "Q.{question_number}", "Q{question_number}", "Question {question_number}", "{question_number}.", "{question_number}-", "{question_number})" or similar patterns to identify where Question {question_number} starts.
- STOP extracting ONLY when you see a label that contains the word "Question", "Q.", "Q-", or "Q " immediately followed by a number DIFFERENT from {question_number} (e.g. "Question 2", "Q.3", "Q-02"). That is the ONLY valid question boundary.
- SCENARIO NUMBERING IS NEVER A QUESTION BOUNDARY: Patterns such as "1-", "2-", "3-", "1.", "2.", "1)", "(1)", "Case 1", "Case 2", "Scenario 1", "Scenario 2" etc. are sub-parts of the current question. NEVER stop extraction or treat them as the start of a new top-level question.
- In auditing or case-study questions, students frequently label separate company scenarios as "1- Company Name" and "2- Another Company". These are ALWAYS sub-parts of the same question, never a new question.
- Content that appears BEFORE the start of Question {question_number} belongs to a different question — EXCLUDE it entirely.
- Content that appears AFTER Question {question_number} ends (i.e., after a new TOP-LEVEL question label) — EXCLUDE it entirely.
- Within Question {question_number}, extract ALL content including sub-sections, sub-headings, tables, and workings across ALL provided pages.
- If the pages only contain one question and it matches {question_number}, extract everything.
- If you cannot find Question {question_number} on the provided pages, return minimal output with empty answer content.
- NEVER merge or pull content from any other question.
- Ignore page headers, footers, "Continued...", watermarks, candidate numbers, etc.

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
   If the student used any other clear, consistent numbering or labeling system that logically divides the answer into separate scenarios or parts (even if it doesn't match the exact patterns above), treat those as sub_parts.
   Only use this fallback when the structure is obvious and consistent across the pages.

4. OTHERWISE (final fallback)
   If none of the above patterns are present, treat the entire answer as ONE single question.
   → Output exactly ONE sub_part.
   → Use the main question identifier (e.g. "Question 1" or "1") as the question_number.
   → Concatenate all content in order, preserving paragraphs and line breaks.

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
    # Fallback: generic prompt (no question filtering)
    return STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE.format(
        question_number="(unknown)"
    )


# Backward compatibility: default prompt without question filtering
STUDENT_ASSIGNMENT_EXTRACTION_PROMPT = STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE.format(
    question_number="(the main question on these pages)"
)


QUESTION_EXTRACTION_PROMPT = """You are an expert at extracting exam questions from PDF question papers.

CRITICAL OUTPUT REQUIREMENTS:
- Your output MUST contain ONLY clean, standard printable characters.
- Use ONLY: letters, numbers, standard punctuation (. , ; : ! ? - ( ) [ ]), spaces, and newlines.
- NO control characters, NO Unicode artifacts, NO escape sequences.
- Replace bullet symbols with standard dashes (-) or asterisks (*).
- Output clean, readable text that can be directly used without post-processing.

Focus strictly on Question that is present as a whole and extract all of its question content.

Rules:
- Preserve exact original wording, line breaks, bullet points, and formatting.
- Include any introductory scenario/description in the description field.
- If there are no subquestions, create one SubQuestion with the main question number.
- Capture marks exactly as written in the "marks" field (e.g., "(6 marks)", "Total: 20 marks").

TOTAL MARKS (critical):
- For each question item, always populate "total_marks" as a plain number (e.g. 18.0).
- If the question states a total explicitly (e.g. "Total: 18 marks"), use that.
- If only sub-parts carry individual marks (e.g. (a)=8, (b)=10), sum them → 18.
- Recurse into nested sub_questions if needed to find all individual marks.
- Never leave "total_marks" as null when any marks are present in the question.

Return only valid JSON — no markdown, no commentary, no preamble.
"""


MODEL_ANSWER_EXTRACTION_PROMPT_TEMPLATE = """You are an expert in extracting and structuring model answers and marking criteria from exam marking guides, including any handwritten annotations visible in the PDF.

CRITICAL OUTPUT REQUIREMENTS:
- Your output MUST contain ONLY clean, standard printable characters.
- Use ONLY: letters, numbers, standard punctuation (. , ; : ! ? - ( ) [ ]), spaces, and newlines.
- NO control characters, NO Unicode artifacts, NO escape sequences.
- Replace bullet symbols with standard dashes (-) or asterisks (*).
- For currency symbols, use text equivalents (e.g., "GBP" instead of £, "USD" instead of $).
- For special math symbols, use text equivalents (e.g., "x" for ×, "percent" for %).
- Output clean, readable text that can be directly used without post-processing.

TARGET QUESTION: {question_number}
You MUST extract ONLY the model answer and marking criteria for Question {question_number}.

═══════════════════════════════════════════════════
HOW TO IDENTIFY WHERE QUESTION {question_number} STARTS
═══════════════════════════════════════════════════
Question headers in marking guides appear in many formats. All of the following mean the SAME thing — the start of Question {question_number}:
  Ans.{question_number}   Ans {question_number}   Answer {question_number}   Answer:{question_number}
  Q.{question_number}     Q {question_number}      Q{question_number}         Q-{question_number}
  Q-0{question_number}    {question_number}.        ({question_number})        [{question_number}]
Look for ANY of these patterns (case-insensitive, with or without punctuation/spaces) followed by content.

IMPORTANT — the question header line may also contain the FIRST sub-section name on the same line
(e.g. "Ans.5 Military research project (5 Marks)"). In that case:
- question_title = "Ans.5" (the question label only)
- "Military research project (5 Marks)" is the FIRST sub-section — it goes into the answers array as the first entry
- Any further sub-sections on subsequent lines or pages (e.g. "Fire (5 Marks)") go as additional entries
Do NOT put sub-section names into question_title.

═══════════════════════════════════════════════════
CRITICAL — WHEN TO STOP EXTRACTING
═══════════════════════════════════════════════════
A question boundary is determined by ONE thing only: a DIFFERENT question number appearing.

STOP only when you see a heading that:
  (1) matches one of the question header formats listed above, AND
  (2) carries a question number that is NOT {question_number}.

A heading that does NOT contain a question number is NEVER a question boundary — it is always a sub-section or topic heading within the current question. Do not stop for it.

═══════════════════════════════════════════════════
SUB-SECTION DETECTION (CRITICAL for theoretical/essay answers)
═══════════════════════════════════════════════════
- If the model answer for Question {question_number} contains named sub-sections with their own marks allocation (e.g., "Part (a)", "Part (b)", "Issue 1 (4 Marks)", a topic heading followed by a marks value in parentheses), you MUST extract each sub-section as a SEPARATE answer entry in the "answers" array.
- This applies even when sub-sections span across multiple pages — extract ALL of them.
- Each sub-section answer entry should have:
    - question_number: the sub-section identifier (e.g., "SL", "Fire", "Overdraft", "1(a)", "Issue 1")
    - answer: the full model answer text for that sub-section
    - marking_criteria: any formal marking criteria for that sub-section (may be null/empty for theoretical answers)
    - total_marks_available: the marks allocated to that sub-section (e.g., "3", "5 Marks", "(4)")
- Do NOT merge all sub-sections into one giant answer entry. Keep them separate.
- If there are NO distinct sub-sections (just one continuous answer), use a single answer entry.

═══════════════════════════════════════════════════
PAGE COVERAGE (CRITICAL — multi-page answers)
═══════════════════════════════════════════════════
- ALL provided pages belong to Question {question_number} unless a DIFFERENT question header appears.
- Process every page you are given. Do not stop at the first page.
- A sub-section heading at the top of a new page (e.g., "Fire (5 Marks)" on page 2) is the CONTINUATION of Question {question_number}, not a new question.
- Scan every page fully from top to bottom before deciding what to include.

You are provided a PDF file containing:
- Model answers
- Printed marking criteria
- Headings and subheadings
- Handwritten annotations (usually in red ink)

Your task is to extract all content for Question {question_number} directly from the PDF and return structured JSON that strictly follows the provided schema.

HIERARCHICAL MARKING CRITERIA (IMPORTANT):
- If the marking guide includes a broad criterion worth multiple marks (e.g., 4 marks) that is then broken down into smaller bullet points (e.g., 0.5 each, 1 each, etc.), represent it as ONE marking_criteria object with:
    - marks = the broad total for that criterion (e.g., 4)
    - description = the broad criterion text (e.g., "Assimilate and apply technical knowledge")
    - sub_criteria = a list of smaller MarkingPoint objects for each individual markable point under that broad criterion.
- When sub_criteria is present and non-empty, the parent criterion should NOT be duplicated as separate micro-criteria elsewhere.
- For any criterion line like "1/2 – Correctly calculates goodwill" or "1/2 mk each max 4":
    - Put the marking value into the marks field (e.g., 0.5 or "1/2 mk each max 4")
    - Put ONLY the descriptive assessment text into description (e.g., "Correctly calculates goodwill").
    - NEVER set description to a pure marking notation like "1/2" or "1/4".

TABLE/MARKING GRID EXTRACTION (CRITICAL):
- Marking criteria often appear in a grid/table where marks are in one column and descriptions are in another.
- You MUST read across the full row: extract BOTH the marks value AND the associated descriptive text.
- If you see a marks value but cannot find any descriptive text on the same row, DO NOT output that criterion.
- It is forbidden for any marking_criteria item to have description equal to the marks value.
- Every description must be meaningful and tied to a specific markable point. If the printed criterion/label is genuinely short (e.g., "Goodwill"), keep it verbatim.

HANDWRITTEN ANNOTATIONS (CRITICAL):
- Handwritten annotations (red ink) usually indicate awarded marks (e.g., "1/4", "1/2", "2") or short notes (e.g., "Add - Land", "HR").
- NEVER output generic placeholders like "Handwritten annotation: 1/4" with no associated criterion.
- If a handwritten MARK (e.g., "1/4") is written next to a specific printed line/sentence/table row in the model answer or marking guide:
    - Create a marking_criteria item where:
        - marks = the handwritten mark value
        - description = the EXACT nearby printed text that the mark applies to (verbatim)
    - This applies even when the page is NOT a formal marking grid (e.g., margin marks beside the worked solution).
- If a handwritten NOTE (not a mark) is present (e.g., "Add - Land", "HR"):
    - Create a marking_criteria item with marks = null and description = the exact handwritten note text (verbatim).
- Do NOT inject handwritten notes into the model answer text. Keep handwritten content in marking_criteria only.

IMPORTANT GLOBAL RULES:
- Preserve wording EXACTLY as written. No rewriting, rephrasing, summarizing, or adding interpretations.
- NEVER omit marking criteria for any subsection if marking criteria text or annotations exist anywhere in the PDF.
- ALWAYS extract marking criteria for the main question AND for each subsection separately if present.
- Handwritten annotations (red ink) must be merged directly into the marking_criteria array as separate objects.
- If marking criteria or annotations apply to multiple subsections or the entire main question, replicate the block across all relevant parts.

Return only valid JSON — no markdown, no commentary, no preamble.
"""


def get_model_answer_extraction_prompt(question_number: str = None) -> str:
    """Return the model answer extraction prompt, optionally scoped to a specific question number."""
    if question_number:
        return MODEL_ANSWER_EXTRACTION_PROMPT_TEMPLATE.format(
            question_number=question_number
        )
    # Fallback: generic prompt (no question filtering)
    return MODEL_ANSWER_EXTRACTION_PROMPT_TEMPLATE.format(
        question_number="(the main question on these pages)"
    )


# Backward compatibility
MODEL_ANSWER_EXTRACTION_PROMPT = MODEL_ANSWER_EXTRACTION_PROMPT_TEMPLATE.format(
    question_number="(the main question on these pages)"
)