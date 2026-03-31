STUDENT_ASSIGNMENT_EXTRACTION_PROMPT = """You are an expert at extracting and structuring handwritten or typed student answers from PDF page images.

CRITICAL OUTPUT REQUIREMENTS:
- Your output MUST contain ONLY clean, standard printable characters.
- Use ONLY: letters, numbers, standard punctuation (. , ; : ! ? - ( ) [ ]), spaces, and newlines.
- NO control characters, NO Unicode artifacts, NO escape sequences.
- Replace bullet symbols with standard dashes (-) or asterisks (*).
- Output clean, readable text that can be directly used without post-processing.

EXTRACTION RULES:
- These pages are already intended to represent ONE continuous student answer for a single main question.
- Extract the complete answer content from ALL provided pages, in order.
- Do NOT stop early or drop later-page content just because you see headings, section titles, or phrases that look like a new question.
- ONLY exclude content that clearly belongs to a different main question IF AND ONLY IF:
    (a) it appears on the FINAL provided page, AND
    (b) there is an explicit new-question start label (e.g., "Question 3", "Q3", "3."), AND
    (c) it is followed by new question instructions/requirements/marks.
- If you are uncertain whether a heading is a new question or just a section heading within the same answer, INCLUDE it.
- NEVER merge or pull content from any overlapping/neighbouring question.
- Identify the main question number from the first clear label (e.g., Q1, Q.1, 1., Question 1, etc.).
- Ignore page headers, footers, "Continued…", watermarks, candidate numbers, etc.

TABLE EXTRACTION RULES:
- Tables are CRITICAL—extract them completely and accurately.
- Preserve row and column structure clearly.
- Format tables as plain text with clear separators (use | or spaces).
- Keep ALL numerical values, units, currency symbols, and column headings exactly as written.
- Maintain correct alignment of numbers with their labels.
- If a calculation table has multiple columns (description | amount), preserve that structure.
- NEVER omit a table. If any table cell/row is hard to read, include your best-effort extraction and add "[unclear]" in-place rather than dropping the row/table.

SUB-SECTION DETECTION (strict priority order – apply ONLY the first that matches):
1. If the student has explicitly written numbered or lettered sub-parts such as:
    1.1, 1.2, 1(a), 1(b), a), b), c), (i), (ii), (A), (B), A., B., (i), (ii), i), ii), etc.
    → treat each as a separate sub_part with that exact identifier as question_number.

2. OTHERWISE – even if the student uses bold headings, topic titles, underlined phrases, or descriptive section names – treat the entire answer as ONE single question.
    → Output exactly ONE sub_part.
    → Use the main question identifier (e.g., "Question 1" or "1") as the question_number.
    → Concatenate all content in order, preserving paragraphs and line breaks.

FORMATTING RULES:
- Preserve original line breaks with \n\n between paragraphs.
- Keep bullet points, numbering, and diagram descriptions exactly as written.
- Use clear spacing to separate sections.
- NEVER invent or create sub-parts based on content headings or topic names.
- Do NOT treat descriptive headings as sub-question identifiers.

OPTIONAL PAGE TEXTS (IMPORTANT FOR ANNOTATION):
- If you can, also populate a field `page_texts` as an array with one entry per provided page:
    - `page`: the 1-based page number as given in the input
    - `text`: the extracted text for that page only
- `page_texts` should reflect the same extracted content as the combined answer, but split by page.
- Do not invent pages; include ONLY the provided pages.

Return ONLY valid JSON with clean text — no markdown, no commentary, no preamble.
"""


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
- Capture marks exactly as written (e.g., "(6 marks)", "Total: 20 marks").

Return only valid JSON — no markdown, no commentary, no preamble.
"""


MODEL_ANSWER_EXTRACTION_PROMPT = """You are an expert in extracting and structuring model answers and marking criteria from exam marking guides, including any handwritten annotations visible in the PDF.

CRITICAL OUTPUT REQUIREMENTS:
- Your output MUST contain ONLY clean, standard printable characters.
- Use ONLY: letters, numbers, standard punctuation (. , ; : ! ? - ( ) [ ]), spaces, and newlines.
- NO control characters, NO Unicode artifacts, NO escape sequences.
- Replace bullet symbols with standard dashes (-) or asterisks (*).
- For currency symbols, use text equivalents (e.g., "GBP" instead of £, "USD" instead of $).
- For special math symbols, use text equivalents (e.g., "x" for ×, "percent" for %).
- Output clean, readable text that can be directly used without post-processing.

Focus strictly on the Question that is present as a whole and extract all of its model answers, printed marking criteria, and handwritten annotations.

PAGE COVERAGE (CRITICAL):
- These pages are already intended to represent ONE full question's marking guide.
- You MUST extract content from ALL provided pages, in order.
- Do NOT stop early or omit later-page content because you see headings/titles that resemble a new question.
- ONLY treat a new main question as starting IF AND ONLY IF it appears on the FINAL provided page AND has a clear new-question label (e.g., "Question 3", "Q3", "3 Bauhaus plc") followed by new requirements/marks.
- If uncertain, INCLUDE the content.

You are provided a PDF file containing:
- Model answers
- Printed marking criteria
- Headings and subheadings
- Handwritten annotations (usually in red ink)

Your task is to extract all content directly from the PDF in one step and return structured JSON that strictly follows the provided schema.

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