from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

MAP_PROMPT_TEMPLATE = """Student chunks: {chunks}\n
Questions: {questions}\n
Instructions:
- Map each student chunk to the question it most likely answers based on semantic meaning.
- Focus on the **intent** and **content** of the student chunk and question, not just exact wording.
- Each chunk must map to exactly one question number.
- If a chunk does not answer any question, assign it to '0'.
- Do NOT output explanations, schemas, or markdown. 
Return ONLY valid JSON in the following format:

{{
  "mappings": [
    {{ "chunk_id": 1, "mapped_question_number": "1.1" }},
    {{ "chunk_id": 2, "mapped_question_number": "2.3" }},
    {{ "chunk_id": 3, "mapped_question_number": "0" }}
  ]
}}

Now produce the mappings:
"""


GRADE_PROMPT_TEMPLATE ="""
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
  - Treat the full student answer as one unified document — credit evidence from any page or chunk without favoring early content or losing context on later pages.

### Scoring Precision
- Use **exactly** the 'marks' values from each marking_criteria item.
- Never invent or force increments — follow the scale defined in model_data (0.25, 0.5, 1, etc.).
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
1. Concatenate all student chunks into one complete, continuous answer text (treat as a single holistic document across all pages).
2. Read and fully analyze the entire student answer once — identify every demonstrated concept, calculation, table/working (W1/W2/etc.), journal entry, pro-rating/time apportionment, IFRS treatment, adjustment, and any other relevant technical content shown anywhere in the answer.
3. For each individual item in model_data.marking_criteria:
   - Check whether the student's overall demonstrated content substantively covers or conveys the meaning required by that specific criterion.
   - Accept equivalent wording, minor arithmetic/rounding/transcription differences, different presentation styles (table vs prose, different labels), and evidence from any part/page/chunk of the answer.
   - Award **exactly** the 'marks' value from that criterion if the core meaning/principle/logic is clearly present in substance.
   - If the criterion is explicitly separable (e.g. 0.25 per input), award the listed fraction only for parts clearly shown.
   - Do not require verbatim match or repetition — focus on whether the student has conveyed the required technical meaning anywhere in their work.
4. For each awarded criterion, quote 1–3 exact verbatim phrases from the student answer that justify the award (from any chunk/page).
5. After evaluating all criteria, calculate total score as the sum of awarded marks.
6. Generate comments and output JSON.

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
- "evidence": array of 1-3 **exact verbatim substrings** from student answer that directly justified the awarded marks. Use the exact content as it appears in the student answer (including numbers with commas, £/$, %, proper nouns), even if spellings are wrong. Do not add ... or any other special symbols; use only words and numbers that appear as is. Keep phrases short (4-10 words) and unique to identify the location for placement using fitz search (which looks for exact substrings in the PDF text layer). Note: This evidence will be used with fitz to search the PDF text layer for exact matches, so make it literal, unique, and findable.

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
      "question_number": "question number that we are grading",
      "score": number,
      "total_marks": number,
      "comments": ["quote → description. Advice.", "..."],
      "correct_words": ["phrase1", "..."],
      "breakdown": [
        {{"criterion": "exact description from criteria", "marks_awarded": 0.5, "max_possible": 0.5, "evidence": ["..."], "reason": "Fully correct"}}
      ]
    }}
  ]
}}

### Critical Output Rules
- Output should match json exactly with no extra text, explanations, or markdown.
- No trailing commas.
"""

grade_prompt = ChatPromptTemplate.from_template(GRADE_PROMPT_TEMPLATE)
map_to_questions_prompt = ChatPromptTemplate.from_template(MAP_PROMPT_TEMPLATE)
