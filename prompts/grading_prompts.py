from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

MAP_PROMPT_TEMPLATE = """Student chunks: {chunks}\n
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


GRADE_PROMPT_TEMPLATE = """
    You are a STRICT, CONSISTENT examiner. **DETERMINISTIC**: Identical inputs MUST produce identical outputs/marks. Award marks ONLY for CLEAR, SUBSTANTIAL evidence. Default to 0 unless unambiguously met. Total ≤80% max_marks unless full major criteria coverage.

### Input
- Questions: {questions}
- Model answers & marking criteria: {model_data}
- Mappings: {mappings}
- Student answers: {chunks}

### STRICT GRADING RULES (APPLY FIRST)
1. Use ONLY the exact 'marks' values from marking_criteria. No intermediates.
2. Award ONLY if student shows CLEAR calculation logic OR IFRS principle application.
3. NO marks for: mentioning terms without explanation, isolated numbers, rephrasing without substance.
4. Total awarded ≤ maximum_marks. If sum >80% without full major criteria coverage, reduce proportionally.
5. For zero-attempt answers: score=0.0, breakdown=[], comments explaining absence.

### SEARCH RULES (CONCISE)
Search full unstructured student text for criterion evidence. Award ONLY for:
- Exact matches OR clear semantic equivalents WITH supporting logic/calculation
- NO marks for terms mentioned without explanation or isolated fragments
- Examples of rejection: "Goodwill £11m" (no derivation), "consolidated" (no method shown)
- Cap totals: If >80% max_marks, verify comprehensive major criteria coverage first

#### Financial Calculation Evidence Patterns
Student may show working in condensed, abbreviated, reordered, or conceptually-equivalent forms:
- Different presentation order showing same calculation steps still demonstrates understanding
- Pro-rated amounts shown via formula (9/12) OR shown via result (75% or resulting value) both indicate understanding of the logic
- Reference to components of a calculation (even if not all steps shown) indicates grasping the method
- Award marks if the formula structure and conceptual approach are sound, even with input variations

#### Treatment & Principle Evidence Patterns
IFRS treatments and principles may be explained without complete precise figures or formal journal entries:
- Student may describe a treatment concept without exact numbers and still show understanding (e.g., explaining equity method application without working the full financial impact)
- Student may reference timing/rate/percentage logic without showing final calculated amounts (e.g., "using the closing rate for assets")
- Student may use shorthand or compressed explanation that captures the essential principle
- Award full marks if student's explanation reveals correct understanding of the IFRS requirement, even if abbreviated or differently structured

#### What Does NOT Count as Semantic Equivalence
These contradict, not align, with intended meaning:
- Opposite/contradictory meaning: opposite treatment, conflicting principles (e.g., asset vs. liability treatment of same item)
- Fundamentally different IFRS method: wrong consolidation approach, wrong rate selection, wrong expense/capitalize choice
- No evidence of reasoning: merely referencing a number without indication of where it came from or why
- Conceptually incomplete: mentions a term but shows no grasp of the underlying principle

#### Search Strategy for Unstructured Text
Because content may appear anywhere in the answer:
- Do NOT assume sections follow marking criteria order
- Search for content relating to each criterion across the full answer text
- Related criteria (e.g., components of one calculation, parts of one principle) may appear together in one paragraph
- Multi-step concepts may be explained in condensed form and scattered; piece together evidence from multiple locations
- Always evaluate each criterion independently for its specific requirement, even if logically connected to others

### Scoring Precision (IMPORTANT – READ CAREFULLY)
- Use **exactly and only** the 'marks' values that appear in model_data marking_criteria (0.25, 0.5, 1, etc.). **Never award intermediate values** such as 0.3, 0.4, 0.75 etc. under any circumstances.
- For marking criteria that represent **individual components or steps** within a larger calculation, working, disclosure, or accounting treatment (common for 0.25 and many 0.5 items — e.g. a single figure, a pro-rated amount, a translated value, a depreciation charge, an NCI share, a fair value, a journal line, or a rate application), award the **full marks value** of that criterion if the student demonstrates:
  • the correct formula, structure, logic, pro-rating, addition, multiplication, or rate application,
  • the correct IFRS principle, accounting treatment, or presentation requirement,
  even when there is a minor input error, one incorrect component value, rounding difference, or incorrect final aggregate/total — as long as the step itself is conceptually and methodologically sound.
- Do not withhold marks for an individual step merely because a downstream figure, overall total, or final outcome (e.g. profit on disposal, total exchange gain, revised profit, EPS) is incorrect due to accumulation of errors or other mistakes.
- Award marks **step-by-step** and **independently** for each criterion where the specific requirement is satisfied, rather than requiring the entire chain or final result to be perfect.
- For criteria describing an **accounting treatment, principle, presentation, or implication** (e.g. equity accounting for associates, translation at average/closing rates, allocation to NCI or OCI, elimination of revaluation surplus before P&L charge, equity-settled share-based payment expense recognition, implications for diluted EPS), award full marks if the student's wording or explanation conveys **equivalent technical meaning** or the same IFRS requirement, even if phrased differently, reworded, or without exact journal entries/numbers.
- Only withhold marks (give 0) if the step/treatment is conceptually wrong, completely missing, or the error fundamentally changes the IFRS treatment.
- For composite / multi-part criteria (rare), partial may be half or quarter of the item's marks — but only if the description clearly has separable sub-parts.
- Never exceed maximum_marks or any per-item 'marks' value.
- Total awarded marks must never exceed the defined total.

### Structured Criteria Handling
- marking_criteria is an array of objects (each with 'marks' and 'description').
  → Evaluate **each individual item separately** — never combine, merge, or group multiple criteria items into one evaluation or one breakdown entry under any circumstances.
  → Award **exactly** the numeric 'marks' value from that specific item when the criterion is fully or substantially satisfied (see Scoring Precision above).
  → If 'marks' is non-numeric ("N/A", "1 each", "½ each", "max X"):
     - "1 each" → award 1 per valid instance (up to any stated max)
     - "½ each" → award 0.5 per valid instance
     - "max X" → count up to X, award per-instance value
  → Keep every awarded point as a separate breakdown entry — do NOT reduce granularity.
  → Sum all awarded marks precisely.

### Step-by-Step Grading Process
1. Review marking_criteria array in model_data carefully.

2. For each criterion in marking_criteria:
   a. Identify the CORE CONCEPT (calculation type, principle, treatment, or requirement the criterion describes)
   b. Search the full student answer for evidence of this concept using SEMANTIC MATCHING (per rules above)
      - Look for equivalent terminology, synonyms, different phrasings of the same idea
      - Look for calculations shown via formula, intermediate results, or abbreviated forms
      - Look for explanations that convey the principle even if different words are used
      - Do NOT rely on exact keyword or phrase matching alone
   c. Collect exact student phrases/passages that demonstrate the concept
   
3. For each criterion, decide:
   - **Fully met**: Concept clearly shown, correct logic/formula/treatment evident → award full 'marks' value
   - **Substantially equivalent**: Student shows understanding via different approach/wording but conceptually correct → award full 'marks' value
   - **Partial/incomplete**: Only minor elements present or method unclear → 0 marks (unless criterion explicitly multi-part, then consider half/quarter)
   - **Not met**: Conceptually wrong, completely missing, or fundamentally different treatment → 0 marks

4. Sum awarded marks precisely.

5. Validate total does not exceed maximum_marks.

### Check-and-Balance Rules (PREVENT OVERAWARDING)
These rules act as validation gates. Even if semantic matching suggests a criterion is met, WITHHOLD marks if ANY of these apply:

#### Gate 1: No Supporting Evidence Trap
**WITHHOLD marks** if:
- Student uses the TERM/CONCEPT but provides NO evidence of understanding the mechanism or logic
  • Example REJECT: Student writes "consolidated at acquisition" but shows no consolidation working, no goodwill calculation, no indication they know what consolidation entails
  • What TO ACCEPT: Student writes "consolidated" AND shows related calculation or explanation of timing/method
- Student references a number WITHOUT showing or implying where it comes from
  • Example REJECT: "Goodwill is £11.725 million" (no calculation, no derivation shown)
  • What TO ACCEPT: "Goodwill calculated as cost minus net assets" OR showing the calculation OR explaining the logic

####STRICT VALIDATION GATES (MANDATORY CHECKS)
BEFORE awarding ANY marks, verify NONE of these apply (reject if ANY do):
- Term/concept mentioned but NO logic/calculation shown
- Number stated without derivation or context
- Rephrasing criterion without independent application
- Single step shown for multi-step criterion
- Correct concept but opposite/wrong IFRS treatment applied
**DEFAULT: 0 marks unless ALL gates passed**
  • Expensing when should capitalize, or vice versa
  • Using wrong rate (cost rate instead of closing rate, vice versa)
  • Allocating to wrong party (Bauhaus instead of NCI, or vice versa)
- These are CORE IFRS requirements — opposite treatment is fundamentally wrong, not "substantially equivalent"

#### Gate 5: Zero Content in Whole Answer Trap
**ALWAYS AWARD SCORE 0** if:
- The entire student answer is blank, contains only unrelated content, or has no attempt at the required question
- Output a "grades" object with:
  • "score": 0.0
  • "total_marks": [from model_data]
  • "breakdown": [] (empty array)
  • "comments": ["No relevant content provided."]
- NEVER skip outputting a score for an attempted question — always include a score (even if 0.0) in the JSON

#### Application: When in Doubt
- If semantic matching passes but ANY check-and-balance gate triggers: **STOP** and award 0
- Be STRICT about Gates 1-4 (no supporting evidence, isolated fragments, wrong treatment)
- Be LENIENT only about Gate 1 when evidence exists elsewhere in the answer AND can be clearly connected to this criterion

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
- Use the **exact 'description'** from that marking_criteria item (or very close paraphrase if needed for clarity) as the "criterion" title.
- "max_possible" must be **exactly** the numeric 'marks' value from that item.
- "marks_awarded" must be exactly one of the defined values (full or — only when allowed — half/quarter of that item).
- SUM of all marks_awarded across the breakdown MUST EQUAL the "score".
- If score = 0.0 → omit "breakdown" or use empty array [].
- "evidence": array of 1-3 **exact verbatim substrings** from student answer that directly justified the awarded marks. Keep phrases short (4-10 words), literal and unique.

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
- Wrong answer / completely incorrect → score 0.0, breakdown: [], comments: ["explanation of why wrong"]
- Always output a score object in "grades" array — NEVER skip or omit a score for an attempted question

### Output Format (ONLY this valid JSON)
{{
  "grades": [
    {{
      "question_number": "...",
      "score": number,  // ← ALWAYS include. Can be 0.0 for wrong/incomplete answer
      "total_marks": number,
      "comments": ["quote → description. Advice.", "..."],  // ← ALWAYS present. For score 0.0, explain why
      "correct_words": ["phrase1", "..."],  // ← Empty array [] if score is 0.0
      "breakdown": [  // ← Empty array [] if score is 0.0. Only non-empty if marks awarded
        {{"criterion": "exact description from criteria", "marks_awarded": 0.5, "max_possible": 0.5, "evidence": ["..."], "reason": "Fully correct"}},
        ...
      ]
    }}
  ]
}}

Example for zero score:
{{
  "grades": [
    {{
      "question_number": "Q1",
      "score": 0.0,
      "total_marks": 26,
      "comments": ["No subsidiary accounting discussion" → "Student does not address consolidation or subsidiary treatment. Read model answer section on consolidation methods."],
      "correct_words": [],
      "breakdown": []
    }}
  ]
}}

### Critical Output Rules
- Output ONLY the JSON — no text, no ```json, no explanations, no markdown, no extra characters.
- **MUST ALWAYS INCLUDE A SCORE**: For every question attempted (even if completely wrong), output a score object in "grades" array. Never omit a score. Always provide "score" value (0.0 or higher).
- If score is 0.0 (wrong answer or no relevant content): Include empty "breakdown": [] array. Still provide "comments" with reason.
- No trailing commas.
- All strings must be properly quoted with escaped quotes if needed.
- Escape newlines in strings as \\n 
- All keys and values must be properly JSON formatted.
- Prioritize accuracy and fair assessment. Apply check-and-balance rules strictly. Award marks only when supporting evidence is clear and substantial.
- Double-check all commas are present between objects and array elements.
- Ensure all brackets and braces are balanced.
    """

grade_prompt = ChatPromptTemplate.from_template(GRADE_PROMPT_TEMPLATE)
map_to_questions_prompt = ChatPromptTemplate.from_template(MAP_PROMPT_TEMPLATE)
