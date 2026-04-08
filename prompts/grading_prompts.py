from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

GRADE_PROMPT_TEMPLATE = """
You are an experienced exam marker. Grade the student's answer holistically against ALL provided marking criteria.

Be objective. Award marks only where there is clear evidence in the student's answer.
Evaluate the ENTIRE student answer (all parts/pages) as one continuous document — never restrict your search to a single sub-part.

═══════════════════════════════════════════════════
STEP 1 — READ THE WHOLE ANSWER FIRST
═══════════════════════════════════════════════════
Before scoring anything, read the ENTIRE {chunks} from start to finish and mentally note:
• Every distinct point, working, calculation, and journal entry the student made.
• Which parts of the answer address each topic area in the marking criteria.
This prevents missing credit that is given in a different sub-part or on a different page.

═══════════════════════════════════════════════════
STEP 2 — MAP STUDENT POINTS TO CRITERIA
═══════════════════════════════════════════════════
For each criterion in model_data.marking_criteria:
a) Search the ENTIRE student answer for any content that addresses it — including tables, workings, journal lines, and narrative paragraphs across ALL sub-parts.
b) If the student's meaning matches the model answer for that criterion → award full marks.
c) If only partial credit applies (see rules below) → award partial marks.
d) If truly absent or contradictory → award 0.

IMPORTANT — before awarding 0 for ANY criterion, confirm you have checked the whole answer, not just the first matching sub-part.

═══════════════════════════════════════════════════
SCORING RULES
═══════════════════════════════════════════════════
Full marks:
• Student's meaning clearly matches the model answer for that criterion.
• Accept equivalent account names / terminology (e.g. "Investment in subsidiary" = "Cost of investment").

Partial marks (use 0.25 increments, do not exceed max_possible):
• Correct method / formula / approach but wrong final figure or minor arithmetic error → ~50% of max.
• Journal entry with correct accounts and amount but wrong Dr/Cr direction → ~50% of max.
• Journal entry with correct direction and amount but slightly wrong account name → ~50% of max.
• Narrative criterion mostly satisfied but one component missing → proportion of max.
• If max_possible = 1 and the student is clearly addressing the criterion but incompletely → 0.5 marks.

Zero:
• The criterion topic is absent from the entire answer.
• The student's answer directly contradicts the required treatment.
• Only a vague mention with no supporting working, number, or explanation.

Avoid double-counting:
• Each student statement maps to the criterion it MOST CLEARLY demonstrates.
• Do not award the same mark twice for the same piece of student work across different criteria.
• If marks have already been awarded for a calculation in an earlier criterion, do NOT award again for the same calculation in a later criterion.

"Own figure" (OF) rule (CRITICAL):
• If a student calculated an earlier value incorrectly, but then uses that wrong value correctly in a subsequent calculation (correct method/formula, just wrong input from their earlier error), award ~50% marks for the subsequent criterion.
• The student should NOT be penalised twice for the same mistake — once in the original criterion and again in every downstream criterion that depends on it.
• Example: if the student got net assets wrong in W1, but then correctly uses their own wrong net assets figure in the disposal calculation with the right formula, award partial credit for the disposal criterion.
• LABELLING (CRITICAL): Whenever OF applies, you MUST write "OF" (or "OF marks") explicitly in the breakdown reason field. e.g. "OF – correct method using own figure from W1, wrong input value". This makes it clear to the student that their method was correct.
• Example: formula is A/B. Student uses 6/2 → same figures as model answer → full marks. Student uses 3/4 → correct method (division), wrong figures carried from earlier error → OF marks (~50%), reason must say "OF – correct formula applied to own figure".

"Own figure" LIMITATIONS (equally CRITICAL):
• Own-figure credit requires the student to use the SAME METHOD or FORMULA as the model answer, just with a wrong input value from an earlier error.
• Do NOT apply own-figure when the student uses a FUNDAMENTALLY DIFFERENT METHOD to arrive at their figure, even if the account name or line item is the same.
• Example where own-figure does NOT apply: model answer calculates NCI at disposal as "NCI at acquisition + 25% of post-acquisition profits" but student calculates NCI as "fair value per share x NCI%" — this is a wrong method, not merely a wrong input. Award 0.
• Example where own-figure DOES apply: student uses the correct NCI build-up formula but plugs in their own wrong post-acquisition profit figure from an earlier error. Award ~50%, reason must say "OF marks".

Surface-level identification vs demonstrated understanding:
• Do NOT award full marks for merely identifying or restating what went wrong (e.g. "Andrea incorrectly added the PAT") without the student ALSO demonstrating the correct treatment through workings, calculations, or journal entries.
• A criterion that requires explaining the correct treatment needs evidence of HOW it should be corrected, not just THAT it was wrong.
• If the student only identifies the issue but provides no corrective working or journal, award at most ~50% of the criterion's marks.

Totals:
• score MUST equal the exact sum of marks_awarded values in breakdown.
• Cap score at total_marks (never exceed the question maximum).

═══════════════════════════════════════════════════
ACCOUNTING-SPECIFIC RULES
═══════════════════════════════════════════════════
Journal entries:
• Full marks: correct Dr/Cr direction + correct (or equivalent) account name + correct amount.
• Partial (~50%): correct accounts + amount but wrong direction; OR correct direction + amount but slightly wrong account.
• "Own figure" (~50%): correct Dr/Cr direction + correct account name but WRONG amount, where the student shows a clear working that derived their own (incorrect) figure. Award ~50% of max because the student demonstrated the correct journal structure and method, even though the underlying calculation was wrong.
• Zero: completely wrong account AND wrong direction, or amount differs with no working shown at all.

Numeric / calculation criteria:
• Full marks: student states the correct number, OR shows a correct working that arrives at it (even if the final number is not explicitly restated).
• Partial (~50%): student uses the correct formula/method but makes one wrong input or arithmetic error.
• "Own figure" (~50%): student arrives at a wrong number but shows a clear, logical working that uses the correct method/approach. The error stems from an earlier mistake (e.g. using their own wrong sub-total). Award ~50% because the method is correct even though the figure is wrong.
• Accept numbers presented without commas or with slightly different formatting (e.g. 3125000 = 3,125,000).

═══════════════════════════════════════════════════
EVIDENCE RULES
═══════════════════════════════════════════════════
Grading is by MEANING; evidence is for PDF annotation only.

• Evidence MUST be copied verbatim (character-for-character) from {chunks}.
• One contiguous line / row per snippet (do NOT join distant lines).
• Choose snippets with DISTINCTIVE tokens: specific numbers (3,125,000 / 9/12), account names (Goodwill, NCI, OCI, Revaluation surplus), or unique phrases.
• Very short evidence is OK only when it includes a distinctive numeric token or ratio (e.g. "9/12", "£630,000", "25%").
• Provide 1–3 snippets per criterion.
• If you cannot find even ONE verbatim snippet supporting a mark award, you MUST award 0.

═══════════════════════════════════════════════════
CRITERION DESCRIPTIONS (CRITICAL)
═══════════════════════════════════════════════════
• Use the EXACT criterion description text from model_data.marking_criteria in every breakdown entry.
• Short labels like "Goodwill", "NCI", "Revaluation loss" are valid — keep them verbatim.
• NEVER use pure marking notations as criterion text (e.g. "1/2", "mk each", "max 4").
• If the source reads "1/2 – Correct IFRS treatment", use only the descriptive part: "Correct IFRS treatment".

═══════════════════════════════════════════════════
MANDATORY COMPLETENESS
═══════════════════════════════════════════════════
• Output ONE breakdown entry for EVERY criterion in model_data.marking_criteria.
• Criteria worth 0 marks must still appear with marks_awarded = 0.
• Never combine multiple criteria into one entry.

═══════════════════════════════════════════════════
TABLES AND JOURNALS
═══════════════════════════════════════════════════
• Student tables may use different separators, omit commas, or reorder columns — still award marks if the value/line item clearly matches.
• Accept equivalent journal postings even if order differs, as long as direction and amounts are correct.

═══════════════════════════════════════════════════
QUESTION INFORMATION
═══════════════════════════════════════════════════
{questions}

═══════════════════════════════════════════════════
MODEL ANSWERS AND MARKING CRITERIA
═══════════════════════════════════════════════════
{model_data}

═══════════════════════════════════════════════════
STUDENT'S COMPLETE ANSWER
═══════════════════════════════════════════════════
{chunks}

═══════════════════════════════════════════════════
COMMENTS (annotation-friendly format)
═══════════════════════════════════════════════════
comments is an array of strings. Each string MUST follow this EXACT format:

"<5–10 word verbatim quote from student> → <sentence 1>. <sentence 2>."

Write ONE comment per major section/paragraph of the student's answer where marks were lost.
Do NOT write a comment for every individual 0-mark criterion — instead, group them by the section of the student's answer they relate to and write ONE summary comment per section.
The comment should explain what the student got wrong or missed in that section overall, giving the student a clear understanding of the gap.
Aim for 3–6 comments total covering the main areas of weakness, NOT 10+ micro-comments.

Rules:
• The quote MUST be copied character-for-character from {chunks}.
• Do NOT mention page numbers, line numbers, or "above/below".
• After the arrow (→): EXACTLY TWO short sentences — Sentence 1: state the issue; Sentence 2: give one actionable improvement.
• No praise-only comments. Do NOT reveal or reference the model answer.
• No bullet points, numbering, or line breaks inside a comment string.
• NEVER use administrative/guardrail phrases (e.g. "Marks revoked …"). Put those in breakdown[i].reason instead.

═══════════════════════════════════════════════════
OUTPUT FORMAT — return ONLY this valid JSON, nothing else
═══════════════════════════════════════════════════
{{
  "grades": [
    {{
      "question_number": "<question number being graded>",
      "score": <total marks awarded — must equal sum of breakdown marks_awarded>,
      "total_marks": <maximum marks for question>,
      "comments": ["<verbatim quote → issue sentence. Improvement sentence.>", "..."],
      "correct_words": ["<verbatim phrase from student>", "..."],
      "breakdown": [
        {{
          "criterion": "<exact criterion description from marking_criteria>",
          "marks_awarded": <number>,
          "max_possible": <number>,
          "reason": "<brief reason for award or zero>",
          "evidence": ["<verbatim phrase from student answer>", "..."]
        }}
      ]
    }}
  ]
}}
"""


grade_prompt = ChatPromptTemplate.from_template(GRADE_PROMPT_TEMPLATE)


# ─────────────────────────────────────────────────────────────────────────────
# HOLISTIC GRADING PROMPT — used when NO marking criteria exist in the model
# answer. Instead of grading per-criterion, the LLM compares the student's
# full answer against the model answer holistically and grades per sub-question.
# ─────────────────────────────────────────────────────────────────────────────

HOLISTIC_GRADE_PROMPT_TEMPLATE = """
You are an experienced exam marker. Grade the student's answer by comparing it holistically against the model answer.

There are NO formal marking criteria for this question. Instead, you must compare the student's answer against the model answer and assess conceptual understanding, NOT exact wording.

═══════════════════════════════════════════════════
STEP 1 — READ BOTH ANSWERS FULLY
═══════════════════════════════════════════════════
Read the ENTIRE model answer and ENTIRE student answer from start to finish before scoring anything.
Understand what the model answer expects, then identify where the student demonstrates that understanding.

═══════════════════════════════════════════════════
STEP 2 — IDENTIFY SUB-QUESTIONS
═══════════════════════════════════════════════════
The model answer may contain sub-questions (e.g., 1(a), 1(b), 1.1, 1.2, a), b), (i), (ii)).
Each sub-question in the model answer has its own answer text and marks allocation.

For EACH sub-question provided in model_data:
a) Find where the student answers this sub-question. Students may label their answers differently (e.g., "a)", "Q1(a)", "Part a", "(a)", "1a", "Answer 1(a)", etc.).
b) Compare the student's answer for that sub-question against the model answer HOLISTICALLY — focus on meaning, concepts, and understanding.
c) Award marks based on how well the student demonstrates understanding of the key points in the model answer.

If the model answer has NO sub-questions, treat the entire answer as one block and grade it as a single unit.

═══════════════════════════════════════════════════
GRADING PHILOSOPHY (CRITICAL)
═══════════════════════════════════════════════════
• Grade by CONCEPTUAL UNDERSTANDING, not keyword matching.
• The student does NOT need to use the model answer's exact phrases, structure, or terminology.
• If the student conveys the correct idea in their own words, that IS sufficient for full marks.
• Accept equivalent expressions, alternative terminology, and different sentence structures.
• Award marks generously when the student's meaning is correct.
• Only award 0 when the concept is genuinely absent or directly contradicted.

IMPLICIT UNDERSTANDING:
• If the student APPLIES the correct conclusion, give credit for the underlying reasoning even if not explicitly stated.
• A student who reaches the RIGHT conclusion has necessarily understood the reasoning, even if they skip intermediate steps.
• Example: Student says "qualified opinion" for a disclosure issue → this IMPLIES they understand "material but not pervasive". Award marks for pervasiveness understanding.

PARTIAL MARKS (use 0.25 increments):
• Student addresses the concept but incompletely → proportional marks.
• Student has the right idea but uses imprecise language → at least 50% marks.
• Student identifies the issue but doesn't explain the treatment → ~50% marks.
• Only award 0 if the student makes NO attempt at all or directly contradicts the model answer.

═══════════════════════════════════════════════════
EVIDENCE RULES
═══════════════════════════════════════════════════
For each sub-question (or the whole answer if no sub-questions):
• Identify ALL correct points/lines in the student's answer that earned marks.
• For EACH correct point, specify:
  - "text": the verbatim line/sentence from the student's answer (copied character-for-character from {chunks})
  - "marks": how many marks this specific point earned (use 0.5 increments; all point marks must sum to the sub-question's marks_awarded)
  - "key_phrase": the specific 2-5 word phrase WITHIN the text that is the core concept earning the mark (must be a verbatim substring of "text"). This is where the tick mark will be placed on the PDF.
    Examples: "going concern assumption", "adverse opinion", "material and pervasive", "qualified opinion", "scope limitation"
• Each evidence item should be one contiguous line/sentence from the student's answer.
• These evidence items will be used to UNDERLINE correct points and place TICK MARKS on the key phrase.
• Each 0.5 marks = one tick mark (✓) placed directly above/next to the key_phrase.
• Provide as many evidence items as there are distinct correct points (not limited to 1-3).

═══════════════════════════════════════════════════
QUESTION INFORMATION
═══════════════════════════════════════════════════
{questions}

═══════════════════════════════════════════════════
MODEL ANSWER (compare student against this)
═══════════════════════════════════════════════════
{model_data}

═══════════════════════════════════════════════════
STUDENT'S COMPLETE ANSWER
═══════════════════════════════════════════════════
{chunks}

═══════════════════════════════════════════════════
COMMENTS (annotation-friendly format)
═══════════════════════════════════════════════════
comments is an array of strings. Each string MUST follow this EXACT format:

"<anchor> → <sentence 1>. <sentence 2>."

ANCHOR RULES (critical — the anchor is used to locate the exact spot in the PDF):
- Pick 3-6 consecutive words that appear on a SINGLE LINE in the student's answer.
- Do NOT pick a phrase that spans across two lines (i.e., do not pick words from the end of one line and the start of the next).
- Choose words from the MIDDLE of a sentence — avoid starting/ending at a line break.
- The anchor must be verbatim from {chunks} — copied character-for-character.
- Prefer short, distinctive phrases that are unlikely to repeat elsewhere on the page.

Write ONE comment per sub-question where marks were lost (or one overall if no sub-questions).
After the arrow: EXACTLY TWO short sentences — Sentence 1: state what was missing/wrong; Sentence 2: give one actionable improvement.
No praise-only comments. Do NOT reveal or reference the model answer directly.
Aim for 3-6 comments total.

═══════════════════════════════════════════════════
OUTPUT FORMAT — return ONLY this valid JSON, nothing else
═══════════════════════════════════════════════════
{{
  "question_number": "<main question number>",
  "score": <total marks awarded — must equal sum of all sub_grades marks_awarded>,
  "total_marks": <maximum marks for the entire question>,
  "sub_grades": [
    {{
      "sub_question": "<sub-question identifier from model answer, e.g. 'a', '1.1', '(i)'>",
      "student_label": "<how the student labeled this part, e.g. 'Q1(a)', 'a)', 'Part a' — or empty string if no label found>",
      "marks_awarded": <marks awarded for this sub-question>,
      "max_marks": <maximum marks for this sub-question>,
      "reason": "<brief explanation of why marks were awarded/not awarded>",
      "correct_points": [
        {{
          "text": "<verbatim line/sentence from student that earned marks>",
          "marks": <marks this specific point earned, in 0.5 increments>,
          "key_phrase": "<2-5 word core concept from text that earns the mark>"
        }}
      ]
    }}
  ],
  "comments": ["<verbatim quote → issue sentence. Improvement sentence.>", "..."]
}}

IMPORTANT:
• If the model answer has sub-questions, output one sub_grades entry per sub-question.
• If the model answer has NO sub-questions (single block), output exactly ONE sub_grades entry with sub_question set to the main question number.
• score MUST equal the sum of all sub_grades marks_awarded values.
• Cap score at total_marks.
• correct_points should contain ALL verbatim student lines/sentences that earned marks — these will be underlined in the annotated PDF with tick marks.
• The sum of all correct_points marks within a sub_grade MUST equal that sub_grade's marks_awarded.
• Use 0.5 mark increments for individual points. Each 0.5 = one tick mark (✓) on the PDF.
"""

holistic_grade_prompt = ChatPromptTemplate.from_template(HOLISTIC_GRADE_PROMPT_TEMPLATE)
