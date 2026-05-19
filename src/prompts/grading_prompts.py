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

DUPLICATE POINTS (CRITICAL):
• If the student writes the SAME calculation, journal, or narrative point twice (e.g. repeated for emphasis, copied above and below, or restated in another section), award marks ONLY ONCE.
• Use the FIRST occurrence as the evidence for that criterion (so the score + underline land on the first one).
• For the DUPLICATE occurrence, do NOT add it as evidence on any criterion. Instead, emit a COMMENT in the format:
  "<5-10 word verbatim quote from the duplicate> → Marks already given above for this point. <one-sentence improvement>."
  (use "below" if the duplicate is earlier than the primary occurrence).
• Two genuinely DISTINCT calculations that happen to share wording (e.g. two different journal entries with the same account name but different amounts) are NOT duplicates — keep them separate.

"Marks given above / below" — working vs. subsequent use:
• When a student calculates a figure in a WORKING (e.g. W2: NCI at disposal = £6,975,000) and then USES that same figure in a subsequent journal entry (e.g. Dr NCI 6,975,000), the journal entry criterion earns its OWN separate marks — this is a different skill (knowing which account to debit/credit) and is NOT a duplicate.
• However, if the student merely RE-STATES the same calculation a second time without adding new working (e.g. writes the NCI build-up twice in different sections), award marks ONLY for the FIRST occurrence and comment "Marks given above" on the second.
• The rule of thumb: marks follow the WORK, not the conclusion. Award at the location where the student actually performs the calculation or writes the journal. Later references to the same result get no extra marks.

"Own figure" (OF) rule (CRITICAL):
• If a student calculated an earlier value incorrectly, but then uses that wrong value correctly in a subsequent CALCULATION (correct method/formula, just wrong input from their earlier error), award FULL marks for that calculation criterion.
• In UK professional exams (ICAEW/ACCA style), own-figure for CALCULATIONS earns the full mark — the method is what is tested, and the student is NOT penalised twice for one wrong input.
• For JOURNAL entries with wrong amounts (but correct direction and accounts): award ~50% because the journal amount is the assessable element (not just a downstream figure).
• The student should NOT be penalised twice for the same mistake — once in the original criterion and again in every downstream criterion that depends on it.
• Example: if the student got net assets wrong in W1, but then correctly uses their own wrong net assets figure in the disposal calculation with the right formula, award FULL marks for the disposal criterion.
• LABELLING (CRITICAL): Whenever OF applies, you MUST write "OF" (or "OF marks") explicitly in the breakdown reason field. e.g. "OF – correct method using own figure from W1, wrong input value". This makes it clear to the student that their method was correct.
• Example: formula is A/B. Student uses 6/2 → same figures as model answer → full marks. Student uses 3/4 → correct method (division), wrong figures carried from earlier error → FULL OF marks for that step, reason must say "OF – correct formula applied to own figure".

"Own figure" LIMITATIONS (equally CRITICAL):
• Own-figure credit requires the student to use the SAME METHOD or FORMULA as the model answer, just with a wrong input value from an earlier error.
• Do NOT apply own-figure when the student uses a FUNDAMENTALLY DIFFERENT METHOD to arrive at their figure, even if the account name or line item is the same.
• Example where own-figure does NOT apply: model answer calculates NCI at disposal as "NCI at acquisition + 25% of post-acquisition profits" but student calculates NCI as "fair value per share x NCI%" — this is a wrong method, not merely a wrong input. Award 0.
• Example where own-figure DOES apply: student uses the correct NCI build-up formula but plugs in their own wrong post-acquisition profit figure from an earlier error. Award ~50%, reason must say "OF marks".

Share-based payment (SBP) "own figure" — wrong fair value input:
• The correct grant-date FV for equity-settled options is the FV at the date of GRANT (e.g. £24 per option). The exercise price (e.g. £210) is WRONG as an FV input.
• If a student uses the correct SBP formula structure (N_employees × N_options × FV × proportion/vesting) but substitutes the exercise price for the grant-date FV, this is an OWN FIGURE scenario — correct method, wrong input.
• Award FULL marks for each SBP calculation criterion where the formula structure is correct but the FV used is the exercise price. Label the reason as "OF – correct formula, wrong fair value (used exercise price instead of grant-date FV)".
• Do NOT award OF marks if the student uses a completely different formula structure (e.g. total proceeds ÷ vesting period), only if the formula has the right shape but wrong FV input.

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
NOT-REQUIRED (OFF-TOPIC) CONTENT
═══════════════════════════════════════════════════
Students sometimes include content the question never asked for — definitions of
unrelated concepts, padding, irrelevant tangents, or material from a different
question. Real markers strike these out with a "Not required" note so the student
knows to drop them in future answers.

For each clearly off-topic sentence/passage in the student's answer:
• Output ONE entry in not_required_points with:
  - "text": the verbatim off-topic sentence/passage from the student answer.
  - "key_phrase": a 3-6 word verbatim substring of "text" — the anchor where the
    "Not required" marker will be placed on the PDF.
  - "reason": ONE short sentence explaining why this content is off-topic
    (e.g. "Question asks for the consolidation entries, not the definition of goodwill.").

Rules:
• not_required_points carry NO marks. They do NOT change marks_awarded for any criterion.
• Do NOT flag content that earned marks elsewhere (it must not appear in evidence AND not_required_points).
• Borderline / weakly relevant content → leave it out. Only flag CLEARLY off-topic.
• If the student is on-topic throughout, return an empty list.

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

"[<top_level_sub_question>] <5–10 word verbatim quote from student> → <sentence 1>. <sentence 2>."

STRICT QUOTA — ONE comment per major sub-section, AT MOST.
• The bracketed prefix must be the TOP-LEVEL sub-section identifier ("1.1", "1.2",
  "(a)", "(b)", etc.) — NEVER a per-criterion or fine-grained label.
• If a sub-section has multiple criteria where marks were lost, write ONE
  consolidated comment that covers the most important gap. Do NOT emit one
  comment per missed criterion.
• If a sub-section is fully credited, omit it from comments entirely.
• Aim for AT MOST one comment per major sub-section in the question. For a
  question with 4 sub-sections, that means at most 4 comments total.

The comment should explain what the student got wrong or missed in that sub-section overall, giving the student a clear understanding of the gap.

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
      ],
      "not_required_points": [
        {{
          "text": "<verbatim off-topic sentence from student>",
          "key_phrase": "<3-6 words verbatim from text>",
          "reason": "<one short sentence why this is off-topic>"
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
You are an experienced exam marker. Grade the student's answer against the model answer.

═══════════════════════════════════════════════════
HOW TO GRADE
═══════════════════════════════════════════════════

STEP 1 — IDENTIFY THE SCOREABLE POINTS

The grading payload looks like:
   model_data.answers[0].sub_questions = [
     {{
       "sub_question": "4.1(a) Consequences",
       "answer": "<model answer text for this leaf>",
       "max_marks": 4.0,
       "marking_criteria": [{{"marks": 1, "description": "...", "keywords": [...]}}, ...],
       "marking_rule": "1 mark per paragraph. Maximum 4.",
       "parent_section": "4.1",
       "section_cap": 8.0
     }},
     ...
   ]
You MUST iterate sub_questions and grade each leaf independently. The
model_data.answers[0].answer is just the concatenated text of all leaves —
use it for context, but score against the criteria of each leaf.

For each sub-question, the scoreable points are:
• If marking_criteria has descriptions that are real model-answer sentences (the
  default) → THOSE are the scoreable points. Each criterion's "marks" field is
  its full value (1, 0.5, 1.5, or 2). Compare the student's writing under that
  sub-question to each criterion and apply the FULL/HALF/ZERO rule below.
• If marking_criteria descriptions are generic labels (rare) → the scoreable
  points are the PARAGRAPHS and BULLETS in the "answer" field. Split by \\n\\n
  or lines starting with -.
• If "marking_rule" is present → apply it (e.g. "1 mark per paragraph, max 4").

FULL / HALF / ZERO RULE (THE CORE OF GRADING):
For each criterion in a sub-question's marking_criteria list:
• FULL marks (= criterion.marks) — the student's writing under this sub-question
  fully conveys the meaning of the criterion. Examples:
  – Criterion: "Cost overruns and projects falling behind schedule will not be
    identified, resulting in delayed remedial action." (1 mark)
    Student: "the lack of comparison means cost overruns won't be caught and
    management can't act in time" → FULL (1 mark).
• HALF marks (= criterion.marks / 2, rounded to 0.25 increments) — the student
  conveys the RIGHT CONCEPT but is missing a key element, only states one half
  of a compound idea, or uses different wording for a phrase the rubric
  specifically demands. Examples:
  – Criterion: "Adverse impact on cash flow AND profitability" (1 mark)
    Student: "shortage of cashflow / poor budgeting in future" → HALF (0.5).
    Reason: covers cash flow but not profitability.
  – Criterion: "Monthly comparison of budget against actual evidenced by
    signature" (1 mark)
    Student: "managers sign a document to show they understand their
    responsibility in comparing COS monthly" → HALF (0.5).
    Reason: signature is for responsibility, not the comparison itself —
    partial match.
• ZERO — the concept is absent, contradicted, or named with the wrong reason:
  – Criterion: "Self-interest threat — reluctant to identify prior-year
    misstatements that could damage reputation" (1 mark)
    Student: "self-interest threat since the CEO seems keen to keep Nicola"
    → ZERO. The reason given is wrong (CEO keeping her, not her reputation).
  – Criterion: "Firm must refuse to provide payroll services" (1 mark)
    Student: "Griffin can accept this with safeguards if fee < 10%" → ZERO.
    Direct contradiction of the required answer.

ROLE OF THE "keywords" FIELD ON EACH CRITERION:
• keywords are PRIMARILY placement anchors for the tick (which student words
  to underline). They help target the visible mark in the PDF.
• keywords are SECONDARILY a sanity check: if NO keyword (or obvious stem
  variant like "expand-/expanded", "prohibit-/prohibits", "cashflow/cash flow")
  appears in the student's text for this sub-question, that's a strong signal
  the criterion is at best a HALF, not a FULL. Use this as a tiebreaker — don't
  treat keywords as a hard match filter.

Determine marks per point:
• marking_rule present → follow it exactly (e.g. "1 mark per paragraph, max 4" = 1 mark each, stop at 4).
• "1 per para" / "one each" / "one per para" → 1 mark per point.
• "half for each item" → 0.5 marks per point.
• Otherwise → total_marks_available ÷ number of points.
• "max N" → award up to N marks for valid matches. Even 1 valid match earns marks.

SECTION CAPS (CRITICAL):
• Some sub-questions carry a "section_cap" and "parent_section" field.
• section_cap = the MAXIMUM marks that ALL sub-questions with the same parent_section can earn COMBINED.
• Track a running total per parent_section. BEFORE awarding marks to a sub-question in that section, check the running total: only award up to (section_cap − running_total). Once the total reaches section_cap, set marks_awarded = 0 for any remaining sub-questions in that section.
• Sub-questions without section_cap are capped only by their own max_marks.
• Also honour each sub-question's own max_marks (e.g. "maximum_marks": 3) even when it is lower than total_marks_available.

WORKED EXAMPLE — section cap with max_marks:
Suppose 4.2 has parent section_cap = 4, and contains two sub-questions:
   - "4.2 Threats" (max_marks 3, four 1-mark criteria available)
   - "4.2 Response" (no max_marks, five 1-mark criteria available)
If the student matches 3 of the 4 threat criteria → award 3 marks for "4.2 Threats" (its own max).
Running total for 4.2 is now 3. Section cap is 4, so only 1 mark remains.
Even if the student matches 4 of the 5 response criteria → award only 1 mark for "4.2 Response" (cap exhausted).
Tick count must equal awarded marks ÷ 0.5; trim extra correct_points so the count matches.

STEP 2 — COMPARE STUDENT WRITING TO THE MODEL ANSWER, THEN ANNOTATE THE MATCHES

You are NOT checking off criteria in isolation. You are COMPARING the student's
writing to the model answer, finding the specific student WORDS that demonstrate
each scoreable point, and placing ticks on those exact words.

The loop, for each criterion in the rubric:

(1) READ THE CRITERION'S SUBSTANCE.
    The criterion's "description" tells you the concept; its "keywords" list
    highlights its DISTINGUISHING ELEMENT (the specific fact/figure/relationship
    that separates this criterion from every other criterion in the rubric —
    e.g. "five years", "reasonable third party", "FRC ES prohibits for listed
    clients", "expanded review of audit work").

(2) FIND IT IN THE STUDENT'S TEXT, OR DON'T.
    Scan the student answer for words/phrases that EXPRESS the distinguishing
    element — verbatim or as clear paraphrase of the substance, not of the
    topic. If you find a substantive match, you have a credit. If you only
    find a generic gesture toward the topic (no diinguishing element), you
    have NO credit. There is no half-credit here — either the substance is
    there or it isn't.

(3) ANCHOR THE TICK ON THE MATCHING STUDENT WORDS.
    The student's matching phrase becomes the `key_phrase` for the tick. Pick
    3-6 verbatim words from the student's text — the slice that contains (or
    paraphrases) the distinguishing element. THAT is where the tick lands.

STRICTNESS — four rules that prevent over-marking:

A. DISTINGUISHING-ELEMENT REQUIRED — LITERAL OR STEM PRESENCE.
   If the criterion's description or keywords contain a SPECIFIC distinguishing
   word/phrase (a precise term that identifies THIS criterion, not a generic
   topic word), the student's text MUST contain that word/phrase — verbatim,
   pluralised, or as a clear morphological/stem variant of the SAME root.

   Semantic synonyms with DIFFERENT word roots are NOT acceptable. A teacher
   following this rubric expects the specific term to appear; equivalent
   concepts expressed in completely different vocabulary do not earn the mark.

   EARN: "audited for 5 yrs" matches "five years" (stem match: 5/five → years/yrs).
   EARN: "expanded the review" matches "expanded review of audit work"
         (stem match: expanded → expanded).
   EARN: "FRC ES prohibits payroll for listed clients" matches "FRC Ethical
         Standard expressly prohibits payroll services for listed clients"
         (literal match on FRC, prohibits, listed).
   ZERO: "external quality review" does NOT match "expanded review of audit
         work" — "external" is a different root from "expanded"; the
         distinguishing element ("expanded"/"audit work"/"discrepancies") is
         absent. Different words, different mark.
   ZERO: "outside observer" does NOT match "reasonable and informed third
         party" — different root words; the distinguishing phrase "third party"
         is absent.
   ZERO: "there is a familiarity threat as she has known the client a while"
         does NOT match "five-year continuous audit partner" — no five-year
         or continuity word present.
   ZERO: "the firm shouldn't do payroll" — no FRC / listed element.

   Rule of thumb: scan the criterion's keywords list. Does ANY keyword (or its
   obvious stem like `expand-`, `prohibit-`, `disclos-`, `scept-`) appear in
   the student's text? If NO keyword has even a stem-match in the student
   text, the criterion is NOT matched — even if a generic synonym is used.

   SPACING / HYPHENATION EQUIVALENCE (read this carefully — common false-strict):
   Treat single-word, hyphenated, and space-separated forms of the SAME root as
   matching. The student is not penalised for spacing choices.

   EARN: "cashflow" matches keyword "cash flow" (single word vs two words).
   EARN: "non-compliant" matches keyword "non compliant" (hyphen vs space).
   EARN: "self review" matches keyword "self-review" (space vs hyphen).
   EARN: "userfriendly" matches keyword "user friendly".
   EARN: "TCWG" matches keyword "those charged with governance" (acronym).
   ZERO: "cash position" does NOT match "cash flow" — "flow" stem absent.

   COMMON PARAPHRASE FAILURES (these are real teacher fail-modes — do NOT
   accept them):
   ZERO: "challenge management" does NOT match "insufficiently sceptical of
         financial statements" — different roots ("challenge" ≠ "scept-"),
         and the substantive object ("financial statements") is absent.
   ZERO: "external quality review" ≠ "expanded review of audit work".
   ZERO: "outside observer" ≠ "reasonable and informed third party".
   ZERO: "sign a document to show responsibility" ≠ "monthly comparison
         evidenced by signature" — the signing is for responsibility, NOT
         for the monthly comparison. Different referent.
   ZERO: "informed that expensing outside X will not be reimbursed" ≠ "do
         not process expense claims that bypassed the website" — different
         remedy (employee communication vs back-office process control).
   ZERO: "authorization controls on working papers" ≠ "confidential and
         secure data filing" — different concept (access control vs
         physical filing system).

B. NO DOUBLE-CREDIT FOR ONE CONCEPT.
   Two criteria that describe causally-linked aspects of the SAME concept
   (familiarity → trust → reduced scepticism) collapse to ONE credit when
   the student treats them as one thought.

   ZERO (double-credit): student writes "familiarity threat as she may be too
   trusting and scepticism is impaired" — DO NOT credit BOTH "familiarity
   threat" AND "too trusting/insufficiently sceptical" criteria. Pick the
   single best match. To earn both, the student must develop each point
   SEPARATELY with distinct substantive content.

B'. RESPECT THE LEAF — NO CROSS-LEAF CREDIT FOR THE SAME SENTENCE.
   When the rubric splits a sub-question into leaves with similar/overlapping
   criteria (e.g. 4.3 has BOTH a "Recruitment Threats" leaf and a "Payroll
   Threats" leaf, both with a "management threat" / "too closely aligned"
   criterion), a single student sentence MUST credit criteria in AT MOST
   ONE leaf — the leaf whose CONTEXT matches the sentence.

   Identify the leaf from the student's context cues:
   • The student's own sub-label (e.g. "4.3(A)" → Recruitment leaf,
     "4.3(b)" → Payroll leaf) is the strongest signal — respect it.
   • Substantive content cues — "decision roles", "candidate selection" →
     Recruitment; "reviewing payroll", "payroll information" → Payroll.

   WRONG (cross-leaf double-credit): student writes under 4.3(A): "management
   threat ... Griffin taking on decision roles ... Griffin's goals aligning
   to closely to Yeti's." Credit ONLY the Recruitment Threats leaf's
   management/closely-aligned criterion. DO NOT also credit the Payroll
   Threats leaf's "closely aligned with views and interests of management"
   criterion using the same sentence — that's the same concept in a leaf
   the student is not addressing.

   RIGHT: each leaf earns marks ONLY from sentences whose context matches
   that leaf. If the student wrote nothing about payroll's "closely aligned"
   point, the payroll leaf earns 0 for it — even if a recruitment sentence
   contains matching keywords.

B''. EVIDENCE LOCALITY — NO CROSS-SECTION EVIDENCE LEAKAGE (CRITICAL).
   Evidence supporting any criterion in sub-question 4.X MUST be drawn from
   the student's writing UNDER section 4.X. You are NOT allowed to harvest a
   sentence the student wrote under 4.2 and use it as evidence for a 4.3
   criterion, even if the words happen to match a 4.3 keyword.

   How to determine which student section a sentence belongs to:
   1. Section headings in the student's text — "4.2", "4.3", "(a)", "(b)",
      "A)", "b)" — define a boundary. Every sentence between heading X and
      the next heading belongs to X.
   2. If a sentence appears BEFORE the first heading, attribute it to the
      first sub-question listed in the rubric (usually 4.1(a)).
   3. The student's own paragraph layout matters more than keyword matches —
      a sentence under "4.2" is 4.2's, even if it mentions payroll.

   COMMON LEAK CASES TO AVOID:
   • "should refuse Nicola to continue on this engagement" lives in 4.2 (the
     auditor-rotation question). It is NOT evidence for 4.3's "must refuse
     payroll services" or for any 4.3 criterion.
   • "as an external quality review" lives in 4.2. NOT evidence for 4.3
     or 4.4.
   • "poor management decisions are made re. costs" lives in 4.1(a). NOT
     evidence for 4.3's "management threat" criterion.
   • "authorization controls on audit working papers" lives in 4.4. NOT
     evidence for 4.1.

   Before adding a correct_point with a given `text`, ask: which section of
   the student's answer did this sentence come from? If it isn't the
   sub-question you're currently grading, REJECT the evidence — do not
   award the credit, even if a keyword matches.

C. NO CREDIT FOR LABEL WITHOUT SUBSTANCE.
   "there is a self-interest threat" — naming the threat without explaining
   why it arises → 0. Criterion names by themselves are not assessable points.

   EXTRA EXAMPLES (these are the failure modes we have seen):
   • "There may also be a self-interest threat since the CEO seems keen to
     keep Nicola." → 0 for the self-interest criterion. The reason given
     ("CEO keen to keep Nicola") does NOT match the model answer's reason
     ("Nicola may be reluctant to identify prior-year misstatements that
     could damage her reputation"). Wrong rationale = label only.
   • "management threat" appearing alone without explaining HOW the firm
     becomes aligned → 0. The student must state the mechanism
     (decision-making, candidate selection, alignment with management's
     views) to earn the half-mark.

   To earn a label-bearing half-mark, the student's sentence must contain
   the LABEL token AND a substantive clause that matches the rubric's
   stated reason for that label. If only the label is present, or the
   reason is wrong, award 0.

D. WRONG-ANSWER CONTAMINATION VOIDS THE POINT (AND OFTEN THE WHOLE LEAF).
   If the student includes a CORRECT phrase alongside an INCORRECT conclusion
   on the same point, do not award the point.

   ENFORCEMENT — the "how to address" leaves of 4.3 are the canonical case:
   When the model answer says the firm MUST REFUSE / DECLINE a service (e.g.
   FRC ES prohibition on payroll services for listed clients), and the
   student says the firm CAN ACCEPT WITH SAFEGUARDS (or any variant that
   reverses the directive — "acceptable if fee < X%", "okay with team
   segregation", etc.), you MUST award 0 for the entire "How to address" leaf
   for that service, regardless of how many keywords match. The student has
   misunderstood the rule, not learned it. Do not award even 0.5 for naming
   the safeguard name correctly when the safeguard is the WRONG remedy.

   The corresponding "Threats" leaf is graded separately — the student may
   still earn marks there for correctly identifying threats (self-review,
   management, self-interest) even if their conclusion is wrong. The
   contamination only voids the leaf where the wrong conclusion lives.

E. EXACT-PHRASE INERTIA FOR DISTINGUISHING TERMS.
   Some criteria are anchored on a SPECIFIC LEXICAL TERM the rubric author
   chose deliberately ("expanded review", "FRC Ethical Standard", "five years
   continuously", "reasonable and informed third party", "evidenced by
   signature", "confidential and secure data filing", etc.). For these
   criteria, the student's text should contain a keyword stem variant of the
   SAME ROOT word — otherwise downgrade FULL → HALF (not always zero).

   Apply the FULL/HALF/ZERO mapping for concept-level paraphrase mismatches:
   • "external quality review" vs "expanded review" → HALF (right concept,
     wrong specific term). If the student also fails other parts, → ZERO.
   • "outside observer" vs "reasonable and informed third party" → HALF.
   • "managers should sign acknowledgement of responsibility" vs "monthly
     comparison evidenced by signature" → HALF (signing is happening but
     for the wrong thing). Could go to ZERO if the comparison itself isn't
     described.
   • "authorization controls on working papers" vs "confidential and secure
     data filing" → ZERO (different procedural concept — access control vs
     filing system).
   • Wrong-direction conclusions (Rule D) → ZERO regardless of keyword match.

DUPLICATE OCCURRENCES of the SAME point by the student (repeated for emphasis)
— credit ONLY the first occurrence; on the second, do not output another
correct_point. Optionally add a comment "<anchor> → Marks already given above
for this point. <improvement>."

STEP 3 — PLACE THE TICKS ON THE STUDENT'S MATCHING WORDS

Each correct_point = 0.5 marks = ONE tick (✓) on the annotated PDF, drawn over
the EXACT student words that demonstrated the criterion. The tick is the
visible record of the comparison: "this phrase in your answer matched this
point in the model answer".

Marks → tick count (each tick = 0.5 marks visible on PDF):
• Award FULL marks for a criterion = output (criterion.marks / 0.5) correct_points.
  Examples: 1 mark FULL → 2 ticks. 1.5 marks FULL → 3 ticks. 0.5 mark FULL → 1 tick. 2 marks FULL → 4 ticks.
• Award HALF marks for a criterion = output (criterion.marks / 0.5 / 2) correct_points.
  Examples: 1 mark HALF (= 0.5 awarded) → 1 tick. 1.5 marks HALF (= 0.75 awarded — round to 0.5) → 1 tick. 0.5 mark HALF (= 0.25 awarded — round to 0.5) → 1 tick.
• Award ZERO → no correct_points for that criterion.
• Every correct_point in the output JSON MUST have `marks: 0.5`. The number of
  correct_points × 0.5 must equal the sub-question's marks_awarded.

WHERE THE TICK LANDS (key_phrase selection):
1. The key_phrase MUST be a 5-8 word verbatim slice of the student's sentence
   that captures the WHOLE substantive clause demonstrating the criterion —
   the part a teacher would underline with a red pen. NEVER 1-3 words and
   AVOID 4-word fragments unless the 4 words clearly stand alone as a clause.
   Short anchors render as a tick on a single word in the PDF (e.g. on
   "a", "the", "of") and look like noise. Underline the WHOLE point, not
   just the cue word.

   Examples of GOOD anchor selection (5-8 words, full substantive clause):
   • Criterion: "Adverse impact on cash flow"
     GOOD:  "shortage of cashflow / poor budgeting in future"
     BAD:   "shortage of cashflow" (too short — extend to the predicate)
   • Criterion: "Will not hit stated target of net zero"
     GOOD:  "inability to meet net zero goals and Yeti"
     BAD:   "inability to meet net zero" (cuts off before the consequence)
   • Criterion: "Increased scope for fraud"
     GOOD:  "also a risk of fraud, due to uncontrolled"
     BAD:   "risk of fraud" (3 words — too short)
   • Criterion: "Familiarity (trust) threat"
     GOOD:  "Familiarity threat to objectivity, since Nicola has"
     BAD:   "Familiarity threat" (2 words — too short)
2. The slice must contain the SUBSTANTIVE WORDS that demonstrate the criterion.
   If the criterion is "five-year continuous audit partner" and the student wrote
   "audit partner for 5 yrs", the key_phrase should be the WHOLE clause
   like "Nicola has been on the audit for 5 yrs" (7 words) — the full
   distinguishing clause with subject + predicate, not just "5 yrs".
3. For 1-mark criteria (2 ticks), pick TWO DIFFERENT 5-8 word spans inside the
   same sentence — each must independently be a meaningful clause. Do not split
   one short clause into two even shorter halves.
4. If the criterion has a `keywords` list AND a keyword (or its obvious
   paraphrase) appears in the student text, the key_phrase MUST contain that
   keyword/paraphrase plus enough surrounding context to make a clause.
5. If no keyword appears verbatim (pure paraphrase case), pick the 5-8 word
   slice that most directly demonstrates the criterion's substance.

BAD key_phrases — DO NOT EMIT:
• "a", "the", "to", "of" — articles/prepositions
• 1-3 word fragments that lack a subject or predicate
  – "reviewing payroll" (2 words, no subject — expand to "audit team will be reviewing payroll", 6 words)
  – "closely to Yeti's" (3 words, no subject — expand to "Griffins goals aligning to closely to Yeti's", 7 words)
  – "Griffins goals aligning" (3 words, no completion — expand to "Griffins goals aligning to closely to Yeti's", 7 words)
  – "risk of fraud" (3 words — expand to "also a risk of fraud, due to uncontrolled", 8 words)
  – "Familiarity threat" (2 words — expand to "Familiarity threat to objectivity, since Nicola has", 7 words)

GOOD key_phrases (5-8 words, meaningful clause with subject + predicate):
• "Nicola has been on the audit for 5 yrs" (8 words)
• "audit team will be reviewing payroll" (6 words)
• "Griffin should decline to assist with recruitment" (6 words)
• "decisions on behalf of management" (5 words)
• "shortage of cashflow / poor budgeting in future" (7 words)
• "inability to meet net zero goals" (6 words)

key_phrase mechanics:
• 5-8 words only. Shorter underlines look like ticks on stray articles; longer phrases span PDF lines and cannot be found.
• Verbatim substring of "text" (the student sentence).
• Two key_phrases from the same text must target STRICTLY DIFFERENT word spans — no word may appear in both.

GOOD (criterion keywords were ["shortage of cashflow", "budgeting"]):
   "shortage of cashflow/ poor" + "poor budgeting in future"
BAD overlap (the word "non-compliant" appears in both — REJECT):
   "non-compliant" + "Yeti may been non-compliant"
BAD overlap (same span, no anchor split):
   "shortage of cashflow/ poor budgeting" + "cashflow/ poor budgeting in future"

marks_awarded for the sub-question = count of correct_points × 0.5.
Cap at max_marks, then apply section_cap (Step 1). score = sum of all marks_awarded.

STEP 3.5 — SELF-AUDIT BEFORE RETURNING (do this for every sub-question):
For each sub-question you just built:
1. List the criteria you classified FULL or HALF (from STEP 2 reasoning).
2. Sum: for each FULL → +criterion.marks; for each HALF → +criterion.marks/2.
   That total is expected_marks.
3. Count your correct_points for this sub-question → tick_count.
4. Verify: tick_count == expected_marks / 0.5. If LESS, you under-marked — add
   the missing tick(s) on a different anchor word in the same sentence.
   If MORE, you over-marked — remove ticks until balanced.
5. Apply max_marks and section_cap LAST. Trim ticks to match the capped
   marks_awarded.
6. Final check: marks_awarded must EXACTLY equal len(correct_points) × 0.5.
   If not, fix one or the other before returning.

STEP 3.6 — TEACHER-PASS (theory only — last sanity check before returning):
Read each AWARDED criterion one more time with a teacher's mindset:
  a) Does the student's writing in this sub-question REALLY cover the
     model-answer point, or only gesture at the topic?
  b) Is the conclusion correct (e.g. "must refuse payroll" vs "can accept
     with safeguards")?
  c) For label-bearing points like "self-interest threat" — is the REASON
     the student gives the same as the model answer's reason?
If the answer to any of these is "no", downgrade FULL → HALF or HALF → ZERO,
and remove ticks accordingly. The teacher does NOT credit:
  • "external quality review" for "expanded review of audit work" (different
    word root) — downgrade to HALF at most, often ZERO if accompanied by
    other errors.
  • "sign a document showing responsibility" for "monthly comparison
    evidenced by signature" (signing the wrong thing) — HALF.
  • "self-interest threat" with the wrong reason — ZERO.
  • "can accept payroll with safeguards" for "must refuse payroll" — ZERO
    (entire "How to address" leaf goes to zero per Rule D).

After this teacher-pass, RE-VERIFY the marks_awarded == len(correct_points)
× 0.5 invariant before returning.

STEP 4 — IDENTIFY NOT-REQUIRED (OFF-TOPIC) CONTENT

Students sometimes include content that the question never asked for — definitions
of unrelated concepts, padding, tangents, or material from a different question.
Real teachers mark these areas with "Not required" so the student knows to drop
them in future answers.

For each off-topic sentence/passage in the student's answer:
• Output ONE entry in not_required_points with:
  - "text": the verbatim off-topic sentence/passage from the student answer
  - "key_phrase": a 3-6 word verbatim substring of "text" — the anchor where the
    "Not required" marker will be placed on the PDF
  - "reason": ONE short sentence explaining why this content is off-topic
    (e.g. "Question asks about audit procedures, not internal control design.")

Rules:
• not_required_points carry NO marks. They do NOT reduce marks_awarded.
• Do NOT mark a point as "not required" if it earned credit elsewhere (would never
  appear in correct_points AND not_required_points).
• Borderline / weakly relevant content → leave it out. Only flag CLEARLY off-topic.
• If the student is on-topic throughout, return an empty list.

═══════════════════════════════════════════════════
QUESTION INFORMATION
═══════════════════════════════════════════════════
{questions}

═══════════════════════════════════════════════════
MODEL ANSWER
═══════════════════════════════════════════════════
{model_data}

═══════════════════════════════════════════════════
STUDENT'S ANSWER
═══════════════════════════════════════════════════
{chunks}

═══════════════════════════════════════════════════
COMMENTS
═══════════════════════════════════════════════════
Array of strings, each formatted as:
"[<top_level_sub_question>] <3-5 word verbatim quote from student> → <what was wrong>. <one improvement>."

QUOTA — ONE comment per TOP-LEVEL sub-question where marks were lost.
• For EACH top-level sub-question that lost marks → emit EXACTLY ONE consolidated
  comment. Do not skip a sub-question that lost marks; do not emit two for one.
• If a top-level sub-question is FULLY credited (no marks lost) → omit it entirely.
• Prefix MUST be the TOP-LEVEL sub-question only ("4.1", "4.2", "4.3", "4.4") —
  NEVER a leaf label ("4.1(a) Consequences", "4.3 Payroll Threats" etc.).

ANCHOR REQUIREMENTS — these are how the PDF annotator finds where to place
the comment popup. Getting these wrong puts the comment in `unanchored_comments`
where the student never sees it.

• 3-5 words ONLY. Longer phrases (6+ words) often span PDF lines and the
  exact-match search fails. Pick the SHORTEST distinctive slice you can.
• MUST be a character-for-character verbatim substring of {chunks}. Copy
  exactly — preserve typos ("hsould", "biith", "specipitcal"), preserve
  punctuation, preserve casing. If you paraphrase or "clean up" the spelling,
  the search will not find the anchor in the PDF.
• MUST come from a single line in {chunks} — never join words across lines.
• MUST be from the SAME sub-question as the prefix. A `[4.3]` comment cannot
  anchor on a phrase the student wrote under 4.4.
• If you can't find a distinctive 3-5 word phrase, pick the first 3-4 words
  of the most relevant sentence in that sub-question — but never fabricate.

Other rules:
• No praise-only comments. Do not reveal the model answer.
• Use full form of words in comments, no abbreviations.

Example (good — top-level prefix, one comment for the whole sub-question):
"[4.1] Cost overruns and projects falling behind → Consequences are limited to
delays and customer dissatisfaction; missed work-in-progress overvaluation, breach
of contract, and net-zero/reputational impact. Recommendations also miss progress
reports to senior management and not processing claims that bypassed the website.
Cover the wider commercial and procedural consequences for full marks."

Example (BAD — leaf prefix, multiple comments per sub-question):
"[4.1(a) Consequences] Cost overruns → ..."  ← REJECT, must be "[4.1]"
"[4.1(a) Recommendations] All large variances → ..."  ← REJECT, redundant with above

═══════════════════════════════════════════════════
OUTPUT — return ONLY valid JSON
═══════════════════════════════════════════════════
{{
  "question_number": "<main question number>",
  "score": <sum of all marks_awarded>,
  "total_marks": <max marks for question>,
  "sub_grades": [
    {{
      "sub_question": "<identifier from model answer>",
      "student_label": "<verbatim label from student's answer, or empty string>",
      "marks_awarded": <number>,
      "max_marks": <number>,
      "reason": "<brief explanation>",
      "correct_points": [
        {{
          "text": "<verbatim student sentence>",
          "marks": 0.5,
          "key_phrase": "<3-6 words verbatim from text>"
        }}
      ],
      "not_required_points": [
        {{
          "text": "<verbatim off-topic sentence>",
          "key_phrase": "<3-6 words verbatim from text>",
          "reason": "<one short sentence why this is off-topic>"
        }}
      ]
    }}
  ],
  "comments": ["<quote → issue. improvement.>"]
}}

Constraints:
• One sub_grades entry per model answer sub-question.
• marks_awarded = count of correct_points × 0.5 (capped at max_marks).
• score = sum of all marks_awarded (capped at total_marks).
• Every correct_point has marks: 0.5. Every key_phrase is 3-6 words.
• student_label must be verbatim from {chunks} or empty string.
"""

holistic_grade_prompt = ChatPromptTemplate.from_template(HOLISTIC_GRADE_PROMPT_TEMPLATE)
