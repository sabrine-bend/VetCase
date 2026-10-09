
import json
import re

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

model = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
    max_retries=0,
)

VETERINARY_FIELDS = [
    "Small Animal",
    "Farm Animal",
    "Equine",
    "Exotic & Zoo",
    "General Veterinary Medicine",
]

LANGUAGES = ["English", "Français"]

SPECIES_BY_FIELD = {
    "Small Animal": [
        "dogs", "cats", "rabbits", "guinea pigs", "hamsters",
        "rats", "mice", "ferrets",
    ],
    "Farm Animal": [
        "cattle", "sheep", "goats", "pigs", "chickens",
        "turkeys", "ducks", "geese", "farm rabbits",
        "farm camelids", "buffalo",
    ],
    "Equine": [
        "horses", "ponies", "donkeys", "mules",
    ],
    "Exotic & Zoo": [
        "parrots", "passerine birds", "raptors", "reptiles",
        "snakes", "lizards", "tortoises", "turtles",
        "amphibians", "primates", "wild mammals",
        "zoo animals", "marine mammals", "ornamental fish",
        "aquatic animals", "wild birds",
    ],
    "General Veterinary Medicine": [
        "dogs", "cats", "cattle", "sheep", "goats", "pigs",
        "horses", "donkeys", "poultry", "rabbits", "rodents",
        "pet birds", "reptiles", "wildlife", "aquatic animals",
    ],
}


# ============================================================
# 2. CASE GENERATION PROMPT
# ============================================================

CASE_GENERATION_PROMPT = """
You are an experienced veterinary clinician and clinical educator.

Generate ONE realistic, internally consistent veterinary clinical case.

Selected field: __FIELD__

Species appropriate for this field:
__SPECIES__

Choose ONE suitable species. Do not choose a species outside the
selected field unless the field is General Veterinary Medicine.

Return ONLY one valid JSON object with exactly these top-level fields:

{
  "title": "...",
  "signalment": "...",
  "presentation": "...",
  "clinical_findings": "...",
  "hidden_diagnosis": "...",
  "differentials": ["...", "...", "..."],
  "key_clues": ["...", "...", "...", "..."],
  "hidden_case_data": {
    "history": {
      "diet": "...",
      "medications": "...",
      "previous_illnesses": "...",
      "vaccination_or_prevention": "...",
      "onset_and_progression": "...",
      "other_relevant_history": "..."
    },
    "additional_clinical_information": [
      {
        "topic": "...",
        "information": "..."
      }
    ],
    "diagnostic_tests": {
      "performed": [
        {
          "test": "...",
          "result": "..."
        }
      ],
      "not_performed": ["..."]
    },
    "other_relevant_information": ["..."]
  }
}

IMPORTANT JSON REQUIREMENTS:
- Return a JSON object, not a list or a wrapped object.
- Use the exact field names shown above.
- Include every required field.
- Use double quotes for all JSON keys and strings.
- Do not include Markdown fences or commentary.
- Do not leave any required value empty or null.
- Ensure the JSON is syntactically valid.

VISIBLE CASE RULES:
- Use clear English.
- Respect the selected field and species physiology.
- Include realistic signalment, duration, history and examination findings.
- Include useful positive and negative findings.
- Do not reveal the diagnosis in the title or visible presentation.
- Choose one defensible most likely diagnosis.
- Provide exactly 3 clinically plausible differential diagnoses.
- Provide exactly 4 key clues supported by the visible case.
- The visible case contains only title, signalment, presentation,
  and clinical_findings.
- Do not claim a diagnosis is definitively confirmed without evidence.

HIDDEN CLINICAL RECORD:
- Provide a coherent, sufficiently detailed fictional patient record.
- Include relevant history, exposures, previous illness, prevention,
  medications, diet, appetite, drinking, elimination and progression
  when appropriate for the species.
- Include additional examination findings that can be discovered
  through reasonable clinical questions.
- Include relevant positive and negative findings.
- Avoid vague statements such as "unknown" when realistic details can
  be provided.
- If information is genuinely unavailable or not applicable, state so.
- The hidden diagnosis, diagnosis name, answer key and explicit diagnostic
  conclusion must not appear in hidden_case_data.
- Do not contradict the visible case.
- Tests in "performed" must have actual recorded results.
- Tests in "not_performed" have no results.
- Never invent test results during investigation.
- Use realistic results consistent with the patient and diagnosis.
- Include enough evidence to make the hidden diagnosis defensible.
- Include evidence that helps distinguish it from competing diagnoses.
- If the evidence only supports a suspected diagnosis, make that limitation
  clear in the record.
- Do not make an unperformed test the only possible way to reason about
  the case when reasonable clinical evidence is available.
- Do not create contradictory or irrelevant test results.
- Ensure the diagnosis is consistent with species, age, clinical course,
  examination and recorded test results.
- Every performed test must contain a test name and a non-empty result.
- Every not_performed entry must be a test name, not a result.
- Do not list the same test as both performed and not performed.
- All history values must be non-empty strings.

SPECIES-SPECIFIC RULES:
- Horses, donkeys and mules cannot vomit normally. Never describe vomiting
  or emesis as a normal clinical sign in equids.
- Colic is a clinical sign of abdominal pain, not a specific diagnosis.
- Respect ruminant anatomy, rumination and digestive physiology.
- Distinguish vomiting from regurgitation in dogs and cats.
- Respect rabbit gastrointestinal physiology and their inability to vomit.
- Use species-appropriate signs, handling and examination for birds.
- Respect reptile temperature dependence, husbandry and species-specific
  physiology.
- Use appropriate species-specific terminology for amphibians and fish.
- Do not automatically apply dog or cat assumptions to exotic species.
- Do not treat all birds, reptiles, fish or wildlife as physiologically
  identical.
- For production animals, consider herd/flock context when relevant.
- Use realistic husbandry, diet, housing and environmental exposures.
- Respect zoonotic and occupational risks where clinically relevant.
- Avoid unsafe or implausible handling procedures.

CASE VARIETY:
- Prefer common, educational conditions.
- Vary species, conditions, clinical presentations and investigations.
- Avoid generating only dogs and cats across cases.
- Include farm, avian, equine, exotic, wildlife and aquatic cases when
  appropriate to the selected field.
- Vary age, sex, breed, clinical setting and case severity.
- Include normal findings where useful.
- Avoid revealing the diagnosis through an obvious title.
- Ensure all information describes the same patient and episode.

Return ONLY the JSON object. No markdown or commentary.
"""


# ============================================================
# 3. JSON AND MODEL HELPERS
# ============================================================

def _extract_json(content):
    """Extract a JSON object from a model response."""

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))

        content = "\n".join(parts)

    text = str(content).strip()

    if not text:
        raise ValueError("The model returned an empty response.")

    # Remove Markdown code fences if present.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    decoder = json.JSONDecoder()
    candidates = []

    # First, try to parse the entire response.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            candidates.append(parsed)

    except json.JSONDecodeError:
        pass

    # If extra text exists, search for JSON objects.
    for index, character in enumerate(text):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(text[index:])

            if isinstance(parsed, dict):
                candidates.append(parsed)

        except json.JSONDecodeError:
            continue

    if not candidates:
        raise ValueError(
            "The model did not return a valid JSON object."
        )

    # Support common wrappers such as {"case": {...}}.
    wrapper_keys = (
        "case",
        "clinical_case",
        "generated_case",
        "result",
        "data",
    )

    possible_objects = []

    for candidate in candidates:
        possible_objects.append(candidate)

        for key in wrapper_keys:
            nested = candidate.get(key)

            if isinstance(nested, dict):
                possible_objects.append(nested)

    # Prefer the object containing the most required case fields.
    required_fields = globals().get("CASE_FIELDS", [])

    if required_fields:
        return max(
            possible_objects,
            key=lambda obj: sum(
                field in obj for field in required_fields
            ),
        )

    return possible_objects[-1]


def _call_model(prompt):
    """Call the configured model and parse its JSON response."""

    response = model.invoke(prompt)
    return _extract_json(response.content)


def _call_text_model(prompt):
    """Call the configured model and return plain text."""

    response = model.invoke(prompt)
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")

                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)

                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content).strip()


# ============================================================
# 4. CASE VALIDATION
# ============================================================

CASE_FIELDS = [
    "title",
    "signalment",
    "presentation",
    "clinical_findings",
    "hidden_diagnosis",
    "differentials",
    "key_clues",
    "hidden_case_data",
]


def _validate_case(case):
    """Validate the generated case structure and clinical record."""

    if not isinstance(case, dict):
        raise ValueError("The generated case must be an object.")

    missing = [
        field for field in CASE_FIELDS
        if field not in case
    ]

    if missing:
        raise ValueError(
            "Generated case is missing fields: "
            + ", ".join(missing)
        )

    # Required top-level strings.
    for field in CASE_FIELDS[:5]:
        value = case[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Case field '{field}' must be a non-empty string."
            )

    # Differential diagnoses.
    if not isinstance(case["differentials"], list):
        raise ValueError("Differentials must be a list.")

    if len(case["differentials"]) != 3:
        raise ValueError(
            "Exactly 3 differential diagnoses are required."
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in case["differentials"]
    ):
        raise ValueError(
            "Each differential must be a non-empty string."
        )

    # Key clues.
    if not isinstance(case["key_clues"], list):
        raise ValueError("Key clues must be a list.")

    if len(case["key_clues"]) != 4:
        raise ValueError("Exactly 4 key clues are required.")

    if not all(
        isinstance(item, str) and item.strip()
        for item in case["key_clues"]
    ):
        raise ValueError(
            "Each key clue must be a non-empty string."
        )

    # Hidden case record.
    hidden_data = case["hidden_case_data"]

    if not isinstance(hidden_data, dict):
        raise ValueError(
            "hidden_case_data must be an object."
        )

    required_hidden_fields = [
        "history",
        "additional_clinical_information",
        "diagnostic_tests",
        "other_relevant_information",
    ]

    missing_hidden = [
        field for field in required_hidden_fields
        if field not in hidden_data
    ]

    if missing_hidden:
        raise ValueError(
            "Hidden case data is missing: "
            + ", ".join(missing_hidden)
        )

    # History.
    history = hidden_data["history"]

    if not isinstance(history, dict):
        raise ValueError(
            "Hidden clinical history must be an object."
        )

    required_history_fields = [
        "diet",
        "medications",
        "previous_illnesses",
        "vaccination_or_prevention",
        "onset_and_progression",
        "other_relevant_history",
    ]

    missing_history = [
        field for field in required_history_fields
        if field not in history
    ]

    if missing_history:
        raise ValueError(
            "History is missing: "
            + ", ".join(missing_history)
        )

    for key, value in history.items():
        if not isinstance(key, str):
            raise ValueError(
                "History keys must be strings."
            )

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"History entry '{key}' must be a non-empty string."
            )

    # Additional clinical information.
    additional = hidden_data[
        "additional_clinical_information"
    ]

    if not isinstance(additional, list):
        raise ValueError(
            "Additional clinical information must be a list."
        )

    for item in additional:
        if not isinstance(item, dict):
            raise ValueError(
                "Each additional clinical detail must be an object."
            )

        topic = item.get("topic")
        information = item.get("information")

        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(
                "Each additional clinical detail needs a topic."
            )

        if (
            not isinstance(information, str)
            or not information.strip()
        ):
            raise ValueError(
                "Each additional clinical detail needs information."
            )

    # Diagnostic tests.
    tests = hidden_data["diagnostic_tests"]

    if not isinstance(tests, dict):
        raise ValueError(
            "Diagnostic tests must be an object."
        )

    if "performed" not in tests or "not_performed" not in tests:
        raise ValueError(
            "Diagnostic tests must contain performed and not_performed."
        )

    if not isinstance(tests["performed"], list):
        raise ValueError("Performed tests must be a list.")

    if not isinstance(tests["not_performed"], list):
        raise ValueError("Not-performed tests must be a list.")

    for item in tests["performed"]:
        if not isinstance(item, dict):
            raise ValueError(
                "Each performed test must be an object."
            )

        test_name = item.get("test")
        result = item.get("result")

        if not isinstance(test_name, str) or not test_name.strip():
            raise ValueError(
                "Each performed test needs a non-empty test name."
            )

        if not isinstance(result, str) or not result.strip():
            raise ValueError(
                f"Performed test '{test_name}' needs a recorded result."
            )

    for item in tests["not_performed"]:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "Each not-performed test must be a non-empty string."
            )

    performed_names = {
        re.sub(r"\s+", " ", item["test"].strip().lower())
        for item in tests["performed"]
    }

    not_performed_names = {
        re.sub(r"\s+", " ", item.strip().lower())
        for item in tests["not_performed"]
    }

    overlap = performed_names.intersection(not_performed_names)

    if overlap:
        raise ValueError(
            "A test cannot be both performed and not performed: "
            + ", ".join(sorted(overlap))
        )

    # Other relevant information.
    other_information = hidden_data[
        "other_relevant_information"
    ]

    if not isinstance(other_information, list):
        raise ValueError(
            "Other relevant information must be a list."
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in other_information
    ):
        raise ValueError(
            "Other relevant information entries must be "
            "non-empty strings."
        )

    # Species-specific safety check.
    visible_text = " ".join(
        case[key]
        for key in (
            "title",
            "signalment",
            "presentation",
            "clinical_findings",
        )
    ).lower()

    is_equine = any(
        re.search(rf"\b{re.escape(word)}\b", visible_text)
        for word in (
            "horse", "mare", "stallion", "foal",
            "equine", "gelding", "donkey", "mule", "pony",
        )
    )

    if is_equine and re.search(
        r"\b(vomiting|vomited|vomit|emesis)\b",
        visible_text,
    ):
        raise ValueError(
            "Invalid equine case: equids cannot vomit normally."
        )

    # Key clues must be supported by visible case information.
    visible_evidence = (
        case["presentation"].lower()
        + " "
        + case["clinical_findings"].lower()
    )

    test_patterns = {
        "radiograph": (
            r"\b(radiograph|radiographs|x-ray|x-rays)\b"
        ),
        "ultrasound": (
            r"\b(ultrasound|ultrasonography|sonogram)\b"
        ),
        "blood test": (
            r"\b(blood test|bloodwork|hematology|haematology|"
            r"biochemistry)\b"
        ),
        "urinalysis": (
            r"\b(urinalysis|urine analysis)\b"
        ),
        "biopsy": (
            r"\b(biopsy|histopathology)\b"
        ),
        "culture": (
            r"\b(culture result|culture showed|culture grew)\b"
        ),
    }

    for clue in case["key_clues"]:
        clue_lower = clue.lower()

        for test_name, pattern in test_patterns.items():
            if re.search(pattern, clue_lower):
                if not re.search(pattern, visible_evidence):
                    raise ValueError(
                        f"A key clue mentions {test_name}, but the "
                        "investigation is not documented in the visible case."
                    )

    return case


# ============================================================
# 5. GENERATE CASE
# ============================================================

def generate_case(field):
    """Generate a veterinary case with validation and repair attempts."""

    if field not in VETERINARY_FIELDS:
        raise ValueError(
            f"Unknown veterinary field: {field}"
        )

    species = SPECIES_BY_FIELD.get(field, [])
    species_text = ", ".join(species)

    base_prompt = (
        CASE_GENERATION_PROMPT
        .replace("__FIELD__", field)
        .replace("__SPECIES__", species_text)
    )

    last_error = None
    previous_response = None

    # One initial generation and two repair attempts.
    for attempt in range(3):
        if attempt == 0:
            prompt = base_prompt
        else:
            prompt = f"""
{base_prompt}

IMPORTANT: The previous response failed validation.

Validation error:
{last_error}

Previous response:
{previous_response}

Correct the specific validation error and return a complete case.

Do not return only the missing fields. Return the entire JSON object.

Check carefully:
- Every required top-level field is present.
- Field names match the original schema exactly.
- Every required field has the correct data type.
- There are exactly 3 differential diagnoses.
- There are exactly 4 key clues.
- Every required history field is present and non-empty.
- hidden_case_data contains every required section.
- Each performed test has a recorded result.
- Each not_performed entry is a test name.
- No test is both performed and not performed.
- All clinical information is consistent with the selected species.
- The JSON is valid and contains no Markdown.

Return ONLY the corrected JSON object.
"""

        try:
            response = model.invoke(prompt)
            previous_response = response.content

            case = _extract_json(previous_response)
            return _validate_case(case)

        except Exception as error:
            last_error = error

    raise ValueError(
        "Could not generate a valid case after 3 attempts. "
        f"Last error: {last_error}"
    )


def build_visible_case(case):
    """Return only the information intended for the student."""

    return {
        "title": case["title"],
        "signalment": case["signalment"],
        "presentation": case["presentation"],
        "clinical_findings": case["clinical_findings"],
    }


# ============================================================
# 6. CLINICAL INVESTIGATION
# ============================================================

def answer_investigation_question(
    case,
    question,
    investigation_history=None,
    language="English",
):
    """Answer clinical questions using the established patient record."""

    if not isinstance(case, dict):
        raise ValueError("Case data must be a dictionary.")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please enter a clinical question.")

    if language not in LANGUAGES:
        raise ValueError("Unsupported language.")

    visible_case = build_visible_case(case)
    history = investigation_history or []
    hidden_data = case.get("hidden_case_data", {})

    language_instruction = (
        "Answer in clear English."
        if language == "English"
        else (
            "Répondez en français clair et adapté "
            "aux étudiants vétérinaires."
        )
    )

    prompt = f"""
You are a veterinary clinical teaching assistant conducting an interactive
clinical investigation with a veterinary student.

LANGUAGE:
{language_instruction}

VISIBLE CASE:
{json.dumps(visible_case, indent=2, ensure_ascii=False)}

PRIVATE CLINICAL RECORD:
{json.dumps(hidden_data, indent=2, ensure_ascii=False)}

PREVIOUS INVESTIGATION QUESTIONS AND ANSWERS:
{json.dumps(history, indent=2, ensure_ascii=False)}

STUDENT'S NEW QUESTION:
{question}

STRICT EVIDENCE RULES:
1. Treat the PRIVATE CLINICAL RECORD as the sole source of patient-specific
   facts. It is a fixed record, not an invitation to invent missing details.
2. Use only facts explicitly present in the record.
3. For a performed diagnostic test, report only its recorded result.
   Do not add measurements, findings or details not explicitly recorded.
4. For a test listed under not_performed, state that it has not been
   performed and no result is available.
5. If a test is not listed as performed or not performed, say its status
   is not documented.
6. Never invent findings, history, results, laboratory values, imaging
   descriptions, medications or exposures.
7. If a detail is absent, say it is not documented. Do not infer a
   specific result from the suspected diagnosis.
8. Previous assistant answers are not evidence. If a previous answer
   conflicts with the record, correct it using the record.
9. You may recommend an examination or test to clarify an unresolved
   question, but label it as a recommendation, not an existing result.
10. Do not reveal or confirm the hidden diagnosis or answer key.
11. Do not disclose hidden differentials or key clues.
12. Do not make a definitive diagnosis or prescribe treatment.
13. Keep the answer concise, relevant and educational.
14. If the record does not contain enough information, say so.
15. Clearly distinguish recorded findings from possible explanations.

Return plain text only.
"""

    answer = _call_text_model(prompt)

    if not answer:
        raise ValueError(
            "The model returned an empty investigation answer."
        )

    return answer


# ============================================================
# 7. EVALUATION PROMPT
# ============================================================

EVALUATION_PROMPT = """
You are an experienced veterinary clinical educator.

Evaluate the student's clinical investigation, differential diagnoses,
and final diagnosis.

LANGUAGE FOR FEEDBACK: __LANGUAGE__

FULL CASE DATA (evaluation only):
__CASE__

STUDENT'S INVESTIGATION HISTORY:
__HISTORY__

STUDENT'S DIFFERENTIAL DIAGNOSES:
__DIFFERENTIALS__

STUDENT'S FINAL DIAGNOSIS:
__FINAL_DIAGNOSIS__

SCORING: EXACTLY 60 POINTS
==========================

A. CLINICAL REASONING: 0–20
Assess relevance and quality of the actual questions, history-taking,
examination requests, investigation choices, interpretation of findings,
and recognition of uncertainty.

B. DIFFERENTIAL DIAGNOSES: 0–20
Assess relevance, plausibility, coverage and consistency with the evidence.
Do not require an exact match to the answer-key list.

C. FINAL DIAGNOSIS: 0–20
Assess the submitted diagnosis against the hidden reference diagnosis
AND the evidence available from the visible case and investigation.

Calibration:
- 19–20: Correct diagnosis, accepted synonym, or clinically equivalent term.
- 15–18: Very close diagnosis with only a minor distinction.
- 10–14: Reasonable alternative supported by important findings, but not
  equivalent to the reference diagnosis.
- 5–9: Partially related diagnosis with limited evidence or contradictions.
- 1–4: Weakly related diagnosis with little supporting evidence.
- 0: Unrelated, nonsensical, or clearly indefensible diagnosis.

Fairness:
- Score only actual submitted work.
- Accept recognized synonyms and equivalent terminology.
- Do not award full marks for unsupported guesses.
- Reward relevant questions even if information is absent or a test was
  not performed.
- Do not expect knowledge of hidden facts before the student asks.
- Do not penalize missing results for tests that were not performed.
- Do not lower investigation or differential scores solely because the
  final diagnosis is wrong.
- Do not treat a suspected diagnosis as definitively confirmed.
- Do not invent evidence.
- If the case does not distinguish two diagnoses adequately, acknowledge
  uncertainty and score fairly.
- A diagnosis appearing in the differential list is not automatically
  correct as a final diagnosis.
- If diagnosis_match is false, final_diagnosis_score must be below 20.
- A clinically reasonable alternative deserves proportionate partial credit.
- total_score must equal the three subscores exactly.

FEEDBACK:
Return exactly 3 concise points:
1. What the student did well.
2. What could be improved, with a specific and fair suggestion.
3. One important clinical learning point.

Keep each point to one or two short sentences.
Do not criticize missing unavailable information.
Use the selected language for all feedback and the correct diagnosis label.

OUTPUT:
Return ONLY valid JSON in this exact structure:

{
  "clinical_reasoning_score": 0,
  "differentials_score": 0,
  "final_diagnosis_score": 0,
  "total_score": 0,
  "correct_diagnosis": "...",
  "diagnosis_match": false,
  "feedback": [
    "...",
    "...",
    "..."
  ]
}

No markdown or commentary.
"""


# ============================================================
# 8. VALIDATE EVALUATION
# ============================================================

def _validate_evaluation(result):
    """Validate score ranges, totals and feedback structure."""

    if not isinstance(result, dict):
        raise ValueError(
            "The evaluation response must be a JSON object."
        )

    score_fields = [
        "clinical_reasoning_score",
        "differentials_score",
        "final_diagnosis_score",
        "total_score",
    ]

    for field in score_fields:
        value = result.get(field)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Invalid evaluation score: {field}"
            )

        if not float(value).is_integer():
            raise ValueError(
                f"Score must be an integer: {field}"
            )

        result[field] = int(value)

    for field in score_fields[:3]:
        if not 0 <= result[field] <= 20:
            raise ValueError(
                f"{field} must be between 0 and 20."
            )

    expected_total = sum(
        result[field] for field in score_fields[:3]
    )

    if result["total_score"] != expected_total:
        raise ValueError(
            "The total score does not equal the sum "
            "of the three scores."
        )

    if not 0 <= result["total_score"] <= 60:
        raise ValueError(
            "Total score must be between 0 and 60."
        )

    if (
        not isinstance(result.get("correct_diagnosis"), str)
        or not result["correct_diagnosis"].strip()
    ):
        raise ValueError(
            "Missing correct diagnosis in evaluation."
        )

    if not isinstance(result.get("diagnosis_match"), bool):
        raise ValueError(
            "diagnosis_match must be true or false."
        )

    if (
        result["diagnosis_match"] is False
        and result["final_diagnosis_score"] == 20
    ):
        raise ValueError(
            "The final diagnosis score cannot be 20 when "
            "diagnosis_match is false."
        )

    feedback = result.get("feedback")

    if not isinstance(feedback, list) or len(feedback) != 3:
        raise ValueError(
            "Evaluation must contain exactly 3 feedback points."
        )

    if not all(
        isinstance(point, str) and point.strip()
        for point in feedback
    ):
        raise ValueError(
            "Each feedback point must be a non-empty string."
        )

    return result


# ============================================================
# 9. EVALUATE THE COMPLETE CASE
# ============================================================

def evaluate_case(
    case,
    investigation_history,
    differentials,
    final_diagnosis,
    language="English",
):
    """
    Evaluate clinical reasoning, differential diagnoses and final diagnosis.

    Accepts differentials as a list or newline-separated text.
    Keeps the function signature compatible with the Streamlit interface.
    """

    if language not in LANGUAGES:
        raise ValueError("Unsupported language.")

    if not isinstance(case, dict):
        raise ValueError("Case data must be a dictionary.")

    if not isinstance(investigation_history, list):
        raise ValueError(
            "Investigation history must be a list."
        )

    if isinstance(differentials, str):
        differential_list = [
            line.strip().lstrip("-• \t")
            for line in differentials.splitlines()
            if line.strip().lstrip("-• \t")
        ]

    elif isinstance(differentials, list):
        differential_list = [
            str(item).strip().lstrip("-• \t")
            for item in differentials
            if item is not None
            and str(item).strip().lstrip("-• \t")
        ]

    else:
        raise ValueError(
            "Differential diagnoses must be a list or a text string."
        )

    if not differential_list:
        raise ValueError(
            "Please submit your differential diagnoses."
        )

    if (
        not isinstance(final_diagnosis, str)
        or not final_diagnosis.strip()
    ):
        raise ValueError(
            "Please submit your final diagnosis."
        )

    case_json = json.dumps(
        case,
        indent=2,
        ensure_ascii=False,
    )

    history_json = json.dumps(
        investigation_history,
        indent=2,
        ensure_ascii=False,
    )

    differentials_text = "\n".join(
        f"- {diagnosis}" for diagnosis in differential_list
    )

    prompt = (
        EVALUATION_PROMPT
        .replace("__LANGUAGE__", language)
        .replace("__CASE__", case_json)
        .replace("__HISTORY__", history_json)
        .replace("__DIFFERENTIALS__", differentials_text)
        .replace("__FINAL_DIAGNOSIS__", final_diagnosis.strip())
    )

    result = _call_model(prompt)

    return _validate_evaluation(result)


# ============================================================
# 10. OPTIONAL BACKEND TEST
# ============================================================

if __name__ == "__main__":
    print("VetCase backend")
    print("================")

    print("\nAvailable fields:")

    for field in VETERINARY_FIELDS:
        print("-", field)

    print("\nSpecies examples by field:")

    for field, species in SPECIES_BY_FIELD.items():
        print(f"- {field}: {', '.join(species)}")

    print("\nGenerating an Equine test case...")

    try:
        test_case = generate_case("Equine")

        print("\nStudent-visible case:")
        print(
            json.dumps(
                build_visible_case(test_case),
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nHidden clinical record generated:")

        hidden_data = test_case.get("hidden_case_data", {})

        print(
            "History sections:",
            len(hidden_data.get("history", {})),
        )

        print(
            "Additional clinical details:",
            len(
                hidden_data.get(
                    "additional_clinical_information", []
                )
            ),
        )

        print(
            "Performed tests:",
            len(
                hidden_data.get("diagnostic_tests", {}).get(
                    "performed", []
                )
            ),
        )

        print("\nBackend validation passed.")

    except Exception as error:
        print("\nCase generation failed:")
        print(error)