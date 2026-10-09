
import inspect
import json

import streamlit as st
from dotenv import load_dotenv

import vetcase_agent_old as backend

load_dotenv()

st.set_page_config(
    page_title="VetCase",
    page_icon="🐾",
    layout="wide",
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FIELDS = [
    "Small Animal",
    "Farm Animal",
    "Equine",
    "Exotic & Zoo",
    "General Veterinary Medicine",
]

TEXT = {
    "English": {
        "subtitle": "Veterinary diagnostic training for students",
        "language": "Language",
        "field": "Select a veterinary field",
        "generate": "Generate a new case",
        "generating": "Generating clinical case...",
        "intro": "Study the case and investigate before proposing your diagnoses.",
        "case": "Clinical Case",
        "signalment": "Signalment",
        "presentation": "Clinical Presentation",
        "findings": "Clinical Findings",
        "investigation": "Investigation",
        "question": "Ask a clinical investigation question",
        "ask": "Submit question",
        "history": "Investigation history",
        "finish_investigation": "Finish Investigation",
        "finish_investigation_help": (
            "When you have finished investigating the case, continue "
            "to your differential diagnoses."
        ),
        "differentials": "Differential Diagnoses",
        "diff_help": "Enter your differential diagnoses, one per line.",
        "submit_differentials": "Submit Differential Diagnoses",
        "final": "Final Diagnosis",
        "final_help": "Enter your most likely diagnosis.",
        "submit_final": "Submit Final Diagnosis",
        "evaluation": "Score & Feedback",
        "error": "An error occurred",
        "empty_question": "Please enter a question.",
        "empty_differentials": "Please enter at least one differential diagnosis.",
        "empty_final": "Please enter your final diagnosis.",
        "educational": (
            "Educational use only. VetCase does not replace professional "
            "veterinary examination or clinical judgment."
        ),
        "new_case_note": "Generate a new case to start an exercise.",
        "back_investigation": "Back to Investigation",
        "back_differentials": "Back to Differential Diagnoses",
        "total_score": "Total score",
        "correct_diagnosis": "Correct Diagnosis",
        "diagnosis_match": "Diagnosis Match",
        "feedback": "Feedback",
    },
    "Français": {
        "subtitle": "Entraînement au raisonnement diagnostique vétérinaire",
        "language": "Langue",
        "field": "Choisissez un domaine vétérinaire",
        "generate": "Générer un nouveau cas",
        "generating": "Génération du cas clinique...",
        "intro": "Étudiez le cas et réalisez les investigations avant de proposer vos diagnostics.",
        "case": "Cas clinique",
        "signalment": "Identification de l'animal",
        "presentation": "Présentation clinique",
        "findings": "Résultats cliniques",
        "investigation": "Investigations",
        "question": "Posez une question clinique",
        "ask": "Envoyer la question",
        "history": "Historique des investigations",
        "finish_investigation": "Terminer les investigations",
        "finish_investigation_help": (
            "Lorsque vous avez terminé les investigations, passez "
            "aux diagnostics différentiels."
        ),
        "differentials": "Diagnostics différentiels",
        "diff_help": "Saisissez vos diagnostics différentiels, un par ligne.",
        "submit_differentials": "Soumettre les diagnostics différentiels",
        "final": "Diagnostic final",
        "final_help": "Saisissez le diagnostic le plus probable.",
        "submit_final": "Soumettre le diagnostic final",
        "evaluation": "Note et commentaires",
        "error": "Une erreur s'est produite",
        "empty_question": "Veuillez saisir une question.",
        "empty_differentials": "Veuillez saisir au moins un diagnostic différentiel.",
        "empty_final": "Veuillez saisir votre diagnostic final.",
        "educational": (
            "Usage pédagogique uniquement. VetCase ne remplace pas un examen "
            "vétérinaire professionnel ni le jugement clinique."
        ),
        "new_case_note": "Générez un nouveau cas pour commencer un exercice.",
        "back_investigation": "Retour aux investigations",
        "back_differentials": "Retour aux diagnostics différentiels",
        "total_score": "Note totale",
        "correct_diagnosis": "Diagnostic correct",
        "diagnosis_match": "Correspondance du diagnostic",
        "feedback": "Commentaires",
    },
}


def get_text(key):
    language = st.session_state.get("vetcase_language", "English")
    return TEXT[language][key]


def json_safe(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    except Exception:
        return str(value)


def get_case_value(case, *keys):
    if not isinstance(case, dict):
        return None

    for key in keys:
        value = case.get(key)
        if value not in (None, "", [], {}):
            return value

    return None


def get_model_text(response):
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)

    if content is not None:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, dict):
                    item_text = item.get("text", "")
                    if item_text:
                        parts.append(str(item_text))
                else:
                    parts.append(str(item))

            return "\n".join(parts).strip()

        return str(content)

    if isinstance(response, dict):
        return str(
            response.get("answer")
            or response.get("response")
            or response.get("content")
            or response
        )

    return str(response)


# --------------------------------------------------
# DISPLAY HELPERS
# --------------------------------------------------

def normalize_label(key):
    """Convert backend keys into readable labels."""
    return str(key).replace("_", " ").strip().title()


def render_feedback(value):
    """
    Render feedback as readable text instead of displaying
    the raw Python list/dictionary representation.
    """

    if value is None:
        return

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return

        # Render ordinary text and Markdown normally.
        st.markdown(text)
        return

    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                render_feedback(item)
            elif item is not None and str(item).strip():
                st.markdown(f"- {str(item).strip()}")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if item is None or item == "":
                continue

            st.markdown(f"**{normalize_label(key)}**")

            if isinstance(item, (dict, list)):
                render_feedback(item)
            elif isinstance(item, bool):
                st.write("Yes" if item else "No")
            else:
                st.markdown(str(item))
        return

    st.write(value)


def render_evaluation_value(key, value):
    """Display scores, feedback, and other evaluation fields."""

    if value is None or value == "":
        return

    normalized_key = str(key).strip().lower().replace("-", "_").replace(" ", "_")

    feedback_keys = {
        "feedback",
        "comments",
        "comment",
        "overall_feedback",
        "clinical_feedback",
        "general_feedback",
        "reasoning_feedback",
        "differential_feedback",
        "diagnosis_feedback",
    }

    if normalized_key in feedback_keys or "feedback" in normalized_key:
        st.markdown(f"**{get_text('feedback') if normalized_key == 'feedback' else normalize_label(key)}**")
        render_feedback(value)
        return

    # Render structured score sections clearly.
    if isinstance(value, dict):
        st.markdown(f"**{normalize_label(key)}**")
        for subkey, subvalue in value.items():
            st.markdown(f"**{normalize_label(subkey)}**")
            render_evaluation_value(subkey, subvalue)
        return

    if isinstance(value, list):
        st.markdown(f"**{normalize_label(key)}**")
        render_feedback(value)
        return

    if isinstance(value, bool):
        readable_value = "Yes" if value else "No"
        st.markdown(f"**{normalize_label(key)}:** {readable_value}")
        return

    st.markdown(f"**{normalize_label(key)}**")
    st.write(value)


# --------------------------------------------------
# INVESTIGATION
# --------------------------------------------------

def answer_investigation_question(
    case,
    question,
    language,
    investigation_history=None,
):
    existing_function = getattr(
        backend,
        "answer_investigation_question",
        None,
    )

    if callable(existing_function):
        signature = inspect.signature(existing_function)

        available = {
            "case": case,
            "clinical_case": case,
            "question": question,
            "user_question": question,
            "language": language,
            "investigation_history": investigation_history or [],
        }

        kwargs = {
            name: available[name]
            for name, parameter in signature.parameters.items()
            if name in available
            and parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
            )
        }

        required_missing = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and name not in kwargs
        ]

        if not required_missing:
            return get_model_text(existing_function(**kwargs))

    model = getattr(backend, "model", None)

    if model is None:
        raise RuntimeError(
            "The backend does not expose a model or investigation function."
        )

    visible_case = backend.build_visible_case(case)

    prompt = f"""
You are VetCase, a veterinary diagnostic-training assistant.

Answer the student's investigation question using only information
available in the visible clinical record.

Rules:
- Respond in {language}.
- Do not reveal, confirm, or hint at the hidden diagnosis.
- Do not reveal hidden differential diagnoses or hidden key clues.
- Never invent history, examination findings, laboratory results,
  imaging results, or other clinical information.
- If the requested information is not available, say that it is not
  provided in the case and may need further investigation.
- You may suggest a relevant test, but clearly distinguish a suggested
  test from an actual result.
- Keep the answer concise and educational.

Visible clinical record:
{json_safe(visible_case)}

Previous investigation history:
{json_safe(investigation_history or [])}

Student's question:
{question}
"""

    return get_model_text(model.invoke(prompt))


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_student(
    case,
    investigation_history,
    differentials,
    final_diagnosis,
    language,
):
    evaluate_function = getattr(backend, "evaluate_case", None)

    if not callable(evaluate_function):
        raise RuntimeError(
            "evaluate_case() was not found in vetcase_agent_old.py. "
            "Check that the correct backend file is being imported."
        )

    if not isinstance(investigation_history, list):
        raise ValueError("Investigation history must be a list.")

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
            if item is not None and str(item).strip().lstrip("-• \t")
        ]

    else:
        raise ValueError(
            "Differential diagnoses must be a list or a text string."
        )

    if not differential_list:
        raise ValueError("Please submit your differential diagnoses.")

    if not isinstance(final_diagnosis, str) or not final_diagnosis.strip():
        raise ValueError("Please submit your final diagnosis.")

    return evaluate_function(
        case=case,
        investigation_history=investigation_history,
        differentials=differential_list,
        final_diagnosis=final_diagnosis.strip(),
        language=language,
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "vetcase_language": "English",
    "vetcase_current_case": None,
    "vetcase_visible_case": None,
    "vetcase_current_field": None,
    "vetcase_stage": "investigation",
    "vetcase_investigation_history": [],
    "vetcase_differentials": [],
    "vetcase_final_diagnosis": "",
    "vetcase_evaluation": None,
    "vetcase_case_error": None,
    "vetcase_investigation_error": None,
    "vetcase_evaluation_error": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# HEADER AND SELECTION
# --------------------------------------------------

st.title("🐾 VetCase")
st.caption(get_text("subtitle"))

language = st.selectbox(
    get_text("language"),
    ["English", "Français"],
    key="vetcase_language",
)

field = st.selectbox(
    get_text("field"),
    FIELDS,
    key="vetcase_selected_field",
)

st.info(get_text("intro"))


# --------------------------------------------------
# GENERATE CASE
# --------------------------------------------------

if st.button(get_text("generate"), type="primary"):
    with st.spinner(get_text("generating")):
        try:
            generated_case = backend.generate_case(field)
            visible_case = backend.build_visible_case(generated_case)

            st.session_state["vetcase_current_case"] = generated_case
            st.session_state["vetcase_visible_case"] = visible_case
            st.session_state["vetcase_current_field"] = field

            st.session_state["vetcase_stage"] = "investigation"
            st.session_state["vetcase_investigation_history"] = []
            st.session_state["vetcase_differentials"] = []
            st.session_state["vetcase_final_diagnosis"] = ""
            st.session_state["vetcase_evaluation"] = None

            st.session_state["vetcase_case_error"] = None
            st.session_state["vetcase_investigation_error"] = None
            st.session_state["vetcase_evaluation_error"] = None

        except Exception as exc:
            st.session_state["vetcase_case_error"] = str(exc)


if st.session_state["vetcase_case_error"]:
    st.error(
        f"{get_text('error')}: "
        f"{st.session_state['vetcase_case_error']}"
    )


# --------------------------------------------------
# DISPLAY CURRENT CASE
# --------------------------------------------------

case = st.session_state["vetcase_current_case"]
visible_case = st.session_state["vetcase_visible_case"]
stage = st.session_state["vetcase_stage"]

if case and visible_case:
    st.divider()
    st.header(get_text("case"))

    title = get_case_value(visible_case, "title", "case_title")
    if title:
        st.subheader(str(title))

    display_fields = [
        (("signalment", "patient", "animal"), "signalment"),
        (
            ("presentation", "clinical_presentation", "history"),
            "presentation",
        ),
        (
            ("clinical_findings", "findings", "examination_findings"),
            "findings",
        ),
    ]

    for keys, label_key in display_fields:
        value = get_case_value(visible_case, *keys)
        if value:
            st.markdown(f"**{get_text(label_key)}**")
            render_feedback(value)

    # ==================================================
    # STAGE 1 — INVESTIGATION
    # ==================================================

    if stage == "investigation":
        st.divider()
        st.header(get_text("investigation"))

        with st.form(
            "vetcase_investigation_form",
            clear_on_submit=True,
        ):
            question = st.text_area(
                get_text("question"),
                height=100,
            )
            ask_clicked = st.form_submit_button(get_text("ask"))

        if ask_clicked:
            if not question.strip():
                st.warning(get_text("empty_question"))
            else:
                with st.spinner(
                    "Preparing an answer..."
                    if language == "English"
                    else "Préparation de la réponse..."
                ):
                    try:
                        answer = answer_investigation_question(
                            case=case,
                            question=question.strip(),
                            language=language,
                            investigation_history=st.session_state[
                                "vetcase_investigation_history"
                            ],
                        )

                        st.session_state[
                            "vetcase_investigation_history"
                        ].append(
                            {
                                "question": question.strip(),
                                "answer": answer,
                            }
                        )

                        st.session_state[
                            "vetcase_investigation_error"
                        ] = None

                    except Exception as exc:
                        st.session_state[
                            "vetcase_investigation_error"
                        ] = str(exc)

        if st.session_state["vetcase_investigation_error"]:
            st.error(
                f"{get_text('error')}: "
                f"{st.session_state['vetcase_investigation_error']}"
            )

        history = st.session_state["vetcase_investigation_history"]

        if history:
            st.subheader(get_text("history"))

            for index, item in enumerate(history, start=1):
                with st.expander(
                    f"{index}. {item['question']}",
                    expanded=(index == len(history)),
                ):
                    render_feedback(item["answer"])

        st.info(get_text("finish_investigation_help"))

        if st.button(
            get_text("finish_investigation"),
            type="primary",
        ):
            st.session_state["vetcase_stage"] = "differentials"
            st.rerun()

    # ==================================================
    # STAGE 2 — DIFFERENTIAL DIAGNOSES
    # ==================================================

    elif stage == "differentials":
        st.divider()
        st.header(get_text("differentials"))

        with st.form("vetcase_differentials_form"):
            differential_text = st.text_area(
                get_text("diff_help"),
                value="\n".join(
                    st.session_state["vetcase_differentials"]
                ),
                height=130,
            )

            submit_differentials = st.form_submit_button(
                get_text("submit_differentials"),
                type="primary",
            )

        if submit_differentials:
            differential_list = [
                line.strip(" \t-•")
                for line in differential_text.splitlines()
                if line.strip(" \t-•")
            ]

            if not differential_list:
                st.warning(get_text("empty_differentials"))
            else:
                st.session_state["vetcase_differentials"] = differential_list
                st.session_state["vetcase_stage"] = "final"
                st.rerun()

        if st.button(get_text("back_investigation")):
            st.session_state["vetcase_stage"] = "investigation"
            st.rerun()

    # ==================================================
    # STAGE 3 — FINAL DIAGNOSIS
    # ==================================================

    elif stage == "final":
        st.divider()
        st.header(get_text("differentials"))

        for diagnosis in st.session_state["vetcase_differentials"]:
            st.markdown(f"- {diagnosis}")

        st.divider()
        st.header(get_text("final"))

        with st.form("vetcase_final_diagnosis_form"):
            final_diagnosis = st.text_input(
                get_text("final_help"),
                value=st.session_state["vetcase_final_diagnosis"],
            )

            submit_final = st.form_submit_button(
                get_text("submit_final"),
                type="primary",
            )

        if submit_final:
            if not final_diagnosis.strip():
                st.warning(get_text("empty_final"))
            else:
                with st.spinner(
                    "Evaluating your answers..."
                    if language == "English"
                    else "Évaluation de vos réponses..."
                ):
                    try:
                        result = evaluate_student(
                            case=case,
                            investigation_history=st.session_state[
                                "vetcase_investigation_history"
                            ],
                            differentials=st.session_state[
                                "vetcase_differentials"
                            ],
                            final_diagnosis=final_diagnosis.strip(),
                            language=language,
                        )

                        st.session_state[
                            "vetcase_final_diagnosis"
                        ] = final_diagnosis.strip()

                        st.session_state["vetcase_evaluation"] = result
                        st.session_state["vetcase_evaluation_error"] = None
                        st.session_state["vetcase_stage"] = "evaluation"

                        st.rerun()

                    except Exception as exc:
                        st.session_state[
                            "vetcase_evaluation_error"
                        ] = str(exc)

        if st.session_state["vetcase_evaluation_error"]:
            st.error(
                f"{get_text('error')}: "
                f"{st.session_state['vetcase_evaluation_error']}"
            )

        if st.button(get_text("back_differentials")):
            st.session_state["vetcase_stage"] = "differentials"
            st.rerun()

    # ==================================================
    # STAGE 4 — SCORE AND FEEDBACK
    # ==================================================

    elif stage == "evaluation":
        st.divider()
        st.header(get_text("evaluation"))

        st.markdown(f"**{get_text('differentials')}**")
        for diagnosis in st.session_state["vetcase_differentials"]:
            st.markdown(f"- {diagnosis}")

        st.markdown(f"**{get_text('final')}**")
        st.write(st.session_state["vetcase_final_diagnosis"])

        evaluation = st.session_state["vetcase_evaluation"]

        if isinstance(evaluation, dict):
            score_keys = {
                "total_score",
                "score",
                "total",
                "final_score",
            }

            score = get_case_value(
                evaluation,
                "total_score",
                "score",
                "total",
                "final_score",
            )

            if score is not None:
                st.metric(get_text("total_score"), str(score))

            # Display each evaluation field in a readable format.
            for key, value in evaluation.items():
                if key in score_keys:
                    continue

                render_evaluation_value(key, value)

        elif isinstance(evaluation, list):
            render_feedback(evaluation)

        else:
            render_feedback(evaluation)

    st.caption(get_text("educational"))

else:
    st.caption(get_text("new_case_note"))