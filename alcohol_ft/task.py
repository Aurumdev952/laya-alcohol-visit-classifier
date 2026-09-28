"""One fixed question shared by training, calibration, and inference."""

LABELS = ("direct", "indirect", "negative")
QUESTION_VERSION = "v1-three-way"
QUESTION = {
    "alcohol_attribution": {
        "type": "choice",
        "instructions": (
            "Read all notes from this hospital visit as one case. Classify alcohol "
            "attribution. Choose direct if any note documents the patient's own "
            "alcohol use, intoxication, alcohol use disorder, withdrawal, or a "
            "condition attributed to the patient's alcohol use, including past use. "
            "Otherwise choose indirect if alcohol use by someone else caused or "
            "contributed to the patient's presentation, including prenatal exposure. "
            "Otherwise choose negative. A denial, screening question, negative test, "
            "family history without a causal link, or an alcohol-containing product "
            "alone is negative. Direct takes precedence over indirect across notes."
        ),
        "criteria": {
            "direct": "Patient's own alcohol use or alcohol-attributed condition is documented.",
            "indirect": "Another person's alcohol use causally affected this patient; no direct evidence.",
            "negative": "No qualifying direct or indirect alcohol attribution is documented.",
        },
    }
}

DECOMPOSED_QUESTIONS = {
    "patient_alcohol": {
        "type": "choice",
        "instructions": (
            "Across all notes for this visit, is there affirmative documentation of the "
            "patient's own alcohol use, alcohol use disorder, intoxication, withdrawal, "
            "or a condition attributed to the patient's drinking, including past use? "
            "A denial, question, negative test, family history, or alcohol-containing "
            "product alone does not establish the patient's own use."
        ),
        "criteria": {
            "documented": "Qualifying personal alcohol involvement is affirmatively documented.",
            "not_documented": "No qualifying personal alcohol involvement is documented.",
        },
    },
    "other_person_causal": {
        "type": "choice",
        "instructions": (
            "Across all notes for this visit, did another person's alcohol use cause "
            "or contribute to this patient's presentation, including prenatal exposure? "
            "An unrelated family history or another person's drinking without a causal "
            "link does not qualify. Answer independently of whether the patient also drank."
        ),
        "criteria": {
            "causal": "Another person's drinking causally affected this patient.",
            "not_causal": "No qualifying causal involvement by another person's drinking.",
        },
    },
}


def questions_for(mode: str) -> dict:
    if mode == "three_way":
        return QUESTION
    if mode == "decomposed":
        return DECOMPOSED_QUESTIONS
    raise ValueError(f"unknown question mode: {mode}")


def target_for(row: dict, question_id: str) -> str:
    if question_id == "alcohol_attribution":
        return row["label"]
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError("decomposed questions require annotation")
    if question_id == "patient_alcohol":
        value = annotation.get("patient_evidence")
        if not isinstance(value, bool):
            raise ValueError("patient_evidence must be boolean")
        return "documented" if value else "not_documented"
    if question_id == "other_person_causal":
        value = annotation.get("other_person_causal")
        if not isinstance(value, bool):
            raise ValueError("other_person_causal must be boolean")
        return "causal" if value else "not_causal"
    raise ValueError(f"unknown question: {question_id}")


def combine_components(patient_documented: bool, other_person_causal: bool) -> str:
    if patient_documented:
        return "direct"
    if other_person_causal:
        return "indirect"
    return "negative"


def combine_probabilities(patient_probability: float, other_probability: float) -> dict[str, float]:
    """Diagnostic joint estimate; assumes independence and requires separate calibration."""
    if not 0 <= patient_probability <= 1 or not 0 <= other_probability <= 1:
        raise ValueError("component probabilities must be in [0, 1]")
    return {"direct": patient_probability,
            "indirect": (1 - patient_probability) * other_probability,
            "negative": (1 - patient_probability) * (1 - other_probability)}


def case_state(encounters: list[dict]) -> str:
    if not encounters:
        raise ValueError("a case needs at least one encounter")
    parts = []
    for index, encounter in enumerate(encounters, 1):
        if not isinstance(encounter, dict) or not isinstance(encounter.get("note"), str):
            raise ValueError("each encounter needs a text note")
        note = encounter["note"].strip()
        if not note:
            raise ValueError("encounter note is blank")
        kind = encounter.get("type", "clinical note")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("encounter type must be text")
        parts.append(f"Encounter {index} ({kind}):\n{note}")
    return "Hospital visit case report\n\n" + "\n\n".join(parts)
