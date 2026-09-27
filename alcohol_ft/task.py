"""One fixed question shared by training, calibration, and inference."""

LABELS = ("direct", "indirect", "negative")
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
