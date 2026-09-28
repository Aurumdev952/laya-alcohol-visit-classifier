"""Generate 30,000 paired synthetic visits for attribution experiments.

These are development examples, not clinically validated records. All three
variants of a source scenario stay in one split. No model is needed to run this.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from alcohol_ft.data import validate_splits, write_jsonl
from alcohol_ft.task import LABELS, combine_components

SEED = 20260928
SPLITS = {"train": 7000, "validation": 1000, "calibration": 1000, "test": 1000}
FAMILIES = (
    ("wrist", "wrist pain and swelling"), ("ankle", "ankle injury"),
    ("scalp", "scalp laceration"), ("shoulder", "shoulder pain"),
    ("knee", "knee contusion"), ("ribs", "rib pain"),
    ("face", "facial bruising"), ("neck", "neck pain"),
    ("back", "lower back pain"), ("hand", "hand injury"),
    ("foot", "foot pain"), ("elbow", "elbow pain"),
    ("hip", "hip pain"), ("finger", "finger injury"),
    ("shin", "shin contusion"), ("forearm", "forearm pain"),
    ("developmental", "developmental concerns"),
    # These three presentation families occur only in the locked test.
    ("thigh", "thigh pain"), ("jaw", "jaw pain"),
    ("neurodevelopment", "cognitive and developmental findings"),
)
SEEN_FAMILIES = FAMILIES[:17]
UNSEEN_FAMILIES = FAMILIES[17:]

DIRECT = (
    "Patient says they drank {amount} of {drink} {when}.",
    "Social history records {amount} of {drink} {when} for this patient.",
    "Patient describes drinking {drink} {when}; intake is {amount}.",
    "The patient acknowledged consuming {amount} of {drink} {when}.",
    "Patient's own ETOH intake: {amount} of {drink} {when}.",
    "They report personal alcohol use, usually {amount} of {drink} {when}.",
    "During review, the patient disclosed {amount} of {drink} {when}.",
    "Patient confirms they had {amount} of {drink} {when}.",
    "Interview documents a personal drinking history of {amount} of {drink} {when}.",
    "Alcohol history: the patient previously consumed {amount} of {drink} {when}.",
    "Prior alcohol use disorder is documented for the patient; abstinent for {interval}.",
    "The patient's alcohol dependence is in remission, sober for {interval}.",
    "Patient stopped drinking {interval} ago after years of use.",
    "History lists alcohol withdrawal in this patient's prior admission.",
    "This patient is monitored for withdrawal after stopping alcohol {interval} ago.",
    "Clinician attributes the patient's {condition} to their own past drinking.",
    "Assessment links {condition} to alcohol previously consumed by this patient.",
    "The patient's {condition} is described as alcohol related from personal use.",
    "Patient admits drinking, despite a prior intake form marking no alcohol use.",
    "Earlier denial corrected: patient now reports {amount} of {drink} {when}.",
    "A later note clarifies that the patient, not the companion, drank {drink}.",
    "Ethanol testing is positive and the patient confirms consuming {drink}.",
    "Patient reports they drank {drink}; the relative did not drink.",
    "The patient had used alcohol before arrival, according to their own account.",
)
INDIRECT = (
    "A driver impaired by alcohol struck this patient; the patient was a passenger.",
    "The patient was injured by an intoxicated driver and reports no personal drinking.",
    "Police report links the collision to the other driver's alcohol use.",
    "The patient's injury followed a collision caused by a driver impaired by drinking.",
    "An intoxicated partner pushed the patient, causing the current injury.",
    "An alcohol-impaired stranger assaulted the patient, leading to this visit.",
    "A relative who was drunk knocked the patient down; the patient had not drunk.",
    "The patient fell while avoiding a person impaired by alcohol.",
    "The patient's injury occurred when an alcohol-impaired coworker dropped equipment.",
    "An intoxicated neighbor caused the fall that prompted the patient's evaluation.",
    "A caregiver's alcohol impairment caused the event for which the patient is seen.",
    "The patient was a sober pedestrian hit by a motorist under the influence.",
    "The alcohol-impaired person operating the vehicle, not the patient, caused the collision.",
    "Witnesses said the assailant's intoxication contributed to the attack that injured the patient.",
    "The patient's presentation resulted from violence worsened by the assailant's intoxication.",
    "The other person's alcohol impairment caused the incident that injured the patient.",
    "The patient's own alcohol screen was negative; the injuring driver had been drinking.",
    "A companion's intoxication led to the accident that injured the patient.",
    "The patient was hurt when an alcohol-impaired person lost control of a cart.",
    "Trauma assessment attributes the event to another person's drinking, with no patient use documented.",
    "During {period}, a {actor} who had been drinking caused the patient's {complaint}.",
    "The {actor}'s alcohol impairment precipitated the event that resulted in {complaint}.",
    "A witness says {drink} impaired the person who injured the patient {period}.",
    "The patient sustained {complaint} after an alcohol-impaired {actor} caused an accident.",
    "The other participant reported {amount} of {drink}; their impairment caused the patient's injury.",
    "Emergency staff link the patient's {complaint} to a {actor} impaired by alcohol.",
    "The alcohol exposure belonged to the {actor}; their actions caused the patient's presentation.",
    "The patient was affected by the {actor}'s drinking {period}, as documented by witnesses.",
    "Impaired after drinking {drink}, the {actor} caused the event for which the patient is being treated.",
    "Report attributes this patient's injury to an alcohol-impaired {actor} at {location}.",
)
PRENATAL = (
    "The patient's developmental findings are attributed to the birth mother's alcohol use during pregnancy.",
    "Birth history documents prenatal alcohol exposure from the mother's drinking, contributing to this patient's findings.",
    "Clinician links the patient's developmental concerns to alcohol consumed by the birth mother during gestation.",
    "The patient has sequelae of fetal alcohol exposure caused by maternal drinking during pregnancy.",
    "A prenatal record confirms the birth mother used alcohol; this exposure contributed to the patient's current findings.",
    "The patient's condition is related to in-utero alcohol exposure from someone else's consumption, not their own.",
    "Assessment attributes the patient's developmental differences to the birth parent's drinking while pregnant.",
    "Maternal alcohol use during gestation caused the fetal exposure linked to this patient's presentation.",
    "The consult describes fetal alcohol spectrum findings due to the birth mother's alcohol use.",
    "Prenatal alcohol exposure from maternal consumption is identified as contributing to the patient's condition.",
)
NEGATIVE = (
    "Patient denies drinking alcohol; evaluation concerns {complaint}.",
    "The patient reports never using alcohol.",
    "Alcohol use screening was completed and recorded as none.",
    "Intake asks about ETOH; the patient's answer is no.",
    "Ethanol testing was below detection; no drinking is otherwise documented.",
    "No personal alcohol use or alcohol-related mechanism is documented.",
    "The chart's alcohol field is blank; no use is described elsewhere.",
    "Only a routine question about alcohol appears on the admission form.",
    "A generic medication leaflet advises avoiding alcohol during treatment.",
    "The nurse gave standard advice not to combine alcohol with this medication.",
    "Discharge paperwork includes a routine warning about alcohol.",
    "An alcohol wipe was used to clean the skin before venipuncture.",
    "The clinician used an isopropyl alcohol swab during the procedure.",
    "Alcohol-based hand rub was used before examination.",
    "The supply list includes rubbing alcohol; no patient ingestion is described.",
    "Family history mentions an uncle with alcohol use disorder; no causal link to this visit.",
    "A sibling drinks alcohol, but this is unrelated to the patient's presentation.",
    "A companion reported their own drinking, with no connection to the patient's injury.",
    "Patient asked whether alcohol interacts with antibiotics; no use was reported.",
    "A preventive health handout discusses alcohol, without any patient use documented.",
    "The patient denies alcohol use; a later note repeats the denial.",
    "The word alcohol appears only in a product name in the procedure record.",
    "Current complaint is {complaint}; the notes contain no alcohol history.",
    "Assessment addresses {complaint} without an alcohol-related finding.",
)
NEGATIVE_CATEGORIES = (
    "denial", "denial", "screening", "screening", "negative_test", "no_evidence",
    "blank_field", "screening", "medication_warning", "medication_warning",
    "medication_warning", "product", "product", "product", "product",
    "family_history", "family_history", "unrelated_other", "question",
    "prevention", "denial", "product", "no_mention", "no_mention",
)

DRINKS = ("beer", "wine", "cider", "spirits", "liquor")
AMOUNTS = ("a glass", "two drinks", "several drinks", "one bottle", "three cans")
WHEN = ("most evenings", "last weekend", "occasionally", "daily", "last month")
INTERVALS = ("six months", "two years", "five weeks", "one year")
CONDITIONS = ("pancreatitis", "cirrhosis", "gastritis", "peripheral neuropathy")
ACTORS = ("driver", "coworker", "partner", "neighbor", "caregiver", "bystander")
PERIODS = ("earlier that day", "before the incident", "that evening", "shortly before the accident")
LOCATIONS = ("the crossing", "the workplace", "home", "the parking area")
NOTE_TYPES = ("ED physician note", "nursing note", "consult", "discharge summary", "triage", "progress note")
NEUTRAL_NOTES = (
    "Repeat examination found stable findings. Available studies and symptom control were discussed with the patient.",
    "Medication list and allergies were reviewed. The patient understood the follow-up plan and return precautions.",
    "Nursing reassessment documented stable breathing and orientation. Pain and mobility were checked before disposition.",
    "The clinician reviewed the examination, available studies, and options for supportive treatment with the patient.",
    "A further progress entry records no new complaint. The team planned routine observation and outpatient review.",
    "On reassessment the patient was alert and conversant. Discharge instructions were reviewed and questions answered.",
    "Further studies were considered for the presenting concern. The team compared findings with the earlier examination.",
    "The consultation summarized the presenting symptoms, current medication list, examination, and follow-up needs.",
    "The receiving team compared the earlier examination with the current findings and confirmed the planned disposition.",
    "A separate nursing entry notes that oral intake and mobility were reviewed before the patient left the unit.",
    "The treating clinician explained expected symptom progression and when another assessment would be needed.",
    "Records from the current visit were reviewed by the next shift, with no change to the documented treatment plan.",
    "The patient described the timing of symptoms and the area of discomfort during a repeat history.",
    "The care team reviewed the patient's symptoms again and documented the response to supportive measures.",
    "A clinician reviewed available laboratory information and discussed routine follow-up with the patient.",
    "The discharge review included return precautions, pending results, and instructions for symptom management.",
)


def _backgrounds(rng: random.Random, complaint: str, style: int) -> list[str]:
    age = rng.randrange(18, 91)
    pulse = rng.randrange(55, 111)
    systolic = rng.randrange(98, 159)
    diastolic = rng.randrange(57, 99)
    common = [
        f"Patient aged {age} evaluated for {complaint}.",
        f"Vitals: pulse {pulse}/min, BP {systolic}/{diastolic} mmHg.",
        f"Examination for {complaint} completed; follow-up arranged.",
    ]
    if style == 0:
        return [common[0], common[1]]
    if style == 1:
        return [f"Chief complaint: {complaint}. {common[0]}", common[1], common[2]]
    if style == 2:
        return [common[1], common[0], common[2]]
    if style == 3:
        return [f"Triage: {complaint}. Patient alert and oriented.", common[1]]
    if style == 4:
        return [common[0], f"Consult review: {complaint}; {common[1]}"]
    if style == 5:
        return [f"ED presentation: {complaint}; age {age}.", common[1], common[2]]
    if style == 6:
        return [f"Referral for {complaint}. {common[1]}", common[2]]
    if style == 7:
        return [f"Reason for attendance is {complaint}.", common[1], common[2]]
    return [f"At arrival, {complaint} was the chief concern.", common[1], common[2]]


def _render_case(split: str, group_index: int, label: str, family: tuple[str, str],
                 style: int, rng: random.Random, common: list[str]) -> dict:
    family_id, complaint = family
    fields = {"complaint": complaint, "drink": rng.choice(DRINKS),
              "amount": rng.choice(AMOUNTS), "when": rng.choice(WHEN),
              "interval": rng.choice(INTERVALS), "condition": rng.choice(CONDITIONS),
              "actor": rng.choice(ACTORS), "period": rng.choice(PERIODS),
              "location": rng.choice(LOCATIONS)}
    prenatal_family = family_id in {"developmental", "neurodevelopment"}
    bank = {"direct": DIRECT, "indirect": PRENATAL if prenatal_family else INDIRECT,
            "negative": NEGATIVE}[label]
    signal_index = rng.randrange(len(bank))
    signal = bank[signal_index].format(**fields)
    patient_evidence = label == "direct"
    other_causal = label == "indirect"
    evidence = [signal] if label != "negative" else []
    decoys = [signal] if label == "negative" else []
    notes = list(common) + [signal]
    if label == "direct" and group_index % 9 == 0:
        second = rng.choice(PRENATAL if prenatal_family else INDIRECT).format(**fields)
        notes.append(second)
        evidence.append(second)
        other_causal = True
    if label == "indirect" and group_index % 4 == 0:
        denial = rng.choice(("Patient denies personal alcohol use.", "Personal ETOH history: none reported."))
        notes.append(denial)
        decoys.append(denial)
    if label == "negative" and group_index % 7 == 0:
        notes.append("No note links any other person's drinking to this presentation.")
    if combine_components(patient_evidence, other_causal) != label:
        raise AssertionError("generator label mismatch")
    rng.shuffle(notes)
    encounters = [{"type": rng.choice(NOTE_TYPES), "note": note} for note in notes]
    evidence_refs = [{"encounter_index": next(i for i, item in enumerate(encounters, 1)
                                           if sentence in item["note"]), "text": sentence}
                     for sentence in evidence]
    group_id = f"v2-{split}-{group_index:05d}"
    if label == "negative":
        category = NEGATIVE_CATEGORIES[signal_index]
    elif label == "direct":
        category = "mixed_direct_indirect" if other_causal else (
            "past_disorder" if 10 <= signal_index <= 12 else
            "withdrawal" if 13 <= signal_index <= 14 else
            "alcohol_condition" if 15 <= signal_index <= 17 else
            "contradiction" if 18 <= signal_index <= 20 else "personal_use")
    else:
        category = "prenatal" if prenatal_family else "other_person_causal"
    return {"case_id": f"{group_id}-{label}",
            "patient_id": f"{group_id}-{label}",
            "group_id": group_id, "label": label, "source": "fully_synthetic_v2",
            "scenario": family_id, "signal_category": category,
            "family_seen_in_training": family in SEEN_FAMILIES,
            "render_style": style, "encounters": encounters,
            "annotation": {"patient_evidence": patient_evidence,
                           "other_person_causal": other_causal,
                           "evidence": evidence_refs, "decoys": decoys}}


def generate_split(split: str, groups: int, seed: int = SEED) -> list[dict]:
    if split not in SPLITS or groups < 1:
        raise ValueError("invalid split or count")
    rng = random.Random(f"{seed}-{split}")
    rows = []
    for group_index in range(groups):
        if split == "test" and group_index >= groups // 2:
            family = UNSEEN_FAMILIES[(group_index - groups // 2) % len(UNSEEN_FAMILIES)]
        else:
            family = SEEN_FAMILIES[group_index % len(SEEN_FAMILIES)]
        styles = {"train": (0, 1, 2, 3, 4), "validation": (5,),
                  "calibration": (6,), "test": (7, 8)}
        style = rng.choice(styles[split])
        common = _backgrounds(rng, family[1], style)
        extra = 16 if group_index % 20 == 0 else 2 if group_index % 5 == 0 else 0
        if extra:
            common.extend(rng.sample(NEUTRAL_NOTES, extra))
        for label in LABELS:
            rows.append(_render_case(split, group_index, label, family, style, rng, common))
    rng.shuffle(rows)
    return rows


def build_dataset(output_dir: Path, seed: int = SEED) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty directory: {output_dir}")
    paths = {}
    stats = {}
    for split, groups in SPLITS.items():
        rows = generate_split(split, groups, seed)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = path
        stats[split] = {"groups": groups, "families": dict(Counter(r["scenario"] for r in rows)),
                        "styles": dict(Counter(str(r["render_style"]) for r in rows))}
    summary = validate_splits(paths)
    manifest = {"version": 2, "source": "fully_synthetic", "seed": seed,
                "counts": summary, "coverage": stats,
                "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                           for key, path in paths.items()},
                "caveat": "Synthetic development data; no clinical performance claim."}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/alcohol_synthetic_v2"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = build_dataset(args.output_dir, args.seed)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
