"""Deterministic synthetic bootstrap data; never presented as clinical validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from alcohol_ft.data import validate_splits, write_jsonl
from alcohol_ft.task import LABELS

DIRECT = {
    "train": [
        "Patient reports {amount} of {drink} {frequency}.",
        "Social history: drinks {amount} {drink} {frequency}.",
        "Assessment: {condition} attributed to the patient's alcohol use.",
        "The patient arrived after drinking {amount} of {drink}.",
        "History of alcohol use disorder; last drink {time}.",
        "Patient stopped drinking {time} after years of heavy use.",
        "Serum ethanol positive; patient admits consuming {drink}.",
        "Treatment plan includes monitoring for alcohol withdrawal after patient's last drink {time}.",
    ],
    "calibration": [
        "The patient acknowledges having {amount} of {drink} {frequency}.",
        "Clinician documents prior alcohol dependence; sober since {time}.",
        "Working diagnosis is {condition} from the patient's prior drinking.",
    ],
    "test": [
        "During the interview, patient disclosed {frequency} consumption of {amount} {drink}.",
        "Discharge summary records remote alcohol misuse, abstinent since {time}.",
        "Physician links {condition} to the patient's own drinking history.",
    ],
}
INDIRECT = {
    "train": [
        "Patient was injured when a driver who had been drinking struck their vehicle; patient was a sober passenger.",
        "Assault by an intoxicated partner caused the patient's injury; patient denies drinking.",
        "Child has prenatal alcohol exposure from birth mother's drinking; child has never consumed alcohol.",
        "Patient was hurt by a coworker impaired by alcohol; patient reports no alcohol use.",
        "The patient's burns followed a fire started by an intoxicated neighbor; patient was not drinking.",
        "Patient fell while avoiding an intoxicated person; patient denies personal alcohol use.",
    ],
    "calibration": [
        "Trauma note: sober pedestrian hit by a motorist under the influence of alcohol.",
        "Pediatric consult attributes developmental findings to maternal drinking during pregnancy; the child is the patient.",
        "Patient's injury occurred during an attack by a drunk relative; patient did not drink.",
    ],
    "test": [
        "Patient, a nondrinking bicyclist, was struck by an alcohol-impaired driver.",
        "Neonatal chart describes fetal alcohol exposure due to maternal consumption; infant is the patient.",
        "The sober patient was pushed down stairs by an intoxicated roommate.",
    ],
}
NEGATIVE = {
    "train": [
        "Patient denies alcohol use; screening completed for routine admission.",
        "Alcohol use: none. Presenting complaint is {complaint}.",
        "Serum ethanol was undetectable; no history of drinking documented.",
        "Discharge instructions say to avoid alcohol while taking medication; no use reported.",
        "Family history of alcohol use disorder in an uncle; no contribution to this visit and patient denies drinking.",
        "Alcohol swab used before venipuncture; patient denies alcohol intake.",
        "Patient asked whether alcohol is allowed with antibiotics; no drinking documented.",
        "Clinical history contains no alcohol information. Visit for {complaint}.",
    ],
    "calibration": [
        "Routine alcohol screen was negative. Evaluation concerns {complaint}.",
        "Chart says no ETOH use. Patient seen for {complaint}.",
        "Alcohol-based hand rub documented in procedure note, with no alcohol exposure history.",
    ],
    "test": [
        "Patient explicitly reports never consuming alcohol; visit concerns {complaint}.",
        "Preoperative form asks about drinking and records 'no'; assessment is {complaint}.",
        "Lab result ethanol below detection; there is no documented patient drinking or alcohol-related injury.",
    ],
}
BACKGROUNDS = {
    "train": [
        "Triage: alert, breathing comfortably; visit for {complaint}.",
        "Nursing note: vital signs stable; {complaint} under evaluation.",
        "Consult: examination and medication list reviewed for {complaint}.",
        "Progress note: pain controlled; follow-up arranged for {complaint}.",
    ],
    "calibration": [
        "Admission note: evaluation underway for {complaint}; no acute distress.",
        "Ward review: treatment options discussed for {complaint}.",
    ],
    "test": [
        "ED assessment: focused examination for {complaint} completed.",
        "Follow-up note: symptoms from {complaint} reassessed before discharge.",
    ],
}
COMPLAINTS = ["ankle sprain", "abdominal pain", "migraine", "cellulitis", "chest discomfort", "fracture", "asthma flare", "kidney stone", "fever", "rash", "laceration", "back pain"]
DRINKS = ["beer", "wine", "spirits", "cider"]
AMOUNTS = ["one glass", "two drinks", "several glasses", "three cans"]
FREQUENCIES = ["on weekends", "each evening", "most days", "occasionally"]
CONDITIONS = ["pancreatitis", "cirrhosis", "gastritis", "peripheral neuropathy"]
TIMES = ["yesterday", "six months ago", "two years ago", "last week"]


def generate_split(split: str, per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        bank = {"direct": DIRECT, "indirect": INDIRECT, "negative": NEGATIVE}[label][split]
        for i in range(per_label):
            fields = dict(amount=rng.choice(AMOUNTS), drink=rng.choice(DRINKS),
                          frequency=rng.choice(FREQUENCIES), condition=rng.choice(CONDITIONS),
                          time=rng.choice(TIMES), complaint=rng.choice(COMPLAINTS))
            signal = rng.choice(bank).format(**fields)
            backgrounds = [rng.choice(BACKGROUNDS[split]).format(**fields) for _ in range(rng.randint(0, 3))]
            # Neutral measurements diversify case reports without encoding the label.
            vitals = (f"Vitals: pulse {rng.randrange(54, 121)}/min, "
                      f"BP {rng.randrange(95, 157)}/{rng.randrange(55, 101)} mmHg, "
                      f"temperature {rng.randrange(360, 391) / 10:.1f} C.")
            notes = [signal + " " + vitals] + backgrounds
            rng.shuffle(notes)
            encounters = [{"type": rng.choice(["ED physician note", "nursing note", "consult", "discharge summary"]),
                           "note": note} for note in notes]
            case_id = f"syn-{split}-{label}-{i:05d}"
            rows.append({"case_id": case_id, "patient_id": case_id,
                         "label": label, "source": "synthetic", "scenario": label,
                         "encounters": encounters})
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/alcohol_synthetic"))
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    paths = {}
    for split, per_label in (("train", 8000), ("calibration", 1000), ("test", 1000)):
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, generate_split(split, per_label, args.seed))
        paths[split] = path
    summary = validate_splits(paths)
    manifest = {"source": "fully_synthetic", "seed": args.seed,
                "total_cases": sum(part["total"] for part in summary.values()),
                "splits": summary,
                "sha256": {split: hashlib.sha256(path.read_bytes()).hexdigest()
                           for split, path in paths.items()}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
