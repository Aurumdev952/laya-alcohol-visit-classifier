"""Generate 30,000 synthetic visits with matched causal and incidental alcohol contexts.

The v3 hard groups add two-note mechanisms. Their negative variants mention
alcohol but attribute the visit to a separate event; indirect variants connect
another person's drinking to the event. Previously authored evaluation cases
are never read or copied by this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from alcohol_ft.data import validate_splits, write_jsonl
from alcohol_ft.generate_v2 import (DIRECT, FAMILIES, NEUTRAL_NOTES, NOTE_TYPES,
                                    SEEN_FAMILIES, SPLITS, _backgrounds,
                                    generate_split as generate_v2_split)
from alcohol_ft.task import LABELS

SEED = 20260929

# Each context pairs a non-alcohol mechanism and an incidental alcohol mention
# with an event actually caused by another person's drinking. The same
# presentation is used for all three labels in a counterfactual group.
HARD_CONTEXTS = (
    ("mechanical_collision", "a steering linkage failed during travel",
     "another motorist crossed the center line and struck the patient",
     "Both involved drivers' alcohol screens were negative",
     "the other motorist was impaired by alcohol and failed a roadside screen while crossing the lane"),
    ("stairway", "a loose stair tread gave way beneath the patient",
     "a caretaker pushed a cart into the patient on the stairs",
     "A visiting relative's drinking was recorded after the fall, without involvement in it",
     "the caretaker was impaired by liquor while moving the cart"),
    ("sports", "a collision occurred during recreational play",
     "a spectator entered the playing area and knocked the patient down",
     "Family history lists alcohol dependence in a parent; the game had already started",
     "the spectator was intoxicated when entering the playing area"),
    ("electrical_fire", "an electrical short started the fire",
     "a visitor left a heater running and started the fire",
     "Alcohol antiseptic was used later during wound care",
     "the visitor was impaired by drinking when operating the heater"),
    ("wet_floor", "water from a leaking pipe caused a slip",
     "a coworker scattered supplies across the walkway and the patient fell",
     "Routine discharge paperwork warns against combining medication with alcohol",
     "the coworker was impaired by drinking while handling the supplies"),
    ("curb", "the patient's foot caught on an uneven curb edge",
     "a pedestrian collided with the patient at the curb",
     "The patient denied EtOH and the other pedestrian's screen was negative",
     "the pedestrian was visibly drunk and had consumed beer just before crossing"),
    ("shelf", "a loose shelf bracket collapsed beside the patient",
     "a maintenance worker dislodged the shelf onto the patient",
     "The worker's prior drinking appears in an unrelated workplace history",
     "the maintenance worker was impaired by drinking when handling the shelf"),
    ("debris", "wind-blown debris struck the patient",
     "a bystander threw an object that struck the patient",
     "A bystander smelling of liquor arrived only after the debris had hit",
     "the bystander was intoxicated when throwing the object"),
    ("exercise", "the patient fell while exercising alone",
     "the patient's partner shoved them during an argument",
     "The partner's drinking was discussed for housing support; it preceded neither the exercise nor fall",
     "the partner was impaired by alcohol during the argument"),
    ("pothole", "a pothole forced the patient's vehicle off the road",
     "a second driver forced the patient's vehicle off the road",
     "A driver at a nearby, separate scene had a positive alcohol test",
     "the second driver was impaired by drinking when forcing the patient off the road"),
    ("workplace", "an unsecured tool fell from a storage rack",
     "a contractor dropped a tool onto the patient",
     "The contractor discussed drinking at a party last month; the rack failure was documented separately",
     "the contractor was unsteady from drinking when dropping the tool"),
    ("bus", "a mechanical brake defect made the bus stop abruptly",
     "the bus operator made an unsafe turn that injured the patient",
     "The bus operator and patient both had negative alcohol screens",
     "the bus operator was impaired by alcohol while making the unsafe turn"),
)

# These mechanisms appear only in the locked v3 test. They probe transfer
# beyond both the sentence surfaces and the particular training mechanisms.
TEST_ONLY_CONTEXTS = (
    ("elevator", "an elevator door sensor failed and the door closed unexpectedly",
     "a building attendant overrode the elevator door sensor and the door struck the patient",
     "A cleaning log lists denatured alcohol for the elevator controls after the event",
     "the attendant was impaired by drinking while overriding the sensor"),
    ("ladder", "a ladder foot slipped on loose gravel",
     "a coworker shifted the ladder while the patient was on it",
     "The coworker's social history includes drinking after work on another day",
     "the coworker was intoxicated while shifting the ladder"),
    ("playground", "a playground hinge broke during ordinary use",
     "a supervisor left a gate unsecured and the patient was hurt",
     "A parent's prior alcohol counseling is mentioned on an unrelated form",
     "the supervisor was impaired by drinking when leaving the gate unsecured"),
    ("trolley", "a wheel bearing broke and the trolley rolled into the patient",
     "a depot worker released the trolley toward the patient",
     "An alcohol-based surface cleaner was used on the trolley after the injury",
     "the depot worker was impaired by alcohol when releasing the trolley"),
)

DEVELOPMENTAL_CONTEXTS = (
    ("prenatal_record", "a genetic finding explains the current developmental concerns",
     "prenatal alcohol exposure is linked to the current developmental concerns",
     "The birth parent's alcohol use began years after the pregnancy",
     "records document that the birth parent drank during this pregnancy"),
    ("prenatal_screen", "the current findings are attributed to a documented chromosomal condition",
     "the current findings are attributed to alcohol exposure during gestation",
     "Pregnancy records show no alcohol exposure; a later family-history form mentions drinking",
     "the gestational parent used alcohol during the pregnancy"),
)

SURFACE = {
    "train": (
        "Assessment of {complaint}: {event}.",
        "Additional history: {alcohol}.",
    ),
    "validation": (
        "The documented mechanism for {complaint} was that {event}.",
        "A separate entry records: {alcohol}.",
    ),
    "calibration": (
        "During review of {complaint}, clinicians recorded that {event}.",
        "The alcohol-related entry states: {alcohol}.",
    ),
    "test": (
        "For the current {complaint}, the chart identifies the following event: {event}.",
        "A later note adds: {alcohol}.",
    ),
}


def _v3_id(split: str, group_index: int) -> str:
    return f"v3-{split}-{group_index:05d}"


def _upgrade_base(row: dict, split: str, group_index: int) -> dict:
    row = dict(row)
    group_id = _v3_id(split, group_index)
    row.update(group_id=group_id, case_id=f"{group_id}-{row['label']}",
               patient_id=f"{group_id}-{row['label']}", source="fully_synthetic_v3",
               hard_group=False, hard_category=None)
    return row


def _hard_group(split: str, group_index: int, family_id: str, complaint: str,
                style: int, seed: int) -> list[dict]:
    rng = random.Random(f"{seed}-{split}-{group_index}-hard")
    developmental = family_id in {"developmental", "neurodevelopment"}
    contexts = (DEVELOPMENTAL_CONTEXTS if developmental else
                TEST_ONLY_CONTEXTS if split == "test" and group_index % 4 == 0 else
                HARD_CONTEXTS)
    category, negative_event, indirect_event, incidental, other_drinking = contexts[
        (group_index // 2) % len(contexts)]
    event_template, alcohol_template = SURFACE[split]
    common = _backgrounds(rng, complaint, style)
    extra = 16 if group_index % 20 == 0 else 2 if group_index % 5 == 0 else 0
    if extra:
        common.extend(rng.sample(NEUTRAL_NOTES, extra))
    direct_signal = rng.choice(DIRECT).format(
        complaint=complaint, drink=rng.choice(("beer", "wine", "cider")),
        amount=rng.choice(("one glass", "two drinks", "several drinks")),
        when=rng.choice(("last weekend", "most evenings", "last month")),
        interval=rng.choice(("six months", "two years")),
        condition=rng.choice(("pancreatitis", "gastritis")))
    group_id = _v3_id(split, group_index)
    rows = []
    for label in LABELS:
        event = indirect_event if label == "indirect" else negative_event
        alcohol = (other_drinking if label == "indirect" else
                   direct_signal if label == "direct" else incidental)
        event_note = event_template.format(complaint=complaint, event=event)
        alcohol_note = alcohol_template.format(alcohol=alcohol)
        notes = list(common) + [event_note, alcohol_note]
        rng.shuffle(notes)
        encounters = [{"type": rng.choice(NOTE_TYPES), "note": note} for note in notes]
        evidence_sentences = ([alcohol_note] if label == "direct" else
                              [event_note, alcohol_note] if label == "indirect" else [])
        evidence = [{"encounter_index": next(i for i, item in enumerate(encounters, 1)
                                               if sentence == item["note"]), "text": sentence}
                    for sentence in evidence_sentences]
        rows.append({
            "case_id": f"{group_id}-{label}", "patient_id": f"{group_id}-{label}",
            "group_id": group_id, "label": label, "source": "fully_synthetic_v3",
            "scenario": family_id,
            "signal_category": (f"hard_{category}" if label == "negative" else
                                "hard_other_person_causal" if label == "indirect" else
                                "hard_patient_use"),
            "family_seen_in_training": family_id in {x[0] for x in SEEN_FAMILIES},
            "render_style": style, "hard_group": True, "hard_category": category,
            "encounters": encounters,
            "annotation": {"patient_evidence": label == "direct",
                           "other_person_causal": label == "indirect",
                           "evidence": evidence,
                           "decoys": [alcohol_note] if label == "negative" else []},
        })
    return rows


def generate_split(split: str, groups: int, seed: int = SEED) -> list[dict]:
    if split not in SPLITS or groups < 1:
        raise ValueError("invalid split or count")
    base_rows = generate_v2_split(split, groups, seed)
    base_groups = {}
    for row in base_rows:
        group_index = int(row["group_id"].rsplit("-", 1)[1])
        base_groups.setdefault(group_index, []).append(row)
    rows = []
    complaints = dict(FAMILIES)
    for group_index in range(groups):
        base_group = base_groups[group_index]
        if group_index % 2:
            rows.extend(_upgrade_base(row, split, group_index) for row in base_group)
        else:
            family_id = base_group[0]["scenario"]
            rows.extend(_hard_group(split, group_index, family_id, complaints[family_id],
                                    base_group[0]["render_style"], seed))
    random.Random(f"{seed}-{split}-shuffle-v3").shuffle(rows)
    return rows


def build_dataset(output_dir: Path, seed: int = SEED) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty directory: {output_dir}")
    paths, stats = {}, {}
    for split, groups in SPLITS.items():
        rows = generate_split(split, groups, seed)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = path
        stats[split] = {"groups": groups,
                        "hard_groups": len({r["group_id"] for r in rows if r["hard_group"]}),
                        "hard_negative_categories": dict(Counter(
                            r["hard_category"] for r in rows
                            if r["hard_group"] and r["label"] == "negative")),
                        "families": dict(Counter(r["scenario"] for r in rows))}
    summary = validate_splits(paths)
    manifest = {"version": 3, "source": "fully_synthetic", "seed": seed,
                "counts": summary, "coverage": stats,
                "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                           for key, path in paths.items()},
                "caveat": "Synthetic development data; no clinical performance claim."}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/alcohol_synthetic_v3"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = build_dataset(args.output_dir, args.seed)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
