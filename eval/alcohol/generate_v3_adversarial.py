"""Freeze a small challenge set with mechanisms outside the v3 generator.

These are authored synthetic diagnostics, not clinician-labeled records. The
cases are defined here before scoring the v3 model and never enter training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from alcohol_ft.data import validate_splits, write_jsonl
from alcohol_ft.task import LABELS

# name, complaint, nonalcohol event, incidental mention, other-person event,
# other-person causal alcohol evidence, patient's own use.
CONTEXTS = (
    ("marina", "wrist injury", "A corroded cleat failed and a mooring line snapped.",
     "A visitor described wine with dinner last week; that visitor handled no lines.",
     "A deckhand released a taut line into the patient's arm.",
     "Harbor staff judged the deckhand impaired by rum while releasing the line.",
     "Medication review records the patient's own regular wine consumption."),
    ("theater", "scalp wound", "A lighting mount fractured above the patient.",
     "Alcohol solvent was used later to clean the lighting console.",
     "A stagehand dropped a lighting unit onto the patient.",
     "The stagehand was unsteady from drinking while carrying the unit.",
     "The patient reports having several beers at home the previous night."),
    ("restaurant", "forearm burn", "A tray leg broke and tipped hot tea onto the patient.",
     "The printed aftercare sheet gives a standard warning about alcohol with analgesics.",
     "A server tipped a full teapot onto the patient's arm.",
     "Witnesses describe the server as intoxicated and losing balance while carrying it.",
     "The patient acknowledges drinking cider on most weekends."),
    ("roof", "shoulder pain", "A roof tile came loose in a wind gust and fell.",
     "A relative's distant alcohol dependence is listed on a family-history form.",
     "A roofer dislodged a tile that fell onto the patient.",
     "The roofer had been drinking and was impaired while working on the roof.",
     "The patient disclosed a prior period of heavy liquor use."),
    ("dental", "jaw injury", "The dental chair's hydraulic lock failed during positioning.",
     "The clinician used an alcohol swab for an unrelated injection afterward.",
     "A technician moved the chair suddenly and injured the patient.",
     "The technician was impaired by alcohol while operating the chair controls.",
     "The patient says they drank wine earlier that day."),
    ("farm", "knee injury", "An animal broke through a weak fence and knocked the patient down.",
     "A neighbor who had been drinking arrived only after the animal was secured.",
     "A farm worker left a gate open and the animal ran into the patient.",
     "The worker's alcohol impairment led them to leave the gate open, according to the incident review.",
     "The patient reports personal beer intake several times each week."),
    ("pool", "head injury", "A loose pool ladder rung gave way beneath the patient.",
     "Both the patient and lifeguard had negative alcohol screens.",
     "A lifeguard moved the ladder while the patient was climbing it.",
     "The lifeguard was impaired by drinking when moving the ladder.",
     "The patient confirms consuming spirits before arriving at the pool."),
    ("museum", "ankle injury", "An aging display rail broke as the patient passed.",
     "A visitor smelling of beer stood near the exit, away from the display.",
     "A guide pushed a display stand into the patient.",
     "The guide was visibly intoxicated when pushing the stand.",
     "The patient reports an established history of alcohol use disorder."),
    ("bakery", "hand injury", "A mixer safety latch failed during normal use.",
     "A coworker mentioned drinking after last week's shift; the latch failure was documented separately.",
     "A coworker restarted the mixer while the patient's hand was nearby.",
     "The coworker was impaired by alcohol when restarting the mixer.",
     "The patient describes daily alcohol intake in the social history."),
    ("library", "rib pain", "A bookcase anchor failed and the case tipped.",
     "A prevention leaflet about drinking was among the papers given at discharge.",
     "A volunteer pulled the bookcase toward the patient.",
     "The volunteer was intoxicated when pulling it, according to the incident report.",
     "The patient acknowledges drinking two glasses of wine on the weekend."),
    ("ferry", "back pain", "A gangway hinge failed as the patient boarded.",
     "A passenger's drinking was discussed in a separate welfare note; that passenger did not use the gangway.",
     "A crew member released the gangway before the patient had cleared it.",
     "The crew member was impaired by alcohol while operating the release.",
     "The patient reports past alcohol withdrawal during a prior admission."),
    ("workshop", "foot injury", "A pneumatic hose split and displaced a heavy tool.",
     "The patient denied drinking and a visitor's alcohol test was negative.",
     "An instructor dropped the tool onto the patient's foot.",
     "The instructor was impaired by drinking while handling the tool.",
     "The patient reports regular personal consumption of beer."),
)


def build_rows() -> list[dict]:
    rows = []
    for context_index, (name, complaint, neutral_event, incidental,
                        other_event, other_alcohol, patient_alcohol) in enumerate(CONTEXTS):
        for style in range(2):
            group_id = f"v3-adversarial-{context_index:02d}-{style}"
            for label in LABELS:
                event = other_event if label == "indirect" else neutral_event
                alcohol = (other_alcohol if label == "indirect" else
                           patient_alcohol if label == "direct" else incidental)
                if style == 0:
                    encounters = [
                        {"type": "triage", "note": f"Seen for {complaint}. {event}"},
                        {"type": "clinician assessment", "note": alcohol},
                        {"type": "discharge", "note": "The team reviewed symptoms and the follow-up plan."},
                    ]
                else:
                    encounters = [
                        {"type": "history", "note": alcohol},
                        {"type": "incident review", "note": event},
                        {"type": "progress note", "note": f"Current visit addresses {complaint}. Follow-up was arranged."},
                    ]
                rows.append({"case_id": f"{group_id}-{label}",
                             "patient_id": f"{group_id}-{label}",
                             "group_id": group_id, "label": label,
                             "source": "authored_synthetic_v3_adversarial",
                             "scenario": name, "encounters": encounters})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval/alcohol/v3_adversarial.jsonl"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite frozen challenge set: {args.output}")
    rows = build_rows()
    write_jsonl(args.output, rows)
    validate_splits({"v3_adversarial": args.output})
    manifest = {"cases": len(rows), "groups": len(rows) // 3,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "source": "authored_synthetic_v3_adversarial",
                "caveat": "Not clinician-labeled; frozen before v3 model evaluation."}
    args.output.with_name("v3_adversarial_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
