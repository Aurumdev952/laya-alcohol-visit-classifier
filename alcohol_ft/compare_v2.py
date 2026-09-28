"""Compare two v2 reports with paired, counterfactual-group bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from alcohol_ft.benchmark import metrics
from alcohol_ft.data import read_cases


def compare(reference: dict, candidate: dict, data_dir: Path, draws=2000, seed=20260928) -> dict:
    if reference["split"] != candidate["split"]:
        raise ValueError("reports use different splits")
    split = reference["split"]
    cases = read_cases(data_dir / f"{split}.jsonl")
    by_ref = {p["case_id"]: p for p in reference["predictions"]}
    by_new = {p["case_id"]: p for p in candidate["predictions"]}
    case_ids = {case["case_id"] for case in cases}
    if set(by_ref) != case_ids or set(by_new) != case_ids:
        raise ValueError("reports do not cover the same dataset")
    groups = defaultdict(list)
    for case in cases:
        groups[case.get("group_id", case["case_id"])].append(case["case_id"])
    paired_groups = list(groups.values())
    rng = random.Random(seed)
    deltas = {"accuracy": [], "macro_f1": [], "negative_recall": []}
    for _ in range(draws):
        sample = [cid for _ in paired_groups for cid in paired_groups[rng.randrange(len(paired_groups))]]
        old = metrics([by_ref[cid] for cid in sample])
        new = metrics([by_new[cid] for cid in sample])
        deltas["accuracy"].append(new["accuracy"] - old["accuracy"])
        deltas["macro_f1"].append(new["macro_f1"] - old["macro_f1"])
        deltas["negative_recall"].append(new["per_label"]["negative"]["recall"] -
                                         old["per_label"]["negative"]["recall"])
    result = {}
    for key, values in deltas.items():
        values.sort()
        result[key] = {"delta": candidate["metrics"][key] - reference["metrics"][key]
                       if key != "negative_recall" else
                       candidate["metrics"]["per_label"]["negative"]["recall"] -
                       reference["metrics"]["per_label"]["negative"]["recall"],
                       "group_bootstrap_95_interval": [values[int(0.025 * draws)],
                                                       values[int(0.975 * draws)]]}
    return {"split": split, "groups": len(paired_groups), "draws": draws,
            "reference": reference["model"], "candidate": candidate["model"], "deltas": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic_v2"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("draws must be at least 100")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare(reference, candidate, args.data_dir, args.draws)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
