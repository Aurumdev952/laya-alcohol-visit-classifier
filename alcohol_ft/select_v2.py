"""Check the frozen development targets against a validation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(reference: dict, candidate: dict, negative_target=0.90, recall_tolerance=0.02) -> dict:
    if reference["split"] != "validation" or candidate["split"] != "validation":
        raise ValueError("selection requires validation reports")
    if reference["data_dir"] != candidate["data_dir"]:
        raise ValueError("reports reference different datasets")
    old = reference["metrics"]["per_label"]
    new = candidate["metrics"]["per_label"]
    checks = {"negative_recall": new["negative"]["recall"] >= negative_target,
              "direct_recall": new["direct"]["recall"] >= old["direct"]["recall"] - recall_tolerance,
              "indirect_recall": new["indirect"]["recall"] >= old["indirect"]["recall"] - recall_tolerance}
    return {"candidate": candidate["model"], "reference": reference["model"],
            "passes_research_targets": all(checks.values()), "checks": checks,
            "candidate_macro_f1": candidate["metrics"]["macro_f1"],
            "candidate_recalls": {label: new[label]["recall"] for label in new},
            "reference_recalls": {label: old[label]["recall"] for label in old},
            "negative_to_direct": candidate["metrics"]["negative_to_direct"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(json.loads(args.reference.read_text(encoding="utf-8")),
                    json.loads(args.candidate.read_text(encoding="utf-8")))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
