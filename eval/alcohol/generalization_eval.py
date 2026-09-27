"""One-shot subject-attribution generalization test on locked counterfactual trios."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from alcohol_ft.benchmark import metrics, paired_accuracy_interval, run_model
from alcohol_ft.data import read_cases, validate_splits
from alcohol_ft.task import LABELS


def group_metrics(cases: list[dict], predictions: list[dict]) -> dict:
    by_group = defaultdict(list)
    for case, prediction in zip(cases, predictions):
        if case["case_id"] != prediction["case_id"]:
            raise ValueError("case order mismatch")
        by_group[case["group_id"]].append(prediction)
    if not by_group:
        raise ValueError("no counterfactual groups")
    exact = distinct = 0
    failures = []
    for group_id, rows in sorted(by_group.items()):
        if len(rows) != 3 or {row["expected"] for row in rows} != set(LABELS):
            raise ValueError(f"{group_id}: expected one case per label")
        all_correct = all(row["expected"] == row["predicted"] for row in rows)
        exact += all_correct
        distinct += len({row["predicted"] for row in rows}) == 3
        if not all_correct:
            failures.append(group_id)
    return {"groups": len(by_group), "all_three_correct": exact,
            "all_three_correct_rate": exact / len(by_group),
            "predicted_three_distinct_labels": distinct,
            "predicted_three_distinct_rate": distinct / len(by_group),
            "failed_groups": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--base", default="convaiinnovations/laya")
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic"))
    parser.add_argument("--curated", type=Path, default=Path("eval/alcohol/curated.jsonl"))
    parser.add_argument("--generalization", type=Path, default=Path("eval/alcohol/generalization.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/laya-alcohol-001/generalization.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    paths = {split: args.data_dir / f"{split}.jsonl" for split in ("train", "calibration", "test")}
    paths.update(curated=args.curated, generalization=args.generalization)
    print("Split validation:", validate_splits(paths), flush=True)
    cases = read_cases(args.generalization)
    base, base_seconds = run_model(args.base, cases, args.batch_size)
    tuned, tuned_seconds = run_model(args.finetuned, cases, args.batch_size)
    report = {
        "source": "authored_counterfactual",
        "dataset_sha256": hashlib.sha256(args.generalization.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "base": {**metrics(base), **group_metrics(cases, base), "elapsed_seconds": base_seconds},
        "finetuned": {**metrics(tuned), **group_metrics(cases, tuned), "elapsed_seconds": tuned_seconds},
        "paired_accuracy": paired_accuracy_interval(base, tuned),
        "predictions": {"base": base, "finetuned": tuned},
    }
    for name in ("base", "finetuned"):
        rows = report["predictions"][name]
        report[name]["negative_as_direct"] = sum(
            row["expected"] == "negative" and row["predicted"] == "direct" for row in rows)
        report[name]["high_confidence_errors_0_9"] = sum(
            row["expected"] != row["predicted"] and max(row["probabilities"].values()) >= 0.9
            for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ("base", "finetuned"):
        print(f"{name}: accuracy={report[name]['accuracy']:.3f}, macro F1={report[name]['macro_f1']:.3f}, "
              f"complete trios={report[name]['all_three_correct']}/{report[name]['groups']}", flush=True)
    print("Report:", args.output)


if __name__ == "__main__":
    main()
