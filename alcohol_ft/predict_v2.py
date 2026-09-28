"""Classify unlabeled visit reports using a prepared v2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alcohol_ft.benchmark_v2 import predict_classifier, predict_laya
from alcohol_ft.task import case_state


def read_input(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("empty input")
    try:
        value = json.loads(raw)
        cases = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    prepared = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("each case must be a JSON object")
        prepared.append({"case_id": case.get("case_id", f"case-{index+1}"),
                         "state": case_state(case["encounters"])})
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--review-below", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.review_below <= 1 or args.batch_size < 1:
        parser.error("invalid review threshold or batch size")
    cases = read_input(args.input)
    if (args.model / "alcohol_config.json").exists():
        predictions, mode = predict_classifier(str(args.model), cases, args.batch_size), "three_way"
    else:
        predictions, mode = predict_laya(str(args.model), cases, args.batch_size)
    for row in predictions:
        confidence = row["probabilities"][row["predicted"]]
        result = {"case_id": row["case_id"], "label": row["predicted"],
                  "probabilities": row["probabilities"], "confidence": confidence,
                  "review_required": confidence < args.review_below}
        if mode == "decomposed":
            result["components"] = row["components"]
            result["probability_note"] = "Approximate joint probability; validate calibration before threshold use."
        print(json.dumps(result))


if __name__ == "__main__":
    main()
