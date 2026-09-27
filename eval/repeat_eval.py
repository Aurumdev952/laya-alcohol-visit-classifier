"""Repeat the same 50-case GPU evaluation and summarize its scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from eval.run_eval import DEFAULT_CASES, evaluate, load_cases


DEFAULT_OUTPUT = Path(__file__).with_name("repeated_results.json")


def summarize_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("at least one run is required")
    accuracies = [run["accuracy"] for run in runs]
    macro_f1s = [run["macro_f1"] for run in runs]
    case_ids = [row["id"] for row in runs[0]["cases"]]
    changed_cases = []
    for case_id in case_ids:
        labels = {
            next(row["predicted"] for row in run["cases"] if row["id"] == case_id)
            for run in runs
        }
        if len(labels) > 1:
            changed_cases.append(case_id)
    return {
        "runs": len(runs),
        "cases_per_run": runs[0]["total"],
        "mean_accuracy": round(mean(accuracies), 4),
        "accuracy_stddev": round(pstdev(accuracies), 4),
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
        "mean_macro_f1": round(mean(macro_f1s), 4),
        "macro_f1_stddev": round(pstdev(macro_f1s), 4),
        "prediction_changes": changed_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--require-gpu", default="5090")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")

    try:
        cases = load_cases(args.cases)
        runs = []
        for index in range(1, args.runs + 1):
            run = evaluate(args.base_url, cases, args.timeout, args.require_gpu, progress=False)
            runs.append(run)
            print(f"Run {index}/{args.runs}: {run['correct']}/{run['total']} "
                  f"({run['accuracy']:.1%}), macro F1 {run['macro_f1']:.4f}", flush=True)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Repeated evaluation failed: {exc}\n")

    summary = summarize_runs(runs)
    report = {
        "model": runs[0]["model"],
        "gpu": runs[0]["gpu"],
        "summary": summary,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Mean accuracy: {summary['mean_accuracy']:.1%} "
          f"(SD {summary['accuracy_stddev']:.4f})")
    print(f"Mean macro F1: {summary['mean_macro_f1']:.4f} "
          f"(SD {summary['macro_f1_stddev']:.4f})")
    print(f"Cases with changed predictions: {len(summary['prediction_changes'])}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

