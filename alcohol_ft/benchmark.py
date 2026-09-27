"""Run a locked visit-level benchmark for base and fine-tuned Laya."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from alcohol_ft.data import read_cases, validate_splits
from alcohol_ft.task import LABELS, QUESTION


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        raise ValueError("empty benchmark")
    matrix = {true: {pred: 0 for pred in LABELS} for true in LABELS}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    per_label = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[actual][label] for actual in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"support": support, "precision": precision,
                            "recall": recall, "f1": f1}
    correct = [int(row["expected"] == row["predicted"]) for row in rows]
    brier = statistics.mean(sum((row["probabilities"][label] - (label == row["expected"])) ** 2
                                for label in LABELS) for row in rows)
    bins = []
    ece = 0.0
    for low in range(0, 10):
        bucket = [row for row in rows if min(9, int(max(row["probabilities"].values()) * 10)) == low]
        if bucket:
            confidence = statistics.mean(max(row["probabilities"].values()) for row in bucket)
            accuracy = statistics.mean(row["expected"] == row["predicted"] for row in bucket)
            ece += len(bucket) / n * abs(accuracy - confidence)
            bins.append({"range": [low / 10, (low + 1) / 10], "count": len(bucket),
                         "accuracy": accuracy, "mean_confidence": confidence})
    coverage = {}
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        selected = [row for row in rows if max(row["probabilities"].values()) >= threshold]
        coverage[str(threshold)] = {"count": len(selected), "coverage": len(selected) / n,
                                    "accuracy": statistics.mean(r["expected"] == r["predicted"]
                                                                for r in selected) if selected else None}
    slices = {}
    for name in sorted({row.get("scenario", "other") for row in rows}):
        selected = [row for row in rows if row.get("scenario", "other") == name]
        slices[name] = {"count": len(selected),
                        "accuracy": statistics.mean(r["expected"] == r["predicted"] for r in selected)}
    return {"total": n, "accuracy": statistics.mean(correct), "macro_f1": statistics.mean(
        value["f1"] for value in per_label.values()), "per_label": per_label,
        "confusion_matrix": matrix, "brier": brier, "ece_10bin": ece,
        "calibration_bins": bins, "confidence_coverage": coverage, "slices": slices,
        "direct_missed": sum(r["expected"] == "direct" and r["predicted"] != "direct" for r in rows),
        "indirect_missed": sum(r["expected"] == "indirect" and r["predicted"] != "indirect" for r in rows)}


def paired_accuracy_interval(base: list[dict], tuned: list[dict], seed=20260927, replicates=2000):
    if [r["case_id"] for r in base] != [r["case_id"] for r in tuned]:
        raise ValueError("paired runs have different case order")
    rng = random.Random(seed)
    n = len(base)
    delta = [(t["predicted"] == t["expected"]) - (b["predicted"] == b["expected"])
             for b, t in zip(base, tuned)]
    draws = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates))
    return {"delta": statistics.mean(delta), "bootstrap_95_interval":
            [draws[int(0.025 * replicates)], draws[int(0.975 * replicates)]]}


def run_model(model_id: str, cases: list[dict], batch_size: int):
    import laya
    import torch
    from laya.common import build_sequence

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; benchmark requires GPU")
    agent = laya.load(model_id, device="cuda")
    if agent.device.type != "cuda":
        raise RuntimeError("Laya did not load on CUDA")
    q = QUESTION["alcohol_attribution"]
    internal = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
    max_len = agent.cfg.get("max_len", 512)
    empty, _ = build_sequence(agent.tok, "", internal, max_len, agent.cfg.get("head_max_len", 192))
    budget = max_len - len(empty)
    for case in cases:
        state = case["state"].replace(agent.tok.mask_token, " ") if agent.tok.mask_token else case["state"]
        length = len(agent.tok(state, add_special_tokens=False)["input_ids"])
        if length > budget:
            raise ValueError(f"{case['case_id']}: {length} tokens exceed model budget {budget}")
    started = time.perf_counter()
    answers = agent.predict_batch([case["state"] for case in cases], QUESTION, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    if len(answers) != len(cases) or agent.device.type != "cuda":
        raise RuntimeError("incomplete predictions or CUDA fallback")
    rows = []
    for case, response in zip(cases, answers):
        answer = response["answers"]["alcohol_attribution"]
        predicted = answer["choice"]
        probs = answer["probabilities"]
        if predicted not in LABELS or set(probs) != set(LABELS):
            raise ValueError(f"invalid answer for {case['case_id']}")
        if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in probs.values()):
            raise ValueError(f"invalid probabilities for {case['case_id']}")
        if abs(sum(float(p) for p in probs.values()) - 1) > 0.01:
            raise ValueError(f"probabilities do not sum to 1 for {case['case_id']}")
        rows.append({"case_id": case["case_id"], "scenario": case.get("scenario", "other"),
                     "expected": case["label"], "predicted": predicted,
                     "probabilities": {label: float(probs[label]) for label in LABELS}})
    del agent
    torch.cuda.empty_cache()
    return rows, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", required=True, help="local final checkpoint path")
    parser.add_argument("--base", default="convaiinnovations/laya")
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic"))
    parser.add_argument("--curated", type=Path, default=Path("eval/alcohol/curated.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/laya-alcohol/benchmark.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    paths = {split: args.data_dir / f"{split}.jsonl" for split in ("train", "calibration", "test")}
    paths["curated"] = args.curated
    print("Data validation:", validate_splits(paths), flush=True)
    report = {"models": {"base": args.base, "finetuned": args.finetuned}, "datasets": {}}
    for name in ("test", "curated"):
        cases = read_cases(paths[name])
        base, base_seconds = run_model(args.base, cases, args.batch_size)
        tuned, tuned_seconds = run_model(args.finetuned, cases, args.batch_size)
        report["datasets"][name] = {
            "base": {**metrics(base), "elapsed_seconds": base_seconds},
            "finetuned": {**metrics(tuned), "elapsed_seconds": tuned_seconds},
            "paired_accuracy": paired_accuracy_interval(base, tuned),
            "predictions": {"base": base, "finetuned": tuned},
        }
        print(f"{name}: base macro F1 {report['datasets'][name]['base']['macro_f1']:.3f}; "
              f"fine-tuned macro F1 {report['datasets'][name]['finetuned']['macro_f1']:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Report:", args.output)


if __name__ == "__main__":
    main()
