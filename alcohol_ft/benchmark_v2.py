"""Score one v2 checkpoint, including complete counterfactual groups and slices."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from alcohol_ft.benchmark import metrics
from alcohol_ft.data import read_cases
from alcohol_ft.task import LABELS, combine_components, combine_probabilities, questions_for


def summarize(cases: list[dict], predictions: list[dict]) -> dict:
    by_case = {row["case_id"]: row for row in predictions}
    if len(by_case) != len(cases):
        raise ValueError("incomplete or duplicate predictions")
    ordered = [by_case[case["case_id"]] for case in cases]
    report = metrics(ordered)
    report["negative_to_direct"] = sum(r["expected"] == "negative" and r["predicted"] == "direct"
                                       for r in ordered)
    report["high_confidence_errors_90"] = sum(r["expected"] != r["predicted"] and
                                              r["probabilities"][r["predicted"]] >= 0.9 for r in ordered)
    if any("components" in row for row in ordered):
        report["brier"] = None
        report["ece_10bin"] = None
        report["calibration_bins"] = []
        report["confidence_coverage"] = {}
        report["probability_note"] = "Two marginal answers give a heuristic joint distribution; calibration metrics are withheld."
    grouped = defaultdict(list)
    for case, prediction in zip(cases, ordered):
        grouped[case.get("group_id", case["case_id"])].append(prediction)
    complete = [group for group in grouped.values() if len(group) == 3 and
                {row["expected"] for row in group} == set(LABELS)]
    report["counterfactual"] = {"groups": len(complete),
                                 "complete_correct": sum(all(row["expected"] == row["predicted"]
                                                             for row in group) for group in complete),
                                 "complete_fraction": statistics.mean(
                                     all(row["expected"] == row["predicted"] for row in group)
                                     for group in complete) if complete else None}
    slices = {}
    for name, selected in {
        "seen_family": [c for c in cases if c.get("family_seen_in_training") is True],
        "unseen_family": [c for c in cases if c.get("family_seen_in_training") is False],
        "direct_and_indirect": [c for c in cases if c.get("annotation", {}).get("patient_evidence")
                                and c.get("annotation", {}).get("other_person_causal")],
        "negative_with_alcohol_word": [c for c in cases if c["label"] == "negative" and
                                       "alcohol" in c["state"].lower()],
        "negative_without_alcohol_word": [c for c in cases if c["label"] == "negative" and
                                          "alcohol" not in c["state"].lower()],
    }.items():
        if selected:
            values = [by_case[c["case_id"]] for c in selected]
            slices[name] = {"count": len(values), "accuracy": statistics.mean(
                row["expected"] == row["predicted"] for row in values),
                "negative_to_direct": sum(row["expected"] == "negative" and row["predicted"] == "direct"
                                          for row in values)}
    report["targeted_slices"] = slices
    categories = {}
    for category in sorted({case.get("signal_category", "other") for case in cases}):
        values = [by_case[c["case_id"]] for c in cases if c.get("signal_category", "other") == category]
        categories[category] = {"count": len(values), "accuracy": statistics.mean(
            row["expected"] == row["predicted"] for row in values),
            "negative_to_direct": sum(row["expected"] == "negative" and row["predicted"] == "direct"
                                      for row in values)}
    report["category_slices"] = categories
    return report


def predict_laya(model_path: str, cases: list[dict], batch_size: int) -> tuple[list[dict], str]:
    import laya
    import torch
    from laya.common import build_sequence

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    agent = laya.load(model_path, device="cuda")
    if agent.device.type != "cuda":
        raise RuntimeError("checkpoint did not load on GPU")
    mode = agent.cfg.get("alcohol_question_mode", "three_way")
    questions = questions_for(mode)
    for qid, question in questions.items():
        internal = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
        empty, _ = build_sequence(agent.tok, "", internal,
                                  agent.cfg["max_len"], agent.cfg["head_max_len"])
        budget = agent.cfg["max_len"] - len(empty)
        for case in cases:
            state = case["state"].replace(agent.tok.mask_token, " ") if agent.tok.mask_token else case["state"]
            length = len(agent.tok(state, add_special_tokens=False)["input_ids"])
            if length > budget:
                raise ValueError(f"{case['case_id']} / {qid}: input exceeds {budget} tokens")
    responses = agent.predict_batch([case["state"] for case in cases], questions, batch_size=batch_size)
    if len(responses) != len(cases):
        raise RuntimeError("incomplete Laya response")
    rows = []
    for case, response in zip(cases, responses):
        answers = response["answers"]
        if mode == "three_way":
            answer = answers["alcohol_attribution"]
            probabilities = {label: float(answer["probabilities"][label]) for label in LABELS}
            predicted = answer["choice"]
            components = None
        else:
            p_patient = float(answers["patient_alcohol"]["probabilities"]["documented"])
            p_other = float(answers["other_person_causal"]["probabilities"]["causal"])
            probabilities = combine_probabilities(p_patient, p_other)
            predicted = combine_components(p_patient >= 0.5, p_other >= 0.5)
            components = {"patient_evidence": p_patient, "other_person_causal": p_other}
        if predicted not in LABELS or abs(sum(probabilities.values()) - 1) > 0.01:
            raise RuntimeError("invalid prediction")
        row = {"case_id": case["case_id"], "expected": case.get("label"),
               "predicted": predicted, "probabilities": probabilities,
               "scenario": case.get("scenario", "other")}
        if components is not None:
            row["components"] = components
        rows.append(row)
    if agent.device.type != "cuda":
        raise RuntimeError("Laya fell back to CPU")
    return rows, mode


def predict_classifier(model_path: str, cases: list[dict], batch_size: int) -> list[dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    path = Path(model_path)
    cfg = json.loads((path / "alcohol_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda")
    model.eval()
    prepared = []
    for case in cases:
        ids = tokenizer(case["state"], add_special_tokens=True)["input_ids"]
        if len(ids) > cfg["max_len"]:
            raise ValueError(f"{case['case_id']}: input exceeds {cfg['max_len']} tokens")
        prepared.append(ids)
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(cases), batch_size):
            batch = prepared[offset:offset + batch_size]
            width = max(map(len, batch))
            ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids)
            for i, sequence in enumerate(batch):
                ids[i, :len(sequence)] = torch.tensor(sequence, device="cuda")
                mask[i, :len(sequence)] = 1
            logits = model(input_ids=ids, attention_mask=mask).logits.float() / cfg["temperature"]
            probs = torch.softmax(logits, -1).cpu().tolist()
            for case, values in zip(cases[offset:offset + batch_size], probs):
                probabilities = dict(zip(LABELS, values))
                rows.append({"case_id": case["case_id"], "expected": case.get("label"),
                             "predicted": max(probabilities, key=probabilities.get),
                             "probabilities": probabilities,
                             "scenario": case.get("scenario", "other")})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", choices=("laya", "classifier"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic_v2"))
    parser.add_argument("--split", choices=("validation", "test", "custom"), required=True)
    parser.add_argument("--cases", type=Path, help="labeled JSONL for a custom diagnostic set")
    parser.add_argument("--unlock-test", action="store_true",
                        help="explicitly allow the once-per-finalist locked test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.unlock_test:
        parser.error("test split is locked; pass --unlock-test only for finalized candidates")
    if (args.split == "custom") != (args.cases is not None):
        parser.error("--cases is required exactly when --split custom is used")
    if args.batch_size < 1 or args.output.exists():
        parser.error("batch size must be positive and output path must be new")
    cases_path = args.cases if args.cases is not None else args.data_dir / f"{args.split}.jsonl"
    cases = read_cases(cases_path)
    start = time.perf_counter()
    if args.family == "laya":
        rows, mode = predict_laya(args.model, cases, args.batch_size)
    else:
        rows, mode = predict_classifier(args.model, cases, args.batch_size), "three_way"
    report = {"model": args.model, "model_family": args.family, "question_mode": mode,
              "data_dir": str(args.data_dir), "split": args.split,
              "cases_path": str(cases_path),
              "elapsed_seconds": time.perf_counter() - start,
              "metrics": summarize(cases, rows), "predictions": rows}
    if mode == "decomposed":
        report["probability_note"] = "Joint visit probabilities assume independent components; calibrate separately."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{args.split}: macro_f1={report['metrics']['macro_f1']:.4f}; "
          f"negative_recall={report['metrics']['per_label']['negative']['recall']:.4f}; "
          f"complete_groups={report['metrics']['counterfactual']['complete_correct']}/"
          f"{report['metrics']['counterfactual']['groups']}")


if __name__ == "__main__":
    main()
