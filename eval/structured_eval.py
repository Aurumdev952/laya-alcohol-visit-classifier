"""Evaluate Laya's bounded JSON-schema extraction on three 50-case tasks."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).with_name("structured")
DEFAULT_OUTPUT = DATA_DIR / "results.json"
MODEL_ID = "convaiinnovations/laya"


def field_values(property_schema: dict) -> tuple:
    if "enum" in property_schema:
        return tuple(property_schema["enum"])
    if property_schema["type"] == "boolean":
        return (False, True)
    if property_schema["type"] == "integer":
        return tuple(range(property_schema["minimum"], property_schema["maximum"] + 1))
    raise ValueError("unsupported evaluation field")


def load_tasks(data_dir: Path = DATA_DIR) -> dict:
    tasks = json.loads((data_dir / "tasks.json").read_text(encoding="utf-8"))
    if set(tasks) != {"support", "education", "mining"}:
        raise ValueError("expected support, education, and mining tasks")
    for name, task in tasks.items():
        properties = task["schema"]["properties"]
        if len(properties) != 3 or any(not field_values(prop) for prop in properties.values()):
            raise ValueError(f"{name}: expected three bounded fields")
    return tasks


def load_cases(task_name: str, schema: dict, data_dir: Path = DATA_DIR) -> list[dict]:
    path = data_dir / f"{task_name}.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 50 or len({case["id"] for case in cases}) != 50:
        raise ValueError(f"{task_name}: expected 50 unique cases")
    properties = schema["properties"]
    for case in cases:
        if not isinstance(case["text"], str) or not case["text"].strip():
            raise ValueError(f"{task_name}/{case['id']}: empty text")
        expected = case["expected"]
        if set(expected) != set(properties):
            raise ValueError(f"{task_name}/{case['id']}: expected fields do not match schema")
        for field, prop in properties.items():
            value = expected[field]
            if type(value) is not type(field_values(prop)[0]) or value not in field_values(prop):
                raise ValueError(f"{task_name}/{case['id']}: invalid {field} value")
    return cases


def score_records(rows: list[dict], schema: dict) -> dict:
    properties = schema["properties"]
    exact = sum(row["expected"] == row["predicted"] for row in rows)
    by_field = {}
    for field, prop in properties.items():
        values = field_values(prop)
        correct = sum(row["expected"][field] == row["predicted"][field] for row in rows)
        matrix = {
            str(actual): {
                str(predicted): sum(
                    row["expected"][field] == actual and row["predicted"][field] == predicted
                    for row in rows
                ) for predicted in values
            } for actual in values
        }
        by_field[field] = {
            "correct": correct,
            "accuracy": round(correct / len(rows), 4),
            "confusion_matrix": matrix,
        }
    return {
        "total": len(rows),
        "exact_records": exact,
        "exact_accuracy": round(exact / len(rows), 4),
        "field_accuracy": round(sum(field["correct"] for field in by_field.values()) /
                                (len(rows) * len(by_field)), 4),
        "by_field": by_field,
        "mistakes": [row for row in rows if row["expected"] != row["predicted"]],
    }


def evaluate_task(agent, task_name: str, schema: dict, cases: list[dict], batch_size: int) -> dict:
    from laya.common import build_sequence
    from laya.structured import answers_to_json, questions_from_json_schema

    questions = questions_from_json_schema(schema)
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    budgets = []
    for question in questions.values():
        internal = agent._to_internal(question)
        empty, _ = build_sequence(agent.tok, "", internal, max_len, head_max_len)
        budgets.append(max_len - len(empty))
    max_text_tokens = min(budgets)
    for case in cases:
        count = len(agent.tok(case["text"], add_special_tokens=False)["input_ids"])
        if count > max_text_tokens:
            raise ValueError(f"{task_name}/{case['id']}: text would be truncated")

    started = time.perf_counter()
    results = agent.predict_batch([case["text"] for case in cases], questions, batch_size=batch_size)
    if agent.device.type != "cuda":
        raise RuntimeError(f"{task_name}: inference left CUDA for {agent.device}")
    if len(results) != len(cases):
        raise RuntimeError(f"{task_name}: wrong number of Laya responses")

    rows = []
    for case, result in zip(cases, results):
        answers = result["answers"]
        values = answers_to_json(answers, schema)
        if set(values) != set(schema["properties"]):
            raise RuntimeError(f"{task_name}/{case['id']}: incomplete structured answer")
        rows.append({
            "id": case["id"],
            "text": case["text"],
            "expected": case["expected"],
            "predicted": values,
            "raw_answers": answers,
        })
    report = score_records(rows, schema)
    report.update({"schema": schema, "elapsed_seconds": round(time.perf_counter() - started, 3),
                   "cases": rows})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--require-gpu", default="5090")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    try:
        import laya
        import torch

        tasks = load_tasks(args.data_dir)
        cases = {name: load_cases(name, task["schema"], args.data_dir)
                 for name, task in tasks.items()}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        gpu = torch.cuda.get_device_name(0)
        if args.require_gpu and args.require_gpu.lower() not in gpu.lower():
            raise RuntimeError(f"GPU is {gpu!r}, not {args.require_gpu!r}")

        agent = laya.load(MODEL_ID, device="cuda")
        if agent.device.type != "cuda":
            raise RuntimeError(f"Laya loaded on {agent.device}, expected CUDA")
        reports = {}
        for name, task in tasks.items():
            report = evaluate_task(agent, name, task["schema"], cases[name], args.batch_size)
            if agent.device.type != "cuda":
                raise RuntimeError(f"{name}: Laya fell back to {agent.device}")
            reports[name] = report
            print(f"{name}: exact {report['exact_records']}/50 ({report['exact_accuracy']:.1%}); "
                  f"field accuracy {report['field_accuracy']:.1%}", flush=True)

        total = sum(report["total"] for report in reports.values())
        exact = sum(report["exact_records"] for report in reports.values())
        field_correct = sum(sum(field["correct"] for field in report["by_field"].values())
                            for report in reports.values())
        output = {
            "model": MODEL_ID,
            "laya_version": laya.__version__,
            "torch_version": torch.__version__,
            "device": agent.device.type,
            "gpu": gpu,
            "autocast_dtype": str(agent.dtype),
            "batch_size": args.batch_size,
            "summary": {
                "tasks": len(reports), "total_records": total,
                "exact_records": exact, "exact_accuracy": round(exact / total, 4),
                "field_correct": field_correct,
                "field_total": total * 3,
                "field_accuracy": round(field_correct / (total * 3), 4),
            },
            "tasks_report": reports,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Structured evaluation failed: {exc}\n")

    print(f"Overall exact: {exact}/{total} ({output['summary']['exact_accuracy']:.1%})")
    print(f"Overall fields: {field_correct}/{total * 3} "
          f"({output['summary']['field_accuracy']:.1%})")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
