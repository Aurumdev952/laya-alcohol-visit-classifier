"""Run five 50-case sector routing evaluations on a local CUDA Laya checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from eval.run_eval import score_cases


DATA_DIR = Path(__file__).with_name("sectors")
DEFAULT_OUTPUT = DATA_DIR / "results.json"
MODEL_ID = "convaiinnovations/laya"


def load_tasks(path: Path = DATA_DIR / "tasks.json") -> dict:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not tasks:
        raise ValueError("no sector tasks found")
    for sector, task in tasks.items():
        if not task.get("instructions") or len(task.get("criteria", {})) != 5:
            raise ValueError(f"{sector}: expected instructions and exactly five categories")
        if any(not label or not description for label, description in task["criteria"].items()):
            raise ValueError(f"{sector}: labels and descriptions must be nonempty")
    return tasks


def load_sector_cases(sector: str, labels: tuple[str, ...], data_dir: Path = DATA_DIR) -> list[dict]:
    path = data_dir / f"{sector}.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [case["id"] for case in cases]
    counts = Counter(case["label"] for case in cases)
    if len(cases) != 50 or len(set(ids)) != 50 or counts != {label: 10 for label in labels}:
        raise ValueError(f"{sector}: expected 50 unique cases, 10 per category")
    if any(not isinstance(case["text"], str) or not case["text"].strip() for case in cases):
        raise ValueError(f"{sector}: every case needs nonempty text")
    return cases


def evaluate_sector(agent, sector: str, task: dict, cases: list[dict], batch_size: int) -> dict:
    from laya.common import build_sequence

    labels = tuple(task["criteria"])
    question = {
        "type": "choice",
        "instructions": task["instructions"],
        "criteria": task["criteria"],
    }
    internal = {"t": "choice", "ins": question["instructions"], "crit": question["criteria"]}
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    empty_sequence, _ = build_sequence(agent.tok, "", internal, max_len, head_max_len)
    max_text_tokens = max_len - len(empty_sequence)
    for case in cases:
        count = len(agent.tok(case["text"], add_special_tokens=False)["input_ids"])
        if count > max_text_tokens:
            raise ValueError(f"{sector}/{case['id']}: text exceeds the model input window")

    if agent.device.type != "cuda":
        raise RuntimeError("Laya is not running on CUDA")
    started = time.perf_counter()
    responses = agent.predict_batch(
        [case["text"] for case in cases], {"category": question}, batch_size=batch_size
    )
    if agent.device.type != "cuda":
        raise RuntimeError(f"{sector}: Laya fell back to CPU")
    if len(responses) != len(cases):
        raise RuntimeError(f"{sector}: Laya returned the wrong number of answers")

    rows = []
    for case, response in zip(cases, responses):
        answer = response["answers"]["category"]
        predicted = answer["choice"]
        probabilities = answer["probabilities"]
        if predicted not in labels or set(probabilities) != set(labels):
            raise RuntimeError(f"{sector}/{case['id']}: invalid model answer")
        rows.append({
            "id": case["id"],
            "text": case["text"],
            "expected": case["label"],
            "predicted": predicted,
            "probabilities": probabilities,
        })

    report = score_cases(rows, labels)
    report.update({
        "sector": sector,
        "instructions": task["instructions"],
        "criteria": task["criteria"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cases": rows,
    })
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

        tasks = load_tasks(args.data_dir / "tasks.json")
        cases_by_sector = {
            sector: load_sector_cases(sector, tuple(task["criteria"]), args.data_dir)
            for sector, task in tasks.items()
        }
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        gpu = torch.cuda.get_device_name(0)
        if args.require_gpu and args.require_gpu.lower() not in gpu.lower():
            raise RuntimeError(f"GPU is {gpu!r}, not {args.require_gpu!r}")
        agent = laya.load(MODEL_ID, device="cuda")
        if agent.device.type != "cuda":
            raise RuntimeError("Laya loaded on CPU")

        sectors = {}
        for sector, task in tasks.items():
            report = evaluate_sector(agent, sector, task, cases_by_sector[sector], args.batch_size)
            sectors[sector] = report
            print(f"{sector}: {report['correct']}/50 ({report['accuracy']:.1%}), "
                  f"macro F1 {report['macro_f1']:.4f}", flush=True)

        total_cases = sum(report["total"] for report in sectors.values())
        correct = sum(report["correct"] for report in sectors.values())
        summary = {
            "sectors": len(sectors),
            "cases_per_sector": 50,
            "total_cases": total_cases,
            "correct": correct,
            "overall_accuracy": round(correct / total_cases, 4),
            "mean_sector_macro_f1": round(mean(report["macro_f1"] for report in sectors.values()), 4),
        }
        output = {
            "model": MODEL_ID,
            "laya_version": laya.__version__,
            "torch_version": torch.__version__,
            "gpu": gpu,
            "device": agent.device.type,
            "autocast_dtype": str(agent.dtype),
            "batch_size": args.batch_size,
            "summary": summary,
            "sectors": sectors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"Sector evaluation failed: {exc}\n")

    print(f"Overall: {summary['correct']}/{summary['total_cases']} "
          f"({summary['overall_accuracy']:.1%})")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
