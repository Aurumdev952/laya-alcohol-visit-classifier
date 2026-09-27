"""Evaluate 50 labeled cases against the running CUDA sentiment API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib import error, request


LABELS = ("positive", "neutral", "negative")
DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")
DEFAULT_OUTPUT = Path(__file__).with_name("results.json")


def load_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [case["id"] for case in cases]
    if len(cases) != 50 or len(set(ids)) != 50:
        raise ValueError("evaluation data must contain exactly 50 cases with unique IDs")
    for case in cases:
        if case["label"] not in LABELS or not isinstance(case["text"], str) or not case["text"].strip():
            raise ValueError(f"invalid evaluation case: {case['id']}")
    return cases


def score_cases(rows: list[dict], labels: tuple[str, ...] = LABELS) -> dict:
    matrix = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1

    by_label = {}
    for label in labels:
        tp = matrix[label][label]
        actual = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_label[label] = {
            "support": actual,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    correct = sum(matrix[label][label] for label in labels)
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "macro_f1": round(sum(by_label[label]["f1"] for label in labels) / len(labels), 4),
        "per_label": by_label,
        "confusion_matrix": matrix,
        "mistakes": [row for row in rows if row["expected"] != row["predicted"]],
    }


def get_json(url: str, payload: dict | None, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc


def evaluate(base_url: str, cases: list[dict], timeout: float, required_gpu: str, progress: bool = True) -> dict:
    base_url = base_url.rstrip("/")
    health = get_json(f"{base_url}/health", None, timeout)
    gpu = health.get("gpu", "")
    if health.get("status") != "ok" or health.get("device") != "cuda":
        raise RuntimeError(f"API is not reporting a healthy CUDA model: {health}")
    if required_gpu and required_gpu.lower() not in gpu.lower():
        raise RuntimeError(f"API reports GPU {gpu!r}; expected a name containing {required_gpu!r}")

    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        response = get_json(f"{base_url}/sentiment", {"text": case["text"]}, timeout)
        predicted = response.get("sentiment")
        probabilities = response.get("probabilities", {})
        if predicted not in LABELS or set(probabilities) != set(LABELS):
            raise RuntimeError(f"invalid API response for {case['id']}: {response}")
        rows.append({
            "id": case["id"],
            "text": case["text"],
            "expected": case["label"],
            "predicted": predicted,
            "probabilities": probabilities,
        })
        if progress and index % 10 == 0:
            print(f"Evaluated {index}/{len(cases)} cases", flush=True)

    report = score_cases(rows)
    report.update({
        "model": health.get("model"),
        "gpu": gpu,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cases": rows,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--require-gpu", default="5090")
    args = parser.parse_args()

    try:
        report = evaluate(args.base_url, load_cases(args.cases), args.timeout, args.require_gpu)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Accuracy: {report['correct']}/{report['total']} ({report['accuracy']:.1%})")
    print(f"Macro F1: {report['macro_f1']:.3f}")
    print(f"Mistakes: {len(report['mistakes'])}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
