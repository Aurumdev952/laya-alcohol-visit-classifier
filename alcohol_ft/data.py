"""Validate visit-level JSONL and prevent patient/case leakage across splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from alcohol_ft.task import LABELS, case_state, combine_components


def read_cases(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row["label"] not in LABELS:
                raise ValueError("invalid label")
            if not isinstance(row["case_id"], str) or not row["case_id"].strip() or not isinstance(row["patient_id"], str) or not row["patient_id"].strip():
                raise ValueError("missing case_id or patient_id")
            if not isinstance(row["encounters"], list):
                raise ValueError("encounters must be a list")
            row["state"] = case_state(row["encounters"])
            annotation = row.get("annotation")
            if annotation is not None:
                if not isinstance(annotation, dict):
                    raise ValueError("annotation must be an object")
                patient = annotation.get("patient_evidence")
                other = annotation.get("other_person_causal")
                if not isinstance(patient, bool) or not isinstance(other, bool):
                    raise ValueError("component labels must be boolean")
                if combine_components(patient, other) != row["label"]:
                    raise ValueError("component labels disagree with visit label")
                spans = annotation.get("evidence", [])
                if not isinstance(spans, list):
                    raise ValueError("evidence must be a list")
                for evidence in spans:
                    if not isinstance(evidence, dict):
                        raise ValueError("evidence entries must be objects")
                    index = evidence.get("encounter_index")
                    span = evidence.get("text")
                    if not isinstance(index, int) or not 1 <= index <= len(row["encounters"]) or not isinstance(span, str) or span not in row["encounters"][index - 1]["note"]:
                        raise ValueError("evidence span is missing from referenced encounter")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no cases")
    return rows


def validate_splits(paths: dict[str, Path]) -> dict:
    seen_case, seen_patient, seen_text, seen_group = {}, {}, {}, {}
    summary = {}
    for split, path in paths.items():
        rows = read_cases(path)
        counts = Counter()
        for row in rows:
            counts[row["label"]] += 1
            for key, value, seen in (
                ("case_id", row["case_id"], seen_case),
                ("state", hashlib.sha256(row["state"].encode()).hexdigest(), seen_text),
            ):
                if value in seen:
                    raise ValueError(f"{key} leakage: {value} in {seen[value]} and {split}")
                seen[value] = split
            patient_id = row["patient_id"]
            if patient_id in seen_patient and seen_patient[patient_id] != split:
                raise ValueError(f"patient_id leakage: {patient_id} in {seen_patient[patient_id]} and {split}")
            seen_patient[patient_id] = split
            group_id = row.get("group_id")
            if group_id is not None:
                if not isinstance(group_id, str) or not group_id:
                    raise ValueError("group_id must be nonempty text")
                if group_id in seen_group and seen_group[group_id] != split:
                    raise ValueError(f"group_id leakage: {group_id} in {seen_group[group_id]} and {split}")
                seen_group[group_id] = split
        summary[split] = {"total": len(rows), "labels": dict(counts)}
    return summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
