"""Audit the frozen synthetic v2 data before any training experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from alcohol_ft.data import read_cases, validate_splits
from alcohol_ft.generate_v2 import SEEN_FAMILIES, SPLITS, UNSEEN_FAMILIES
from alcohol_ft.task import LABELS


def audit(data_dir: Path, tokenizer_path: str | None = None) -> dict:
    paths = {part: data_dir / f"{part}.jsonl" for part in SPLITS}
    summary = validate_splits(paths)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != 2:
        raise ValueError("unexpected dataset version")
    report = {"counts": summary, "splits": {}, "manifest_sha256_verified": True}
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    for part, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][part]:
            raise ValueError(f"manifest hash mismatch: {part}")
        rows = read_cases(path)
        if len(rows) != SPLITS[part] * 3 or Counter(r["label"] for r in rows) != {x: SPLITS[part] for x in LABELS}:
            raise ValueError(f"wrong v2 count or class balance: {part}")
        groups = defaultdict(list)
        for row in rows:
            if row.get("source") != "fully_synthetic_v2" or not row.get("group_id"):
                raise ValueError("missing v2 provenance")
            groups[row["group_id"]].append(row)
            annotation = row["annotation"]
            if bool(annotation["evidence"]) != (row["label"] != "negative"):
                raise ValueError(f"missing/unexpected evidence: {row['case_id']}")
            if part == "test":
                expected_seen = row["scenario"] in {x[0] for x in SEEN_FAMILIES}
                if row["family_seen_in_training"] != expected_seen:
                    raise ValueError("incorrect seen-family tag")
            elif row["scenario"] in {x[0] for x in UNSEEN_FAMILIES}:
                raise ValueError("reserved family leaked into model development")
        for group in groups.values():
            if len(group) != 3 or {row["label"] for row in group} != set(LABELS):
                raise ValueError(f"incomplete counterfactual group: {group[0]['group_id']}")
            if len({row["scenario"] for row in group}) != 1:
                raise ValueError("counterfactual family mismatch")
        lengths = [len(tokenizer(r["state"], add_special_tokens=False)["input_ids"])
                   for r in rows] if tokenizer else []
        report["splits"][part] = {
            "groups": len(groups), "families": dict(Counter(r["scenario"] for r in rows)),
            "render_styles": dict(Counter(str(r["render_style"]) for r in rows)),
            "negative_alcohol_word": sum(r["label"] == "negative" and "alcohol" in r["state"].lower()
                                         for r in rows),
            "mixed_direct_indirect": sum(r["annotation"]["patient_evidence"] and
                                         r["annotation"]["other_person_causal"] for r in rows),
            "token_length": {"max": max(lengths), "p95": sorted(lengths)[int(len(lengths) * 0.95)]}
                            if lengths else None,
        }
    if set(report["splits"]["train"]["render_styles"]) & set(report["splits"]["test"]["render_styles"]):
        raise ValueError("train/test render style overlap")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic_v2"))
    parser.add_argument("--tokenizer", help="optional tokenizer path or model ID for length audit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.data_dir, args.tokenizer)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "groups": {key: value["groups"]
                      for key, value in report["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
