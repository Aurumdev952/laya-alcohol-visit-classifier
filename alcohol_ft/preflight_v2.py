"""Check Laya question formatting and token budgets without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alcohol_ft.data import read_cases
from alcohol_ft.train_v2 import prepare_items


def run(data_dir: Path, tokenizer_path: str, mode: str, max_len: int, head_max_len: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cfg = {"max_len": max_len, "head_max_len": head_max_len}
    report = {"question_mode": mode, "tokenizer": tokenizer_path, "max_len": max_len,
              "head_max_len": head_max_len, "splits": {}}
    for split in ("train", "validation", "calibration", "test"):
        rows = read_cases(data_dir / f"{split}.jsonl")
        count, maximum, minimum = 0, 0, max_len
        for offset in range(0, len(rows), 256):
            items = prepare_items(rows[offset:offset + 256], tokenizer, cfg, mode)
            count += len(items)
            maximum = max(maximum, *(len(item["ids"]) for item in items))
            minimum = min(minimum, *(len(item["ids"]) for item in items))
        report["splits"][split] = {"visits": len(rows), "questions": count,
                                    "min_sequence_tokens": minimum, "max_sequence_tokens": maximum}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/alcohol_synthetic_v2"))
    parser.add_argument("--tokenizer", default="runs/laya-alcohol-001/final/tokenizer")
    parser.add_argument("--mode", choices=("three_way", "decomposed"), required=True)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--head-max-len", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.data_dir, args.tokenizer, args.mode, args.max_len, args.head_max_len)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
