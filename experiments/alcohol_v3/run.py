"""Run a v3 Laya experiment; dry-run by default.

Examples:
  python experiments/alcohol_v3/run.py laya_rlcd
  python experiments/alcohol_v3/run.py laya_rlcd --execute
  python experiments/alcohol_v3/run.py laya_rlcd --test --execute
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="stem of experiments/alcohol_v3/<name>.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="score a frozen finalist on the locked test")
    parser.add_argument("--execute", action="store_true", help="start GPU work")
    args = parser.parse_args()
    config_path = CONFIG_DIR / f"{args.name}.json"
    if args.seed < 0 or not config_path.is_file():
        parser.error("seed must be nonnegative and experiment config must exist")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = args.seed
    run_dir = ROOT / "runs" / "alcohol-v3" / f"{args.name}-seed{args.seed}"
    effective = ROOT / "runs" / "alcohol-v3" / "configs" / f"{args.name}-seed{args.seed}.json"
    final = run_dir / "final"
    split = "test" if args.test else "validation"
    result = run_dir / f"{split}_benchmark.json"
    if args.test:
        if not final.is_dir():
            parser.error(f"final checkpoint missing: {final}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"run already started; use trainer --resume: {run_dir}")
    if result.exists():
        parser.error(f"benchmark already exists: {result}")
    cmd_benchmark = [sys.executable, "-m", "alcohol_ft.benchmark_v2",
                     "--model", str(final), "--family", "laya",
                     "--data-dir", config["data_dir"], "--split", split,
                     "--output", str(result)]
    if args.test:
        cmd_benchmark.append("--unlock-test")
        commands = [cmd_benchmark]
    else:
        commands = [[sys.executable, "-m", "alcohol_ft.train_v2",
                     "--config", str(effective), "--output-dir", str(run_dir)],
                    cmd_benchmark]
    print(json.dumps({"config": config, "commands": commands,
                      "will_execute": args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    if not args.test:
        effective.parent.mkdir(parents=True, exist_ok=True)
        effective.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
