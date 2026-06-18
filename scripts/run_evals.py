#!/usr/bin/env python3
"""Run the evaluation for every model in models.yaml, then summarise.

    uv run python scripts/run_evals.py            # all models in models.yaml
    uv run python scripts/run_evals.py --mock     # network-free dry run of all

Each model is evaluated independently (a failure on one does not abort the rest)
and its result lands in evals/<id>.json. Finally summarize_results.py is invoked.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-file", default=str(ROOT / "models.yaml"))
    p.add_argument("--mock", action="store_true", help="use the MockProvider for every model")
    return p.parse_args()


def main():
    args = parse_args()
    models_file = yaml.safe_load(Path(args.models_file).read_text())
    model_ids = [m["id"] for m in models_file.get("models", [])]
    if not model_ids:
        print("no models enabled in models.yaml — uncomment some or use --mock with mock id")

    failed = []
    for mid in model_ids:
        print(f"\n=== {mid} ===")
        cmd = [sys.executable, str(ROOT / "scripts" / "model.py"), "--model", mid]
        if args.mock:
            cmd.append("--mock")
        if subprocess.run(cmd, check=False).returncode != 0:
            print(f"!!! FAILED: {mid}")
            failed.append(mid)

    print(f"\nrun summary: failed={failed or 'none'}")
    print("=== summarise ===")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_results.py")], check=False)


if __name__ == "__main__":
    main()
