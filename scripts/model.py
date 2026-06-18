#!/usr/bin/env python3
"""Run the political-bias evaluation for ONE model and write its JSON result.

    uv run python scripts/model.py --model llama3-8b
    uv run python scripts/model.py --model mock --mock      # network-free dry run

Reads config.yaml + models.yaml, builds a LiteLLM (or mock) provider, runs the
configured methods over the CEO/CIS datasets, and writes evals/<id>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from political_alignment import survey_alignment  # noqa: E402
from political_alignment.dataset import load_datasets  # noqa: E402
from political_alignment.providers import build_provider  # noqa: E402

EVALS = ROOT / "evals"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="model id from models.yaml (or 'mock')")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--models-file", default=str(ROOT / "models.yaml"))
    p.add_argument("--mock", action="store_true", help="use the network-free MockProvider")
    p.add_argument("--out", default=None, help="output JSON path (default evals/<id>.json)")
    return p.parse_args()


def find_model_cfg(models_file: dict, model_id: str, mock: bool) -> dict:
    if mock or model_id == "mock":
        return {"id": model_id, "mock": True}
    for m in models_file.get("models", []):
        if m["id"] == model_id:
            return m
    raise SystemExit(f"model id {model_id!r} not found in models.yaml")


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    models_file = yaml.safe_load(Path(args.models_file).read_text())

    model_cfg = find_model_cfg(models_file, args.model, args.mock)
    provider = build_provider(model_cfg, config)

    methods = config.get("methods", ["survey_alignment"])
    result = {"model_id": provider.id, "methods": {}}

    if "survey_alignment" in methods:
        items = load_datasets(config["datasets"])
        result["methods"]["survey_alignment"] = survey_alignment.evaluate(provider, items, config)

    out = Path(args.out) if args.out else EVALS / f"{provider.id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
