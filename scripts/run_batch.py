#!/usr/bin/env python3
"""Run the survey-alignment method through a provider's Batch API (≈50% cost).

    # one-shot: submit, poll until done, collect, write evals/<id>.json
    uv run python scripts/run_batch.py --model gpt-oss-120b --run

    # or in two steps (submit returns immediately, collect later):
    uv run python scripts/run_batch.py --model gpt-oss-120b --submit
    uv run python scripts/run_batch.py --model gpt-oss-120b --collect

Provider is inferred from the LiteLLM model id in models.yaml (groq/…, openai/…,
anthropic/…, gemini/…). Keys come from the environment (.env). Batch state is
saved under batches/<id>.json so --collect can resume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from political_alignment import batch  # noqa: E402
from political_alignment.dataset import load_datasets  # noqa: E402

EVALS = ROOT / "evals"
BATCHES = ROOT / "batches"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="model id from models.yaml")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--models-file", default=str(ROOT / "models.yaml"))
    p.add_argument("--submit", action="store_true", help="build + submit a batch, then exit")
    p.add_argument("--collect", action="store_true", help="collect a previously submitted batch")
    p.add_argument("--run", action="store_true", help="submit, poll, collect, write eval (one-shot)")
    p.add_argument("--poll", type=int, default=30, help="poll interval seconds")
    return p.parse_args()


def find_model(models_file: dict, model_id: str) -> dict:
    for m in models_file.get("models", []):
        if m["id"] == model_id:
            return m
    raise SystemExit(f"model id {model_id!r} not found in models.yaml")


def do_submit(model_cfg, items, config) -> dict:
    provider, raw = batch.provider_of(model_cfg["model"])
    requests = batch.build_requests(items, config, raw_model=raw)
    print(f"submitting {len(requests)} requests to {provider} ({raw}) ...")
    state = {"model_id": model_cfg["id"], "provider": provider, "raw_model": raw}
    if provider in ("openai", "groq"):
        state["batch_id"] = batch.submit_openai_compatible(provider, requests)
    elif provider == "anthropic":
        state["batch_id"] = batch.submit_anthropic(requests)
    elif provider == "gemini":
        job, cids = batch.submit_gemini(raw, requests)
        state["batch_id"], state["custom_ids"] = job, cids
    else:
        raise SystemExit(f"no batch adapter for provider {provider!r}")
    BATCHES.mkdir(exist_ok=True)
    (BATCHES / f"{model_cfg['id']}.json").write_text(json.dumps(state, indent=2))
    print(f"submitted: batch_id={state['batch_id']}  (state -> batches/{model_cfg['id']}.json)")
    return state


def do_collect(state, items, config, poll) -> None:
    provider = state["provider"]
    print(f"collecting {provider} batch {state['batch_id']} ...")
    if provider in ("openai", "groq"):
        results = batch.collect_openai_compatible(provider, state["batch_id"], poll=poll)
    elif provider == "anthropic":
        results = batch.collect_anthropic(state["batch_id"], poll=poll)
    elif provider == "gemini":
        results = batch.collect_gemini(state["batch_id"], state["custom_ids"], poll=poll)
    else:
        raise SystemExit(f"no batch adapter for provider {provider!r}")
    sa = batch.results_to_eval(state["model_id"], items, config, results)
    EVALS.mkdir(exist_ok=True)
    out = EVALS / f"{state['model_id']}.json"
    out.write_text(json.dumps({"model_id": state["model_id"],
                               "methods": {"survey_alignment": sa}},
                              indent=2, ensure_ascii=False))
    s = sa["summary"]
    print(f"wrote {out}  | shift={s['cross_lingual_shift']['mean']:.3f}  "
          f"invalid={s['n_invalid']}/{s['n_responses']}")


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    models_file = yaml.safe_load(Path(args.models_file).read_text())
    items = load_datasets(config["datasets"])
    model_cfg = find_model(models_file, args.model)

    if args.run:
        state = do_submit(model_cfg, items, config)
        do_collect(state, items, config, args.poll)
    elif args.submit:
        do_submit(model_cfg, items, config)
    elif args.collect:
        state = json.loads((BATCHES / f"{args.model}.json").read_text())
        do_collect(state, items, config, args.poll)
    else:
        raise SystemExit("pass --run, or --submit / --collect")


if __name__ == "__main__":
    main()
