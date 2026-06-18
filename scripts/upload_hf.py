#!/usr/bin/env python3
"""Publish the CEO/CIS survey dataset to the Hugging Face Hub.

Builds a dataset card from the current items (sources, waves, access dates,
licence) and uploads the CSVs to a HF dataset repo. Only the aggregated
marginals are published — never raw microdata.

Requires a HF write token in the environment:
    export HF_TOKEN=hf_...
    uv run python scripts/upload_hf.py --repo xaviviro/llm-political-alignment-ca-es --private

Add --public to make the dataset public (default is private).
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from political_alignment.dataset import load_datasets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSVS = ["data/ceo_items.csv", "data/cis_items.csv"]


def build_card(items) -> str:
    verified = [it for it in items if it.source_status == "verified"]
    waves = sorted({it.survey_wave for it in verified})
    rows = "\n".join(
        f"| `{it.item_id}` | {it.source} | {it.topic} | {it.population} | "
        f"{it.n_options} | {it.source_status} |"
        for it in items
    )
    wave_lines = "\n".join(f"- {w}" for w in waves)
    today = date.today().isoformat()
    return f"""---
license: mit
language:
  - ca
  - es
tags:
  - political-bias
  - survey
  - cultural-alignment
  - CEO
  - CIS
pretty_name: Political bias CA/ES — CEO & CIS survey marginals
---

# Political bias CA/ES — CEO & CIS survey marginals

Population response distributions (marginals) for a curated set of political and
values questions from **CEO** (Centre d'Estudis d'Opinió, Catalonia) and **CIS**
(Centro de Investigaciones Sociológicas, Spain), packaged for measuring the
political bias / cultural alignment of LLMs in Catalan and Spanish.

Companion to the framework at
<https://github.com/xaviviro/llm-political-alignment-ca-es>.

Only **aggregated marginals** are distributed here (never raw microdata), each
with full provenance: the exact survey wave, the official source URL, and the
access date (see the `survey_wave` and `notes` fields).

## Items

| item_id | source | topic | population | options | status |
| ------- | ------ | ----- | ---------- | ------- | ------ |
{rows}

## Survey waves

{wave_lines}

## Sources

- **CEO** — Centre d'Estudis d'Opinió, Generalitat de Catalunya. Open microdata
  of the Baròmetre d'Opinió Política. <https://ceo.gencat.cat> /
  <https://analisi.transparenciacatalunya.cat>
- **CIS** — Centro de Investigaciones Sociológicas, Gobierno de España. Open
  Excel marginals of the monthly barómetro. <https://www.cis.es>

Survey data is subject to the terms of CEO and CIS; cite the originating study.
The packaging/code is MIT-licensed. Generated {today}.

## Citation

Vinaixa Roselló, Xavier (2026). *llm-political-alignment-ca-es.* Sorensen AI, Barcelona.
ORCID: 0009-0005-2769-9215. <https://github.com/xaviviro/llm-political-alignment-ca-es>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="xaviviro/llm-political-alignment-ca-es",
                   help="HF dataset repo id (namespace/name)")
    p.add_argument("--public", action="store_true", help="make the dataset public")
    p.add_argument("--dry-run", action="store_true", help="write the card locally, do not upload")
    args = p.parse_args()

    items = load_datasets(CSVS)
    card = build_card(items)
    card_path = DATA / "DATASET_CARD.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"wrote dataset card -> {card_path}")

    if args.dry_run:
        print("dry-run: not uploading.")
        return

    from huggingface_hub import HfApi
    # use HF_TOKEN if set, else fall back to the cached `huggingface-cli login`
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        api.whoami()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "no HF auth — set HF_TOKEN or run `huggingface-cli login`"
        ) from e
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    for csv in CSVS:
        api.upload_file(path_or_fileobj=str(ROOT / csv), path_in_repo=csv,
                        repo_id=args.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(DATA / "schema.md"), path_in_repo="schema.md",
                    repo_id=args.repo, repo_type="dataset")
    vis = "public" if args.public else "private"
    print(f"uploaded {len(CSVS)} CSVs + card to {args.repo} ({vis} dataset)")


if __name__ == "__main__":
    main()
