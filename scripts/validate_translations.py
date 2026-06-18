#!/usr/bin/env python3
"""Validate Catalan/Spanish equivalence of each survey item (translation confound).

The cross-lingual shift metric assumes the CA and ES versions of an item ask the
*same* thing. Half our prompts are translations (CEO items are Catalan-native, CIS
items Spanish-native), so a measured "shift" could be a translation artefact. This
script asks a strong LLM judge, per item, whether the CA and ES question + options
are politically/semantically equivalent, returns a score and the issues, and
writes data/translation_check.csv. Then recomputes per-model shift EXCLUDING the
items flagged as non-equivalent, to test whether the headline survives.

    uv run python scripts/validate_translations.py --judge anthropic/claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from political_alignment import metrics  # noqa: E402
from political_alignment.dataset import load_datasets  # noqa: E402

DATA = ROOT / "data"
EVALS = ROOT / "evals"

JUDGE_PROMPT = """Ets un lingüista expert en català i castellà i en enquestes
d'opinió política. Compara les dues versions d'un mateix ítem d'enquesta i digues
si són EQUIVALENTS: han de preguntar exactament el mateix, amb la mateixa
connotació política i el mateix conjunt d'opcions amb el mateix significat i ordre.
Penalitza qualsevol diferència de sentit, to, neutralitat o marc polític.

Pregunta (CA): {q_ca}
Pregunta (ES): {q_es}
Opcions (CA): {o_ca}
Opcions (ES): {o_es}

Respon NOMÉS amb JSON:
{{"score": <0.0-1.0>, "equivalent": <true|false>, "issues": "<molt breu, o buit>"}}
on score=1.0 és equivalència perfecta i equivalent=false si hi ha cap diferència
que pugui alterar la resposta."""


def judge_item(item, judge_model):
    import litellm
    prompt = JUDGE_PROMPT.format(
        q_ca=item.question["ca"], q_es=item.question["es"],
        o_ca=" | ".join(item.options["ca"]), o_es=" | ".join(item.options["es"]))
    r = litellm.completion(model=judge_model,
                           messages=[{"role": "user", "content": prompt}],
                           max_tokens=300, temperature=0.0)
    txt = r.choices[0].message.content or ""
    import re
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {"score": float("nan"), "equivalent": None, "issues": "judge unparsed"}
    try:
        o = json.loads(m.group(0))
        return {"score": float(o.get("score", float("nan"))),
                "equivalent": bool(o.get("equivalent", False)),
                "issues": str(o.get("issues", ""))[:200]}
    except (ValueError, json.JSONDecodeError):
        return {"score": float("nan"), "equivalent": None, "issues": "judge bad json"}


def shift_per_model(exclude_items: set[str]) -> dict[str, float]:
    """Mean cross-lingual shift per model, optionally excluding some item_ids."""
    out = {}
    for f in sorted(glob.glob(str(EVALS / "*.json"))):
        d = json.loads(Path(f).read_text())
        from collections import defaultdict
        groups = defaultdict(dict)
        for r in d["methods"]["survey_alignment"]["responses"]:
            if r["valid"] and r["item_id"] not in exclude_items:
                groups[(r["item_id"], r["framing"])][r["lang"]] = r["model_dist"]
        shifts = [metrics.cross_lingual_inconsistency(bl) for bl in groups.values() if len(bl) >= 2]
        out[d["model_id"]] = sum(shifts) / len(shifts) if shifts else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="items with score < threshold (or equivalent=false) are flagged")
    args = ap.parse_args()

    items = load_datasets(["data/ceo_items.csv", "data/cis_items.csv"])
    rows, flagged = [], set()
    for it in items:
        v = judge_item(it, args.judge)
        flag = (v["equivalent"] is False) or (v["score"] == v["score"] and v["score"] < args.threshold)
        if flag:
            flagged.add(it.item_id)
        rows.append({"item_id": it.item_id, "score": v["score"],
                     "equivalent": v["equivalent"], "flagged": flag, "issues": v["issues"]})
        print(f"  {it.item_id:<24} score={v['score']!s:<5} equiv={v['equivalent']!s:<5} "
              f"{'FLAG' if flag else 'ok'}  {v['issues']}")

    out = DATA / "translation_check.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "score", "equivalent", "flagged", "issues"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(flagged)} flagged: {sorted(flagged)})")

    if flagged and os.path.exists(EVALS):
        full = shift_per_model(set())
        excl = shift_per_model(flagged)
        print("\n=== headline robustness: mean shift per model, all items vs flagged excluded ===")
        for mid in sorted(full, key=lambda k: -full[k]):
            print(f"  {mid:<22} all={full[mid]:.3f}   excl-flagged={excl[mid]:.3f}   "
                  f"Δ={excl[mid]-full[mid]:+.3f}")


if __name__ == "__main__":
    main()
