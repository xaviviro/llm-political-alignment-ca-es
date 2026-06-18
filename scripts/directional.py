#!/usr/bin/env python3
"""Directional language-pull analysis — the part distinct from MENAValues.

We have TWO reference populations (CEO = Catalonia, CIS = Spain). For a concept
measured on BOTH, we can ask: does prompting the model in Catalan pull its answer
toward the Catalan population, and prompting in Spanish toward the Spanish one?

For a dual-reference concept we place every distribution on a common normalised
left--right axis [0,1] (its expected value over the substantive options, dropping
don't-know / no-answer). Then, per model and language:

    pull_to_CEO(lang) = dist(model_lang, CIS_pop) - dist(model_lang, CEO_pop)

(positive = the model's answer in that language sits closer to the Catalan
population). The directional effect is pull_to_CEO(ca) - pull_to_CEO(es): positive
supports the hypothesis that Catalan pulls toward CEO and Spanish toward CIS.

With the current data only **ideology** (left--right self-placement) is cleanly
dual-reference; this is a proof of concept, reported with that caveat.
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"

# dual-reference concepts: (prompt item used for the model, CEO-pop item, CIS-pop item)
CONCEPTS = [
    {"concept": "ideology", "prompt_item": "ceo_ideology",
     "ceo_item": "ceo_ideology", "cis_item": "cis_ideology"},
]


def _numeric_codes(options: list[str]) -> list[float | None]:
    """Leading integer of each option label, or None for don't-know/no-answer."""
    out = []
    for o in options:
        m = re.match(r"\s*(\d+)", o)
        out.append(float(m.group(1)) if m else None)
    return out


def norm_mean(dist, codes) -> float:
    """Expected value over substantive options, normalised to [0,1]."""
    sub = [(p, c) for p, c in zip(dist, codes, strict=True) if c is not None]
    tot = sum(p for p, _ in sub)
    if tot <= 0:
        return float("nan")
    e = sum(p * c for p, c in sub) / tot
    lo = min(c for _, c in sub)
    hi = max(c for _, c in sub)
    return (e - lo) / (hi - lo) if hi > lo else float("nan")


def load_items():
    import sys
    sys.path.insert(0, str(ROOT))
    from political_alignment.dataset import load_datasets
    return {it.item_id: it for it in
            load_datasets([str(ROOT / "data" / "ceo_items.csv"),
                           str(ROOT / "data" / "cis_items.csv")])}


def main():
    items = load_items()
    for c in CONCEPTS:
        ceo, cis = items[c["ceo_item"]], items[c["cis_item"]]
        ceo_pos = norm_mean(ceo.pop_dist, _numeric_codes(ceo.options["ca"]))
        cis_pos = norm_mean(cis.pop_dist, _numeric_codes(cis.options["es"]))
        print(f"=== {c['concept']} ===")
        print("reference positions on the normalised left-right axis [0=left,1=right]:")
        print(f"  CEO (Catalonia) = {ceo_pos:.3f}   CIS (Spain) = {cis_pos:.3f}   "
              f"|gap| = {abs(ceo_pos - cis_pos):.3f}")
        if abs(ceo_pos - cis_pos) < 0.05:
            print("  (!) the two populations are very close on this axis -> weak directional test")

        prompt_item = c["prompt_item"]
        codes = _numeric_codes(items[prompt_item].options["ca"])
        print(f"\n  {'model':<22}{'CA':>6}{'ES':>6}  {'pull→CEO ca/es':>16}  {'directional':>12}")
        rows = []
        for f in sorted(glob.glob(str(EVALS / "*.json"))):
            d = json.loads(Path(f).read_text())
            by_lang = defaultdict(list)
            for r in d["methods"]["survey_alignment"]["responses"]:
                if r["item_id"] == prompt_item and r["valid"]:
                    by_lang[r["lang"]].append(r["model_dist"])
            if "ca" not in by_lang or "es" not in by_lang:
                continue
            ca = norm_mean(np.mean(by_lang["ca"], axis=0), codes)
            es = norm_mean(np.mean(by_lang["es"], axis=0), codes)
            pull_ca = abs(ca - cis_pos) - abs(ca - ceo_pos)
            pull_es = abs(es - cis_pos) - abs(es - ceo_pos)
            directional = pull_ca - pull_es
            rows.append((d["model_id"], ca, es, pull_ca, pull_es, directional))
        for mid, ca, es, pca, pes, dr in sorted(rows, key=lambda x: -x[5]):
            arrow = "→CEO" if dr > 0.01 else ("→CIS" if dr < -0.01 else "≈")
            print(f"  {mid:<22}{ca:>6.2f}{es:>6.2f}  {pca:>+7.2f}/{pes:>+6.2f}  {dr:>+9.3f} {arrow}")
        if rows:
            mean_dir = float(np.mean([r[5] for r in rows]))
            n_pos = sum(1 for r in rows if r[5] > 0.01)
            print(f"\n  mean directional effect = {mean_dir:+.3f}  "
                  f"({n_pos}/{len(rows)} models lean CA→CEO / ES→CIS as hypothesised)")


if __name__ == "__main__":
    main()
