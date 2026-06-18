#!/usr/bin/env python3
"""Aggregate evals/*.json into results.json and an HTML table.

    uv run python scripts/summarize_results.py

Reads every per-model result, extracts the survey-alignment headline numbers,
and renders site/index.html from templates/table_template.jinja plus a machine-
readable results.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from political_alignment import metrics  # noqa: E402

EVALS = ROOT / "evals"
SITE = ROOT / "site"
TEMPLATES = ROOT / "templates"


def fmt_ci(ci: dict) -> str:
    if ci is None or ci.get("mean") != ci.get("mean"):  # NaN check
        return "—"
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def shift_from_responses(responses: list[dict]) -> tuple[dict, list[dict]]:
    """Cross-lingual shift (JSD between languages) computed from stored responses.

    Works on any eval that kept per-response ``model_dist`` — so it is robust to
    older evals that lack a summary ``cross_lingual_shift`` field. Returns the
    per-model shift CI plus a per-item shift list (which items move most across
    languages), both restricted to valid, verified items.
    """
    # group valid+verified responses by (item, framing): {(...)}[lang] = dist
    groups: dict[tuple, dict[str, list]] = defaultdict(dict)
    item_topic = {}
    for r in responses:
        if not r.get("valid", True) or r.get("source_status") != "verified":
            continue
        groups[(r["item_id"], r["framing"])][r["lang"]] = r["model_dist"]
        item_topic[r["item_id"]] = r.get("topic", "")

    per_group, per_item = [], defaultdict(list)
    for (item_id, _framing), by_lang in groups.items():
        if len(by_lang) >= 2:
            shift = metrics.cross_lingual_inconsistency(by_lang)
            per_group.append(shift)
            per_item[item_id].append(shift)

    shift_ci = _ci(per_group)
    by_item = sorted(
        ({"item_id": k, "topic": item_topic.get(k, ""),
          "shift": sum(v) / len(v)} for k, v in per_item.items()),
        key=lambda d: -d["shift"],
    )
    return shift_ci, by_item


def _ci(values: list[float]) -> dict:
    mean, lo, hi = metrics.bootstrap_mean_ci(values, n_boot=2000, seed=42)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}


def refusal_controlled_shifts(model_responses: dict, n_redraws: int = 20, seed: int = 42) -> dict:
    """Disentangle refusal from genuine cross-lingual shift.

    A model that refuses more keeps fewer valid samples per item, which inflates
    the plug-in JSD *mechanically* (less data -> noisier -> higher JSD), not
    because it really shifts more. So we (a) report the effective valid sample
    size per model and (b) recompute the shift with every model sub-sampled to
    the COMMON minimum effective ``n``, so "who shifts most" is judged at equal
    data. Returns ``{model_id: {n_valid, shift_raw, shift_equalized}}`` plus a
    ``"_n_common"`` key.
    """
    per_model = {}
    for mid, responses in model_responses.items():
        groups, neffs = defaultdict(dict), []
        for r in responses:
            if not r.get("valid", True) or r.get("source_status") != "verified":
                continue
            n_eff = max(0, int(r.get("n_samples", 0)) - int(r.get("parse_fail", 0)))
            groups[(r["item_id"], r["framing"])][r["lang"]] = (
                np.asarray(r["model_dist"], dtype=np.float64), n_eff)
            neffs.append(n_eff)
        per_model[mid] = {"groups": groups,
                          "n_valid": float(np.median(neffs)) if neffs else 0.0}
    valid_ns = [m["n_valid"] for m in per_model.values() if m["n_valid"] > 0]
    n_common = int(max(1, min(valid_ns))) if valid_ns else 1
    rng = np.random.default_rng(seed)
    out = {"_n_common": n_common}
    for mid, m in per_model.items():
        raw, eq = [], []
        for by_lang in m["groups"].values():
            if len(by_lang) < 2:
                continue
            raw.append(metrics.cross_lingual_inconsistency(
                {lg: d for lg, (d, _n) in by_lang.items()}))
            redraws = []
            for _ in range(n_redraws):
                sub = {lg: rng.multinomial(n_common, d) / n_common
                       for lg, (d, _n) in by_lang.items()}
                redraws.append(metrics.cross_lingual_inconsistency(sub))
            eq.append(float(np.mean(redraws)))
        out[mid] = {
            "n_valid": m["n_valid"],
            "shift_raw": float(np.mean(raw)) if raw else float("nan"),
            "shift_equalized": float(np.mean(eq)) if eq else float("nan"),
        }
    return out


def collect() -> tuple[list[dict], dict]:
    rows, model_responses = [], {}
    for path in sorted(EVALS.glob("*.json")):
        data = json.loads(path.read_text())
        sa = data.get("methods", {}).get("survey_alignment")
        if not sa:
            continue
        s = sa["summary"]
        model_responses[data["model_id"]] = sa.get("responses", [])
        shift_ci, shift_by_item = shift_from_responses(sa.get("responses", []))
        rows.append({
            "model_id": data["model_id"],
            # headline: cross-lingual shift (the language-induced displacement)
            "cross_lingual_shift": s.get("cross_lingual_shift", shift_ci),
            "cross_lingual_shift_floor": s.get("cross_lingual_shift_floor"),
            "cross_lingual_shift_net": s.get("cross_lingual_shift_net"),
            "cross_lingual_emd": s.get("cross_lingual_emd"),
            "shift_by_item": shift_by_item,
            "alignment_overall": s["alignment_overall"],
            "by_source": s["by_source"],
            "by_framing": s["by_framing"],
            "by_language": s["by_language"],
            "cross_lingual_consistency": s["cross_lingual_consistency"],
            "mean_template_sd": s.get("mean_template_sd"),
            "refusal_rate": s.get("refusal_rate", 0.0),
            "refused_items": s.get("refused_items", []),
            "n_verified": s["n_verified"],
            "n_example": s["n_example"],
            "n_invalid": s.get("n_invalid", 0),
            "n_responses": s.get("n_responses", 0),
            "headline_excludes_examples": s["headline_excludes_examples"],
        })
    return rows, model_responses


def main():
    rows, model_responses = collect()
    # refusal-controlled robustness: shift judged at a common effective n
    control = refusal_controlled_shifts(model_responses)
    n_common = control.pop("_n_common", None)
    for r in rows:
        c = control.get(r["model_id"], {})
        r["n_valid_eff"] = c.get("n_valid")
        r["shift_equalized"] = c.get("shift_equalized")
    SITE.mkdir(exist_ok=True)
    (SITE / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["ci"] = fmt_ci
    template = env.get_template("table_template.jinja")
    any_example = any(r["n_example"] for r in rows)
    html = template.render(rows=rows, any_example=any_example, n_common=n_common)
    (SITE / "index.html").write_text(html)
    print(f"wrote {SITE / 'results.json'} and {SITE / 'index.html'}  ({len(rows)} models)")


if __name__ == "__main__":
    main()
