#!/usr/bin/env python3
"""Generate the vector (PDF) figures for the paper from evals/*.json.

Outputs into paper/figs/:
  shift_by_model.pdf   cross-lingual shift per model (95% CI)
  shift_by_item.pdf    cross-lingual shift per survey item (mean across models)
  refusal_by_model.pdf refusal rate per model
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
EXCLUDE = set()  # models to drop from the figures/table, by id

ITEM_CA = {
    "ceo_independence": "Independència (CEO)",
    "ceo_identity": "Identitat nacional (CEO)",
    "ceo_ideology": "Ideologia esq–dre (CEO)",
    "ceo_monarchy": "Monarquia/república (CEO)",
    "ceo_state_model": "Model d'estat (CEO)",
    "ceo_trust_gov_spain": "Confiança Govern Espanya (CEO)",
    "ceo_trust_gov_catalonia": "Confiança Govern Catalunya (CEO)",
    "ceo_trust_monarchy": "Confiança monarquia (CEO)",
    "ceo_trust_parties": "Confiança partits (CEO)",
    "ceo_trust_eu": "Confiança UE (CEO)",
    "ceo_trust_church": "Confiança Església (CEO)",
    "ceo_trust_congress": "Confiança Congrés Diputats (CEO)",
    "ceo_trust_parliament_cat": "Confiança Parlament cat. (CEO)",
    "ceo_trust_courts": "Confiança tribunals (CEO)",
    "ceo_trust_unions": "Confiança sindicats (CEO)",
    "ceo_trust_police": "Confiança policia (CEO)",
    "ceo_trust_mossos": "Confiança Mossos (CEO)",
    "ceo_trust_army": "Confiança exèrcit (CEO)",
    "ceo_trust_un": "Confiança ONU (CEO)",
    "cis_ideology": "Ideologia esq–dre (CIS)",
    "cis_economy": "Situació econòmica (CIS)",
}


NAME = {
    "gemini-flash-lite": "gemini-3.1-flash-lite",
    "gemini-flash": "gemini-3.5-flash",
}


def load():
    import sys
    sys.path.insert(0, str(ROOT))
    from political_alignment import metrics
    models, per_item = [], defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "evals" / "*.json"))):
        d = json.loads(Path(f).read_text())
        if d["model_id"] in EXCLUDE:
            continue
        s = d["methods"]["survey_alignment"]["summary"]
        models.append({
            "id": d["model_id"],
            "shift": s["cross_lingual_shift"]["mean"],
            "lo": s["cross_lingual_shift"]["lo"], "hi": s["cross_lingual_shift"]["hi"],
            "align": s["alignment_overall"]["mean"],
            "tplsd": s["mean_template_sd"]["mean"],
            "refusal": 100 * s.get("refusal_rate", 0.0),
            "net": s.get("cross_lingual_shift_net", {}).get("mean", float("nan")),
        })
        # per-item shift from responses
        groups = defaultdict(dict)
        for r in d["methods"]["survey_alignment"]["responses"]:
            if r["valid"]:
                groups[(r["item_id"], r["framing"])][r["lang"]] = r["model_dist"]
        tmp = defaultdict(list)
        for (iid, _fr), bl in groups.items():
            if len(bl) >= 2:
                tmp[iid].append(metrics.cross_lingual_inconsistency(bl))
        for iid, v in tmp.items():
            per_item[iid].append(float(np.mean(v)))
    models.sort(key=lambda m: m["shift"])
    return models, per_item


def _color(mid):
    if mid.startswith("gemini"):
        return "#d62728"
    if mid.startswith("claude"):
        return "#9467bd"
    if mid.startswith("gpt"):
        return "#2ca02c"
    return "#1f77b4"


def fig_shift_by_model(models):
    ids = [m["id"] for m in models]
    sh = [m["shift"] for m in models]
    err = [[m["shift"] - m["lo"] for m in models], [m["hi"] - m["shift"] for m in models]]
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    y = np.arange(len(ids))
    ax.barh(y, sh, color=[_color(m) for m in ids], xerr=err, capsize=3, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_xlabel("Desplaçament translingüe (JSD ca ↔ es)", fontsize=8)
    ax.set_xlim(0, max(m["hi"] for m in models) * 1.1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "shift_by_model.pdf")
    fig.savefig(FIGS / "shift_by_model.png", dpi=200)
    plt.close(fig)


def fig_shift_by_item(per_item):
    items = sorted(per_item.items(), key=lambda kv: np.mean(kv[1]))
    labels = [ITEM_CA.get(k, k) for k, _ in items]
    means = [np.mean(v) for _, v in items]
    cols = ["#ff7f0e" if k.startswith("cis") else "#1f77b4" for k, _ in items]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    y = np.arange(len(labels))
    ax.barh(y, means, color=cols, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Desplaçament translingüe mitjà (JSD ca ↔ es)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="CEO (Catalunya)"),
                       Patch(color="#ff7f0e", label="CIS (Espanya)")],
              fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "shift_by_item.pdf")
    fig.savefig(FIGS / "shift_by_item.png", dpi=200)
    plt.close(fig)


def fig_refusal(models):
    ms = sorted(models, key=lambda m: m["refusal"])
    ids = [m["id"] for m in ms]
    rf = [m["refusal"] for m in ms]
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    y = np.arange(len(ids))
    ax.barh(y, rf, color=[_color(m) for m in ids], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_xlabel("Taxa de rebuig (% mostres sense resposta)", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "refusal_by_model.pdf")
    fig.savefig(FIGS / "refusal_by_model.png", dpi=200)
    plt.close(fig)


def write_tables(models):
    """Emit the LaTeX table rows + the \\nmodels count, so the paper auto-updates."""
    paper = ROOT / "paper"
    ranked = sorted(models, key=lambda m: -m["shift"])
    top2 = {m["id"] for m in ranked[:2]}
    rows = []
    for m in ranked:
        name = NAME.get(m["id"], m["id"]).replace("_", r"\_")
        sh = f"\\textbf{{{m['shift']:.3f}}}" if m["id"] in top2 else f"{m['shift']:.3f}"
        net = f"{m['net']:.3f}" if m["net"] == m["net"] else "--"
        rows.append(f"{name} & {sh} & [{m['lo']:.2f}, {m['hi']:.2f}] & {net} & "
                    f"{m['align']:.3f} & {m['refusal']:.1f}\\,\\% & {m['tplsd']:.3f} \\\\")
    table = (
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "\\textbf{Model} & \\textbf{Desplaç.} & \\textbf{[IC 95\\%]} & "
        "\\textbf{Net} & \\textbf{Alineació} & \\textbf{Rebuig} & \\textbf{SD plant.} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    (paper / "table_models.tex").write_text(table, encoding="utf-8")
    from political_alignment.dataset import load_datasets
    nitems = len(load_datasets([str(ROOT / "data" / "ceo_items.csv"),
                                str(ROOT / "data" / "cis_items.csv")]))
    (paper / "paper_data.tex").write_text(
        f"\\newcommand{{\\nmodels}}{{{len(models)}}}\n"
        f"\\newcommand{{\\nitems}}{{{nitems}}}\n"
        f"\\newcommand{{\\nresp}}{{{nitems * 6}}}\n", encoding="utf-8")


def fig_emd_by_item():
    """Ordinal-aware cross-lingual shift (normalised Wasserstein-1 / EMD) per
    scale item, averaged across models. JSD treats every category swap equally;
    on the 0--10 trust/ideology scales EMD instead charges by distance moved.
    Skipped when the evals carry no ordinal items.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from political_alignment import metrics
    from political_alignment.dataset import load_datasets
    items = {it.item_id: it for it in load_datasets(
        [str(ROOT / "data" / "ceo_items.csv"), str(ROOT / "data" / "cis_items.csv")])}
    per_item = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "evals" / "*.json"))):
        d = json.loads(Path(f).read_text())
        if d["model_id"] in EXCLUDE:
            continue
        sa = d.get("methods", {}).get("survey_alignment")
        if not sa:
            continue
        groups = defaultdict(dict)
        for r in sa["responses"]:
            if r.get("valid"):
                groups[(r["item_id"], r["framing"])][r["lang"]] = r["model_dist"]
        tmp = defaultdict(list)
        for (iid, _fr), bl in groups.items():
            it = items.get(iid)
            if it is None or not it.is_ordinal or len(bl) < 2:
                continue
            codes = it.ordinal_codes
            langs = sorted(bl)
            subs = [metrics.ordinal_view(bl[lg], codes) for lg in langs]
            pairs = [metrics.wasserstein1(subs[i][0], subs[j][0], positions=subs[i][1],
                                          normalize=True)
                     for i in range(len(langs)) for j in range(i + 1, len(langs))]
            if pairs:
                tmp[iid].append(float(np.mean(pairs)))
        for iid, v in tmp.items():
            per_item[iid].append(float(np.mean(v)))
    if not per_item:
        print("  (no ordinal items in evals; skipping emd_by_item figure)")
        return
    items_sorted = sorted(per_item.items(), key=lambda kv: np.mean(kv[1]))
    labels = [ITEM_CA.get(k, k) for k, _ in items_sorted]
    means = [float(np.mean(v)) for _, v in items_sorted]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    y = np.arange(len(labels))
    ax.barh(y, means, color="#7b3294", height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("EMD translingüe (Wasserstein-1 normalitzat, ca ↔ es)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "emd_by_item.pdf")
    fig.savefig(FIGS / "emd_by_item.png", dpi=200)
    plt.close(fig)
    print(f"  wrote emd_by_item.pdf ({len(labels)} ordinal items)")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    models, per_item = load()
    fig_shift_by_model(models)
    fig_shift_by_item(per_item)
    fig_refusal(models)
    fig_emd_by_item()
    write_tables(models)
    print(f"wrote figures + tables to {ROOT / 'paper'}  ({len(models)} models)")


if __name__ == "__main__":
    main()
