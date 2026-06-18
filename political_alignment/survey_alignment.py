"""Survey-grounded cultural-alignment method (MENAValues-style).

For each survey item, each language and each perspective framing, the model's
answer distribution over the options is elicited and compared to the real
population distribution from CEO/CIS. Reports:

- **alignment** — 1 - JSD(model, population); how close the model is to the
  population it is being compared against (per item's population);
- **cross-lingual consistency** — 1 - mean pairwise JSD of the model's answers
  to the SAME item across languages (the language-induced shift);

broken down by survey source, language and framing, with bootstrap CIs. Example
(illustrative) items are counted and reported separately so they never inflate a
headline number.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import metrics
from .dataset import SurveyItem
from .framings import render_variants
from .providers import Provider


def evaluate(provider: Provider, items: list[SurveyItem], config: dict) -> dict:
    """Live (synchronous) evaluation: each paraphrase is probed via the provider."""
    def template_probe(item, lang, framing):
        return [provider.answer_distribution(prompt, letters)
                for _i, prompt, letters in render_variants(item, lang, framing)]
    return aggregate(provider.id, items, config, template_probe)


def aggregate(model_id: str, items: list[SurveyItem], config: dict, template_probe) -> dict:
    """Shared aggregation for the survey-alignment method.

    ``template_probe(item, lang, framing)`` returns a list of ``(distribution,
    info)`` — one entry per paraphrase template, where ``info`` has ``n`` (samples)
    and ``n_fail`` (parse failures). Used by both the live evaluator and the batch
    pipeline (which precomputes the distributions from a submitted batch).
    """
    languages = config.get("languages", ["ca", "es"])
    framings = config.get("framings", ["neutral", "personalised", "observer"])
    boot = config.get("bootstrap", {})
    n_boot = boot.get("n_boot", 2000)
    seed = boot.get("seed", 42)

    responses: list[dict] = []
    # dists_by_item_framing[(item_id, framing)][lang] = model distribution
    dists: dict[tuple[str, str], dict[str, list]] = defaultdict(dict)
    # neff[(item_id, framing)][lang] = effective valid sample size (for the JSD floor)
    neff: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)

    for item in items:
        for framing in framings:
            for lang in languages:
                # probe every paraphrase of this (item, lang, framing)
                tmpl_dists, tmpl_aligns = [], []
                total_fail = total_n = 0
                fail_reasons: dict[str, int] = defaultdict(int)
                fail_examples: list[str] = []
                for dist, info in template_probe(item, lang, framing):
                    total_fail += info["n_fail"]
                    total_n += info["n"]
                    for reason, c in info.get("fail_reasons", {}).items():
                        fail_reasons[reason] += c
                    for ex in info.get("fail_examples", []):
                        if len(fail_examples) < 3:
                            fail_examples.append(ex)
                    # a template is invalid if every sample failed to parse
                    if not (info["n"] > 0 and info["n_fail"] >= info["n"]):
                        tmpl_dists.append(np.asarray(dist, dtype=np.float64))
                        tmpl_aligns.append(metrics.alignment_score(dist, item.pop_dist))

                valid = len(tmpl_dists) > 0
                if valid:
                    model_dist = np.mean(np.vstack(tmpl_dists), axis=0)
                    # between-template spread of alignment (the Röttger check)
                    template_sd = float(np.std(tmpl_aligns, ddof=0)) if len(tmpl_aligns) > 1 else 0.0
                else:
                    n = len(item.pop_dist)
                    model_dist = np.full(n, 1.0 / n)
                    template_sd = float("nan")
                align = metrics.alignment_score(model_dist, item.pop_dist)
                responses.append({
                    "item_id": item.item_id,
                    "source": item.source,
                    "population": item.population,
                    "topic": item.topic,
                    "dimension": item.dimension,
                    "source_status": item.source_status,
                    "lang": lang,
                    "framing": framing,
                    "model_dist": [float(x) for x in model_dist],
                    "pop_dist": [float(x) for x in item.pop_dist],
                    "alignment": align,
                    "jsd": 1.0 - align,
                    "template_sd": template_sd,
                    "n_templates": len(tmpl_dists),
                    "parse_fail": total_fail,
                    "n_samples": total_n,
                    "fail_reasons": dict(fail_reasons),
                    "fail_examples": fail_examples,
                    "valid": valid,
                })
                if valid:
                    dists[(item.item_id, framing)][lang] = model_dist
                    neff[(item.item_id, framing)][lang] = max(0, total_n - total_fail)

    floor_boot = boot.get("floor_boot", 300)
    items_by_id = {it.item_id: it for it in items}
    summary = _summarise(responses, dists, neff, items_by_id, n_boot, seed, floor_boot)
    return {
        "model_id": model_id,
        "method": "survey_alignment",
        "config": {"languages": languages, "framings": framings,
                   "distribution_method": config.get("distribution_method")},
        "responses": responses,
        "summary": summary,
    }


def _ci(values, n_boot, seed):
    mean, lo, hi = metrics.bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}


def _summarise(responses, dists, neff, items_by_id, n_boot, seed, floor_boot=300):
    valid = [r for r in responses if r.get("valid", True)]
    n_invalid = len(responses) - len(valid)
    verified = [r for r in valid if r["source_status"] == "verified"]
    # headline excludes example items when verified ones exist, and always
    # excludes parse-failed (invalid) responses
    pool = verified if verified else valid

    by_source = {}
    for src in sorted({r["source"] for r in pool}):
        by_source[src] = _ci([r["alignment"] for r in pool if r["source"] == src], n_boot, seed)
    by_framing = {}
    for fr in sorted({r["framing"] for r in pool}):
        by_framing[fr] = _ci([r["alignment"] for r in pool if r["framing"] == fr], n_boot, seed)
    by_language = {}
    for lg in sorted({r["lang"] for r in pool}):
        by_language[lg] = _ci([r["alignment"] for r in pool if r["lang"] == lg], n_boot, seed)

    # cross-lingual shift/consistency per (item, framing) over its languages.
    # shift = mean pairwise JSD between languages = the language-induced
    # displacement (our headline); consistency = 1 - shift. We also subtract the
    # per-item JSD noise floor (the bias of the plug-in JSD at this sample size)
    # to report the *net* shift, and a hierarchical-bootstrap CI for it.
    consistencies, shifts = [], []
    shift_floors, net_shifts, items_langs, shift_by_item = [], [], [], []
    emd_shifts = []   # ordinal-aware cross-lingual shift (scale items only)
    for (item_id, framing), by_lang in dists.items():
        if len(by_lang) >= 2:
            raw = metrics.cross_lingual_inconsistency(by_lang)
            consistencies.append(1.0 - raw)
            shifts.append(raw)
            langs_pn, per_lang_floor = {}, []
            for lang, dist in by_lang.items():
                n_eff = int(neff.get((item_id, framing), {}).get(lang, 0))
                langs_pn[lang] = (np.asarray(dist, dtype=np.float64), n_eff)
                counts = np.asarray(dist, dtype=np.float64) * n_eff
                per_lang_floor.append(metrics.jsd_noise_floor(counts, n_boot=floor_boot, seed=seed))
            floor = float(np.mean(per_lang_floor)) if per_lang_floor else 0.0
            net = max(0.0, raw - floor)
            shift_floors.append(floor)
            net_shifts.append(net)
            items_langs.append(langs_pn)
            # ordinal-aware shift: Wasserstein-1 between languages on the scale
            # support (0--10 trust / left--right), normalised to [0,1]. None for
            # nominal items, where one-step != extreme has no meaning.
            item = items_by_id.get(item_id)
            emd = None
            if item is not None and item.is_ordinal:
                codes = item.ordinal_codes
                langs = sorted(by_lang)
                subs = [metrics.ordinal_view(by_lang[lg], codes) for lg in langs]
                pairs = [metrics.wasserstein1(subs[i][0], subs[j][0],
                                              positions=subs[i][1], normalize=True)
                         for i in range(len(langs)) for j in range(i + 1, len(langs))]
                if pairs:
                    emd = float(np.mean(pairs))
                    emd_shifts.append(emd)
            shift_by_item.append({"item_id": item_id, "framing": framing,
                                  "shift": raw, "floor": floor, "net_shift": net,
                                  "emd_shift": emd, "ordinal": emd is not None})
    net_mean, net_lo, net_hi = metrics.hierarchical_net_shift_ci(
        items_langs, shift_floors, n_boot=n_boot, seed=seed)

    # between-template robustness (Röttger): mean SD of alignment across the
    # paraphrases of each response. Higher = more prompt-sensitive (less robust).
    tmpl_sds = [r["template_sd"] for r in pool
                if r.get("n_templates", 0) > 1 and r["template_sd"] == r["template_sd"]]

    # refusal / failure accounting (the model declines to pick an option)
    total_samples = sum(r.get("n_samples", 0) for r in responses)
    refusal_samples = sum(r.get("fail_reasons", {}).get("refusal", 0) for r in responses)
    empty_samples = sum(r.get("fail_reasons", {}).get("empty", 0) for r in responses)
    refused_items = sorted(
        {r["item_id"] for r in responses
         if not r["valid"] and r.get("fail_reasons", {}).get("refusal", 0) > 0})

    return {
        "alignment_overall": _ci([r["alignment"] for r in pool], n_boot, seed),
        "by_source": by_source,
        "by_framing": by_framing,
        "by_language": by_language,
        "cross_lingual_shift": _ci(shifts, n_boot, seed),
        "cross_lingual_shift_floor": _ci(shift_floors, n_boot, seed),
        "cross_lingual_shift_net": {"mean": net_mean, "lo": net_lo, "hi": net_hi,
                                    "n": len(net_shifts)},
        "cross_lingual_emd": _ci(emd_shifts, n_boot, seed),
        "shift_by_item": shift_by_item,
        "cross_lingual_consistency": _ci(consistencies, n_boot, seed),
        "mean_template_sd": _ci(tmpl_sds, n_boot, seed),
        "refusal_rate": (refusal_samples / total_samples) if total_samples else 0.0,
        "refusal_samples": refusal_samples,
        "empty_samples": empty_samples,
        "total_samples": total_samples,
        "refused_items": refused_items,
        "n_responses": len(responses),
        "n_valid": len(valid),
        "n_invalid": n_invalid,
        "n_verified": len(verified),
        "n_example": len(valid) - len(verified),
        "headline_excludes_examples": bool(verified),
    }
