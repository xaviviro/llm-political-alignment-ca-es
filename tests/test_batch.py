"""Batch request-generation and results-aggregation (pure, no network)."""

from political_alignment import batch
from political_alignment.dataset import load_datasets


def _config(n_samples=2):
    return {
        "languages": ["ca", "es"],
        "framings": ["neutral", "personalised", "observer"],
        "sampling": {"n_samples": n_samples, "temperature": 1.0, "max_tokens": 512},
        "bootstrap": {"n_boot": 100, "seed": 42},
        "datasets": ["data/ceo_items.csv", "data/cis_items.csv"],
    }


def test_provider_of_splits_litellm_id():
    assert batch.provider_of("groq/openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert batch.provider_of("anthropic/claude-opus-4-8") == ("anthropic", "claude-opus-4-8")
    assert batch.provider_of("gemini/gemini-2.5-flash-lite") == ("gemini", "gemini-2.5-flash-lite")


def test_build_requests_count_and_unique_ids():
    items = load_datasets(["data/ceo_items.csv"])[:1]  # one item
    reqs = batch.build_requests(items, _config(n_samples=2), raw_model="m")
    # 1 item x 2 langs x 3 framings x 3 templates x 2 samples = 36
    assert len(reqs) == 36
    ids = [r["custom_id"] for r in reqs]
    assert len(set(ids)) == len(ids)
    assert all(r["model"] == "m" for r in reqs)
    # round-trip the custom_id encoding
    item_id, lang, framing, tmpl, sample = batch._parse_cid(ids[0])
    assert lang in ("ca", "es") and isinstance(tmpl, int) and isinstance(sample, int)


def test_results_to_eval_aggregates_letters():
    items = load_datasets(["data/ceo_items.csv", "data/cis_items.csv"])
    cfg = _config(n_samples=2)
    reqs = batch.build_requests(items, cfg, raw_model="m")
    # simulate every completion answering option "A"
    results = {r["custom_id"]: "A" for r in reqs}
    sa = batch.results_to_eval("batchmodel", items, cfg, results)
    assert sa["model_id"] == "batchmodel"
    s = sa["summary"]
    assert s["n_invalid"] == 0  # every sample parsed
    assert 0.0 <= s["cross_lingual_shift"]["mean"] <= 1.0
    # answering "A" in every language -> no cross-lingual shift
    assert s["cross_lingual_shift"]["mean"] == 0.0
