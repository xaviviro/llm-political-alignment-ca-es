"""End-to-end survey-alignment run with the network-free MockProvider."""

import numpy as np
import pytest

from political_alignment import survey_alignment
from political_alignment.dataset import load_datasets
from political_alignment.providers import MockProvider


def _config():
    return {
        "languages": ["ca", "es"],
        "framings": ["neutral", "personalised", "observer"],
        "distribution_method": "sampling",
        "bootstrap": {"n_boot": 200, "seed": 42},
    }


def test_survey_alignment_end_to_end_mock():
    items = load_datasets(["data/ceo_items.csv", "data/cis_items.csv"])
    result = survey_alignment.evaluate(MockProvider(id="mock"), items, _config())

    # n_items x 3 framings x 2 langs responses
    expected = len(items) * 3 * 2
    assert len(result["responses"]) == expected
    s = result["summary"]
    assert 0.0 <= s["alignment_overall"]["mean"] <= 1.0
    assert 0.0 <= s["cross_lingual_consistency"]["mean"] <= 1.0
    # headline pool excludes example items when verified ones exist, so by_source
    # reflects whatever sources are in the pool (CEO is verified)
    assert "CEO" in s["by_source"]
    assert set(s["by_language"]) == {"ca", "es"}
    assert set(s["by_framing"]) == {"neutral", "personalised", "observer"}
    assert s["n_verified"] + s["n_example"] == expected


def test_mock_is_deterministic():
    items = load_datasets(["data/ceo_items.csv"])
    cfg = _config()
    r1 = survey_alignment.evaluate(MockProvider(id="mock", seed=1), items, cfg)
    r2 = survey_alignment.evaluate(MockProvider(id="mock", seed=1), items, cfg)
    assert r1["responses"][0]["model_dist"] == r2["responses"][0]["model_dist"]


class _NoiseProvider:
    """Samples every prompt from the SAME uniform distribution, so the Catalan and
    Spanish answers to an item differ ONLY by sampling noise. The raw cross-lingual
    shift is then pure noise and the *net* shift (raw - floor) must be ~0."""

    def __init__(self, n=60, seed=0):
        self.id = "noise"
        self.n = n
        self._rng = np.random.default_rng(seed)

    def answer_distribution(self, prompt, letters):
        k = len(letters)
        draw = self._rng.multinomial(self.n, np.full(k, 1.0 / k)).astype(float)
        return draw / draw.sum(), {"method": "sampling", "n": self.n, "n_fail": 0}


def test_net_shift_is_zero_when_only_sampling_noise():
    items = load_datasets(["data/ceo_items.csv", "data/cis_items.csv"])
    cfg = _config()
    cfg["bootstrap"] = {"n_boot": 200, "seed": 1, "floor_boot": 80}
    s = survey_alignment.evaluate(_NoiseProvider(n=60, seed=3), items, cfg)["summary"]
    # raw shift and floor are both positive (finite-sample noise)...
    assert s["cross_lingual_shift"]["mean"] > 0.0
    assert s["cross_lingual_shift_floor"]["mean"] > 0.0
    # ...but the noise floor explains it: the net shift collapses to ~0.
    assert s["cross_lingual_shift_net"]["mean"] == pytest.approx(0.0, abs=0.05)
