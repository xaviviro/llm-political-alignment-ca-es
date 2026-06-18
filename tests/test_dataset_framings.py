import numpy as np
import pytest

from political_alignment.dataset import SurveyItem, load_datasets
from political_alignment.framings import render, render_variants


def test_datasets_load_and_validate():
    items = load_datasets(["data/ceo_items.csv", "data/cis_items.csv"])
    assert len(items) >= 1
    for it in items:
        assert abs(it.pop_dist.sum() - 1.0) < 1e-3
        assert it.n_options == len(it.pop_dist)
        assert it.source_status in {"verified", "example"}
    # the CEO items are real survey marginals, not placeholders
    ceo = [it for it in items if it.source == "CEO"]
    assert ceo and all(it.source_status == "verified" for it in ceo)


def test_load_rejects_bad_distribution(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "item_id,source,survey_wave,population,topic,question_ca,question_es,"
        "options,options_es,pop_dist,dimension,source_status,notes\n"
        "x1,CEO,w,catalonia,t,qca,qes,A|B,A|B,0.9|0.5,national,example,n\n"
    )
    with pytest.raises(ValueError, match="sums to"):
        load_datasets([csv])


def _item():
    return SurveyItem(
        item_id="t1", source="CEO", survey_wave="w", population="catalonia",
        topic="independence",
        question={"ca": "Pregunta?", "es": "¿Pregunta?"},
        options={"ca": ["Sí", "No"], "es": ["Sí", "No"]},
        pop_dist=np.array([0.5, 0.5]), dimension="national",
        source_status="example", notes="",
    )


def test_render_neutral_has_options_and_letters():
    prompt, letters = render(_item(), "ca", "neutral")
    assert letters == ["A", "B"]
    assert "A) Sí" in prompt and "B) No" in prompt
    assert "Pregunta?" in prompt


def test_render_observer_injects_place():
    prompt, _ = render(_item(), "es", "observer")
    assert "Cataluña" in prompt


def test_render_personalised_differs_from_neutral():
    p_neutral, _ = render(_item(), "ca", "neutral")
    p_personal, _ = render(_item(), "ca", "personalised")
    assert p_neutral != p_personal
    assert "opinió" in p_personal


def test_render_variants_are_multiple_and_distinct():
    variants = render_variants(_item(), "ca", "observer")
    assert len(variants) >= 2  # multiple paraphrases per framing (Röttger check)
    prompts = [p for _i, p, _l in variants]
    assert len(set(prompts)) == len(prompts)  # all paraphrases differ
    # template indices are 0..k-1 and letters are stable across paraphrases
    assert [i for i, _p, _l in variants] == list(range(len(variants)))
    assert all(letters == ["A", "B"] for _i, _p, letters in variants)
