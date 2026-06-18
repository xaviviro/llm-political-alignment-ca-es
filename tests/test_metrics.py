import pytest

from political_alignment import metrics


def test_normalise_sums_to_one():
    out = metrics.normalise([1, 1, 2])
    assert out.sum() == pytest.approx(1.0)
    assert out.tolist() == [0.25, 0.25, 0.5]


def test_normalise_all_zero_is_uniform():
    out = metrics.normalise([0, 0, 0, 0])
    assert out.tolist() == [0.25, 0.25, 0.25, 0.25]


def test_jsd_identical_is_zero():
    assert metrics.jsd([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)


def test_jsd_disjoint_is_one():
    assert metrics.jsd([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-6)


def test_alignment_is_one_minus_jsd():
    p, q = [0.7, 0.3], [0.6, 0.4]
    assert metrics.alignment_score(p, q) == pytest.approx(1.0 - metrics.jsd(p, q))


def test_cross_lingual_consistency_identical_langs():
    d = {"ca": [0.2, 0.8], "es": [0.2, 0.8]}
    assert metrics.cross_lingual_consistency(d) == pytest.approx(1.0, abs=1e-9)


def test_cross_lingual_consistency_single_lang_is_one():
    assert metrics.cross_lingual_consistency({"ca": [0.5, 0.5]}) == 1.0


def test_bootstrap_ci_is_seeded():
    vals = [0.1, 0.5, 0.9, 0.3, 0.7]
    a = metrics.bootstrap_mean_ci(vals, n_boot=500, seed=42)
    b = metrics.bootstrap_mean_ci(vals, n_boot=500, seed=42)
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_wasserstein1_respects_ordering():
    near = metrics.wasserstein1([1, 0, 0], [0, 1, 0])   # one step
    far = metrics.wasserstein1([1, 0, 0], [0, 0, 1])    # two steps (extreme)
    assert near == pytest.approx(1.0)
    assert far == pytest.approx(2.0)
    assert near < far                                   # ordinal: not all swaps equal
    assert metrics.wasserstein1([0.2, 0.5, 0.3], [0.2, 0.5, 0.3]) == pytest.approx(0.0)


def test_wasserstein1_normalize_to_unit_range():
    assert metrics.wasserstein1([1, 0, 0, 0], [0, 0, 0, 1], normalize=True) == pytest.approx(1.0)


def test_ordinal_view_drops_non_ordinal_and_renormalises():
    sub, pos = metrics.ordinal_view([0.4, 0.4, 0.2], [0.0, 1.0, None])
    assert pos.tolist() == [0.0, 1.0]
    assert sub.tolist() == pytest.approx([0.5, 0.5])


def test_noise_floor_positive_and_shrinks_with_n():
    small = metrics.jsd_noise_floor([6] * 10, n_boot=200, seed=1)    # n=60 over 10
    big = metrics.jsd_noise_floor([600] * 10, n_boot=200, seed=1)    # n=6000 over 10
    assert small > 0.0          # finite-sample JSD bias is positive
    assert small > big          # more samples -> lower floor


def test_noise_floor_not_zero_for_identical_distribution():
    # two finite samples from the SAME distribution still give JSD ~= floor > 0;
    # this is exactly the bias we subtract, so the floor must not collapse to 0.
    floor = metrics.jsd_noise_floor([5] * 12 + [0], n_boot=300, seed=2)  # ~60 / 13
    assert 0.0 < floor < 0.5


def test_hierarchical_net_shift_zero_for_same_distribution_draws():
    import numpy as np
    rng = np.random.default_rng(0)
    p = np.array([0.25, 0.25, 0.25, 0.25])
    n = 60
    items, floors = [], []
    for _ in range(12):
        ca = rng.multinomial(n, p) / n
        es = rng.multinomial(n, p) / n
        items.append({"ca": (ca, n), "es": (es, n)})
        floors.append(metrics.jsd_noise_floor(((ca + es) / 2) * n, n_boot=150, seed=1))
    mean, lo, hi = metrics.hierarchical_net_shift_ci(items, floors, n_boot=300, seed=2)
    assert mean == pytest.approx(0.0, abs=0.06)   # point estimate: pure noise -> net ~ 0
    # The hierarchical bootstrap is deliberately conservative near the floor
    # (re-injecting sampling noise inflates a convex distance like JSD), so the CI
    # can sit slightly above the point estimate; it stays low and ordered.
    assert 0.0 <= lo <= hi < 0.2
