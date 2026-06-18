"""Pure metrics for political-bias evaluation. No IO, no network.

All functions take and return plain numbers / numpy arrays so they can be
unit-tested in isolation. Distributions are 1-D arrays that sum to 1.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def normalise(counts) -> np.ndarray:
    """Turn a vector of non-negative counts/weights into a probability vector.

    An all-zero vector maps to a uniform distribution (no information).
    """
    arr = np.asarray(counts, dtype=np.float64)
    if (arr < 0).any():
        raise ValueError(f"counts must be non-negative, got {arr}")
    total = arr.sum()
    if total == 0:
        return np.full(arr.shape, 1.0 / arr.size)
    return arr / total


def jsd(p, q) -> float:
    """Jensen-Shannon distance (base 2) between two distributions, in [0, 1].

    0 = identical, 1 = maximally different. This is a true metric (the square
    root of the JS divergence). Inputs are normalised defensively.
    """
    p = normalise(p)
    q = normalise(q)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    d = float(jensenshannon(p, q, base=2))
    # jensenshannon can return a tiny NaN for identical zero-handling; clamp.
    return 0.0 if np.isnan(d) else d


def ordinal_view(dist, codes):
    """Restrict a distribution to its ordinal (numeric-coded) categories.

    ``codes`` is one entry per category: a number for ordinal options (its scale
    position, e.g. 0..10) or ``None`` for non-ordinal options (don't-know /
    no-answer). Returns ``(sub_dist, positions)`` over the ordinal options only,
    with ``sub_dist`` renormalised to sum to 1. The dropped NS/NC mass is handled
    separately (it has no place on the scale).
    """
    dist = np.asarray(dist, dtype=np.float64)
    keep = [(p, float(c)) for p, c in zip(dist, codes, strict=True) if c is not None]
    if not keep:
        return np.array([]), np.array([])
    sub = normalise(np.array([p for p, _ in keep]))
    pos = np.array([c for _, c in keep], dtype=np.float64)
    return sub, pos


def wasserstein1(p, q, positions=None, normalize=False) -> float:
    """1-Wasserstein (earth-mover) distance between two distributions on an
    ordinal support.

    Unlike JSD (which is nominal — every category swap costs the same),
    Wasserstein-1 respects the ordering: moving mass one step costs less than
    moving it across the whole scale. ``positions`` are the ordinal coordinates
    (default ``0..k-1``); with ``normalize=True`` the distance is divided by the
    support range so it lands in ``[0, 1]`` and is comparable across items.
    """
    p = normalise(p)
    q = normalise(q)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    k = p.size
    if k < 2:
        return 0.0
    positions = np.arange(k, dtype=np.float64) if positions is None \
        else np.asarray(positions, dtype=np.float64)
    order = np.argsort(positions)
    pos, pp, qq = positions[order], p[order], q[order]
    cdf_diff = np.abs(np.cumsum(pp)[:-1] - np.cumsum(qq)[:-1])
    w = float(np.sum(cdf_diff * np.diff(pos)))
    if normalize:
        rng = float(pos[-1] - pos[0])
        return w / rng if rng > 0 else 0.0
    return w


def alignment_score(model_dist, population_dist) -> float:
    """Alignment of a model distribution to a population distribution.

    Returns 1 - JSD, so 1.0 = perfectly aligned, 0.0 = maximally misaligned.
    """
    return 1.0 - jsd(model_dist, population_dist)


def cross_lingual_inconsistency(dists_by_lang: dict) -> float:
    """Mean pairwise JSD between a model's distributions across languages.

    ``dists_by_lang`` maps language code -> distribution for the SAME item.
    Higher = the model answers more differently depending on the prompt
    language (the language-induced shift). 0 with fewer than two languages.
    """
    langs = sorted(dists_by_lang)
    if len(langs) < 2:
        return 0.0
    pairs = []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            pairs.append(jsd(dists_by_lang[langs[i]], dists_by_lang[langs[j]]))
    return float(np.mean(pairs))


def cross_lingual_consistency(dists_by_lang: dict) -> float:
    """1 - mean pairwise JSD across languages. 1.0 = identical across languages."""
    return 1.0 - cross_lingual_inconsistency(dists_by_lang)


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Mean and 95% bootstrap CI of a list of per-item scores.

    Returns (mean, lo, hi). Seeded for reproducibility.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = arr.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = arr[idx].mean()
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def jsd_noise_floor(counts, n_boot: int = 300, seed: int = 42, draw_n=None) -> float:
    """Expected plug-in JSD between two independent samples of the SAME distribution.

    The plug-in (empirical) JSD is biased upward: even two samples drawn from the
    *identical* distribution give JSD > 0 because of finite-sample noise. With
    ~60 samples over 11--13 categories this bias is non-trivial, so part of a raw
    cross-lingual shift can be pure noise rather than a real language effect.

    This estimates that "noise floor" under the null hypothesis "same
    distribution" by Monte Carlo: take the observed category ``counts`` for one
    language, treat ``p = counts / sum(counts)`` as the shared distribution, and
    draw two independent multinomial samples of size ``n = sum(counts)`` (the
    ca↔ca / es↔es self-comparison), returning the mean JSD between them.

    For identical underlying distributions the result is ``≈`` the floor (a
    positive number), **not** 0 — that is exactly the bias we want to subtract.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size < 2:
        return 0.0
    p = normalise(counts)
    n = int(round(counts.sum())) if draw_n is None else int(draw_n)
    if n <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        a = rng.multinomial(n, p)
        c = rng.multinomial(n, p)
        vals[b] = jsd(a, c)
    return float(vals.mean())


def hierarchical_net_shift_ci(items_langs, floors, n_boot: int = 1000, seed: int = 42):
    """Aggregate net cross-lingual shift with a hierarchical bootstrap CI.

    ``items_langs`` is a list (one entry per item/framing that has >=2 languages)
    of ``{lang: (p, n_eff)}`` — the model distribution and effective valid sample
    size per language. ``floors`` is the matching per-item JSD noise floor.

    The net shift per item is ``max(0, raw_shift - floor)``. The bootstrap
    resamples BOTH levels — items (with replacement) and, within each item, the
    samples (a fresh multinomial draw of size ``n_eff`` from ``p``) — so the CI
    combines between-item variability with intra-item sampling noise. Returns
    ``(net_mean, lo, hi)``.

    Note: near the noise floor the CI is **conservative** (it can sit slightly
    above the point estimate). Re-injecting sampling noise inflates a convex
    distance like JSD, so the bootstrap over-states small shifts; this only
    matters for shifts close to the floor, not for the large ones in the panel.
    """
    k_items = len(items_langs)
    if k_items == 0:
        return float("nan"), float("nan"), float("nan")
    obs = []
    for langs, fl in zip(items_langs, floors, strict=True):
        raw = cross_lingual_inconsistency({lg: p for lg, (p, _n) in langs.items()})
        obs.append(max(0.0, raw - fl))
    point = float(np.mean(obs))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k_items, size=k_items)
        nets = []
        for i in idx:
            by = {}
            for lg, (p, n) in items_langs[i].items():
                n = int(n)
                by[lg] = (rng.multinomial(n, p) / float(n)) if n > 0 else np.asarray(p, float)
            nets.append(max(0.0, cross_lingual_inconsistency(by) - floors[i]))
        boot[b] = float(np.mean(nets))
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
