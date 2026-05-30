"""Tests for the Tarquin algorithm.

Run with `pytest` from the repo root (pyproject sets pythonpath = ["."]) or
after `pip install -e .`.
"""
import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

import tarquin as tq


def gaussian_example():
    """The README's worked Gaussian example as a single-component joint GMM."""
    mu = np.array([1.0, 0.5, -0.2])
    cov = np.array([
        [1.0, 0.3, 0.15],
        [0.3, 1.0, 0.5],
        [0.15, 0.5, 2.0],
    ])
    g = GaussianMixture(n_components=1, covariance_type="full")
    g.weights_ = np.array([1.0])
    g.means_ = mu[None, :]
    g.covariances_ = cov[None, :, :]
    g.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(g.covariances_))
    return g


def test_reproduces_readme_thresholds():
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    v_star, _ = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0)
    # README order (v_2*, v_1*, v_0*).
    assert v_star[0] == pytest.approx(0.2886, abs=1e-3)
    assert v_star[1] == pytest.approx(0.1204, abs=1e-3)
    assert v_star[2] == pytest.approx(1.0, abs=1e-9)  # v_0* = t


def test_value_functions_are_monotone():
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    _, tab = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0)
    for p in tab["p"]:
        assert np.all(np.diff(p) >= -1e-9)  # nondecreasing (Prop. 3)


def test_infer_walks_then_shortcircuits():
    v_star = np.array([0.0, 0.0, 0.0])
    assert list(tq.infer_tarquin(v_star, [1.0, 1.0, 1.0])) == [1, 1, 1]
    assert list(tq.infer_tarquin(v_star, [1.0, -1.0, 1.0])) == [1, 0, 0]
    assert list(tq.infer_tarquin(v_star, [-1.0, 1.0, 1.0])) == [0, 0, 0]


def test_empty_endorsement_when_cost_dominates():
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    # A huge top-level cost (c_1) makes proceeding from V_2 never worthwhile.
    v_star, _ = tq.train_tarquin(pairs, np.array([1e6, 0.1]), t=1.0)
    assert np.isinf(v_star[0])  # S_2 empty => v_2* = +inf


def test_enumerate_abridgements_count_and_shape():
    col_order = (0, 1, 2, 3)
    abr = list(tq.enumerate_abridgements(col_order))
    assert len(abr) == 2 ** (len(col_order) - 1) - 2
    for book in abr:
        assert book[-1] == col_order[-1]        # keeps the payoff prophecy V_0
        assert len(book) >= 2                   # not the empty singleton
        assert len(set(book)) == len(book)      # no repeats


def test_train_book_abridgement_runs():
    cost = np.array([np.nan, 0.05, 0.1])
    v_star, _ = tq.train_book(gaussian_example(), (0, 2), cost, t=1.0)
    assert v_star.shape == (2,)
    assert v_star[-1] == pytest.approx(1.0, abs=1e-9)


def test_make_demo_data_deterministic_and_shaped():
    a = tq.make_demo_data()
    b = tq.make_demo_data()
    assert a.shape == (2056, 3)
    assert np.array_equal(a, b)        # deterministic for a fixed seed
    assert (a[:, 0] > 0).all()         # V_2 positive
    assert (a[:, 1] > 0).all()         # V_1 positive
    assert (a[:, 2] < 0).any()         # V_0 can go negative
    r = np.corrcoef(a.T)
    assert 0.2 < r[0, 1] < 0.5         # adjacent correlations in the right range
    assert 0.4 < r[1, 2] < 0.7


def test_fit_pairwise_then_train_runs():
    arr = tq.make_demo_data()
    pairs = tq.fit_pairwise_gmms(arr, n_components=5)
    assert len(pairs) == 2
    v_star, _ = tq.train_tarquin(
        pairs, np.array([100.0, 100.0]), t=float(np.median(arr[:, 2]))
    )
    assert v_star.shape == (3,)
    assert np.isfinite(v_star[2])
