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
        pairs, np.array([100.0, 100.0]), t=float(np.median(arr[:, 2])), monotone="off"
    )
    assert v_star.shape == (3,)
    assert np.isfinite(v_star[2])


# --- regression / robustness tests for the fixes -----------------------------


def _nonmonotone_data(n=20000, seed=1):
    """V_0 a U-shaped (non-FOSD) function of V_1 -- breaks stochastic monotonicity."""
    rng = np.random.default_rng(seed)
    v2 = rng.normal(0, 1, n)
    v1 = 0.8 * v2 + 0.6 * rng.normal(0, 1, n)
    v0 = v1**2 + 0.3 * rng.normal(0, 1, n)
    return np.column_stack([v2, v1, v0])


def test_nonmonotone_detected_and_handled():
    pairs = tq.fit_pairwise_gmms(_nonmonotone_data(), n_components=6)
    c, t = np.array([0.01, 0.01]), 0.0
    # "check" warns but still returns.
    with pytest.warns(UserWarning, match="not nondecreasing"):
        tq.train_tarquin(pairs, c, t, monotone="check")
    # "raise" errors out.
    with pytest.raises(ValueError, match="not nondecreasing"):
        tq.train_tarquin(pairs, c, t, monotone="raise")
    # "project" warns and yields monotone value functions.
    with pytest.warns(UserWarning, match="Projecting"):
        _, tab = tq.train_tarquin(pairs, c, t, monotone="project")
    for p in tab["p"]:
        assert np.all(np.diff(p) >= -1e-9)


def test_default_monotone_is_project_and_noop_on_gaussian():
    # Single-component Gaussian is already monotone: project must not change the
    # README thresholds, and must emit no warning.
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning -> failure
        v_star, tab = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0)  # default project
    assert v_star[0] == pytest.approx(0.2886, abs=1e-3)
    assert v_star[1] == pytest.approx(0.1204, abs=1e-3)
    for p in tab["p"]:
        assert np.all(np.diff(p) >= -1e-9)


def test_negative_cost_and_bad_shape_raise():
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    with pytest.raises(ValueError, match="nonnegative"):
        tq.train_tarquin(pairs, np.array([-0.1, 0.1]), t=1.0)
    with pytest.raises(ValueError, match="length 2"):
        tq.train_tarquin(pairs, np.array([0.1, 0.1, 0.1]), t=1.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        tq.infer_tarquin([0.0, 0.0, 0.0], [1.0, 1.0])


def test_cost_order_lockin():
    # train_book maps per-prophecy costs to README order; lock it with asymmetric costs.
    g = gaussian_example()
    pairs = tq.pairs_from_joint(g, (0, 1, 2))
    cost_per_col = np.array([np.nan, 0.2, 0.01])  # cols (V_2 free, V_1=0.2, V_0=0.01)
    v_book, _ = tq.train_book(g, (0, 1, 2), cost_per_col, t=1.0)
    # README order c = (c_1, c_0) = (cost[V_1], cost[V_0]) = (0.2, 0.01).
    v_direct, _ = tq.train_tarquin(pairs, np.array([0.2, 0.01]), t=1.0)
    assert v_book == pytest.approx(v_direct, abs=1e-9)


def test_marginalize_gmm_is_usable():
    # precisions_cholesky_ is now set, so sklearn's scoring/sampling path works.
    g = gaussian_example()
    m = tq.marginalize_gmm(g, [0, 1])
    X, _ = m.sample(100)
    assert X.shape == (100, 2)
    assert np.all(np.isfinite(m.score_samples(X)))


def test_thresholds_beat_perturbations_mc():
    # The learned thresholds should maximize E[pi] under the MC policy: perturbing
    # either nontrivial threshold up or down should not increase the payoff.
    g = gaussian_example()
    pairs = tq.pairs_from_joint(g, (0, 1, 2))
    v_star, _ = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0)
    cost = np.array([np.nan, 0.05, 0.1])
    rng = np.random.default_rng(0)
    S = rng.multivariate_normal(g.means_[0], g.covariances_[0], size=1_000_000)
    base = tq.evaluate_policy_mc(S, (0, 1, 2), v_star, cost, t=1.0).mean()
    for j in (0, 1):  # v_2*, v_1* (v_0* = t is fixed by Prop. 1)
        for d in (-0.1, 0.1):
            vp = v_star.copy()
            vp[j] += d
            alt = tq.evaluate_policy_mc(S, (0, 1, 2), vp, cost, t=1.0).mean()
            assert alt <= base + 5e-4  # within MC noise


def test_abridgement_dominance_regression():
    # README claim: the full book (0,1,2) is dominated by both 2-prophecy abridgements.
    g = gaussian_example()
    cost = np.array([np.nan, 0.05, 0.1])
    rng = np.random.default_rng(0)
    S = rng.multivariate_normal(g.means_[0], g.covariances_[0], size=1_000_000)
    means = {}
    for book in [(0, 1, 2), (0, 2), (1, 2)]:
        v, _ = tq.train_book(g, book, cost, t=1.0)
        means[book] = tq.evaluate_policy_mc(S, book, v, cost, t=1.0).mean()
    assert means[(1, 2)] > means[(0, 1, 2)]
    assert means[(0, 2)] > means[(0, 1, 2)]


# --- fixes from the review -----------------------------------------------------


def test_always_proceed_threshold_is_neg_inf():
    # Zero costs with a modest t: proceeding is worthwhile even at the lowest modeled
    # signal, so S_n covers the grid and the threshold saturates to -inf (not grid[0]).
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    with pytest.warns(UserWarning, match="saturated to -inf"):
        v_star, _ = tq.train_tarquin(pairs, np.array([0.0, 0.0]), t=1.0, monotone="off")
    assert v_star[0] == -np.inf  # v_2*
    assert v_star[1] == -np.inf  # v_1*
    assert v_star[2] == pytest.approx(1.0, abs=1e-9)  # v_0* = t, still finite


def test_diagnose_fosd_flags_violation():
    # FOSD-true Markov demo: small violation. U-shaped (non-FOSD) data: large.
    assert tq.diagnose_fosd(tq.make_demo_data())["max"] < 0.15
    assert tq.diagnose_fosd(_nonmonotone_data())["max"] > 0.5


def test_diagnose_sufficiency_on_markov_demo():
    # make_demo_data is a genuine Markov chain, so partial correlations are ~0.
    out = tq.diagnose_sufficiency(tq.make_demo_data())
    assert out["partial_corr"].shape == (1,)  # one interior column for N=2
    assert out["max_abs"] < 0.1


def test_bic_selects_components_and_trains():
    arr = tq.make_demo_data()
    pairs = tq.fit_pairwise_gmms(arr, n_components=range(1, 5), n_init=1)
    assert len(pairs) == 2
    assert all(1 <= p.n_components <= 4 for p in pairs)
    v_star, _ = tq.train_tarquin(
        pairs, np.array([100.0, 100.0]), t=float(np.median(arr[:, 2])), monotone="off"
    )
    assert np.isfinite(v_star[2])


def test_fit_empty_candidate_list_raises():
    with pytest.raises(ValueError, match="positive int or a nonempty sequence"):
        tq.fit_pairwise_gmms(tq.make_demo_data(), n_components=[])


def test_marginalize_handles_nonfull_covariance():
    # A diag-covariance joint must still marginalize and stay sklearn-usable.
    g = tq.fit_joint_gmm(tq.make_demo_data(), n_components=3, covariance_type="diag")
    m = tq.marginalize_gmm(g, [0, 1])
    X, _ = m.sample(50)
    assert X.shape == (50, 2)
    assert np.all(np.isfinite(m.score_samples(X)))


def test_holdout_split_is_disjoint_partition():
    data = tq.make_demo_data()
    train, test = tq.holdout_split(data, test_frac=0.25, seed=0)
    assert train.shape[0] + test.shape[0] == data.shape[0]
    assert test.shape[0] == round(data.shape[0] * 0.25)
    # union of rows equals the original set (disjoint partition)
    assert {tuple(r) for r in np.vstack([train, test])} == {tuple(r) for r in data}
    with pytest.raises(ValueError, match="test_frac"):
        tq.holdout_split(data, test_frac=1.5)
