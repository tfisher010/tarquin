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


# --- fixes from the second review --------------------------------------------


@pytest.mark.parametrize("ct", ["full", "tied", "diag", "spherical"])
def test_fit_pairwise_all_covariance_types_train(ct):
    # Every advertised covariance_type must feed train_tarquin: a non-"full" fit is
    # densified to the (K, D, D) layout the recursion indexes (previously crashed).
    arr = tq.make_demo_data()
    pairs = tq.fit_pairwise_gmms(arr, n_components=3, covariance_type=ct)
    assert all(p.covariance_type == "full" for p in pairs)
    v_star, _ = tq.train_tarquin(
        pairs, np.array([100.0, 100.0]), t=float(np.median(arr[:, 2])), monotone="off"
    )
    assert v_star.shape == (3,)
    assert np.isfinite(v_star[2])


def test_degenerate_grid_raises():
    # A conditioning column with zero variance collapses its grid to a point; the
    # builder must raise rather than later divide by dx = 0.
    g = GaussianMixture(n_components=1, covariance_type="full")
    g.weights_ = np.array([1.0])
    g.means_ = np.array([[0.0, 0.0]])
    g.covariances_ = np.array([[[0.0, 0.0], [0.0, 1.0]]])  # V_1 (col 0) has zero variance
    with pytest.raises(ValueError, match="degenerate grid"):
        tq.train_tarquin([g], np.array([0.1]), t=0.0)


def test_bootstrap_thresholds_shapes_and_ci():
    arr = tq.make_demo_data()
    c, t = np.array([100.0, 100.0]), float(np.median(arr[:, 2]))
    out = tq.bootstrap_thresholds(arr, c, t, n_boot=20, n_components=4, seed=0)
    assert out["point"].shape == (3,)
    assert out["samples"].shape == (20, 3)
    for key in ("mean", "std", "ci_low", "ci_high", "n_finite"):
        assert out[key].shape == (3,)
    # v_0* = t exactly on every replicate: zero spread, CI collapses to t.
    assert out["ci_low"][2] == pytest.approx(t, abs=1e-9)
    assert out["ci_high"][2] == pytest.approx(t, abs=1e-9)
    # Where finite, the CI brackets the mean (ordering sanity).
    for i in range(3):
        if out["n_finite"][i] and np.isfinite(out["ci_low"][i]):
            assert out["ci_low"][i] <= out["mean"][i] + 1e-9
            assert out["mean"][i] <= out["ci_high"][i] + 1e-9
    with pytest.raises(ValueError, match="ci must be"):
        tq.bootstrap_thresholds(arr, c, t, n_boot=2, ci=1.5)


# --- fixes from the third review ---------------------------------------------


def test_v0_star_is_t_even_when_t_outside_grid():
    # Prop. 1 says v_0^* = t for any t. Reading the threshold off the V_0 grid would
    # saturate to +/-inf when t lies outside the modeled support; set it directly.
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    for t in (1.0, 20.0, -20.0):  # 20 / -20 sit well outside the V_0 +-10*sigma grid
        v_star, _ = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=t, monotone="off")
        assert v_star[2] == pytest.approx(t, abs=1e-12)
        assert np.isfinite(v_star[2])


def _ar1_chain_gmm(rho, d, var=1.0):
    """Single-component joint GMM for a Gaussian AR(1) chain on d variables.

    AR(1) covariance Sigma_{ij} = var * rho^|i-j| has tridiagonal precision, so the
    sequence in index order is exactly a Markov chain (sufficiency holds) and each
    Gaussian conditional is FOSD-increasing (stochastic monotonicity holds).
    """
    idx = np.arange(d)
    cov = var * rho ** np.abs(idx[:, None] - idx[None, :])
    g = GaussianMixture(n_components=1, covariance_type="full")
    g.weights_ = np.array([1.0])
    g.means_ = np.zeros((1, d))
    g.covariances_ = cov[None, :, :]
    g.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(g.covariances_))
    return g


def test_n3_chain_trains_and_is_well_posed():
    # End-to-end on N=3 (a 4-prophecy book): exercises the linear-in-N recursion and
    # the grid-union logic past the N=2 cases the rest of the suite uses.
    g = _ar1_chain_gmm(rho=0.6, d=4)
    pairs = tq.pairs_from_joint(g, (0, 1, 2, 3))
    assert len(pairs) == 3
    c = np.array([0.02, 0.02, 0.02])  # (c_2, c_1, c_0) in README order
    t = 0.0
    # An exact Gaussian chain is FOSD-true, so no monotonicity projection is needed.
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        v_star, tab = tq.train_tarquin(pairs, c, t)
    assert v_star.shape == (4,)
    assert v_star[3] == pytest.approx(t, abs=1e-12)        # v_0^* = t
    assert np.all(np.isfinite(v_star))                      # all thresholds resolve
    for p in tab["p"]:
        assert np.all(np.diff(p) >= -1e-9)                  # nondecreasing (Prop. 3)
    # Inference walks top-down and short-circuits on the first sub-threshold signal.
    assert list(tq.infer_tarquin(v_star, [5, 5, 5, 5])) == [1, 1, 1, 1]
    assert list(tq.infer_tarquin(v_star, [5, -5, 5, 5])) == [1, 0, 0, 0]


# --- P0: selection-aware (ragged) fitting on truncated deployment samples ----


def _gaussian_example_samples(n, seed=0):
    """Large i.i.d. draw from the README's Gaussian joint (the clean full-joint sample)."""
    mu = np.array([1.0, 0.5, -0.2])
    cov = np.array([[1.0, 0.3, 0.15], [0.3, 1.0, 0.5], [0.15, 0.5, 2.0]])
    return np.random.default_rng(seed).multivariate_normal(mu, cov, size=n)


def test_simulate_incumbent_truncation_is_monotone_missing():
    data = _gaussian_example_samples(5000, seed=1)
    trunc = tq.simulate_incumbent_truncation(data, [0.0, 0.0])  # (tau_2, tau_1)
    assert np.all(np.isfinite(trunc[:, 0]))                     # V_2 always observed (free)
    miss1 = ~np.isfinite(trunc[:, 1])                           # V_1 missing
    miss0 = ~np.isfinite(trunc[:, 2])                           # V_0 missing
    assert np.all(miss0[miss1])                                 # V_1 missing => V_0 missing
    # V_1 revealed iff the draw proceeded past step 2: V_2 > tau_2 = 0.
    assert np.array_equal(np.isfinite(trunc[:, 1]), data[:, 0] > 0.0)
    with pytest.raises(ValueError, match="length 2"):
        tq.simulate_incumbent_truncation(data, [0.0])


def test_fit_pairwise_warns_on_ragged_but_not_on_clean():
    data = _gaussian_example_samples(20_000, seed=2)
    trunc = tq.simulate_incumbent_truncation(data, [0.0, 0.0])
    with pytest.warns(UserWarning, match="ragged/truncated"):
        pairs = tq.fit_pairwise_gmms(trunc, n_components=1)
    assert len(pairs) == 2
    import warnings as _w

    with _w.catch_warnings(record=True) as w:  # clean data must not raise the ragged warning
        _w.simplefilter("always")
        tq.fit_pairwise_gmms(data, n_components=1)
    assert not any("ragged" in str(x.message) for x in w)


def test_fit_pairwise_all_missing_pair_raises():
    data = _gaussian_example_samples(2000, seed=4)
    # A threshold above every V_2 reveals no V_1 at all -> pair (V_2, V_1) unfittable.
    trunc = tq.simulate_incumbent_truncation(data, [data[:, 0].max() + 1.0, 0.0])
    with pytest.raises(ValueError, match="no row reveals both"):
        tq.fit_pairwise_gmms(trunc, n_components=1)


def test_ragged_fit_beats_complete_case_under_truncation():
    # The validation harness: truncate a clean draw with a *loose* incumbent (tau below the
    # true thresholds, so v* stays identified), then check ragged fitting recovers the clean
    # thresholds while naive complete-case fitting is biased on the level whose response is
    # truncated (v_2^*, whose conditional's outcome V_1 complete-case cuts at tau_1).
    import warnings as _w

    data = _gaussian_example_samples(300_000, seed=0)
    c, t = np.array([0.05, 0.1]), 1.0
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        v_clean, _ = tq.train_tarquin(tq.fit_pairwise_gmms(data, n_components=1), c, t)
        trunc = tq.simulate_incumbent_truncation(data, [0.0, 0.0])  # loose incumbent
        v_ragged, _ = tq.train_tarquin(tq.fit_pairwise_gmms(trunc, n_components=1), c, t)
        cc = trunc[np.all(np.isfinite(trunc), axis=1)]              # complete-case (fully revealed)
        v_cc, _ = tq.train_tarquin(tq.fit_pairwise_gmms(cc, n_components=1), c, t)
    err_ragged = abs(v_ragged[0] - v_clean[0])   # index 0 = v_2^*
    err_cc = abs(v_cc[0] - v_clean[0])
    assert err_ragged < err_cc            # ragged removes the response-truncation bias
    assert err_ragged < 0.05              # and lands near the clean threshold (~0.2886)
    assert v_clean[0] == pytest.approx(0.2886, abs=0.02)  # sanity: clean recovers the analytic v_2*


def test_diagnose_saturation_flags_cost_trivial():
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")  # zero cost -> -inf saturation warnings
        v_star, tab = tq.train_tarquin(pairs, np.array([0.0, 0.0]), t=1.0, monotone="off")
    out = tq.diagnose_saturation(v_star, tab, np.array([0.0, 0.0]))
    assert out["cost_trivial"] is True
    assert all(lv["status"].startswith("always_proceed") for lv in out["levels"])
    assert "cost-trivial" in out["summary"]


def test_diagnose_saturation_finite_not_trivial():
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    v_star, tab = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0)
    out = tq.diagnose_saturation(v_star, tab, np.array([0.05, 0.1]))
    assert out["cost_trivial"] is False
    assert any(lv["status"] == "finite" for lv in out["levels"])
    with pytest.raises(ValueError, match="length 2"):
        tq.diagnose_saturation(v_star, tab, np.array([0.05, 0.1, 0.1]))


# --- P1: monotone-bounded extrapolation + partial-identification (bounds) ------


def test_fit_pairwise_records_support_lo():
    data = _gaussian_example_samples(20_000, seed=2)
    trunc = tq.simulate_incumbent_truncation(data, [0.3, -0.2])  # (tau_2, tau_1)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        pairs = tq.fit_pairwise_gmms(trunc, n_components=1)
    # pairs[1] = (V_2, V_1): conditioning col V_2 is observed only where V_2 > tau_2 = 0.3.
    assert pairs[1].tarquin_support_lo_ == pytest.approx(0.3, abs=0.03)
    # pairs[0] = (V_1, V_0): conditioning col V_1 observed only where V_1 > tau_1 = -0.2.
    assert pairs[0].tarquin_support_lo_ == pytest.approx(-0.2, abs=0.03)


def test_bounds_equal_point_when_fully_identified():
    # pairs_from_joint carry no support attribute -> a_n = -inf -> identical to the point fit.
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    v_pt, _ = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0)
    v_lo, v_hi, tab = tq.train_tarquin(
        pairs, np.array([0.05, 0.1]), t=1.0, identification="bounds")
    assert v_lo == pytest.approx(v_pt, abs=1e-9)
    assert v_hi == pytest.approx(v_pt, abs=1e-9)
    assert "p_lo" in tab and "p_hi" in tab


def test_bounds_collapse_when_incumbent_loose():
    import warnings as _w

    data = _gaussian_example_samples(200_000, seed=0)
    c, t = np.array([0.05, 0.1]), 1.0
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        v_clean, _ = tq.train_tarquin(tq.fit_pairwise_gmms(data, n_components=1), c, t)
        loose = tq.simulate_incumbent_truncation(data, [0.0, 0.0])  # below the true thresholds
        v_lo, v_hi, _ = tq.train_tarquin(
            tq.fit_pairwise_gmms(loose, n_components=1), c, t, identification="bounds")
    for i in (0, 1, 2):                       # identified everywhere -> interval collapses
        assert v_lo[i] == pytest.approx(v_hi[i], abs=2e-2)
    assert v_lo[0] == pytest.approx(v_clean[0], abs=0.05)


def test_bounds_bracket_unidentified_threshold():
    import warnings as _w

    data = _gaussian_example_samples(200_000, seed=0)
    c, t = np.array([0.05, 0.1]), 1.0
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        v_clean, _ = tq.train_tarquin(tq.fit_pairwise_gmms(data, n_components=1), c, t)
        tight = tq.simulate_incumbent_truncation(data, [0.5, 0.0])  # tau_2=0.5 > true v_2*~0.29
        v_lo, v_hi, _ = tq.train_tarquin(
            tq.fit_pairwise_gmms(tight, n_components=1), c, t, identification="bounds")
    # v_2^* (index 0) is below the identified support: bracketed by -inf and ~the support edge.
    assert v_lo[0] == -np.inf
    assert np.isfinite(v_hi[0])
    assert v_lo[0] <= v_clean[0] <= v_hi[0]            # the truth lies inside the interval
    assert v_hi[0] == pytest.approx(0.5, abs=0.1)      # ~ incumbent floor / observed min of V_2
    assert v_lo[1] == pytest.approx(v_hi[1], abs=2e-2)  # v_1^* stays identified (tau_1 loose)
    assert v_lo[2] == pytest.approx(t, abs=1e-9)        # v_0^* = t exactly, both bounds


def test_extrapolation_clamp_and_flag():
    import warnings as _w

    data = _gaussian_example_samples(200_000, seed=0)
    c, t = np.array([0.05, 0.1]), 1.0
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        pairs = tq.fit_pairwise_gmms(tq.simulate_incumbent_truncation(data, [0.5, 0.0]),
                                     n_components=1)
        a2 = pairs[1].tarquin_support_lo_                 # identified edge of V_2
        v_clamp, _ = tq.train_tarquin(pairs, c, t, extrapolation="clamp")
        v_gmm, _ = tq.train_tarquin(pairs, c, t, extrapolation="gmm")
    assert np.isfinite(v_clamp[0])
    assert v_clamp[0] == pytest.approx(a2, abs=0.05)      # conservative: report at the edge
    assert v_gmm[0] < a2                                  # free gmm extrapolates below it
    with pytest.warns(UserWarning, match="below the identified support edge"):
        tq.train_tarquin(pairs, c, t, extrapolation="flag")


def test_bounds_support_lo_override():
    # An explicit support_lo overrides the per-pair attribute: forcing a high a_2 makes v_2^*
    # unidentified even on clean pairs (no truncation), bracketing it at [-inf, ~a_2].
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    v_lo, v_hi, _ = tq.train_tarquin(
        pairs, np.array([0.05, 0.1]), t=1.0, identification="bounds",
        support_lo=np.array([2.0, np.nan, np.nan]))  # README order; force a_2 (V_2 slot) = 2.0
    assert v_lo[0] == -np.inf
    assert v_hi[0] == pytest.approx(2.0, abs=0.1)
    with pytest.raises(ValueError, match="support_lo of length 3"):
        tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0, identification="bounds",
                         support_lo=np.array([2.0, 0.0]))


def test_evaluate_policy_mc_hand_computed():
    # Lock the cost accounting against a payoff computed by hand on 3 deterministic rows.
    # Book (col 0, col 1); v* = (1.0 for col 0, 0.0 = t for col 1); acquiring col 1 costs 2.
    samples = np.array([[10.0, 5.0], [10.0, -5.0], [0.0, 5.0]])
    cost_per_prophecy = np.array([np.nan, 2.0])  # col 0 free (top), col 1 costs 2
    v_star = np.array([1.0, 0.0])
    pi = tq.evaluate_policy_mc(samples, (0, 1), v_star, cost_per_prophecy, t=0.0)
    # Row 0: pass v0>1, pay 2 for col1, pass v1>0, collect 5  -> -2 + 5 = 3
    # Row 1: pass v0>1, pay 2 for col1, fail v1>0             -> -2
    # Row 2: fail v0>1 (exit before any cost)                 ->  0
    assert pi == pytest.approx([3.0, -2.0, 0.0], abs=1e-12)
