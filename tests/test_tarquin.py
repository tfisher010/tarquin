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
    # "project" (default) corrects SILENTLY -- no monotonicity warning -- and yields
    # monotone value functions. (Saturation warnings may still fire; only the
    # monotonicity warning must be absent.)
    import warnings as _w

    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        _, tab = tq.train_tarquin(pairs, c, t, monotone="project")
    assert not any(
        "nondecreasing" in str(x.message) or "Projecting" in str(x.message) for x in rec
    )
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


# --- P0 grid_bounds override + observed_support; P2 sample_weight --------------


def test_observed_support_ignores_nan():
    data = np.array([[1.0, 5.0, 9.0],
                     [2.0, np.nan, 8.0],
                     [3.0, 7.0, np.nan]])
    sup = tq.observed_support(data)
    assert sup[0] == (1.0, 3.0)
    assert sup[1] == (5.0, 7.0)        # NaN ignored
    assert sup[2] == (8.0, 9.0)
    allnan = np.full((4, 2), np.nan)
    assert tq.observed_support(allnan) == [None, None]


def test_grid_bounds_override_sets_grid_span():
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    # Override the V_2 level (README index 0) grid; leave others to the GMM-derived span.
    bounds = [(-4.0, 4.0), None, None]
    v_star, tab = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0, grid_bounds=bounds)
    g2 = tab["grids"][2]               # level 2 = V_2
    assert g2[0] == pytest.approx(-4.0) and g2[-1] == pytest.approx(4.0)
    # other levels unaffected (not equal to the override range)
    assert not (tab["grids"][1][0] == pytest.approx(-4.0))
    with pytest.raises(ValueError, match="grid_bounds of length 3"):
        tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0, grid_bounds=[(-4.0, 4.0)])


@pytest.mark.slow
def test_grid_bounds_rescues_heavy_truncation_grid():
    # A heavy incumbent cut shrinks the survivor fit's sigma so the default +-10sigma grid can
    # miss the true low support; supplying the uncensored support via observed_support fixes it.
    import warnings as _w

    data = _gaussian_example_samples(200_000, seed=0)
    sup = tq.observed_support(data)    # true (uncensored) per-column support
    tight = tq.simulate_incumbent_truncation(data, [1.5, 0.0])  # heavy cut on V_2
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        pairs = tq.fit_pairwise_gmms(tight, n_components=1)
        _, tab = tq.train_tarquin(pairs, np.array([0.05, 0.1]), t=1.0, grid_bounds=sup,
                                  monotone="off")
    g2 = tab["grids"][2]
    assert g2[0] == pytest.approx(sup[0][0]) and g2[-1] == pytest.approx(sup[0][1])


def test_sample_weight_resamples_and_validates():
    import warnings as _w

    arr = tq.make_demo_data()
    n = arr.shape[0]
    w = np.ones(n)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        pairs = tq.fit_pairwise_gmms(arr, n_components=2, sample_weight=w)
    assert len(pairs) == 2
    # shape / sign validation
    with pytest.raises(ValueError, match="sample_weight must have shape"):
        tq.fit_pairwise_gmms(arr, n_components=2, sample_weight=np.ones(n - 1))
    bad = np.ones(n)
    bad[0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        tq.fit_pairwise_gmms(arr, n_components=2, sample_weight=bad)


def test_sample_weight_shifts_the_fit():
    # Weighting toward high-V_2 rows should raise the fitted mean of V_2 in its pair.
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, (40_000, 3))       # iid columns: V_2, V_1, V_0
    flat = tq.fit_pairwise_gmms(data, n_components=1)
    w = (data[:, 0] > 0).astype(float) + 1e-6      # upweight high-V_2 (data col 0 = V_2)
    up = tq.fit_pairwise_gmms(data, n_components=1, sample_weight=w)
    # pairs[1] = (V_2, V_1); pair col 0 = V_2 = data col 0. Its fitted mean should rise.
    assert up[1].means_[0, 0] > flat[1].means_[0, 0] + 0.2


def test_regime_overlap_reports_floor_gain():
    rng = np.random.default_rng(0)
    full = rng.normal(0, 1, (40_000, 3))                  # iid columns V_2, V_1, V_0
    half = full.shape[0] // 2
    a = tq.simulate_incumbent_truncation(full[:half], [1.0, -3.0])   # regime A: tau_2 = 1.0
    b = tq.simulate_incumbent_truncation(full[half:], [-1.0, -3.0])  # regime B: tau_2 = -1.0
    data = np.vstack([a, b])
    regime = np.array(["A"] * half + ["B"] * (full.shape[0] - half))
    out = tq.diagnose_regime_overlap(data, regime)
    # conditioning col 0 = V_2: the pair (V_2,V_1) floor is the incumbent tau_2 per regime.
    pr = out[0]["per_regime"]
    assert pr["A"][0] == pytest.approx(1.0, abs=0.1)
    assert pr["B"][0] == pytest.approx(-1.0, abs=0.1)
    assert out[0]["floor_gain"] == pytest.approx(2.0, abs=0.2)   # pooling reaches ~2 lower
    assert out[0]["prophecy"] == "V_2"
    # V_1's pair has the same loose tau_1 in both regimes -> negligible gain.
    assert out[1]["floor_gain"] < 0.3
    assert out["max_floor_gain"] == pytest.approx(out[0]["floor_gain"], abs=1e-9)
    assert 2 not in out                                          # payoff V_0 is not a conditioner
    with pytest.raises(ValueError, match="regime length"):
        tq.diagnose_regime_overlap(data, np.array(["A"]))


# --- P0 payoff diagnostics: calibration + circularity --------------------------


def test_payoff_calibration_clean_vs_inflated():
    rng = np.random.default_rng(0)
    v0 = rng.normal(1.0, 1.0, 50_000)
    y_cal = v0 + rng.normal(0, 0.5, v0.size)          # calibrated: E[Y | v0] = v0
    out = tq.diagnose_payoff_calibration(v0, y_cal)
    assert abs(out["bias"]) < 0.02
    assert out["slope"] == pytest.approx(1.0, abs=0.05)
    assert out["ece"] < 0.05
    assert out["inflated"] is False
    y_inf = v0 - 0.5 + rng.normal(0, 0.5, v0.size)    # realized below forecast: V_0 inflated
    out2 = tq.diagnose_payoff_calibration(v0, y_inf)
    assert out2["bias"] < -0.4                         # E[Y] - E[V_0] strongly negative
    assert out2["inflated"] is True
    assert "INFLATED" in out2["summary"]
    assert out2["ece"] == pytest.approx(0.5, abs=0.05)


def test_payoff_calibration_matured_mask_and_support():
    rng = np.random.default_rng(1)
    v0 = rng.normal(0, 1, 1000)
    y = v0 + rng.normal(0, 0.3, 1000)
    y[::2] = np.nan                                    # half un-matured
    out = tq.diagnose_payoff_calibration(v0, y)        # matured = finite y
    assert out["n_matured"] == 500
    assert out["n_total"] == 1000
    lo, hi = out["matured_support"]
    assert lo < hi
    mask = np.zeros(1000, dtype=bool)
    mask[:100] = True                                  # explicit mask overrides
    out2 = tq.diagnose_payoff_calibration(v0, v0, matured_mask=mask)
    assert out2["n_matured"] == 100
    with pytest.raises(ValueError, match=">= 2 matured"):
        tq.diagnose_payoff_calibration(np.array([1.0]), np.array([1.0]),
                                       matured_mask=np.array([False]))


def test_payoff_circularity_flags_rank_duplicate():
    rng = np.random.default_rng(2)
    g = rng.normal(0, 1, 5000)                # a gating prophecy
    indep = rng.normal(0, 1, 5000)            # an independent column
    payoff_mono = np.exp(0.5 * g)             # monotone transform of the gate -> Spearman 1
    data = np.column_stack([g, indep, payoff_mono])   # cols: 0=gate, 1=indep, 2=payoff
    out = tq.diagnose_payoff_circularity(data, payoff_col=2, gate_cols=[0, 1])
    assert out["rank_corr"][0] == pytest.approx(1.0, abs=1e-6)   # monotone dup of the gate
    assert abs(out["rank_corr"][1]) < 0.1                        # independent
    assert out["flagged"] == [0]
    assert "circular" in out["summary"]


def test_payoff_circularity_clean():
    rng = np.random.default_rng(3)
    data = rng.normal(0, 1, (3000, 3))        # all independent
    out = tq.diagnose_payoff_circularity(data, payoff_col=2, gate_cols=[0, 1])
    assert out["flagged"] == []
    assert out["max_abs"] < 0.1
    assert "Necessary-but-not-sufficient" in out["summary"]


def test_inflated_payoff_drives_upstream_saturation_to_neg_inf():
    # End-to-end on the README/diagnostics claim: an inflated payoff forecast V_0 (one with
    # E[Y | V_0] < V_0) makes proceeding worthwhile even at the lowest modeled signal, so every
    # upstream threshold saturates to -inf ("never cut") and only the terminal v_0^* = t binds;
    # a calibrated forecast leaves the upstream thresholds finite. diagnose_payoff_calibration
    # separates the two cases on (forecast, realized) data.
    def chain_gmm(v0_mean):
        mu = np.array([0.0, 0.0, v0_mean])
        cov = np.array([[1.0, 0.6, 0.36], [0.6, 1.0, 0.6], [0.36, 0.6, 1.0]])
        g = GaussianMixture(n_components=1, covariance_type="full")
        g.weights_ = np.array([1.0])
        g.means_ = mu[None, :]
        g.covariances_ = cov[None, :, :]
        g.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
        return g

    c, t = np.array([0.05, 0.05]), 0.5
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")  # the inflated case raises -inf saturation warnings
        v_cal, _ = tq.train_tarquin(
            tq.pairs_from_joint(chain_gmm(0.0), (0, 1, 2)), c, t, monotone="off")
        v_inf, _ = tq.train_tarquin(
            tq.pairs_from_joint(chain_gmm(8.0), (0, 1, 2)), c, t, monotone="off")
    assert np.all(np.isfinite(v_cal[:2]))                   # calibrated: real cuts on V_2, V_1
    assert v_inf[0] == -np.inf and v_inf[1] == -np.inf      # inflated: never cut upstream
    assert v_inf[2] == pytest.approx(t, abs=1e-9)           # only the terminal rule binds

    rng = np.random.default_rng(0)
    y = rng.normal(0.5, 1.0, 40_000)                        # realized outcome
    v0_cal = y + rng.normal(0, 0.3, y.size)                 # calibrated: E[Y | V_0] ~ V_0
    v0_inf = y + 8.0 + rng.normal(0, 0.3, y.size)           # inflated forecast (+8)
    assert tq.diagnose_payoff_calibration(v0_cal, y)["inflated"] is False
    assert tq.diagnose_payoff_calibration(v0_inf, y)["inflated"] is True


def test_rearrangement_can_break_markov_sufficiency():
    # Abridgements preserve the Markov chain (dropping interior nodes of a chain leaves a
    # chain), but a genuine rearrangement need not: train_book on a reordering is only
    # approximate. Lock the premise of that caveat -- the worked example is Markov in the
    # natural order (V_2->V_1->V_0) yet not under the (V_1, V_2, V_0) permutation, which
    # puts the weakly-linked pair adjacent.
    g = gaussian_example()
    rng = np.random.default_rng(0)
    S = rng.multivariate_normal(g.means_[0], g.covariances_[0], size=200_000)
    assert tq.diagnose_sufficiency(S)["max_abs"] < 0.05            # natural order: Markov
    assert tq.diagnose_sufficiency(S[:, [1, 0, 2]])["max_abs"] > 0.1  # reordered: not Markov
    # train_book still *runs* on the reordering (it is a valid policy to score); the caveat is
    # only about exactness of its thresholds, which the MC ranking does not depend on. With the
    # reorder (1, 0, 2) the new free top is original column 1, so columns 0 and 2 need finite
    # costs (col 1's slot is the unused free one).
    cost = np.array([0.05, np.nan, 0.1])
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")  # a non-Markov order may trip monotonicity projection
        v_reordered, _ = tq.train_book(g, (1, 0, 2), cost, t=1.0)
    assert v_reordered.shape == (3,)
    assert v_reordered[-1] == pytest.approx(1.0, abs=1e-9)         # v_0^* = t regardless


def test_nan_cost_raises_clear_error():
    # A NaN in the cost vector (e.g. a misaligned cost_per_prophecy) must raise a clear error,
    # not propagate into the value functions and crash opaquely in the isotonic projection.
    pairs = tq.pairs_from_joint(gaussian_example(), (0, 1, 2))
    with pytest.raises(ValueError, match="cost vector must be finite"):
        tq.train_tarquin(pairs, np.array([np.nan, 0.1]), t=1.0)


def test_underresolved_grid_warns():
    # A near-deterministic conditional (corr ~1) has a tiny conditional std; the grid is sized
    # to the marginal spread, so it under-resolves that component and train_tarquin must warn.
    # This is distinct from the edge under-coverage that mass renormalization already handles.
    g = GaussianMixture(n_components=1, covariance_type="full")
    g.weights_ = np.array([1.0])
    g.means_ = np.array([[0.0, 0.0]])                              # pair (V_1, V_0)
    cov = np.array([[[1.0, 0.99999], [0.99999, 1.0]]])            # corr ~1 -> tiny cond. std
    g.covariances_ = cov
    g.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))
    with pytest.warns(UserWarning, match="under-resolves"):
        tq.train_tarquin([g], np.array([0.1]), t=0.0, monotone="off")
    # A well-resolved Gaussian must NOT trip it (guards against a spurious warning).
    import warnings as _w

    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        tq.train_tarquin(tq.pairs_from_joint(gaussian_example(), (0, 1, 2)),
                         np.array([0.05, 0.1]), t=1.0)
    assert not any("under-resolves" in str(x.message) for x in rec)


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
