"""Tarquin Algorithm: optimal information acquisition under sequential sufficiency.

Implements Algorithm 1 (training) and Algorithm 2 (inference) from README.md.

The value-of-proceeding recursion is a Snell envelope (the backward induction of
an optimal-stopping / American-option problem):

    p_n^T(v_n) = E[ (p_{n-1}^T(V_{n-1}))^+ | V_n = v_n ] - c_{n-1},   p_0^T(v_0) = v_0 - t,

i.e. the value of proceeding is the conditional expectation of the *positive part*
of the next value, less cost. p_1^T is exactly the Bachelier (normal) call price.

Two structural facts drive the implementation:
  * By sufficiency (the Markov chain V_N -> ... -> V_0) the recursion only ever uses
    the N adjacent-pair conditionals f_{n-1|n}; the full joint is never needed. We
    therefore model each adjacent pair (V_n, V_{n-1}) with its own small 2-D Gaussian
    mixture (analytic Gaussian conditionals) -- see `fit_pairwise_gmms`.
  * p_n^T is carried as a *tabulated* function on a grid and built bottom-up, each
    level integrating the previous level against the conditional. This is linear in N,
    versus the exponential cost of evaluating the nested recursion pointwise.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from sklearn.isotonic import isotonic_regression
from sklearn.mixture import GaussianMixture

__version__ = "0.1.0"

__all__ = [
    "train_tarquin",
    "infer_tarquin",
    "fit_pairwise_gmms",
    "fit_joint_gmm",
    "pairs_from_joint",
    "marginalize_gmm",
    "train_book",
    "enumerate_abridgements",
    "evaluate_policy_mc",
    "make_demo_data",
]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _gauss_pdf(x, loc, scale):
    """Normal pdf via numpy (much faster than scipy.stats.norm.pdf on large arrays)."""
    z = (x - loc) / scale
    return _INV_SQRT_2PI / scale * np.exp(-0.5 * z * z)


def _trapz_weights(n: int) -> np.ndarray:
    """Trapezoidal-rule weights on a uniform grid of `n` points (endpoints halved)."""
    w = np.ones(n)
    if n >= 2:
        w[0] = w[-1] = 0.5
    return w


def _mono_tol(p: np.ndarray) -> float:
    """Tolerance below which a downward step in p counts as numerical noise, not a
    genuine monotonicity violation. Scaled to the value function's range."""
    return 1e-7 * max(1.0, float(np.ptp(p)))


def marginalize_gmm(gmm: GaussianMixture, dims) -> GaussianMixture:
    """Marginalize a joint GMM over selected dimensions."""
    dims = np.asarray(dims)
    out = GaussianMixture(n_components=gmm.n_components, covariance_type="full")
    out.weights_ = gmm.weights_.copy()
    out.means_ = gmm.means_[:, dims]
    covs = gmm.covariances_[:, dims[:, None], dims]
    out.covariances_ = covs
    out.n_features_in_ = len(dims)
    # Compute precisions_cholesky_ so the marginal GMM is fully usable (e.g. .sample,
    # .score), matching sklearn's "full"-covariance convention precision = U @ U.T with
    # U = inv(L).T and cov = L @ L.T. The recursion here reads means_/covariances_/
    # weights_ directly, but a marginal that silently breaks scoring is a footgun.
    chols = np.linalg.cholesky(covs)  # lower L per component, cov = L @ L.T
    out.precisions_cholesky_ = np.linalg.inv(chols).transpose(0, 2, 1)
    return out


def _conditional_params(pair: GaussianMixture, v_grid: np.ndarray):
    """Per-component conditional f(V_{n-1} | V_n = v) for V_n on a grid.

    `pair` is a 2-D GMM over (V_n, V_{n-1}) (column 0 = V_n, the conditioning
    variable). Returns, for the conditional mixture of V_{n-1}:
      weights : (G, K)  -- mixture weights at each grid point (rows sum to 1)
      means   : (G, K)  -- conditional component means (affine in v)
      sigmas  : (K,)    -- conditional component std devs (constant in v)
    """
    mu_c = pair.means_[:, 0]
    mu_o = pair.means_[:, 1]
    S_cc = pair.covariances_[:, 0, 0]
    S_oo = pair.covariances_[:, 1, 1]
    S_oc = pair.covariances_[:, 0, 1]

    sigmas = np.sqrt(S_oo - S_oc**2 / S_cc)  # conditional std, constant in v
    means = mu_o[None, :] + (S_oc / S_cc)[None, :] * (v_grid[:, None] - mu_c[None, :])

    comp_pdf = _gauss_pdf(v_grid[:, None], mu_c[None, :], np.sqrt(S_cc)[None, :])
    weights = pair.weights_[None, :] * comp_pdf
    # Guard the normalization: at grid points in the far tail of every component the
    # densities can all underflow to 0. Leave those rows at 0 (no conditional mass ->
    # the integral contributes nothing, so p_n there is just -c) rather than 0/0 -> NaN.
    wsum = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, wsum, out=np.zeros_like(weights), where=wsum > 0)
    return weights, means, sigmas


def _col_span(gmm: GaussianMixture, col: int, halfwidth: float) -> tuple[float, float]:
    """[lo, hi] covering every component's bulk for one column's marginal."""
    mu = gmm.means_[:, col]
    sigma = np.sqrt(gmm.covariances_[:, col, col])
    return float((mu - halfwidth * sigma).min()), float((mu + halfwidth * sigma).max())


def _build_grids(pairs: list[GaussianMixture], halfwidth: float, size: int) -> list[np.ndarray]:
    """One grid per level 0..N over V_0..V_N.

    Each V_n appears in up to two adjacent pairs -- as the conditioning column
    (col 0 of pairs[n-1]) and as the conditioned column (col 1 of pairs[n]). When
    the pairs are fit independently (`fit_pairwise_gmms`) those two views of V_n's
    marginal can differ; the level-n grid spans the union of both so the conditional
    integrated at step n+1 is not silently truncated by a grid built from the other
    fit. For pairs extracted from a single joint (`pairs_from_joint`) the two views
    agree and the union is a no-op.
    """
    N = len(pairs)
    grids = []
    for m in range(N + 1):
        spans = []
        if m <= N - 1:
            spans.append(_col_span(pairs[m], 1, halfwidth))      # V_m as conditioned col
        if m >= 1:
            spans.append(_col_span(pairs[m - 1], 0, halfwidth))  # V_m as conditioning col
        lo = min(s[0] for s in spans)
        hi = max(s[1] for s in spans)
        grids.append(np.linspace(lo, hi, size))
    return grids


def _enforce_monotone(p: np.ndarray, level: int, mode: str, tol: float) -> np.ndarray:
    """Detect / handle a non-nondecreasing value function (Prop. 3 should guarantee it).

    `p_n^T` is nondecreasing only under stochastic monotonicity (FOSD) of the
    conditionals. A fitted GMM does not enforce FOSD, so a violation signals that
    the modeled conditionals break the assumption -- in which case the endorsement
    set need not be an interval and the single-threshold rule is unreliable.
    `mode` is one of "check" (warn), "raise", "project" (isotonic-project onto the
    monotone cone and warn), or "off".
    """
    if mode == "off":
        return p
    diffs = np.diff(p)
    worst = float(diffs.min()) if diffs.size else 0.0
    if worst >= -tol:
        return p  # nondecreasing up to numerical noise
    msg = (
        f"p_{level}^T is not nondecreasing (min step {worst:.3g} < -{tol:.1g}); the "
        "stochastic-monotonicity (FOSD) assumption (Prop. 3) appears violated by the "
        "fitted conditionals, so the single-threshold rule may be invalid."
    )
    if mode == "raise":
        raise ValueError(msg)
    if mode == "project":
        warnings.warn(msg + " Projecting onto the monotone cone (isotonic).", stacklevel=3)
        return isotonic_regression(p)
    warnings.warn(msg, stacklevel=3)  # mode == "check"
    return p


def _warn_if_near_edge(v: float, grid: np.ndarray, level: int) -> None:
    """Warn when a finite threshold lands within 0.1% of a grid edge (grid too narrow)."""
    if not np.isfinite(v):
        return
    span = grid[-1] - grid[0]
    if v <= grid[0] + 1e-3 * span or v >= grid[-1] - 1e-3 * span:
        warnings.warn(
            f"v_{level}^* = {v:.4g} sits within 0.1% of a grid edge "
            f"[{grid[0]:.4g}, {grid[-1]:.4g}]; widen `halfwidth` or raise `grid_size` "
            "for a reliable threshold.",
            stacklevel=3,
        )


def _threshold_from_grid(grid: np.ndarray, p: np.ndarray) -> float:
    """Smallest v with p(v) >= 0, by linear interpolation of the sign change.

    `p` is assumed nondecreasing (Prop. 3). Returns the grid's lower edge if p is
    already nonnegative there, or +inf if the endorsement set S_n is empty.
    """
    if p[0] >= 0:
        return float(grid[0])
    if p[-1] < 0:
        return np.inf
    i = int(np.argmax(p >= 0))  # first index with p >= 0; i >= 1 given p[0] < 0
    x0, x1, y0, y1 = grid[i - 1], grid[i], p[i - 1], p[i]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def train_tarquin(
    pairs: list[GaussianMixture],
    c: np.ndarray,
    t: float,
    *,
    halfwidth: float = 10.0,
    grid_size: int = 3000,
    monotone: Literal["project", "check", "raise", "off"] = "project",
) -> tuple[np.ndarray, dict]:
    """Algorithm 1 (training) by backward value-function iteration on a grid.

    Parameters
    ----------
    pairs : list of N 2-D GaussianMixture
        pairs[n-1] is the joint over the adjacent pair (V_n, V_{n-1}) with
        column 0 = V_n (the conditioning variable) and column 1 = V_{n-1}.
        Ordered bottom-up: pairs[0] = (V_1, V_0), ..., pairs[N-1] = (V_N, V_{N-1}).
        Build these with `fit_pairwise_gmms` (from data) or `pairs_from_joint`.
    c : array of length N
        Cost vector in README order (c_{N-1}, ..., c_0); c_{n-1} is the cost of
        acquiring V_{n-1} at step n. Must be nonnegative.
    t : float
        Tightness / overhead parameter.
    halfwidth, grid_size : grid covers each marginal's mean +- halfwidth*std with
        `grid_size` points. Widen/refine if thresholds sit near a grid edge (a
        warning is emitted when they do).
    monotone : how to handle a value function p_n^T that fails to be nondecreasing.
        Prop. 3 guarantees monotonicity only under stochastic monotonicity (FOSD),
        which a fitted GMM does not enforce; even on FOSD-true data, finite-sample
        GMM wiggle breaks monotonicity of the estimated p_n^T, so a violation is
        common in practice and means the bare single-threshold rule may be invalid.
        "project" (default) isotonic-projects p_n^T onto the monotone cone -- the
        right estimator since Prop. 3 says the true p_n^T is monotone -- and warns
        when it acts; "check" warns only; "raise" errors; "off" disables the check.
        All four agree on the exact single-component Gaussian case (already monotone,
        so projection is a no-op there).

    Returns
    -------
    v_star : np.ndarray, shape (N+1,)
        Thresholds in README order (v_N^*, ..., v_0^*). v_0^* = t.
    tab : dict with keys "grids" (list of N+1 grids, level 0..N over V_0..V_N) and
        "p" (list of tabulated p_n^T values on those grids). Useful for plotting
        or reusing the value functions; index 0 is level 0 (V_0).
    """
    N = len(pairs)
    c = np.asarray(c, dtype=float)
    if c.shape != (N,):
        raise ValueError(f"expected c of length {N}, got shape {c.shape}")
    if np.any(c < 0):
        raise ValueError(f"cost vector must be nonnegative (README: c in R^N_>=0); got {c}")

    grids = _build_grids(pairs, halfwidth, grid_size)

    # p_0^T(v_0) = v_0 - t (monotone by construction; the call is a no-op here).
    p0 = _enforce_monotone(grids[0] - t, 0, monotone, _mono_tol(grids[0] - t))
    p = [p0]
    v_star_asc = [_threshold_from_grid(grids[0], p0)]  # = t

    for n in range(1, N + 1):
        pair = pairs[n - 1]
        g_n, g_prev = grids[n], grids[n - 1]
        weights, means, sigmas = _conditional_params(pair, g_n)  # (G,K),(G,K),(K,)

        # Integrate (p_{n-1}^T)^+ against each conditional component by the trapezoidal
        # rule against the Gaussian measure on the previous level's grid (endpoints
        # half-weighted). Robust to the kink of the positive part, vectorized over the
        # current grid.
        p_prev_pos = np.maximum(p[n - 1], 0.0) * _trapz_weights(g_prev.size)
        dx = g_prev[1] - g_prev[0]
        integral = np.empty_like(means)  # (G, K)
        for k in range(pair.n_components):
            w_mat = _gauss_pdf(g_prev[None, :], means[:, k][:, None], sigmas[k])
            integral[:, k] = (w_mat @ p_prev_pos) * dx

        c_prev = float(c[N - n])  # README-indexed c_{n-1}
        p_n = (weights * integral).sum(axis=1) - c_prev
        p_n = _enforce_monotone(p_n, n, monotone, _mono_tol(p_n))
        p.append(p_n)
        v_n = _threshold_from_grid(g_n, p_n)
        _warn_if_near_edge(v_n, g_n, n)
        v_star_asc.append(v_n)

    return np.array(v_star_asc[::-1]), {"grids": grids, "p": p}


def infer_tarquin(v_star, v) -> np.ndarray:
    """Algorithm 2 (inference).

    Walks from the highest-index prophecy down. Sets r_n = 1 while v_n > v_n^*;
    on the first failure, sets that r_n = 0 and leaves all lower-indexed (i.e.,
    later-step) decisions at 0, since the buyer has exited.

    Parameters
    ----------
    v_star : array-like, shape (N+1,)
        Thresholds in README order (v_N^*, ..., v_0^*). Typically from train_tarquin.
    v : array-like, shape (N+1,)
        Observed value draw in the same order (v_N, ..., v_0).

    Returns
    -------
    r : np.ndarray of ints, shape (N+1,)
        Decision vector in README order (r_N, ..., r_0). r_0 uses v_0^* = t.
    """
    v_star = np.asarray(v_star, dtype=float)
    v = np.asarray(v, dtype=float)
    if v_star.shape != v.shape:
        raise ValueError(f"shape mismatch: v_star {v_star.shape} vs v {v.shape}")
    r = np.zeros_like(v_star, dtype=int)
    for i in range(len(v)):
        if v[i] > v_star[i]:
            r[i] = 1
        else:
            break
    return r


# --- Fitting, book construction, and policy evaluation helpers -------------


def fit_pairwise_gmms(
    data,
    n_components: int = 5,
    random_state: int = 0,
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
    **kwargs,
) -> list[GaussianMixture]:
    """Fit one 2-D GMM per adjacent pair (V_n, V_{n-1}) -- the model the recursion uses.

    `data` columns must be in README order (V_N, ..., V_0) with V_0 last. By
    sufficiency only adjacent-pair conditionals are needed, so each pair is fit
    on its own two columns (lower-dimensional and more data-efficient than fitting
    the full joint and marginalizing). Returns the pairs bottom-up, ready for
    `train_tarquin`: pairs[0] = (V_1, V_0), ..., pairs[N-1] = (V_N, V_{N-1}).
    """
    data = np.asarray(data)
    N = data.shape[1] - 1
    pairs = []
    for n in range(1, N + 1):
        cols = [N - n, N - n + 1]  # (V_n, V_{n-1}) in README column order
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            covariance_type=covariance_type,
            **kwargs,
        )
        gmm.fit(data[:, cols])
        pairs.append(gmm)
    return pairs


def fit_joint_gmm(
    data,
    n_components: int = 10,
    random_state: int = 0,
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
    **kwargs,
) -> GaussianMixture:
    """Fit a single GMM to the full joint `data` (array or DataFrame).

    Columns must be in README order (V_N, ..., V_0) with V_0 last. Useful when a
    book's abridgements/rearrangements need conditionals for arbitrary adjacent
    pairs (see `pairs_from_joint`); for a fixed ordering prefer `fit_pairwise_gmms`.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        covariance_type=covariance_type,
        **kwargs,
    )
    gmm.fit(np.asarray(data))
    return gmm


def pairs_from_joint(gmm_full: GaussianMixture, col_order) -> list[GaussianMixture]:
    """Extract the adjacent-pair GMMs for a book from a full joint GMM.

    col_order is in README order (V_{M-1}, ..., V_0); consecutive entries are
    adjacent pairs (V_n, V_{n-1}). Returns them bottom-up for `train_tarquin`.
    """
    col_order = list(col_order)
    top_down = [marginalize_gmm(gmm_full, [col_order[i], col_order[i + 1]])
                for i in range(len(col_order) - 1)]
    return top_down[::-1]


def train_book(
    gmm_full: GaussianMixture,
    col_order,
    cost_per_prophecy,
    t: float,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """Train on any subset and/or permutation of columns of `gmm_full`.

    Covers both abridgements (subset of columns) and rearrangements (reordered
    columns) with the same entry point.

    Parameters
    ----------
    col_order : sequence of column indices into gmm_full, in README order
        (V_{M-1}, ..., V_0) -- last entry is the payoff prophecy.
    cost_per_prophecy : array of length D_full; entry j is the cost to acquire
        column j. The top prophecy (col_order[0]) is seen for free; entries for
        col_order[1:] are used.
    """
    col_order = list(col_order)
    cost_per_prophecy = np.asarray(cost_per_prophecy, dtype=float)
    pairs = pairs_from_joint(gmm_full, col_order)
    c_step = np.array([cost_per_prophecy[col_order[i]] for i in range(1, len(col_order))])
    return train_tarquin(pairs, c_step, t, **kwargs)


def enumerate_abridgements(col_order):
    """Yield all (nonempty) abridgements of a book, as col_order tuples.

    Per the README, a book with index set delta of size >= 2 has
    2^(|delta|-1) - 2 abridgements: proper subsets of delta that include V_0
    (the payoff prophecy at col_order[-1]) AND have size >= 2. The size-1
    singleton {V_0} is excluded because it corresponds to seeing V_0 for
    free -- an unphysical upper bound, not a feasible policy. Original
    ordering is preserved within each yielded tuple.
    """
    from itertools import combinations

    head, tail = tuple(col_order[:-1]), col_order[-1]
    M = len(col_order)
    for size in range(1, M - 1):
        for combo in combinations(head, size):
            yield (*combo, tail)


def make_demo_data(n: int = 2056, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic dataset for the demo (replaces the old test.csv).

    A Markov chain V_2 -> V_1 -> V_0 -- so it satisfies the sufficiency
    assumption by construction -- built from a latent standard-normal chain
    pushed through strictly increasing marginal transforms. Monotone maps preserve
    both the Markov property and stochastic monotonicity (FOSD), so the data also
    satisfies assumption 2; the worked recursion is therefore well posed on it. The
    result has positive, right-skewed V_2 and V_1 on an O(100) scale (V_1 floored
    near 68) and a wider, right-skewed V_0 that can go negative, with adjacent
    correlations ~0.34 / ~0.55. Columns are in README order (V_2, V_1, V_0).
    """
    rng = np.random.default_rng(seed)
    z2 = rng.standard_normal(n)
    z1 = 0.42 * z2 + np.sqrt(1 - 0.42**2) * rng.standard_normal(n)
    z0 = 0.66 * z1 + np.sqrt(1 - 0.66**2) * rng.standard_normal(n)
    v2 = np.exp(4.38 + 0.40 * z2)                 # lognormal: mean ~85, skew ~1.1
    v1 = 68.0 + np.exp(3.30 + 0.90 * z1)          # floored, heavy right tail
    v0 = 79.0 + 60.0 * z0 + 8.0 * np.expm1(0.6 * z0)  # strictly increasing, right-skewed, can be < 0
    return np.column_stack([v2, v1, v0])


def evaluate_policy_mc(
    samples,
    col_order,
    v_star,
    cost_per_prophecy,
    t: float,
) -> np.ndarray:
    """Per-sample payoffs under the Tarquinian policy for a given book.

    At step i (top = 0), a row is "alive" iff every threshold check up to
    and including i has passed. Each surviving row pays the cost of
    acquiring col_order[i+1]; the final surviving rows receive v_0 - t.
    """
    samples = np.asarray(samples)
    cost_per_prophecy = np.asarray(cost_per_prophecy, dtype=float)
    col_order = list(col_order)
    M = len(col_order)

    pi = np.zeros(samples.shape[0])
    alive = np.ones(samples.shape[0], dtype=bool)
    for i in range(M):
        col = col_order[i]
        alive = alive & (samples[:, col] > v_star[i])
        if i < M - 1:
            pi -= alive * cost_per_prophecy[col_order[i + 1]]
        else:
            pi += alive * (samples[:, col] - t)
    return pi


if __name__ == "__main__":
    # Reproduce the README's Gaussian example: v_1^* ~= 0.1204, v_2^* ~= 0.2886.
    mu = np.array([1.0, 0.5, -0.2])
    cov = np.array([
        [1.0, 0.3, 0.15],
        [0.3, 1.0, 0.5],
        [0.15, 0.5, 2.0],
    ])
    gmm = GaussianMixture(n_components=1, covariance_type="full")
    gmm.weights_ = np.array([1.0])
    gmm.means_ = mu[None, :]
    gmm.covariances_ = cov[None, :, :]
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    c = np.array([0.05, 0.1])  # (c_1, c_0)
    t = 1.0
    pairs = pairs_from_joint(gmm, (0, 1, 2))  # adjacent pairs (V_1,V_0), (V_2,V_1)
    v_star, _ = train_tarquin(pairs, c, t)
    print("v* (v_2*, v_1*, v_0*):", v_star)
    print("expected approx      : (0.2886, 0.1204, 1.0)")

    # Per-prophecy cost: col 0 (V_2) is the top (free). col 1 = V_1 (c_1). col 2 = V_0 (c_0).
    cost_per_col = np.array([np.nan, c[0], c[1]])

    # Algorithm 2: MC sample payoffs under the learned policy.
    rng = np.random.default_rng(0)
    samples = rng.multivariate_normal(mu, cov, size=2_000_000)  # cols: (v_2, v_1, v_0)
    pi = evaluate_policy_mc(samples, (0, 1, 2), v_star, cost_per_col, t)
    print(f"E[pi_N] under Tarquinian policy ~ {pi.mean():.4f}")

    # Spot-check infer_tarquin on one draw.
    draw = samples[0]
    print(f"draw = {draw},  r = {infer_tarquin(v_star, draw)}")

    # --- Abridgement ranking on the same joint ---
    print("\nBooks ranked by E[pi]:")
    books = [(0, 1, 2), *enumerate_abridgements((0, 1, 2))]
    results = []
    for book in books:
        v_ab, _ = train_book(gmm, book, cost_per_col, t)
        pi_ab = evaluate_policy_mc(samples, book, v_ab, cost_per_col, t)
        results.append((pi_ab.mean(), book, v_ab))
    for mean_pi, book, v_ab in sorted(results, reverse=True):
        print(f"  cols {book}  v*={np.round(v_ab, 3).tolist()}  E[pi]={mean_pi:+.4f}")

    # --- fit_pairwise_gmms smoke test on deterministic synthetic data ---
    arr = make_demo_data()
    data_pairs = fit_pairwise_gmms(arr, n_components=10)
    # Costs/overhead are scaled to this data (V_0 ~ O(100)); with the example's
    # tiny c/t the policy would just "always proceed" (threshold at the grid floor).
    t_data = float(np.median(arr[:, 2]))
    c_data = np.array([100.0, 100.0])
    v_star_data, _ = train_tarquin(data_pairs, c_data, t_data)
    print(f"\nfit_pairwise_gmms on make_demo_data() (t={t_data:.1f}, c=100): "
          f"v* = {np.round(v_star_data, 2).tolist()}  "
          "(-inf=always proceed, +inf=never)")
