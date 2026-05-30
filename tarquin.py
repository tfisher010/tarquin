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
    "diagnose_sufficiency",
    "diagnose_fosd",
    "holdout_split",
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


def _full_covariances(gmm: GaussianMixture) -> np.ndarray:
    """Per-component full (K, D, D) covariances, whatever the fitted covariance_type.

    sklearn stores `covariances_` in four shapes (full: (K,D,D); tied: (D,D); diag:
    (K,D); spherical: (K,)). `marginalize_gmm` slices a (D,D) block per component, so we
    expand to full first -- otherwise a non-"full" joint silently breaks marginalization.
    """
    cov = np.asarray(gmm.covariances_)
    K = gmm.n_components
    D = gmm.means_.shape[1]
    ct = getattr(gmm, "covariance_type", "full")
    if ct == "full":
        return cov
    if ct == "tied":
        return np.broadcast_to(cov, (K, D, D)).copy()
    di = np.arange(D)
    out = np.zeros((K, D, D))
    if ct == "diag":
        out[:, di, di] = cov
    elif ct == "spherical":
        out[:, di, di] = cov[:, None]
    else:
        raise ValueError(f"unknown covariance_type {ct!r}")
    return out


def marginalize_gmm(gmm: GaussianMixture, dims) -> GaussianMixture:
    """Marginalize a joint GMM over selected dimensions (any covariance_type)."""
    dims = np.asarray(dims)
    out = GaussianMixture(n_components=gmm.n_components, covariance_type="full")
    out.weights_ = gmm.weights_.copy()
    out.means_ = gmm.means_[:, dims]
    covs = _full_covariances(gmm)[:, dims[:, None], dims]
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
    # Floor the conditioning variance a hair above 0 so a (near-)degenerate GMM
    # component cannot divide by zero in the regression coefficient / conditional pdf.
    # sklearn's reg_covar keeps fitted components PD, so this is a no-op in practice.
    S_cc = np.maximum(pair.covariances_[:, 0, 0], np.finfo(float).tiny)
    S_oo = pair.covariances_[:, 1, 1]
    S_oc = pair.covariances_[:, 0, 1]

    # Conditional variance, likewise floored: a collapsed component (S_oo*S_cc ~ S_oc^2)
    # then yields a sharp-but-finite density instead of a sqrt-of-negative NaN that would
    # poison the whole recursion.
    cond_var = np.maximum(S_oo - S_oc**2 / S_cc, np.finfo(float).tiny)
    sigmas = np.sqrt(cond_var)  # conditional std, constant in v
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
    n_viol = int((diffs < -tol).sum())
    rel = -worst / max(1.0, float(np.ptp(p)))  # worst dip as a fraction of the value range
    severity = (
        "consistent with finite-sample GMM wiggle (projection recovers the monotone "
        "estimator)"
        if rel < 1e-2
        else "large relative to the value range, suggesting a genuine FOSD violation in "
        "the modeled conditionals; then S_n need not be an interval and the "
        "single-threshold rule may be unreliable even after projection"
    )
    msg = (
        f"p_{level}^T is not nondecreasing (min step {worst:.3g} < -{tol:.1g}, "
        f"{n_viol} violating point(s), rel. magnitude {rel:.2g}); this is {severity}. "
        "The stochastic-monotonicity (FOSD) assumption (Prop. 3) appears violated by the "
        "fitted conditionals."
    )
    if mode == "raise":
        raise ValueError(msg)
    if mode == "project":
        warnings.warn(msg + " Projecting onto the monotone cone (isotonic).", stacklevel=3)
        return isotonic_regression(p)
    warnings.warn(msg, stacklevel=3)  # mode == "check"
    return p


def _warn_if_near_edge(v: float, grid: np.ndarray, level: int) -> None:
    """Warn when a threshold is unresolved within the grid: a finite value within 0.1%
    of an edge, or a low-saturated -inf (p_n^T >= 0 already at the grid floor)."""
    if v == -np.inf:
        warnings.warn(
            f"v_{level}^* saturated to -inf: p_{level}^T >= 0 at the grid floor "
            f"{grid[0]:.4g}, so the endorsement set covers the whole modeled support. "
            "If you expected a finite threshold, widen `halfwidth` / raise `grid_size`.",
            stacklevel=3,
        )
        return
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

    `p` is assumed nondecreasing (Prop. 3). The two saturated cases are returned as
    the honest infinities rather than a finite grid edge:
      * p < 0 across the grid  -> +inf  (endorsement set S_n empty: never proceed).
      * p >= 0 at the grid floor -> -inf (S_n covers the modeled support: always
        proceed). Returning grid[0] here would mislabel a draw below the floor as
        "exit"; -inf is symmetric with the +inf case and is the truthful threshold for
        a saturated endorsement set. The caller warns so a too-narrow grid is visible.
    """
    if p[-1] < 0:
        return np.inf
    if p[0] >= 0:
        return -np.inf
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
        tw = _trapz_weights(g_prev.size)
        p_prev_pos = np.maximum(p[n - 1], 0.0) * tw
        dx = g_prev[1] - g_prev[0]
        integral = np.empty_like(means)  # (G, K)
        for k in range(pair.n_components):
            w_mat = _gauss_pdf(g_prev[None, :], means[:, k][:, None], sigmas[k])
            num = (w_mat @ p_prev_pos) * dx  # int (p_{n-1}^T)^+ f_k over the grid
            # Renormalize by the conditional mass actually captured on the grid, so a
            # component whose conditional mean has been pushed toward a grid edge (large
            # |v_n|) is not silently under-integrated. That truncation is the main source
            # of finite-sample non-monotonicity in p_n^T. With the default +-10*sigma grid
            # the captured mass is ~1 through the bulk, so this is a no-op there and on the
            # exact single-component Gaussian example (mass = 1 to ~23 digits).
            mass = (w_mat @ tw) * dx  # int f_k over the grid (<= 1)
            integral[:, k] = np.divide(num, mass, out=np.zeros_like(num), where=mass > 0)

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


def _fit_one_gmm(X, n_components, random_state, covariance_type, **kwargs):
    """Fit a GMM to `X`, selecting the component count by BIC when `n_components` is a
    sequence (range / list / array). A scalar fits that single count. Extra sklearn
    kwargs (`n_init`, `reg_covar`, ...) flow straight through to `GaussianMixture`."""
    candidates = (
        [int(n_components)] if np.isscalar(n_components)
        else [int(k) for k in n_components]
    )
    if not candidates:
        raise ValueError("n_components must be a positive int or a nonempty sequence")
    best_gmm, best_bic = None, np.inf
    for k in candidates:
        g = GaussianMixture(
            n_components=k,
            random_state=random_state,
            covariance_type=covariance_type,
            **kwargs,
        ).fit(X)
        bic = g.bic(X)
        if bic < best_bic:
            best_gmm, best_bic = g, bic
    return best_gmm


def fit_pairwise_gmms(
    data,
    n_components=5,
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

    `n_components` may be an int or a sequence of candidate counts; a sequence selects
    each pair's count independently by BIC. Pass extra sklearn `GaussianMixture` kwargs
    (notably `n_init` for multiple EM restarts, `reg_covar` to regularize covariances)
    via **kwargs to harden the fit against poor local optima / collapsed components.
    """
    data = np.asarray(data)
    N = data.shape[1] - 1
    pairs = []
    for n in range(1, N + 1):
        cols = [N - n, N - n + 1]  # (V_n, V_{n-1}) in README column order
        pairs.append(
            _fit_one_gmm(data[:, cols], n_components, random_state, covariance_type, **kwargs)
        )
    return pairs


def fit_joint_gmm(
    data,
    n_components=10,
    random_state: int = 0,
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
    **kwargs,
) -> GaussianMixture:
    """Fit a single GMM to the full joint `data` (array or DataFrame).

    Columns must be in README order (V_N, ..., V_0) with V_0 last. Useful when a
    book's abridgements/rearrangements need conditionals for arbitrary adjacent
    pairs (see `pairs_from_joint`); for a fixed ordering prefer `fit_pairwise_gmms`.

    `n_components` may be an int or a sequence of candidates (selected by BIC). Extra
    sklearn kwargs (`n_init`, `reg_covar`, ...) pass through; for downstream use with
    `pairs_from_joint`/`marginalize_gmm` any `covariance_type` is supported.
    """
    return _fit_one_gmm(np.asarray(data), n_components, random_state, covariance_type, **kwargs)


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

    Note: the count is exponential in book size (2^(|delta|-1) - 2), and ranking
    abridgements trains each one, so enumerating a large book is expensive. A warning
    is emitted past ~20 prophecies.
    """
    from itertools import combinations

    head, tail = tuple(col_order[:-1]), col_order[-1]
    M = len(col_order)
    if M - 1 > 20:
        warnings.warn(
            f"enumerating 2^{M - 1} - 2 abridgements of a {M}-prophecy book; this count "
            "(and the cost of ranking them) grows exponentially in book size.",
            stacklevel=2,
        )
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


# --- Assumption diagnostics and train/eval splitting -----------------------


def diagnose_sufficiency(data) -> dict:
    """Necessary-condition check for Markov sufficiency (Assumption 1).

    Columns must be in README order (V_N, ..., V_0). For each interior triple of
    consecutive columns (V_{n+1}, V_n, V_{n-1}) returns the partial correlation of the
    two outer prophecies given the middle one. Under the chain V_N -> ... -> V_0 we have
    V_{n+1} ⊥ V_{n-1} | V_n, so each partial correlation should be ~0.

    This is a *linear, necessary* check: values near 0 are consistent with sufficiency
    but do not prove it (nonlinear residual dependence is invisible to a correlation);
    a large value is positive evidence *against* sufficiency. Pair it with
    `diagnose_fosd` and the training-time monotonicity warning for a fuller picture.

    Returns dict with "partial_corr" (one entry per interior V_n, top-down) and "max_abs".
    """
    data = np.asarray(data, dtype=float)
    D = data.shape[1]
    pcs = []
    for j in range(1, D - 1):
        x, z, y = data[:, j - 1], data[:, j], data[:, j + 1]  # outer, middle, outer
        r_xy = np.corrcoef(x, y)[0, 1]
        r_xz = np.corrcoef(x, z)[0, 1]
        r_yz = np.corrcoef(y, z)[0, 1]
        denom = np.sqrt(max((1 - r_xz**2) * (1 - r_yz**2), np.finfo(float).tiny))
        pcs.append((r_xy - r_xz * r_yz) / denom)
    pcs = np.array(pcs)
    return {"partial_corr": pcs, "max_abs": float(np.abs(pcs).max()) if pcs.size else 0.0}


def diagnose_fosd(data, *, n_bins: int = 10, n_thresholds: int = 25) -> dict:
    """Necessary-condition check for stochastic monotonicity / FOSD (Assumption 2).

    Columns in README order (V_N, ..., V_0). For each adjacent pair (V_n, V_{n-1}) bins
    V_n into `n_bins` quantile bins and compares the empirical conditional CDF of V_{n-1}
    across bins at `n_thresholds` quantile thresholds. FOSD requires a higher V_n to give
    a stochastically larger V_{n-1}, i.e. the conditional CDF is weakly *decreasing* as
    the bin index rises, at every threshold. The reported violation per pair is the
    largest upward CDF step between adjacent bins (0 = no violation, in [0, 1]).

    A necessary, data-level check (the fit may smooth or worsen it). Combine with the
    training-time monotonicity warning, which acts on the *fitted* conditionals.

    Returns dict with "violation" (one per adjacent pair, README order (V_N,V_{N-1})..) and
    "max".
    """
    data = np.asarray(data, dtype=float)
    D = data.shape[1]
    viols = []
    for c in range(D - 1):
        cond, outc = data[:, c], data[:, c + 1]  # V_n, V_{n-1}
        edges = np.quantile(cond, np.linspace(0.0, 1.0, n_bins + 1))
        edges[-1] = np.inf  # right edge open so the max lands in the last bin
        idx = np.clip(np.searchsorted(edges, cond, side="right") - 1, 0, n_bins - 1)
        ts = np.quantile(outc, np.linspace(0.02, 0.98, n_thresholds))
        cdfs = np.full((n_bins, n_thresholds), np.nan)
        for b in range(n_bins):
            sel = outc[idx == b]
            if sel.size:
                cdfs[b] = (sel[:, None] <= ts[None, :]).mean(axis=0)
        steps = np.diff(cdfs, axis=0)  # >0 means CDF rose with V_n: an FOSD violation
        worst = float(np.nanmax(np.maximum(steps, 0.0))) if np.isfinite(steps).any() else 0.0
        viols.append(worst)
    viols = np.array(viols)
    return {"violation": viols, "max": float(viols.max()) if viols.size else 0.0}


def holdout_split(data, test_frac: float = 0.3, seed: int = 0):
    """Shuffle-split rows into (train, test).

    Fit the conditionals on `train` (via `fit_pairwise_gmms` / `fit_joint_gmm`), then
    score the learned policy on `test` with `evaluate_policy_mc`. Scoring on the same
    sample used to fit is optimistically biased; a holdout removes that bias.
    """
    data = np.asarray(data)
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")
    n = data.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(round(n * (1.0 - test_frac)))
    return data[perm[:cut]], data[perm[cut:]]


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
