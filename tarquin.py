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
from collections.abc import Sequence
from typing import Literal, Union

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
    "diagnose_saturation",
    "simulate_incumbent_truncation",
    "holdout_split",
    "bootstrap_thresholds",
    "make_demo_data",
]

# `n_components` is either a fixed count or a set of candidates selected per fit by BIC.
NComponents = Union[int, Sequence[int]]

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


def _as_full_covariance(gmm: GaussianMixture) -> GaussianMixture:
    """Return an equivalent GMM with covariance_type='full'.

    The recursion (`_conditional_params`, `_col_span`) indexes `covariances_` as
    (K, D, D), which only holds for a "full" fit. A pair fit with diag/tied/spherical
    covariances would otherwise crash there; densify once at the source so every
    downstream consumer sees the uniform (K, D, D) layout. A "full" GMM is returned
    unchanged.
    """
    if getattr(gmm, "covariance_type", "full") == "full":
        return gmm
    covs = _full_covariances(gmm)
    out = GaussianMixture(n_components=gmm.n_components, covariance_type="full")
    out.weights_ = gmm.weights_.copy()
    out.means_ = gmm.means_.copy()
    out.covariances_ = covs
    out.n_features_in_ = gmm.means_.shape[1]
    chols = np.linalg.cholesky(covs)  # cov = L @ L.T; precision = U @ U.T, U = inv(L).T
    out.precisions_cholesky_ = np.linalg.inv(chols).transpose(0, 2, 1)
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
        if not hi > lo:
            raise ValueError(
                f"degenerate grid at level {m}: span [{lo:.4g}, {hi:.4g}] has zero width "
                "(a collapsed marginal -> zero conditioning variance). Check the fitted "
                "GMM / `reg_covar`; the trapezoidal step dx would be 0."
            )
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
    # NB: the *magnitude* of this dip does NOT reliably indicate a true FOSD violation.
    # Prop. 3 guarantees monotonicity only under stochastic monotonicity (FOSD), but a
    # fitted GMM does not enforce FOSD, so finite-sample wiggle can break monotonicity of
    # the estimated p_n^T even on FOSD-true data -- and empirically that wiggle is often
    # *larger* (relative to the value range) than the dip a genuinely non-FOSD conditional
    # produces, so a magnitude cutoff here would misclassify both ways. We therefore report
    # the dip factually and defer the actual data-level FOSD question to `diagnose_fosd`.
    msg = (
        f"p_{level}^T is not nondecreasing (min step {worst:.3g} < -{tol:.1g}, "
        f"{n_viol} violating point(s), {rel:.2g} of the value range). Prop. 3 says the "
        "true p_n^T is monotone under stochastic monotonicity (FOSD); a fitted GMM does "
        "not enforce FOSD, so finite-sample wiggle alone can cause this. This dip's size "
        "is not itself evidence of a true FOSD violation -- run `diagnose_fosd` on the "
        "data for that. If FOSD genuinely fails, S_n need not be an interval and the "
        "single-threshold rule is unreliable even after projection."
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
    of an edge, or a saturated endorsement set (-inf: p_n^T >= 0 at the grid floor;
    +inf: p_n^T < 0 at the grid ceiling)."""
    if v == -np.inf:
        warnings.warn(
            f"v_{level}^* saturated to -inf: p_{level}^T >= 0 at the grid floor "
            f"{grid[0]:.4g}, so the endorsement set covers the whole modeled support. "
            "If you expected a finite threshold, widen `halfwidth` / raise `grid_size`.",
            stacklevel=3,
        )
        return
    if v == np.inf:
        warnings.warn(
            f"v_{level}^* saturated to +inf: p_{level}^T < 0 at the grid ceiling "
            f"{grid[-1]:.4g}, so the endorsement set is empty (never proceed). This is "
            "usually genuine (cost exceeds the option value), but a too-narrow grid top "
            "can also produce it; if you expected a finite threshold, widen `halfwidth` / "
            "raise `grid_size`.",
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


def _level_integral(pair: GaussianMixture, g_n: np.ndarray, g_prev: np.ndarray,
                    p_prev: np.ndarray) -> np.ndarray:
    """E[(p_prev)^+ | V_n = v] for every v in g_n: the conditional expectation of the
    positive part of the previous level's value function (the Snell-envelope integrand).

    Trapezoidal rule against each Gaussian conditional component on g_prev (endpoints
    half-weighted), renormalized by the conditional mass actually captured on the grid so a
    component whose conditional mean is pushed toward a grid edge is not under-integrated.
    Returns the integral (before subtracting cost). Shared by the point and bounds paths.
    """
    weights, means, sigmas = _conditional_params(pair, g_n)  # (G,K),(G,K),(K,)
    tw = _trapz_weights(g_prev.size)
    p_prev_pos = np.maximum(p_prev, 0.0) * tw
    dx = g_prev[1] - g_prev[0]
    integral = np.empty_like(means)  # (G, K)
    for k in range(pair.n_components):
        w_mat = _gauss_pdf(g_prev[None, :], means[:, k][:, None], sigmas[k])
        num = (w_mat @ p_prev_pos) * dx  # int (p_prev)^+ f_k over the grid
        mass = (w_mat @ tw) * dx         # int f_k over the grid (<= 1); renormalizes num
        integral[:, k] = np.divide(num, mass, out=np.zeros_like(num), where=mass > 0)
    return (weights * integral).sum(axis=1)


def _resolve_support_lo(pairs: list[GaussianMixture], N: int, support_lo, use_support: bool):
    """Per-level lower edge a_n of the identified region of V_n (the conditioning variable of
    pairs[n-1]), for n=1..N. Below a_n the conditional f_(n-1|n) was never observed (the
    sample stopped), so the recursion would be extrapolating. Sources, in order: an explicit
    `support_lo` array (README order, length N+1; the V_n slot is index N-n), else the
    `tarquin_support_lo_` attribute `fit_pairwise_gmms` records on a ragged fit, else -inf
    (fully identified). When `use_support` is False (the default point/"gmm" path) every a_n
    is -inf, so behavior is unchanged.
    """
    if not use_support:
        return [-np.inf] * N
    if support_lo is not None:
        support_lo = np.asarray(support_lo, dtype=float)
        if support_lo.shape != (N + 1,):
            raise ValueError(f"expected support_lo of length {N + 1}, got {support_lo.shape}")
        return [float(support_lo[N - n]) for n in range(1, N + 1)]
    return [float(getattr(pairs[n - 1], "tarquin_support_lo_", -np.inf)) for n in range(1, N + 1)]


def train_tarquin(
    pairs: list[GaussianMixture],
    c: np.ndarray,
    t: float,
    *,
    halfwidth: float = 10.0,
    grid_size: int = 3000,
    monotone: Literal["project", "check", "raise", "off"] = "project",
    identification: Literal["point", "bounds"] = "point",
    extrapolation: Literal["gmm", "clamp", "flag"] = "gmm",
    support_lo: np.ndarray | None = None,
):
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
        warning is emitted when they do). Cost note: the recursion is linear in N
        *levels*, but each level integrates a dense (grid_size, grid_size) conditional
        per mixture component, so runtime is O(N * K * grid_size^2) and peak memory is
        O(grid_size^2). Doubling grid_size quadruples both; raise it deliberately.
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
    identification : "point" (default) returns a single threshold vector; "bounds" returns
        a (v_lo, v_hi) interval per threshold under the FOSD prior, honest about a region
        the sample does not identify (see `extrapolation` / `support_lo`).
    extrapolation : how to treat V_n below its identified support a_n (where the conditional
        f_(n-1|n) was never observed, e.g. a sample truncated by an incumbent threshold).
        "gmm" (default) lets the fitted GMM extrapolate freely (the original behavior; uses
        no support information). "clamp" applies the FOSD-pessimistic envelope below a_n
        (continuation value floored to 0, so p_n^T = -c there): a threshold that would fall
        below the identified support is reported conservatively at a_n rather than at an
        extrapolated value. "flag" extrapolates like "gmm" but warns when the resolved
        threshold lies below a_n, i.e. depends on extrapolated mass. The honest treatment of
        a genuinely unidentified threshold is `identification="bounds"`, which brackets it.
    support_lo : optional per-level lower edge of the identified region, README order
        (length N+1; the V_n slot is index N-n, the V_0 slot is ignored). If omitted, taken
        from the `tarquin_support_lo_` attribute a ragged `fit_pairwise_gmms` records, else
        -inf (fully identified). Only consulted when identification="bounds" or
        extrapolation in {"clamp", "flag"}.

    Returns
    -------
    If identification="point": (v_star, tab).
      v_star : (N+1,) thresholds in README order (v_N^*, ..., v_0^*); v_0^* = t.
      tab : dict with "grids" (N+1 grids, level 0..N over V_0..V_N) and "p" (tabulated
        p_n^T per level; index 0 is level 0 / V_0).
    If identification="bounds": (v_lo, v_hi, tab).
      v_lo, v_hi : (N+1,) lower/upper bound per threshold (README order); v_lo == v_hi at a
        point-identified level. tab carries "p_lo"/"p_hi" envelopes and the "support_lo" used.
    """
    N = len(pairs)
    c = np.asarray(c, dtype=float)
    if c.shape != (N,):
        raise ValueError(f"expected c of length {N}, got shape {c.shape}")
    if np.any(c < 0):
        raise ValueError(f"cost vector must be nonnegative (README: c in R^N_>=0); got {c}")

    grids = _build_grids(pairs, halfwidth, grid_size)
    use_support = identification == "bounds" or extrapolation in ("clamp", "flag")
    a = _resolve_support_lo(pairs, N, support_lo, use_support)  # a[n-1] = a_n for level n

    if identification == "bounds":
        return _train_bounds(pairs, c, t, grids, a, monotone)

    # p_0^T(v_0) = v_0 - t (monotone by construction; the call is a no-op here).
    p0 = _enforce_monotone(grids[0] - t, 0, monotone, _mono_tol(grids[0] - t))
    p = [p0]
    # v_0^* = t exactly (Prop. 1), set directly rather than read off the grid: the root
    # of v_0 - t is t, but `_threshold_from_grid` would saturate to +/-inf if t fell
    # outside the modeled V_0 support, silently breaking the documented invariant.
    v_star_asc = [float(t)]

    for n in range(1, N + 1):
        pair = pairs[n - 1]
        g_n, g_prev = grids[n], grids[n - 1]
        c_prev = float(c[N - n])  # README-indexed c_{n-1}
        p_n = _level_integral(pair, g_n, g_prev, p[n - 1]) - c_prev
        a_n = a[n - 1]

        below = np.isfinite(a_n) and (g_n < a_n).any()
        if extrapolation == "clamp" and below:
            # FOSD-pessimistic envelope below the identified support: continuation value 0,
            # so p_n^T = -c there. Project only the identified region, then floor below a_n.
            mask = g_n >= a_n
            p_n[mask] = _enforce_monotone(p_n[mask], n, monotone, _mono_tol(p_n[mask]))
            p_n[~mask] = -c_prev
        else:
            p_n = _enforce_monotone(p_n, n, monotone, _mono_tol(p_n))

        p.append(p_n)
        v_n = _threshold_from_grid(g_n, p_n)
        if extrapolation == "flag" and np.isfinite(a_n) and v_n < a_n:
            warnings.warn(
                f"v_{n}^* = {v_n:.4g} lies below the identified support edge a_{n} = "
                f"{a_n:.4g}; its value depends on conditional mass the sample never observed "
                "(extrapolation). Use identification='bounds' to bracket it, or 'clamp' for a "
                "conservative read.",
                stacklevel=2,
            )
        _warn_if_near_edge(v_n, g_n, n)
        v_star_asc.append(v_n)

    return np.array(v_star_asc[::-1]), {"grids": grids, "p": p}


def _train_bounds(pairs, c, t, grids, a, monotone):
    """Partial-identification bounds: propagate an optimistic (p_hi) and a pessimistic (p_lo)
    value-function envelope up the recursion. Below each level's identified support a_n the
    FOSD prior pins p_n^T into [-c_{n-1}, p_n^T(a_n)]: the optimistic envelope clamps to the
    edge value p_n^T(a_n) (no better than the edge), the pessimistic floors to -c_{n-1}
    (continuation value 0). Returns (v_lo, v_hi, tab); v_lo (from the optimistic envelope,
    the most-proceeding) is the lowest the threshold can be, v_hi (pessimistic) the highest.
    """
    N = len(pairs)
    base = grids[0] - t          # V_0 is identified; v_0^* = t exactly
    p_lo, p_hi = [base], [base]
    v_lo_asc, v_hi_asc = [float(t)], [float(t)]

    for n in range(1, N + 1):
        pair = pairs[n - 1]
        g_n, g_prev = grids[n], grids[n - 1]
        c_prev = float(c[N - n])
        a_n = a[n - 1]
        ph = _level_integral(pair, g_n, g_prev, p_hi[n - 1]) - c_prev  # optimistic continuation
        pl = _level_integral(pair, g_n, g_prev, p_lo[n - 1]) - c_prev  # pessimistic continuation

        below = (g_n < a_n) if np.isfinite(a_n) else np.zeros(g_n.size, dtype=bool)
        ident = ~below
        if ident.any():
            ph[ident] = _enforce_monotone(ph[ident], n, monotone, _mono_tol(ph[ident]))
            pl[ident] = _enforce_monotone(pl[ident], n, monotone, _mono_tol(pl[ident]))
        if below.any():
            i = int(np.argmax(ident)) if ident.any() else g_n.size - 1  # first grid pt >= a_n
            ph[below] = ph[i]        # optimistic: no better than the support edge
            pl[below] = -c_prev      # pessimistic: continuation value 0
        p_hi.append(ph)
        p_lo.append(pl)
        v_lo_asc.append(_threshold_from_grid(g_n, ph))  # optimistic (largest p) -> lowest v*
        v_hi_asc.append(_threshold_from_grid(g_n, pl))  # pessimistic -> highest v*

    tab = {"grids": grids, "p_lo": p_lo, "p_hi": p_hi, "support_lo": a}
    return np.array(v_lo_asc[::-1]), np.array(v_hi_asc[::-1]), tab


def infer_tarquin(v_star: np.ndarray, v: np.ndarray) -> np.ndarray:
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
        # Prefer the lowest finite BIC. The first fit is kept as a fallback so a
        # degenerate all-NaN BIC sweep (e.g. a collapsed component) still returns a
        # usable model rather than None; but as soon as any finite BIC appears it
        # replaces a non-finite incumbent -- otherwise a NaN in the *first* slot would
        # latch `best_bic = nan` and block every later (finite, better) candidate.
        if best_gmm is None:
            best_gmm, best_bic = g, bic
        elif np.isfinite(bic) and (not np.isfinite(best_bic) or bic < best_bic):
            best_gmm, best_bic = g, bic
    return best_gmm


def fit_pairwise_gmms(
    data,
    n_components: NComponents = 5,
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

    Any `covariance_type` is accepted: a non-"full" fit is densified to the (K, D, D)
    "full" layout the recursion indexes, so all four types feed `train_tarquin`.

    Selection-aware (ragged) fitting: `data` may carry `np.nan` where a prophecy was not
    acquired -- the shape a sample collected under an incumbent threshold policy actually
    has (V_{n-1} is revealed only for draws that proceeded past step n). Each pair is then
    fit on the rows that reveal *both* its prophecies (proceeded past step n), NOT the
    fully-revealed intersection. By sufficiency this is unbiased wherever V_n was revealed;
    fitting the complete-case intersection instead would truncate the response V_{n-1} from
    below and bias the conditional in the low-V_n region where v_n^* sits. The conditional
    is unidentified below the incumbent threshold (no overlap), so a saturated threshold on
    a truncated sample is a bound, not a point -- check it with `diagnose_saturation`.
    """
    data = np.asarray(data, dtype=float)
    N = data.shape[1] - 1
    n_rows = data.shape[0]
    pairs = []
    for n in range(1, N + 1):
        cols = [N - n, N - n + 1]  # (V_n, V_{n-1}) in README column order
        sub = data[:, cols]
        obs = np.all(np.isfinite(sub), axis=1)  # rows revealing BOTH prophecies of this pair
        n_obs = int(obs.sum())
        if n_obs == 0:
            raise ValueError(
                f"pair (V_{n}, V_{n - 1}): no row reveals both prophecies; cannot fit the "
                "conditional f_(n-1|n). The sample is truncated above this pair's step."
            )
        if n_obs < n_rows:
            warnings.warn(
                f"pair (V_{n}, V_{n - 1}): fitting f_(n-1|n) on {n_obs}/{n_rows} rows that "
                "reveal both prophecies (ragged/truncated sample). The conditional is "
                "identified only where V_n was revealed; below the incumbent threshold it is "
                "extrapolated, so treat a saturated v_n^* as a bound and inspect it with "
                "`diagnose_saturation`.",
                stacklevel=2,
            )
        gmm = _as_full_covariance(
            _fit_one_gmm(sub[obs], n_components, random_state, covariance_type, **kwargs)
        )
        # Record the identified lower edge of V_n (the conditioning column): the smallest v_n
        # for which this pair was observed. Below it f_(n-1|n) is unidentified, which
        # `train_tarquin`'s bounds / clamp / flag paths read via `tarquin_support_lo_`.
        gmm.tarquin_support_lo_ = float(sub[obs, 0].min())
        pairs.append(gmm)
    return pairs


def fit_joint_gmm(
    data,
    n_components: NComponents = 10,
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

    A single joint fit needs every dimension jointly, so rows with any missing prophecy are
    dropped (complete-case). On a sample truncated by an incumbent policy that discards the
    low tail and biases the joint; prefer `fit_pairwise_gmms`, which fits each pair on its
    own observed rows.
    """
    data = np.asarray(data, dtype=float)
    obs = np.all(np.isfinite(data), axis=1)
    n_obs = int(obs.sum())
    if n_obs < data.shape[0]:
        warnings.warn(
            f"fit_joint_gmm: dropping {data.shape[0] - n_obs} row(s) with a missing prophecy "
            "(complete-case). For a sample truncated by an incumbent policy this discards the "
            "low tail and biases the joint; prefer `fit_pairwise_gmms` for ragged data.",
            stacklevel=2,
        )
    return _fit_one_gmm(data[obs], n_components, random_state, covariance_type, **kwargs)


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
    v0 = 79.0 + 60.0 * z0 + 8.0 * np.expm1(0.6 * z0)  # strictly increasing, right-skewed; can be <0
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

    This is a *linear, necessary* check, weak in two distinct ways: (1) it is a
    correlation, so nonlinear residual dependence is invisible to it; (2) it only tests
    *span-2* triples (each outer pair is one step beyond the conditioning variable),
    whereas the full chain also requires V_{n+1} ⊥ {V_{n-1}, ..., V_0} | V_n -- a
    distribution can pass every consecutive-triple check yet have V_{n+1} depend on
    V_{n-2} given V_n. Values near 0 are therefore consistent with sufficiency but do
    not prove it; a large value is positive evidence *against* it. Pair this with
    `diagnose_fosd` and the training-time monotonicity warning for a fuller picture.

    Rule of thumb: |partial_corr| < ~0.1 is reassuring; > ~0.2 is a concrete signal to
    re-examine the ordering / construction of V (these are scale-free correlations, so
    the cutoff is data-agnostic, unlike `diagnose_fosd`'s CDF-step magnitude).

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
    pc = np.array(pcs)
    return {"partial_corr": pc, "max_abs": float(np.abs(pc).max()) if pc.size else 0.0}


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

    Rule of thumb: the violation is a probability-scale CDF step, so it is comparable
    across datasets. < ~0.05-0.1 is consistent with FOSD up to binning/finite-sample
    noise; a violation of several tenths (as the U-shaped non-FOSD fixture produces)
    means the single-threshold rule is unreliable for that pair.

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
    viol = np.array(viols)
    return {"violation": viol, "max": float(viol.max()) if viol.size else 0.0}


def diagnose_saturation(v_star, tab, c) -> dict:
    """Explain saturated (+/-inf) thresholds: artifact of a non-binding cost, or a real edge?

    A saturated v_n^* is easy to over-read. `-inf` ("proceed everywhere") and `+inf`
    ("never proceed") are both legitimate outputs, but on a cost-trivial problem *every*
    upstream threshold saturates to `-inf` and the only binding decision is the terminal
    v_0^* = t -- which is not a tuned multi-step policy. This compares each level's cost
    c_{n-1} against the value of proceeding *before* cost (p_n^T + c_{n-1}) at the saturating
    grid edge, so a tiny cost fraction flags "cost is not binding" rather than "tuned cut".

    Parameters
    ----------
    v_star : (N+1,) thresholds in README order, from `train_tarquin`.
    tab : the dict returned alongside `v_star` (its "p" holds p_n^T per level).
    c : (N,) cost vector in README order, as passed to `train_tarquin`.

    Returns dict with:
      "levels" : per acquisition step n=1..N: {level, threshold, status
                 ("finite"/"always_proceed (-inf)"/"never_proceed (+inf)"), cost,
                 option_value_at_edge, edge, cost_fraction}.
      "cost_trivial" : True iff every upstream threshold saturated to -inf (only the
                 terminal v_0^* = t binds).
      "summary" : a one-line human reading.
    """
    v_star = np.asarray(v_star, dtype=float)
    c = np.asarray(c, dtype=float)
    p = tab["p"]
    N = len(v_star) - 1
    if c.shape != (N,):
        raise ValueError(f"expected c of length {N}, got shape {c.shape}")
    levels = []
    statuses: list[str] = []
    fracs: list[float] = []
    for n in range(1, N + 1):
        v = float(v_star[N - n])
        c_prev = float(c[N - n])  # c_{n-1}
        pn = np.asarray(p[n], dtype=float)
        if v == -np.inf:
            status, edge = "always_proceed (-inf)", "floor"
            option_edge = float(pn[0]) + c_prev   # value of proceeding before cost, at floor
        elif v == np.inf:
            status, edge = "never_proceed (+inf)", "ceiling"
            option_edge = float(pn[-1]) + c_prev  # ... at ceiling
        else:
            status, edge, option_edge = "finite", None, float("nan")
        if edge is None:
            frac = float("nan")
        elif option_edge > 0:
            frac = c_prev / option_edge
        else:
            frac = float("inf")
        statuses.append(status)
        fracs.append(frac)
        levels.append({
            "level": n, "threshold": v, "status": status, "cost": c_prev,
            "option_value_at_edge": option_edge, "edge": edge, "cost_fraction": frac,
        })

    cost_trivial = bool(statuses) and all(s.startswith("always_proceed") for s in statuses)
    if cost_trivial:
        worst = max(fracs)
        summary = (
            f"All {N} acquisition threshold(s) saturated to -inf (proceed everywhere): only "
            f"the terminal decision v_0^* = t binds. Cost is at most {worst:.1%} of the value "
            "of proceeding at the grid floor, so the cost vector is not binding against the "
            "payoff spread -- a cost-trivial regime, not a tuned multi-step policy. (If V_0 is "
            "the quantity t already acts on, max E[(V_0 - t)^+] is near-tautological with the "
            "terminal rule.)"
        )
    else:
        sat = [n for n, s in enumerate(statuses, start=1) if s != "finite"]
        n_finite = sum(s == "finite" for s in statuses)
        summary = (
            f"{n_finite}/{N} acquisition threshold(s) finite; saturated at level(s) "
            f"{sat or 'none'}. See per-level cost_fraction."
        )
    return {"levels": levels, "cost_trivial": cost_trivial, "summary": summary}


def simulate_incumbent_truncation(data, thresholds) -> np.ndarray:
    """Apply an incumbent threshold policy to a full-joint sample, producing the ragged
    (one-sided-truncated) sample a *deployed* system actually collects.

    In deployment a prophecy is revealed only if the draw proceeded past its step: V_{n-1}
    is observed iff V_n exceeded the incumbent threshold (and every upstream prophecy did
    too). This returns a copy of `data` with `np.nan` wherever a draw stopped, so it can be
    fed straight to the selection-aware `fit_pairwise_gmms` to validate that ragged fitting
    recovers the thresholds the clean joint would (and that complete-case fitting does not).

    Parameters
    ----------
    data : (n, N+1) full-joint sample, columns in README order (V_N, ..., V_0).
    thresholds : (N,) incumbent thresholds in README order (tau_N, ..., tau_1); thresholds[i]
        gates revealing column i+1 (i.e. V_N must exceed thresholds[0] to reveal V_{N-1}).
        There is no threshold on V_N (seen for free) or after V_0 (terminal).

    Returns the ragged array (a float copy with NaNs); missingness is monotone down the chain.
    """
    data = np.asarray(data, dtype=float).copy()
    N = data.shape[1] - 1
    thr = np.asarray(thresholds, dtype=float)
    if thr.shape != (N,):
        raise ValueError(f"expected thresholds of length {N}, got shape {thr.shape}")
    alive = np.ones(data.shape[0], dtype=bool)
    for j in range(N):
        alive = alive & (data[:, j] > thr[j])  # proceed past step (N-j): reveal column j+1
        data[~alive, j + 1] = np.nan
    return data


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


def bootstrap_thresholds(
    data,
    c,
    t: float,
    *,
    n_boot: int = 200,
    n_components: NComponents = 5,
    ci: float = 0.95,
    seed: int = 0,
    fit_kwargs: dict | None = None,
    **train_kwargs,
) -> dict:
    """Bootstrap confidence intervals for the thresholds v*.

    The single `train_tarquin` point estimate hides the sampling/fit uncertainty in
    v*. This resamples rows of `data` with replacement `n_boot` times; each replicate
    refits the pairwise conditionals (`fit_pairwise_gmms`) and retrains, yielding a
    bootstrap distribution of v*.

    Columns of `data` must be in README order (V_N, ..., V_0). `c`/`t` are as in
    `train_tarquin`. `n_components` and `fit_kwargs` (a dict of extra `GaussianMixture`
    kwargs, e.g. `reg_covar`, `n_init`, `covariance_type`) control each replicate's
    fit; remaining keyword args pass through to `train_tarquin` (`halfwidth`,
    `grid_size`, `monotone`). Per-replicate warnings (saturation, monotonicity) are
    suppressed -- the saturation shows up as ±inf in `samples` instead.

    Cost note: each replicate is a full fit+train, so wall time is ~`n_boot` ×
    `train_tarquin`; keep `n_components` scalar (not a BIC sweep) for speed.

    Scope note: every replicate refits with the same GMM `random_state` (the
    `fit_pairwise_gmms` default), so the reported spread captures *resampling* (and
    hence data) uncertainty but not EM-initialization variance. Pass a
    `fit_kwargs={"random_state": ...}` per call, or raise `n_init`, if you want the fit's
    own optimization noise reflected too.

    Returns dict with:
      "point"    : (N+1,) v* on the full sample.
      "samples"  : (n_boot, N+1) bootstrap thresholds; a saturated endorsement set on a
                   replicate appears as ±inf (rows in README order).
      "mean"/"std": (N+1,) over the *finite* replicates per threshold (nan if none).
      "ci_low"/"ci_high": (N+1,) percentile interval at level `ci`, over the finite
                   replicates (nan if a threshold never resolved). Read alongside
                   "n_finite": a low finite count means the interval omits saturated
                   replicates and the threshold is genuinely unstable on this sample.
      "n_finite" : (N+1,) count of finite replicates per threshold (of `n_boot`).
    """
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1); got {ci}")
    data = np.asarray(data)
    fit_kwargs = dict(fit_kwargs or {})
    n = data.shape[0]
    rng = np.random.default_rng(seed)

    point, _ = train_tarquin(fit_pairwise_gmms(data, n_components, **fit_kwargs), c, t,
                             **train_kwargs)

    samples = np.empty((n_boot, point.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # saturation/monotonicity noise per replicate
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            pairs = fit_pairwise_gmms(data[idx], n_components, **fit_kwargs)
            samples[b], _ = train_tarquin(pairs, c, t, **train_kwargs)

    finite = np.isfinite(samples)
    n_finite = finite.sum(axis=0)
    masked = np.where(finite, samples, np.nan)
    lo_q, hi_q = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 + ci) / 2.0
    with warnings.catch_warnings():
        # A column with no finite replicate is all-nan -> nan summaries (suppress the
        # numpy "empty slice" / "all-nan" RuntimeWarnings). The CI is over the finite
        # replicates so a handful of saturated (+/-inf) replicates do not poison it.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(masked, axis=0)
        std = np.nanstd(masked, axis=0)
        ci_low = np.nanpercentile(masked, lo_q, axis=0)
        ci_high = np.nanpercentile(masked, hi_q, axis=0)
    return {
        "point": point,
        "samples": samples,
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_finite": n_finite,
    }


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
    # BIC-select the component count per pair rather than fixing it: on ~2k rows a fixed
    # 10-component 2-D fit is over-parameterized and injects spurious p_n^T wiggle.
    arr = make_demo_data()
    data_pairs = fit_pairwise_gmms(arr, n_components=range(1, 11))
    # Costs/overhead are scaled to this data (V_0 ~ O(100)); with the example's
    # tiny c/t the policy would just "always proceed" (threshold at the grid floor).
    t_data = float(np.median(arr[:, 2]))
    c_data = np.array([100.0, 100.0])
    v_star_data, tab_data = train_tarquin(data_pairs, c_data, t_data)
    print(f"\nfit_pairwise_gmms on make_demo_data() (t={t_data:.1f}, c=100): "
          f"v* = {np.round(v_star_data, 2).tolist()}  "
          "(-inf=always proceed, +inf=never)")
    print("  diagnose_saturation:", diagnose_saturation(v_star_data, tab_data, c_data)["summary"])

    # --- fit-on-train, score-on-holdout (avoids optimistic in-sample bias) ---
    train, test = holdout_split(arr, test_frac=0.3, seed=0)
    train_pairs = fit_pairwise_gmms(train, n_components=range(1, 11))
    v_holdout, _ = train_tarquin(train_pairs, c_data, t_data)
    cost_cols = np.array([np.nan, c_data[0], c_data[1]])
    pi_test = evaluate_policy_mc(test, (0, 1, 2), v_holdout, cost_cols, t_data)
    print(f"holdout E[pi] on {test.shape[0]} test rows: {pi_test.mean():+.3f}")

    # --- bootstrap CIs for the thresholds (fit/sampling uncertainty) ---
    # Keep n_components scalar here (a per-replicate BIC sweep would multiply the cost by
    # the candidate count); 5 is modest enough to avoid the over-parameterized wiggle.
    boot = bootstrap_thresholds(arr, c_data, t_data, n_boot=50, n_components=5, seed=0)
    for name, i in (("v_2*", 0), ("v_1*", 1)):
        print(f"{name}: point {boot['point'][i]:.2f}, "
              f"95% CI [{boot['ci_low'][i]:.2f}, {boot['ci_high'][i]:.2f}] "
              f"({boot['n_finite'][i]}/50 finite)")

    # --- deployed/truncated sample: incumbent policy hides the low tail (P0) ---
    # A *loose* incumbent (tau below the true thresholds) keeps v* identified, so ragged
    # fitting on the truncated sample should recover the clean ~[0.289, 0.12, 1.0], while
    # naive complete-case fitting is biased on v_2^* (its response V_1 is cut at tau_1).
    print("\nTruncated-sample (incumbent policy) demo:")
    full = rng.multivariate_normal(mu, cov, size=500_000)
    trunc = simulate_incumbent_truncation(full, thresholds=np.array([0.0, 0.0]))
    print(f"  observed fraction per column (V_2, V_1, V_0): "
          f"{np.round(np.isfinite(trunc).mean(axis=0), 2).tolist()}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ragged-sample / monotonicity notices
        v_ragged, _ = train_tarquin(fit_pairwise_gmms(trunc, n_components=1), c, t)
        cc = trunc[np.all(np.isfinite(trunc), axis=1)]  # naive complete-case (biased)
        v_cc, _ = train_tarquin(fit_pairwise_gmms(cc, n_components=1), c, t)
    print(f"  ragged-fit    v* = {np.round(v_ragged, 3).tolist()}  (selection-aware)")
    print(f"  complete-case v* = {np.round(v_cc, 3).tolist()}  (response-truncation bias)")

    # A *tight* incumbent (tau_2 above the true v_2*) leaves v_2* unidentified; bounds mode
    # reports the honest interval [-inf, ~a_2] instead of a spurious point (P1).
    tight = simulate_incumbent_truncation(full, thresholds=np.array([0.5, 0.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_lo, v_hi, _ = train_tarquin(fit_pairwise_gmms(tight, n_components=1), c, t,
                                      identification="bounds")
    print(f"  tight incumbent, bounds: v_lo = {np.round(v_lo, 3).tolist()}, "
          f"v_hi = {np.round(v_hi, 3).tolist()}  (v_2* unidentified below its floor)")
