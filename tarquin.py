"""Tarquin Algorithm: Bayesian information acquisition under sequential sufficiency.

Implements Algorithm 1 (training) from README.md using a Gaussian mixture model
for the joint density f(v_N, ..., v_0). A GMM makes the conditional densities
f(v_{n-1} | v_n) analytic (also a GMM), leaving only 1-D integrals for quadrature.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def marginalize_gmm(gmm: GaussianMixture, dims) -> GaussianMixture:
    """Marginalize a joint GMM over selected dimensions."""
    dims = np.asarray(dims)
    out = GaussianMixture(n_components=gmm.n_components, covariance_type="full")
    out.weights_ = gmm.weights_.copy()
    out.means_ = gmm.means_[:, dims]
    out.covariances_ = gmm.covariances_[:, dims[:, None], dims]
    out.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(out.covariances_))
    return out


def condition_gmm(gmm: GaussianMixture, cond_dim: int, cond_value: float) -> GaussianMixture:
    """f(X_other | X_{cond_dim} = cond_value) as a GMM, via per-component Gaussian conditioning."""
    K = gmm.n_components
    d = gmm.means_.shape[1]
    other = [i for i in range(d) if i != cond_dim]

    mu_o = gmm.means_[:, other]
    mu_c = gmm.means_[:, cond_dim]
    S_oo = gmm.covariances_[np.ix_(range(K), other, other)]
    S_oc = gmm.covariances_[:, other, cond_dim]
    S_co = gmm.covariances_[:, cond_dim, other]
    S_cc = gmm.covariances_[:, cond_dim, cond_dim]

    new_means = mu_o + (S_oc / S_cc[:, None]) * (cond_value - mu_c)[:, None]
    new_covs = S_oo - (S_oc[:, :, None] * S_co[:, None, :]) / S_cc[:, None, None]

    # Component weights scale by each component's marginal density at cond_value.
    comp_pdf = norm.pdf(cond_value, loc=mu_c, scale=np.sqrt(S_cc))
    new_weights = gmm.weights_ * comp_pdf
    total = new_weights.sum()
    new_weights = new_weights / total if total > 0 else gmm.weights_.copy()

    out = GaussianMixture(n_components=K, covariance_type="full")
    out.weights_ = new_weights
    out.means_ = new_means
    out.covariances_ = new_covs
    out.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(new_covs))
    return out


def train_tarquin(
    gmm: GaussianMixture,
    c: np.ndarray,
    t: float,
    root_bracket: tuple[float, float] = (-1e4, 1e4),
    xtol: float = 1e-6,
) -> tuple[np.ndarray, list[Callable[[float], float]]]:
    """Algorithm 1 (training).

    Parameters
    ----------
    gmm : GaussianMixture
        Joint over (V_N, V_{N-1}, ..., V_0). Column i corresponds to V_{N-i}.
    c : array of length N
        Cost vector; c[i] is the cost of acquiring V_{N-1-i} (i.e. c_{n-1} for n = N-i).
        In README ordering, c = (c_{N-1}, ..., c_0).
    t : float
        Tightness / overhead parameter.
    root_bracket : (lo, hi)
        Bracket for root-finding v_n^*. Widen if thresholds hit the edges.

    Returns
    -------
    v_star : np.ndarray, shape (N+1,)
        Thresholds in README order (v_N^*, ..., v_0^*). v_0^* = t.
    p_T : list of callables, ordered to match v_star (p_T[0] is p_N^T, last is p_0^T).
    """
    D = gmm.means_.shape[1]
    N = D - 1
    c = np.asarray(c, dtype=float)
    assert c.shape == (N,), f"expected c of length {N}, got {c.shape}"

    col_of = lambda n: N - n  # column index of V_n in the GMM

    # n = 0: p_0^T(v_0) = v_0 - t, so v_0^* = t.
    p_T_asc: list[Callable[[float], float]] = [lambda v, _t=t: v - _t]
    v_star_asc: list[float] = [t]

    for n in range(1, N + 1):
        joint = marginalize_gmm(gmm, [col_of(n), col_of(n - 1)])  # cols: (V_n, V_{n-1})
        p_prev = p_T_asc[n - 1]
        v_prev_star = v_star_asc[n - 1]
        c_prev = float(c[(N - 1) - (n - 1)])  # README-indexed c_{n-1}

        p_n = _make_p_T(joint, p_prev, v_prev_star, c_prev)
        p_T_asc.append(p_n)
        v_star_asc.append(_find_threshold(p_n, *root_bracket, xtol=xtol))

    return np.array(v_star_asc[::-1]), p_T_asc[::-1]


def _make_p_T(joint, p_prev, v_prev_star, c_prev):
    """Build p_n^T as a closure so each level carries its own conditional + bound.

    Integrates each mixture component in standardized coordinates z = (v - mu_k)/sigma_k
    so the density is always phi(z), keeping quad stable no matter where v_n lands.
    """
    def p_T_n(vn: float) -> float:
        if np.isinf(v_prev_star):
            return -c_prev  # S_{n-1} empty => integral = 0
        f_cond = condition_gmm(joint, cond_dim=0, cond_value=vn)
        mus = f_cond.means_[:, 0]
        sigmas = np.sqrt(f_cond.covariances_[:, 0, 0])
        weights = f_cond.weights_

        total = 0.0
        for w, mu, sigma in zip(weights, mus, sigmas):
            z_lo = (v_prev_star - mu) / sigma

            def integrand(z, mu=mu, sigma=sigma):
                return p_prev(mu + z * sigma) * norm.pdf(z)

            # Split at 0 so quad's adaptive sampler always sees the peak
            # even when z_lo or the upper end is far from the bulk.
            if z_lo < 0:
                a, _ = quad(integrand, z_lo, 0.0, limit=200)
                b, _ = quad(integrand, 0.0, np.inf, limit=200)
                val = a + b
            else:
                val, _ = quad(integrand, z_lo, np.inf, limit=200)
            total += w * val
        return total - c_prev

    return p_T_n


def infer_tarquin(v_star, v) -> np.ndarray:
    """Algorithm 2 (inference).

    Walks from the highest-index prophecy down. Sets r_n = 1 while v_n >= v_n^*;
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
    assert v_star.shape == v.shape, f"shape mismatch: {v_star.shape} vs {v.shape}"
    r = np.zeros_like(v_star, dtype=int)
    for i in range(len(v)):
        if v[i] >= v_star[i]:
            r[i] = 1
        else:
            break
    return r


def _find_threshold(f, lo, hi, xtol=1e-6) -> float:
    """Smallest v with f(v) >= 0 (f assumed nondecreasing per Prop. 3)."""
    flo, fhi = f(lo), f(hi)
    if flo >= 0:
        return lo
    if fhi < 0:
        return np.inf
    return brentq(f, lo, hi, xtol=xtol)


# --- Fitting, book construction, and policy evaluation helpers -------------


def fit_joint_gmm(
    data,
    n_components: int = 10,
    random_state: int = 0,
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
    **kwargs,
) -> GaussianMixture:
    """Fit a GMM to `data` (array or DataFrame).

    Columns must be in README order (V_N, ..., V_0) with V_0 last.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        covariance_type=covariance_type,
        **kwargs,
    )
    gmm.fit(np.asarray(data))
    return gmm


def train_book(
    gmm_full: GaussianMixture,
    col_order,
    cost_per_prophecy,
    t: float,
    **kwargs,
) -> tuple[np.ndarray, list[Callable[[float], float]]]:
    """Train on any subset and/or permutation of columns of `gmm_full`.

    Covers both abridgements (subset of columns) and rearrangements (reordered
    columns) with the same entry point.

    Parameters
    ----------
    col_order : sequence of column indices into gmm_full, in README order
        (V_{M-1}, ..., V_0) — last entry is the payoff prophecy.
    cost_per_prophecy : array of length D_full; entry j is the cost to acquire
        column j. The top prophecy (col_order[0]) is seen for free; entries for
        col_order[1:] are used.
    """
    col_order = list(col_order)
    cost_per_prophecy = np.asarray(cost_per_prophecy, dtype=float)
    reduced = marginalize_gmm(gmm_full, col_order)
    c_step = np.array([cost_per_prophecy[col_order[i]] for i in range(1, len(col_order))])
    return train_tarquin(reduced, c_step, t, **kwargs)


def enumerate_abridgements(col_order):
    """Yield all abridgements of a book, as col_order tuples.

    Per the README, a book with index set delta has 2^(|delta|-1) - 1
    abridgements: the proper subsets of delta that still include V_0
    (which sits at col_order[-1]). Original ordering is preserved within
    each yielded tuple.
    """
    from itertools import combinations

    head, tail = tuple(col_order[:-1]), col_order[-1]
    M = len(col_order)
    for size in range(0, M - 1):
        for combo in combinations(head, size):
            yield (*combo, tail)


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
        alive = alive & (samples[:, col] >= v_star[i])
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
    v_star, _ = train_tarquin(gmm, c, t)
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

    # --- fit_joint_gmm smoke test on test.csv, if present ---
    from pathlib import Path
    csv = Path(__file__).with_name("test.csv")
    if csv.exists():
        # Columns expected: index, v2, v1, v0 (v2 = V_N, v0 = payoff prophecy).
        arr = np.genfromtxt(csv, delimiter=",", skip_header=1)[:, 1:]
        gmm_data = fit_joint_gmm(arr, n_components=10)
        print(f"\nfit_joint_gmm on {csv.name}: "
              f"log-lik/sample={gmm_data.score(arr):.3f}")
