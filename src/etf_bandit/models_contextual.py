"""Contextual Student-t model.

Per-ETF Bayesian linear regression on a macro context vector z_t, with
Student-t predictive scoring that penalizes coefficient uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as spst

from .beliefs import birth_prior_mean
from .config import EPS_S, ETF_COL, MACRO_COLS, MC_N, NU, RET_COL, TAU0


def make_z_vec_from_df(df_t: pd.DataFrame, z_cols=MACRO_COLS, use_intercept: bool = True) -> np.ndarray:
    """Build z_t from the decision-month frame (macro cols constant across ETFs)."""
    row0 = df_t.iloc[0]
    z = row0[list(z_cols)].astype(float).values
    if use_intercept:
        z = np.concatenate(([1.0], z))
    return z


def init_states() -> dict:
    return {}


def ensure_states(
    states: dict,
    belief_state: dict,
    df_t: pd.DataFrame,
    d: int,
    tau0: float = TAU0,
    use_intercept: bool = True,
) -> None:
    """Initialize new ETFs with m0 (tag-informed intercept only) and V0 = tau0^2 I."""
    first_rows = df_t.drop_duplicates(subset=[ETF_COL])
    for _, row in first_rows.iterrows():
        etf = row[ETF_COL]
        if etf in states:
            continue

        m0 = np.zeros(d, dtype=float)
        intercept_prior = float(birth_prior_mean(belief_state, row))
        if use_intercept:
            m0[0] = intercept_prior
        V0 = (tau0 ** 2) * np.eye(d, dtype=float)
        states[etf] = {"m": m0, "V": V0}


def update_states(states: dict, df_t: pd.DataFrame, z_t: np.ndarray, s_t: float) -> None:
    """Information-form BLR posterior update per ETF.

    V_post = (V^-1 + (1/s^2) z z^T)^-1
    m_post = V_post (V^-1 m + (1/s^2) z x)
    """
    if not np.isfinite(s_t) or s_t <= 0:
        return

    s2 = float(s_t ** 2)
    z = np.asarray(z_t, dtype=float).reshape(-1, 1)

    for _, row in df_t.iterrows():
        etf = row[ETF_COL]
        x = float(row[RET_COL])

        m = states[etf]["m"].reshape(-1, 1)
        V = states[etf]["V"]

        Vinv = np.linalg.inv(V)
        Vinv_post = Vinv + (1.0 / s2) * (z @ z.T)
        V_post = np.linalg.inv(Vinv_post)
        m_post = V_post @ (Vinv @ m + (1.0 / s2) * z * x)

        states[etf]["m"] = m_post.ravel()
        states[etf]["V"] = V_post


def scores_predvar(
    states: dict,
    A_t,
    z_t: np.ndarray,
    s_t: float,
    nu: int = NU,
    mc_N: int = MC_N,
    eps_s: float = EPS_S,
    rng=None,
) -> dict:
    """Contextual Student-t score with predictive variance.

    mu_hat = z^T m;  scale = sqrt(s_t^2 + z^T V z);  X = mu_hat + scale * T_nu.
    score = E[max(X, 0)] / scale.
    """
    if (not np.isfinite(s_t)) or s_t <= 0:
        return {etf: 0.0 for etf in A_t}

    z = np.asarray(z_t, dtype=float).reshape(-1, 1)
    t_draws = spst.t.rvs(df=nu, size=mc_N, random_state=rng).astype(float)

    scores = {}
    for etf in A_t:
        st = states.get(etf)
        if st is None:
            scores[etf] = 0.0
            continue

        m = np.asarray(st["m"], dtype=float).reshape(-1, 1)
        V = np.asarray(st["V"], dtype=float)

        mu_hat = float((z.T @ m)[0, 0])
        var_beta = float((z.T @ V @ z)[0, 0])
        scale = float(np.sqrt(max(s_t ** 2 + var_beta, 1e-12)))
        X = mu_hat + scale * t_draws

        upside = float(np.mean(np.maximum(X, 0.0)))
        score = upside / scale
        scores[etf] = max(0.0, max(score, float(eps_s)) if score > 0 else 0.0)
    return scores


def scores_predvar_topN(
    states: dict,
    A_t,
    z_t: np.ndarray,
    s_t: float,
    N: int = 50,
    nu: int = NU,
    mc_N: int = MC_N,
    eps_s: float = EPS_S,
    rng=None,
) -> dict:
    base = scores_predvar(states, A_t, z_t, s_t, nu=nu, mc_N=mc_N, eps_s=eps_s, rng=rng)
    ranked = sorted(base.items(), key=lambda kv: kv[1], reverse=True)
    ranked_pos = [(etf, s) for etf, s in ranked if s > 0.0]
    keep = set(etf for etf, _ in ranked_pos[: int(N)])
    return {etf: (base[etf] if etf in keep else 0.0) for etf in A_t}
