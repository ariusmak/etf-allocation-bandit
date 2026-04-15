"""Non-contextual Student-t magnitude model.

Per-ETF posterior over the mean excess return mu_i with Normal-Normal
updates, Student-t predictive tails, and V2/V3 (risk-adjusted / Top-N)
scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as spst

from .beliefs import birth_prior_mean
from .config import EPS_S, ETF_COL, MC_N, NU, RET_COL, TAU0


def init_states() -> dict:
    return {}


def ensure_states(states: dict, belief_state: dict, df_t: pd.DataFrame, tau0: float = TAU0) -> None:
    """Initialize any new ETF with prior mean from tags and prior var tau0^2."""
    first_rows = df_t.drop_duplicates(subset=[ETF_COL])
    for _, row in first_rows.iterrows():
        etf = row[ETF_COL]
        if etf not in states:
            m0 = birth_prior_mean(belief_state, row)
            states[etf] = {"m": float(m0), "v": float(tau0 ** 2)}


def update_states(states: dict, df_t: pd.DataFrame, s_t: float) -> None:
    """Normal-Normal posterior update over mu_i."""
    if not np.isfinite(s_t) or s_t <= 0:
        return

    obs_var = float(s_t ** 2)
    for _, row in df_t.iterrows():
        etf = row[ETF_COL]
        x = float(row[RET_COL])
        m_prev = states[etf]["m"]
        v_prev = states[etf]["v"]
        v_post = 1.0 / (1.0 / v_prev + 1.0 / obs_var)
        m_post = v_post * (m_prev / v_prev + x / obs_var)
        states[etf]["m"] = float(m_post)
        states[etf]["v"] = float(v_post)


def scores_v2(
    states: dict,
    A_t,
    s_t: float,
    nu: int = NU,
    mc_N: int = MC_N,
    eps_s: float = EPS_S,
    rng=None,
) -> dict:
    """V2 risk-adjusted upside score: E[max(X,0)] / s_t, with X = mu + s_t * T_nu."""
    if (not np.isfinite(s_t)) or s_t <= 0:
        return {etf: 0.0 for etf in A_t}

    t_draws = spst.t.rvs(df=nu, size=mc_N, random_state=rng).astype(float)
    denom = float(max(s_t, 1e-12))

    scores = {}
    for etf in A_t:
        st = states.get(etf)
        if st is None:
            scores[etf] = 0.0
            continue
        mu = float(st["m"])
        X = mu + float(s_t) * t_draws
        upside = float(np.mean(np.maximum(X, 0.0)))
        score = upside / denom
        scores[etf] = max(0.0, max(score, float(eps_s)) if score > 0 else 0.0)
    return scores


def scores_v3_topN(
    states: dict,
    A_t,
    s_t: float,
    N: int = 50,
    nu: int = NU,
    mc_N: int = MC_N,
    eps_s: float = EPS_S,
    rng=None,
) -> dict:
    base = scores_v2(states, A_t, s_t, nu=nu, mc_N=mc_N, eps_s=eps_s, rng=rng)
    ranked = sorted(base.items(), key=lambda kv: kv[1], reverse=True)
    ranked_pos = [(etf, s) for etf, s in ranked if s > 0.0]
    keep = set(etf for etf, _ in ranked_pos[: int(N)])
    return {etf: (base[etf] if etf in keep else 0.0) for etf in A_t}
