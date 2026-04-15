"""Portfolio policy: scores -> weights, turnover, realized return accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ETF_COL, GROWTH_COL, RF_COL


def scores_to_weights(scores: dict, k: float, eps: float = 1e-16):
    """Convert nonnegative scores at decision month t into (w_etf, w_cash, diag).

    W_etf = (k * mean_score) / (1 + k * mean_score) in (0, 1).
    Within-ETF proportions: score_i / sum(scores_positive).
    Returns full-cash if no arms / no positive scores.
    """
    if scores is None or len(scores) == 0:
        return {}, 1.0, {"mean_score": 0.0, "W_etf": 0.0, "n_pos_scores": 0, "top5_share": 0.0}

    s = {etf: float(max(0.0, v)) for etf, v in scores.items()}
    vals = np.fromiter(s.values(), dtype=float)

    mean_score = float(vals.mean()) if len(vals) else 0.0
    n_pos = int((vals > 0).sum())

    if n_pos == 0 or mean_score <= 0.0:
        return {}, 1.0, {"mean_score": mean_score, "W_etf": 0.0, "n_pos_scores": 0, "top5_share": 0.0}

    W_etf = (k * mean_score) / (1.0 + k * mean_score) if k > 0 else 0.0
    W_etf = float(np.clip(W_etf, 0.0, 1.0))
    w_cash = 1.0 - W_etf

    pos_sum = float(vals[vals > 0].sum())
    if pos_sum <= eps:
        return {}, 1.0, {"mean_score": mean_score, "W_etf": 0.0, "n_pos_scores": 0, "top5_share": 0.0}

    w_etf = {etf: W_etf * (v / pos_sum) for etf, v in s.items() if v > 0}
    top5_share = (
        float(np.sum(sorted(w_etf.values(), reverse=True)[:5])) if w_etf else 0.0
    )
    diag = {
        "mean_score": mean_score,
        "W_etf": W_etf,
        "n_pos_scores": len(w_etf),
        "top5_share": top5_share,
    }
    return w_etf, w_cash, diag


def turnover(prev_weights_full: dict, curr_weights_full: dict) -> float:
    """0.5 * sum |w_new - w_old| over the union of keys (must include CASH)."""
    keys = set(prev_weights_full) | set(curr_weights_full)
    return 0.5 * sum(
        abs(curr_weights_full.get(k, 0.0) - prev_weights_full.get(k, 0.0)) for k in keys
    )


def portfolio_return_for_month(df_hold: pd.DataFrame, w_etf: dict, w_cash: float) -> float:
    """Realized total portfolio return in a holding month (t+1).

    r_p = sum_i w_i * Growth_{i, t+1} + w_cash * rf_{t+1}.
    """
    rf_vals = df_hold[RF_COL].dropna().unique()
    assert len(rf_vals) == 1, f"Expected exactly one rf value, got {rf_vals}"
    rf = float(rf_vals[0])

    realized_growth = dict(zip(df_hold[ETF_COL].values, df_hold[GROWTH_COL].values))
    etf_ret = sum(float(w) * float(realized_growth.get(etf, 0.0)) for etf, w in w_etf.items())
    return etf_ret + float(w_cash) * rf
