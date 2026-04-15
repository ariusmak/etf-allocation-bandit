"""Walk-forward policy simulation loops and simple benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import beliefs, models_contextual, models_studentt
from .config import (
    EPS_S,
    ETF_COL,
    GROWTH_COL,
    MC_N,
    MONTH_COL,
    NU,
    RF_COL,
    VOL_WINDOW,
)
from .policy import portfolio_return_for_month, scores_to_weights, turnover
from .volatility import global_volatility

WEALTH_LOG_COLS = [
    "Month_hold", "port_ret", "wealth", "w_cash", "W_etf",
    "mean_score", "turnover", "n_bet", "top5_share",
]
WEIGHTS_LOG_COLS = ["Month_hold", "ETF", "weight"]
SCORES_LOG_COLS = ["Month_decide", "ETF", "score"]


def run_studentt_topN(
    df: pd.DataFrame,
    k: float,
    N: int = 10,
    initial_wealth: float = 1000.0,
    belief_state_init: dict | None = None,
    states_init: dict | None = None,
    vol_window: int = VOL_WINDOW,
    nu: int = NU,
    mc_N: int = MC_N,
    eps_s: float = EPS_S,
    rng=None,
):
    """Non-contextual Student-t Top-N walk-forward simulation (warm-start capable)."""
    months = sorted(df[MONTH_COL].unique())
    wealth = float(initial_wealth)

    wealth_log, weights_log, scores_log = [], [], []
    belief_state = belief_state_init if belief_state_init is not None else beliefs.init_belief_state()
    states = states_init if states_init is not None else models_studentt.init_states()
    prev_weights_full = {"CASH": 1.0}

    for t_idx in range(len(months) - 1):
        month_t = months[t_idx]
        month_hold = months[t_idx + 1]
        df_t = df[df[MONTH_COL] == month_t]
        df_hold = df[df[MONTH_COL] == month_hold]
        A_t = df_t[ETF_COL].unique().tolist()

        beliefs.update_beliefs(belief_state, df_t)
        models_studentt.ensure_states(states, belief_state, df_t)
        s_t = global_volatility(df, month_t, vol_window=vol_window)
        models_studentt.update_states(states, df_t, s_t)

        scores = models_studentt.scores_v3_topN(
            states, A_t, s_t, N=N, nu=nu, mc_N=mc_N, eps_s=eps_s, rng=rng
        )
        for etf, s in scores.items():
            scores_log.append({"Month_decide": month_t, "ETF": etf, "score": s})

        w_etf, w_cash, diag = scores_to_weights(scores, k)
        for etf, w in w_etf.items():
            weights_log.append({"Month_hold": month_hold, "ETF": etf, "weight": w})
        weights_log.append({"Month_hold": month_hold, "ETF": "CASH", "weight": w_cash})

        port_ret = portfolio_return_for_month(df_hold, w_etf, w_cash)
        wealth *= (1.0 + port_ret)

        curr_weights_full = {**w_etf, "CASH": w_cash}
        turn = turnover(prev_weights_full, curr_weights_full)
        prev_weights_full = curr_weights_full

        wealth_log.append({
            "Month_hold": month_hold,
            "port_ret": port_ret,
            "wealth": wealth,
            "w_cash": w_cash,
            "W_etf": 1.0 - w_cash,
            "mean_score": diag["mean_score"],
            "turnover": turn,
            "n_bet": diag["n_pos_scores"],
            "top5_share": diag["top5_share"],
        })

    return {
        "wealth": pd.DataFrame(wealth_log, columns=WEALTH_LOG_COLS),
        "weights": pd.DataFrame(weights_log, columns=WEIGHTS_LOG_COLS),
        "scores": pd.DataFrame(scores_log, columns=SCORES_LOG_COLS),
        "belief_state": belief_state,
        "states": states,
    }


def run_contextual_topN(
    df: pd.DataFrame,
    k: float,
    N: int = 50,
    initial_wealth: float = 1000.0,
    belief_state_init: dict | None = None,
    states_init: dict | None = None,
    vol_window: int = VOL_WINDOW,
    nu: int = NU,
    mc_N: int = MC_N,
    eps_s: float = EPS_S,
    rng=None,
):
    """Contextual Student-t Top-N walk-forward simulation (warm-start capable)."""
    months = sorted(df[MONTH_COL].unique())
    wealth = float(initial_wealth)

    wealth_log, weights_log, scores_log = [], [], []
    belief_state = belief_state_init if belief_state_init is not None else beliefs.init_belief_state()
    states = states_init if states_init is not None else models_contextual.init_states()
    prev_weights_full = {"CASH": 1.0}

    for t_idx in range(len(months) - 1):
        month_t = months[t_idx]
        month_hold = months[t_idx + 1]
        df_t = df[df[MONTH_COL] == month_t]
        df_hold = df[df[MONTH_COL] == month_hold]
        A_t = df_t[ETF_COL].unique().tolist()

        z_t = models_contextual.make_z_vec_from_df(df_t)
        d = len(np.asarray(z_t).ravel())

        beliefs.update_beliefs(belief_state, df_t)
        models_contextual.ensure_states(states, belief_state, df_t, d=d)
        s_t = global_volatility(df, month_t, vol_window=vol_window)
        models_contextual.update_states(states, df_t, z_t, s_t)

        scores = models_contextual.scores_predvar_topN(
            states, A_t, z_t, s_t, N=N, nu=nu, mc_N=mc_N, eps_s=eps_s, rng=rng
        )
        for etf, s in scores.items():
            scores_log.append({"Month_decide": month_t, "ETF": etf, "score": s})

        w_etf, w_cash, diag = scores_to_weights(scores, k)
        for etf, w in w_etf.items():
            weights_log.append({"Month_hold": month_hold, "ETF": etf, "weight": w})
        weights_log.append({"Month_hold": month_hold, "ETF": "CASH", "weight": w_cash})

        port_ret = portfolio_return_for_month(df_hold, w_etf, w_cash)
        wealth *= (1.0 + port_ret)

        curr_weights_full = {**w_etf, "CASH": w_cash}
        turn = turnover(prev_weights_full, curr_weights_full)
        prev_weights_full = curr_weights_full

        wealth_log.append({
            "Month_hold": month_hold,
            "port_ret": port_ret,
            "wealth": wealth,
            "w_cash": w_cash,
            "W_etf": 1.0 - w_cash,
            "mean_score": diag["mean_score"],
            "turnover": turn,
            "n_bet": diag["n_pos_scores"],
            "top5_share": diag["top5_share"],
        })

    return {
        "wealth": pd.DataFrame(wealth_log, columns=WEALTH_LOG_COLS),
        "weights": pd.DataFrame(weights_log, columns=WEIGHTS_LOG_COLS),
        "scores": pd.DataFrame(scores_log, columns=SCORES_LOG_COLS),
        "belief_state": belief_state,
        "states": states,
    }


def run_cash(df: pd.DataFrame, initial_wealth: float) -> pd.DataFrame:
    months = sorted(df[MONTH_COL].unique())
    wealth = float(initial_wealth)
    log = []
    for t_idx in range(len(months) - 1):
        month_hold = months[t_idx + 1]
        df_hold = df[df[MONTH_COL] == month_hold]
        rf = float(df_hold[RF_COL].iloc[0])
        wealth *= (1.0 + rf)
        log.append({"Month_hold": month_hold, "wealth": wealth})
    return pd.DataFrame(log)


def run_buy_hold_spy(df: pd.DataFrame, initial_wealth: float) -> pd.DataFrame:
    spy = df[df[ETF_COL] == "SPY"].sort_values(MONTH_COL)
    wealth = float(initial_wealth)
    log = []
    for _, row in spy.iterrows():
        wealth *= (1.0 + float(row[GROWTH_COL]))
        log.append({"Month": row[MONTH_COL], "wealth": wealth})
    return pd.DataFrame(log)
