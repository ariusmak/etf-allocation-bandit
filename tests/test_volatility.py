"""Guard: global_volatility must use strictly months < t."""

import numpy as np
import pandas as pd

from etf_bandit.volatility import global_volatility


def _panel():
    months = pd.date_range("2020-01-31", periods=20, freq="ME")
    rows = []
    for i, m in enumerate(months):
        # small excess returns for t < 12, huge for t >= 12 — easy to detect leakage
        val = 0.001 if i < 12 else 10.0
        rows.append({"Month": m, "ETF": "A", "excess_return": val})
    return pd.DataFrame(rows)


def test_strictly_uses_prior_months():
    df = _panel()
    months = sorted(df["Month"].unique())
    # At t = months[12], trailing window of 12 should only cover the small values.
    s = global_volatility(df, months[12], vol_window=12)
    # If current month leaked in, std would be huge. Expect tiny std (~0 but floored).
    assert s < 1.0


def test_nan_before_enough_history():
    df = _panel()
    months = sorted(df["Month"].unique())
    assert np.isnan(global_volatility(df, months[5], vol_window=12))


def test_floored_positive():
    df = pd.DataFrame({
        "Month": pd.date_range("2020-01-31", periods=15, freq="ME"),
        "ETF": ["A"] * 15,
        "excess_return": [0.0] * 15,
    })
    months = sorted(df["Month"].unique())
    s = global_volatility(df, months[13], vol_window=12)
    assert s > 0  # floored to 1e-8, not zero
