"""Guards for the portfolio policy: normalization, cash fallback, turnover."""

import numpy as np
import pandas as pd
import pytest

from etf_bandit.policy import (
    portfolio_return_for_month,
    scores_to_weights,
    turnover,
)


def test_all_zero_scores_go_full_cash():
    w_etf, w_cash, diag = scores_to_weights({"A": 0.0, "B": 0.0}, k=100)
    assert w_etf == {}
    assert w_cash == 1.0
    assert diag["W_etf"] == 0.0


def test_empty_scores_go_full_cash():
    w_etf, w_cash, _ = scores_to_weights({}, k=100)
    assert w_etf == {}
    assert w_cash == 1.0


def test_weights_sum_to_one_with_cash():
    scores = {"A": 0.1, "B": 0.3, "C": 0.0}
    w_etf, w_cash, _ = scores_to_weights(scores, k=100)
    assert "C" not in w_etf  # zero score dropped
    total = sum(w_etf.values()) + w_cash
    assert total == pytest.approx(1.0, abs=1e-12)


def test_all_weights_nonnegative():
    scores = {"A": 0.2, "B": 0.5, "C": 0.1}
    w_etf, w_cash, _ = scores_to_weights(scores, k=50)
    assert w_cash >= 0
    assert all(v >= 0 for v in w_etf.values())


def test_W_etf_monotonic_in_k():
    scores = {"A": 0.5, "B": 0.5}
    _, wc_low, _ = scores_to_weights(scores, k=10)
    _, wc_high, _ = scores_to_weights(scores, k=1000)
    # higher k -> more ETF exposure -> less cash
    assert wc_high < wc_low


def test_turnover_identical_is_zero():
    w = {"A": 0.3, "B": 0.2, "CASH": 0.5}
    assert turnover(w, w) == pytest.approx(0.0)


def test_turnover_full_rotation_is_one():
    prev = {"A": 1.0, "CASH": 0.0}
    curr = {"B": 1.0, "CASH": 0.0}
    assert turnover(prev, curr) == pytest.approx(1.0)


def test_turnover_bounded_zero_one():
    prev = {"A": 0.4, "B": 0.3, "CASH": 0.3}
    curr = {"A": 0.1, "B": 0.5, "C": 0.2, "CASH": 0.2}
    t = turnover(prev, curr)
    assert 0.0 <= t <= 1.0


def test_portfolio_return_uses_growth_and_rf():
    df_hold = pd.DataFrame({
        "ETF": ["A", "B"],
        "Growth": [0.10, -0.05],
        "rf_month": [0.002, 0.002],
    })
    ret = portfolio_return_for_month(df_hold, w_etf={"A": 0.5, "B": 0.3}, w_cash=0.2)
    expected = 0.5 * 0.10 + 0.3 * (-0.05) + 0.2 * 0.002
    assert ret == pytest.approx(expected)


def test_portfolio_return_full_cash_equals_rf():
    df_hold = pd.DataFrame({"ETF": ["A"], "Growth": [0.5], "rf_month": [0.003]})
    ret = portfolio_return_for_month(df_hold, w_etf={}, w_cash=1.0)
    assert ret == pytest.approx(0.003)
