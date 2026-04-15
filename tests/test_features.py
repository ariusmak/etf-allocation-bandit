"""Guards for leak-free feature construction."""

import numpy as np
import pandas as pd
import pytest

from etf_bandit.features import prior_streak
from etf_bandit.panel import apply_zscore, fit_zscore


def test_prior_streak_stores_prior_value_not_current():
    y = [1, 1, 1, 0, 1]
    out = prior_streak(y)
    # The streak at index i is what was seen BEFORE y[i].
    assert list(out) == [0, 1, 2, 3, 0]


def test_prior_streak_all_zeros():
    assert list(prior_streak([0, 0, 0])) == [0, 0, 0]


def test_prior_streak_all_ones():
    assert list(prior_streak([1, 1, 1, 1])) == [0, 1, 2, 3]


def test_zscore_uses_train_stats_only():
    train = pd.DataFrame({"x": [0.0, 2.0, 4.0]})     # mean=2, std=sqrt(8/3)
    test = pd.DataFrame({"x": [100.0, 200.0, 300.0]})  # extreme values
    mean, std = fit_zscore(train, cols=["x"])

    train_z = apply_zscore(train, mean, std, cols=["x"])
    test_z = apply_zscore(test, mean, std, cols=["x"])

    # Standardized train has mean 0 (ddof=0)
    assert train_z["x"].mean() == pytest.approx(0.0, abs=1e-12)
    # Test rows are not forced to zero-mean; stats come from TRAIN only.
    assert test_z["x"].mean() > 10


def test_zscore_zero_std_is_safe():
    train = pd.DataFrame({"x": [3.0, 3.0, 3.0]})
    mean, std = fit_zscore(train, cols=["x"])
    # zero std is replaced with 1.0
    assert float(std["x"]) == 1.0
    out = apply_zscore(train, mean, std, cols=["x"])
    assert np.allclose(out["x"].values, 0.0)
