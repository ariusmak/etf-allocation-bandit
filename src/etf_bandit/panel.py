"""Panel-level validation and leak-free standardization."""

from __future__ import annotations

import pandas as pd

from .config import MACRO_COLS, MONTH_COL, TAG_COLS


def validate_panel(df: pd.DataFrame) -> None:
    """Assertion-based sanity checks mirroring notebook `validate_panel`."""
    required = [MONTH_COL, "ETF", "excess_return", "outperformed", "rf_month"] + TAG_COLS
    missing = set(required) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    assert not df.duplicated(subset=[MONTH_COL, "ETF"]).any(), (
        "Duplicate (Month, ETF) rows detected"
    )
    assert df[MONTH_COL].is_monotonic_increasing, "Month not sorted"

    nan_cols = df[required].columns[df[required].isna().any()]
    assert len(nan_cols) == 0, f"NaNs found in columns: {list(nan_cols)}"

    for c in TAG_COLS:
        assert df[c].isin([0, 1]).all(), f"Invalid values in tag column {c}"
    assert df["outperformed"].isin([0, 1]).all(), "outperformed not binary"

    tag_sums = df.groupby("ETF")[TAG_COLS].sum().sum(axis=1)
    no_tag = tag_sums[tag_sums == 0].index.tolist()
    assert len(no_tag) == 0, f"ETFs with no active tags detected: {no_tag}"


def fit_zscore(df_train: pd.DataFrame, cols: list[str] = MACRO_COLS):
    """Return (mean, std) computed on TRAIN only.

    Zero stds are replaced with 1 so the transform is a no-op on constant
    columns rather than NaN.
    """
    mean = df_train[cols].mean()
    std = df_train[cols].std(ddof=0).replace(0.0, 1.0)
    return mean, std


def apply_zscore(df: pd.DataFrame, mean, std, cols: list[str] = MACRO_COLS) -> pd.DataFrame:
    out = df.copy()
    out[cols] = (out[cols] - mean) / std
    return out


def train_test_split(df: pd.DataFrame, split_date: str = "2014-12-31"):
    df = df.copy()
    df[MONTH_COL] = pd.to_datetime(df[MONTH_COL])
    split = pd.Timestamp(split_date)
    return (
        df[df[MONTH_COL] <= split].copy().reset_index(drop=True),
        df[df[MONTH_COL] > split].copy().reset_index(drop=True),
    )
