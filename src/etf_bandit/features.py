"""Feature engineering: prior-streak context + macro transforms + lag merge.

Mirrors `2_Building_dataset.ipynb`. All macro features are shifted by 1 month
so decisions made at month t only see information available through t-1.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .config import MACRO_SRC_COLS


def prior_streak(y: Iterable[int]) -> np.ndarray:
    """Leak-free prior-month consecutive outperformance streak.

    The value at index i is the streak length BEFORE observing y[i]. Used to
    build `months_outperforming` without leaking the current month's outcome.
    """
    out = np.zeros(len(y), dtype=int)
    streak = 0
    for i, v in enumerate(y):
        out[i] = streak
        streak = streak + 1 if v == 1 else 0
    return out


def add_outperformance_fields(etf_ir: pd.DataFrame) -> pd.DataFrame:
    df = etf_ir.copy()
    df["outperformed"] = (df["Growth"] > df["rf_month"]).astype(int)
    df["excess_return"] = df["Growth"] - df["rf_month"]
    df = df.sort_values(["ETF", "Month"]).reset_index(drop=True)
    df["months_outperforming"] = (
        df.groupby("ETF")["outperformed"].transform(prior_streak)
    )
    return df


def _to_month_end(df: pd.DataFrame, date_col: str = "observation_date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["Month"] = out[date_col] + MonthEnd(0)
    return out.groupby("Month", as_index=False).mean(numeric_only=True)


def build_tb3ms_change(tb3ms_raw: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns [Month, TB3MS, dTB3MS]."""
    tb = _to_month_end(tb3ms_raw).sort_values("Month").reset_index(drop=True)
    tb["dTB3MS"] = tb["TB3MS"].diff()
    return tb[["Month", "TB3MS", "dTB3MS"]]


def build_yield_slope(dgs10_raw: pd.DataFrame, tb3ms_monthly: pd.DataFrame) -> pd.DataFrame:
    dgs = _to_month_end(dgs10_raw).sort_values("Month").reset_index(drop=True)
    merged = pd.merge(dgs, tb3ms_monthly[["Month", "TB3MS"]], on="Month", how="inner")
    merged["yield_slope"] = merged["DGS10"] - merged["TB3MS"]
    return merged[["Month", "yield_slope"]]


def build_credit_spread(baa_raw: pd.DataFrame, dgs10_monthly: pd.DataFrame) -> pd.DataFrame:
    baa = _to_month_end(baa_raw).sort_values("Month").reset_index(drop=True)
    merged = pd.merge(baa, dgs10_monthly, on="Month", how="inner")
    merged["credit_spread"] = merged["BAA"] - merged["DGS10"]
    return merged[["Month", "credit_spread"]]


def build_inflation(cpi_raw: pd.DataFrame) -> pd.DataFrame:
    cpi = _to_month_end(cpi_raw).sort_values("Month").reset_index(drop=True)
    cpi["inflation_yoy"] = cpi["CPIAUCSL"].pct_change(12)
    cpi["inflation_mom"] = cpi["CPIAUCSL"].pct_change(1)
    cpi["inflation_shock"] = cpi["inflation_mom"] - cpi["inflation_mom"].rolling(12).mean()
    return cpi[["Month", "inflation_yoy", "inflation_shock"]]


def _prep_monthly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    out["Month"] = pd.to_datetime(out["Month"])
    return out.drop_duplicates(subset=["Month"]).sort_values("Month").reset_index(drop=True)


def assemble_macro(
    tb3ms_change: pd.DataFrame,
    unrate_monthly: pd.DataFrame,
    inflation: pd.DataFrame,
    yield_slope: pd.DataFrame,
    credit_spread: pd.DataFrame,
    vix_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join all monthly macro tables into a single aligned frame."""
    tb_level = _prep_monthly(tb3ms_change, ["Month", "TB3MS"])
    tb_change = _prep_monthly(tb3ms_change, ["Month", "dTB3MS"])
    unrate = _prep_monthly(unrate_monthly, ["Month", "UNRATE"])
    infl = _prep_monthly(inflation, ["Month", "inflation_yoy", "inflation_shock"])
    ys = _prep_monthly(yield_slope, ["Month", "yield_slope"])
    cs = _prep_monthly(credit_spread, ["Month", "credit_spread"])
    vix = _prep_monthly(vix_monthly, ["Month", "VIXCLS"])

    macro = tb_level
    for df in [tb_change, unrate, infl, ys, cs, vix]:
        macro = pd.merge(macro, df, on="Month", how="inner")
    return macro.sort_values("Month").reset_index(drop=True)


def lag_macro(macro: pd.DataFrame, cols: Iterable[str] = MACRO_SRC_COLS) -> pd.DataFrame:
    """Shift each macro column by one month (no-look-ahead)."""
    out = macro.copy()
    for c in cols:
        out[c + "_lag1"] = out[c].shift(1)
    keep = ["Month"] + [c + "_lag1" for c in cols]
    return out[keep]


def merge_macro_into_panel(etf_ir: pd.DataFrame, macro_lagged: pd.DataFrame) -> pd.DataFrame:
    panel = etf_ir.copy()
    panel["Month"] = pd.to_datetime(panel["Month"])
    panel = panel.sort_values(["ETF", "Month"]).reset_index(drop=True)
    return panel.merge(macro_lagged, on="Month", how="left")
