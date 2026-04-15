"""Pooled cross-sectional volatility estimate s_t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MONTH_COL, RET_COL, VOL_WINDOW


def global_volatility(df: pd.DataFrame, month_t, vol_window: int = VOL_WINDOW) -> float:
    """Sample std of excess returns over the trailing `vol_window` months BEFORE t.

    Strictly uses months < t so no look-ahead. Returns NaN if there is not
    enough history; otherwise a float floored at 1e-8.
    """
    months = sorted(df[MONTH_COL].unique())
    t_idx = months.index(month_t)
    if t_idx < vol_window:
        return np.nan

    trailing = months[t_idx - vol_window : t_idx]
    x = df.loc[df[MONTH_COL].isin(trailing), RET_COL].astype(float).values
    s = float(np.nanstd(x, ddof=1))
    return max(s, 1e-8)
