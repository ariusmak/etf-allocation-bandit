"""Tag-belief layer used for cold-start (birth) priors.

Maintains pooled means of `excess_return` globally and per tag, observed
strictly up through the current decision month. At ETF birth, the intercept
prior mean is initialized as `m0 = m_global + sum_{tag present} (m_tag - m_global)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ETF_COL, RET_COL, TAG_COLS


def init_belief_state() -> dict:
    return {
        "global_n": 0,
        "global_sum": 0.0,
        "tag_n": {tag: 0 for tag in TAG_COLS},
        "tag_sum": {tag: 0.0 for tag in TAG_COLS},
    }


def update_beliefs(state: dict, df_t: pd.DataFrame) -> None:
    x = df_t[RET_COL].astype(float).values
    state["global_n"] += int(len(x))
    state["global_sum"] += float(np.nansum(x))

    tags_mat = df_t[TAG_COLS].astype(int).values
    for j, tag in enumerate(TAG_COLS):
        mask = tags_mat[:, j] == 1
        if mask.any():
            x_k = x[mask]
            state["tag_n"][tag] += int(len(x_k))
            state["tag_sum"][tag] += float(np.nansum(x_k))


def birth_prior_mean(state: dict, tags_row) -> float:
    """Mean-only birth prior using additive tag deviations."""
    n_g = state["global_n"]
    m_g = state["global_sum"] / n_g if n_g > 0 else 0.0

    m = float(m_g)
    for tag in TAG_COLS:
        if int(tags_row[tag]) == 1:
            n_k = state["tag_n"][tag]
            m_k = state["tag_sum"][tag] / n_k if n_k > 0 else m_g
            m += float(m_k - m_g)
    return float(m)
