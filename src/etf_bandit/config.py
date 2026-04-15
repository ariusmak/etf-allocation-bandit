"""Shared constants for the pipeline.

Kept identical to the values used in the notebooks so that modules and
notebooks produce the same numbers.
"""

TAG_COLS = [
    "is_equity",
    "is_international",
    "is_gov",
    "is_credit",
    "is_macro",
]

MACRO_COLS = [
    "TB3MS_lag1",
    "dTB3MS_lag1",
    "UNRATE_lag1",
    "inflation_yoy_lag1",
    "inflation_shock_lag1",
    "yield_slope_lag1",
    "credit_spread_lag1",
    "VIXCLS_lag1",
]

# Unlagged macro source columns (used before the shift(1) step).
MACRO_SRC_COLS = [
    "TB3MS",
    "dTB3MS",
    "UNRATE",
    "inflation_yoy",
    "inflation_shock",
    "yield_slope",
    "credit_spread",
    "VIXCLS",
]

# Panel column names
MONTH_COL = "Month"
ETF_COL = "ETF"
GROWTH_COL = "Growth"
RET_COL = "excess_return"
BIN_COL = "outperformed"
RF_COL = "rf_month"

# Bernoulli baseline params
ESS_B = 10
ALPHA0 = 2.0
BETA0 = 2.0
EPSILON_B = 1e-8

# Student-t params
NU = 5
VOL_WINDOW = 12
MC_N = 2000
EPS_S = 0.0005
TAU0 = 0.05

# Reproducibility
DEFAULT_SEED = 123

# Train / test split
SPLIT_DATE = "2014-12-31"
