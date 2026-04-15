"""Path helpers. Defaults resolve to `<repo_root>/data`.

Notebooks developed in Colab used hardcoded Drive paths. Prefer these helpers
when writing new code so the same code runs locally and in Colab (override
`DATA_DIR` via env var if needed).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("ETF_BANDIT_DATA_DIR", REPO_ROOT / "data"))
COV_DIR = DATA_DIR / "Covariates"
ETF_DIR = DATA_DIR / "etf"

FINAL_DATASET = DATA_DIR / "final_dataset.csv"
TAGGED_FINAL = COV_DIR / "tagged_final.csv"
ETF_IR_CSV = COV_DIR / "ETF+IR.csv"
META_CSV = COV_DIR / "meta.csv"
