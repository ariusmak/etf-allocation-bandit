"""End-to-end runner for the final Contextual Student-t Top-N model.

Loads `data/final_dataset.csv`, applies the train-only z-score, runs the
train grid search (or a single fixed N/k if --fast), warm-starts the test
run with the best train posteriors, and writes artifacts to `outputs/`.

Example:
    python scripts/run_contextual.py --fast --N 5 --k 800
    python scripts/run_contextual.py --grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from etf_bandit import backtest, panel
from etf_bandit.config import DEFAULT_SEED, MACRO_COLS, SPLIT_DATE, TAG_COLS
from etf_bandit.paths import FINAL_DATASET, REPO_ROOT

OUTPUTS = REPO_ROOT / "outputs"

K_GRID = [25, 100, 200, 800]
N_GRID = [5, 10, 20, 50, 100]


def load_panel(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.sort_values(["Month", "ETF"]).reset_index(drop=True)
    for c in TAG_COLS:
        df[c] = df[c].astype(int)
    panel.validate_panel(df)
    return df


def prep_train_test(df: pd.DataFrame):
    df_train, df_test = panel.train_test_split(df, split_date=SPLIT_DATE)
    mean, std = panel.fit_zscore(df_train, cols=MACRO_COLS)
    df_train = panel.apply_zscore(df_train, mean, std, cols=MACRO_COLS)
    df_test = panel.apply_zscore(df_test, mean, std, cols=MACRO_COLS)
    return df_train, df_test


def grid_search(df_train: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    runs = {}
    for N in N_GRID:
        for k in K_GRID:
            print(f"  TRAIN  N={N:<3}  k={k}", flush=True)
            res = backtest.run_contextual_topN(
                df_train, k=k, N=N, initial_wealth=1000.0, rng=rng,
            )
            runs[(N, k)] = res
            w = res["wealth"]
            rows.append({
                "N": int(N), "k": float(k),
                "final_wealth": float(w["wealth"].iloc[-1]),
                "mean_W_etf": float(w["W_etf"].mean()),
                "mean_turnover": float(w["turnover"].mean()),
                "mean_top5_share": float(w["top5_share"].mean()),
                "mean_n_bet": float(w["n_bet"].mean()),
            })
    summary = pd.DataFrame(rows).sort_values("final_wealth", ascending=False).reset_index(drop=True)
    return summary, runs


def pick_best(summary: pd.DataFrame) -> tuple[int, float]:
    best = summary.sort_values(["final_wealth", "mean_turnover"], ascending=[False, True]).iloc[0]
    return int(best["N"]), float(best["k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip grid, use fixed --N/--k")
    ap.add_argument("--grid", action="store_true", help="run full train grid search")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--k", type=float, default=800.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--data", type=Path, default=FINAL_DATASET)
    ap.add_argument("--outputs", type=Path, default=OUTPUTS)
    args = ap.parse_args()

    if not (args.fast or args.grid):
        ap.error("pick one of --fast or --grid")

    args.outputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading panel from {args.data}")
    df = load_panel(args.data)
    print(f"  {len(df):,} rows, {df['ETF'].nunique()} ETFs, "
          f"{df['Month'].min().date()} -> {df['Month'].max().date()}")

    df_train, df_test = prep_train_test(df)
    print(f"Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")

    if args.grid:
        summary, runs = grid_search(df_train, rng)
        summary.to_csv(args.outputs / "train_grid.csv", index=False)
        N_star, k_star = pick_best(summary)
        belief_tr = runs[(N_star, k_star)]["belief_state"]
        states_tr = runs[(N_star, k_star)]["states"]
    else:
        print(f"TRAIN single run  N={args.N}  k={args.k}")
        res_tr = backtest.run_contextual_topN(
            df_train, k=args.k, N=args.N, initial_wealth=1000.0, rng=rng,
        )
        N_star, k_star = args.N, args.k
        belief_tr = res_tr["belief_state"]
        states_tr = res_tr["states"]

    print(f"Best / chosen:  N={N_star}  k={k_star}")
    print("TEST (warm-started)")
    res_test = backtest.run_contextual_topN(
        df_test, k=k_star, N=N_star, initial_wealth=1000.0,
        belief_state_init=belief_tr, states_init=states_tr, rng=rng,
    )
    wealth = res_test["wealth"]
    wealth.to_csv(args.outputs / "test_wealth.csv", index=False)
    res_test["weights"].to_csv(args.outputs / "test_weights.csv", index=False)

    print("\n=== TEST summary ===")
    print(f"N={N_star}  k={k_star}")
    print(f"final_wealth     = {wealth['wealth'].iloc[-1]:.2f}")
    print(f"mean_monthly_ret = {wealth['port_ret'].mean() * 100:.4f}%")
    print(f"mean_W_etf       = {wealth['W_etf'].mean():.3f}")
    print(f"mean_turnover    = {wealth['turnover'].mean():.3f}")
    print(f"mean_top5_share  = {wealth['top5_share'].mean():.3f}")

    # Benchmarks
    cash = backtest.run_cash(df_test, 1000.0)
    spy = backtest.run_buy_hold_spy(df_test, 1000.0)
    print(f"cash_final       = {cash['wealth'].iloc[-1]:.2f}")
    if len(spy):
        print(f"spy_final        = {spy['wealth'].iloc[-1]:.2f}")
    print(f"\nArtifacts written to {args.outputs}/")


if __name__ == "__main__":
    main()
