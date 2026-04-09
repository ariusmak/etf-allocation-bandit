# Bayesian DDDM ETF Allocation (Contextual Student‑t Bandit)

A reproducible **Bayesian decision modelling** pipeline for monthly ETF allocation with an explicit cash option, **cold‑start hierarchical priors**, and a **contextual Student‑t bandit** that conditions decisions on lagged macroeconomic state.

> **What this is (recruiter version):**  
> I built an end‑to‑end system that goes from raw market + macro data → engineered panel dataset → ETF category tagging → online Bayesian learning → portfolio construction → walk‑forward backtesting with diagnostics.

---

## Highlights

- **Decision-first modeling:** This project is about **sequential allocation rules**, not return prediction in isolation.
- **Time-varying ETF universe:** ETFs enter/exit through time; the policy adapts to the available set each month.
- **Cold-start handled correctly:** New ETFs get **tag-informed hierarchical priors** at birth.
- **Contextual bandit:** ETF expected excess returns depend on macro state via per‑ETF Bayesian regression.
- **Risk-aware scoring:** Uses **Student‑t predictive tails** and penalizes uncertainty (predictive variance).
- **Interpretable policy layer:** Scores → cash vs ETF exposure → within‑ETF weights (with Top‑N selection).
- **Strict no look‑ahead:** Decisions at month *t* are applied to returns in month *t+1*.

---

## Data

The working dataset is a monthly ETF panel with:

- **578 ETFs**
- **356 months** from **1993‑01‑31 → 2022‑08‑31**
- **74,168 rows** (one per *(Month, ETF)*)

### Core columns

| Group | Columns |
|------|---------|
| Identifiers | `Month`, `ETF` |
| Returns | `Open`, `Close`, `Growth = Close/Open - 1` |
| Cash benchmark | `rf_month` (monthly risk‑free rate) |
| Excess return targets | `excess_return = Growth - rf_month`, `outperformed = 1{excess_return > 0}` |
| Engineered context | `months_outperforming` (leak‑free prior outperformance streak; not used in the final model but kept for extensions) |
| Lagged macro covariates | `TB3MS_lag1`, `dTB3MS_lag1`, `UNRATE_lag1`, `inflation_yoy_lag1`, `inflation_shock_lag1`, `yield_slope_lag1`, `credit_spread_lag1`, `VIXCLS_lag1` |
| ETF category tags (static) | `is_equity`, `is_international`, `is_gov`, `is_credit`, `is_macro` |

---

## Notebook Workflow

These notebooks were developed in Colab, but are structured as an end‑to‑end pipeline:

1. **`Copy_of_Tagging (1) (1).ipynb`**  
   Creates ETF tags used for cold‑start priors:
   - pulls ETF metadata (`category`, `shortName`, `longName`) via **yfinance**
   - applies regex keyword rules to assign 5 tags (equity / intl / gov / credit / macro)
   - flags ambiguous cases (`needs_review`) and supports **manual overrides**
   - produces `tagged_final.csv`

2. **`Building_dataset (1).ipynb`**  
   Builds the final modeling panel:
   - starts from a prebuilt ETF monthly panel (prices/returns + risk‑free rate)
   - engineers `months_outperforming` (prior streak, leak‑free)
   - constructs macro features (inflation, spreads, VIX, yield slope), then **lags by 1 month**
   - merges macro covariates + tags → exports `final_dataset.csv`

3. **`Non-Contextual_.ipynb`**  
   Groundwork + ablations (what I tried before the final model):
   - **Bernoulli sign-bandit** (Beta‑Bernoulli) baseline
   - **Non‑contextual Student‑t** magnitude model
   - risk adjustment (penalizing high volatility)
   - Top‑N selection to enforce capital scarcity and reduce “over-diversification”
   - hyperparameter sweeps for sizing strength (`k`) and selection (`N`)

4. **`Contextual_studentt (1).ipynb`**  
   Final model: **Contextual Student‑t Top‑N** with predictive variance + warm‑started walk‑forward evaluation.

---

## Methodology

### 1) Decision timing (no look-ahead)

At each month **t**:

- observe data at month **t** for all ETFs in the universe \(A_t\)
- update posteriors using information available at **t**
- compute scores \(s_{i,t}\)
- convert scores into weights to **hold in month t+1**
- apply realized returns in **t+1** and update wealth

This is enforced everywhere in the simulation loop.

---

### 2) Tag-based hierarchical priors (cold start)

ETF category tags are used **only when an ETF first appears**.

- Maintain global and per‑tag running means of `excess_return`
- At ETF “birth”, initialize its **intercept** prior mean as:

```
m0 = global_mean + Σ(tag present) (tag_mean - global_mean)
```

In the contextual model, **macro slopes start at 0**, making the cold‑start prior:
- conservative
- interpretable
- learnable quickly

---

### 3) Global volatility regime estimate

Each month uses a pooled volatility estimate:

- `s_t = std(excess_return)` over the **trailing 12 months**, pooled cross‑sectionally  
- strictly uses months **< t** to avoid leakage

This acts as a regime-adaptive noise scale and helps normalize “risk appetite” over time.

---

## Model Development Path (Groundwork)

I did not jump straight to the final model—these were the critical intermediate findings:

### Baseline 1 — Bernoulli sign bandit (direction only)

- Outcome: `outperformed ∈ {0,1}`
- Posterior per ETF: `Beta(alpha_i, beta_i)`
- Score: `max(0, P(outperform) − p1)`
- Finding: can perform well in frictionless backtests due to broad participation, but it ignores magnitude.

### Baseline 2 — Student‑t magnitude model (non-contextual)

- Posterior per ETF over mean excess return `μ_i` with a Normal update
- Predictive tails modeled with Student‑t (`ν = 5`)
- Score (V1): `E[max(X,0)]` where `X = μ_i + s_t * T_ν`

**Issue discovered:** naive upside scoring can overweight high‑volatility ETFs (fat tails inflate upside).

### Improvement — Risk adjustment

Score (V2): `E[max(X,0)] / s_t`  
This normalizes expected upside by the current volatility regime.

### Improvement — Top‑N selection

Even after risk adjustment, the model tended to hold **hundreds of ETFs** (scores were broadly positive).  
To make the policy express conviction and reduce dilution:

- rank ETFs by score each month
- keep only **Top‑N** positive scores
- allocate only to that subset

This became the backbone of the final model.

---

## Final Model: Contextual Student‑t Top‑N with Predictive Variance

### Context vector

For each decision month `t`, macro covariates are constant across ETFs and form:

- `z_t = [1, macro_1, ..., macro_8]` (intercept + 8 standardized macro features)

**Standardization:** all macro columns are **z‑scored using TRAIN means/std** to prevent leakage.

### Per‑ETF Bayesian regression

For each ETF `i`:

- prior: `β_i ~ N(m_i, V_i)`
- likelihood: `x_{i,t} | β_i ~ N(z_tᵀ β_i, s_t²)`

Posterior update (information form):

```
V_post = (V^-1 + (1/s_t^2) z zᵀ)^-1
m_post = V_post (V^-1 m + (1/s_t^2) z x)
```

### Predictive distribution & scoring

We score each ETF using a Student‑t predictive distribution that **penalizes uncertainty**:

- predicted mean: `μ̂ = zᵀ m`
- predictive variance from coefficient uncertainty: `zᵀ V z`
- predictive scale:
  - `scale = sqrt(s_t^2 + zᵀ V z)`

Then simulate:

- `X = μ̂ + scale * T_ν`

Score:

- `score = E[max(X, 0)] / scale`

Estimated via Monte Carlo (`mc_N = 2000` draws).  
A small floor `eps_s` is applied to avoid allocating on MC noise.

### Top‑N selection

Only the **Top‑N** ETFs by score receive capital; the rest are set to score 0.

---

## Portfolio Policy (Scores → Weights)

Given nonnegative scores at month `t`:

1. Compute mean score `S_t`
2. Determine total ETF exposure using a smooth saturating rule:

```
W_etf = (k * S_t) / (1 + k * S_t)      # in (0,1)
w_cash = 1 - W_etf
```
Where k is a hyperparameter determining market participation aggressiveness

3. Allocate within ETFs proportionally:

```
w_i = W_etf * score_i / Σ score_j
```

4. If all scores are 0 → allocate **100% cash**

### Diagnostics logged each month

- `wealth`
- `W_etf` and `w_cash`
- `turnover = 0.5 * Σ |w_t - w_{t-1}|` (including cash)
- `n_bet` (# of ETFs with positive weight)
- `top5_share` (concentration)

---

## Backtesting Setup

- **Train/Test split:**  
  - Train: **1993‑01 → 2014‑12** (264 months)  
  - Test: **2015‑01 → 2022‑08** (92 months)

- **Warm start evaluation:**  
  Test runs start with **TRAIN posteriors** (belief state + ETF states) to preserve chronology.

- **Benchmarks:**  
  - Cash (risk‑free compounding)
  - Buy & Hold SPY

---

## Results Snapshot (Out-of-sample)

Best TRAIN hyperparameters (grid searched in notebook):

- **N = 5**, **k = 800**

### TEST (2015‑01 → 2022‑08), starting wealth = 1000

| Strategy | Final Wealth | Mean Monthly Return | Beats SPY (months) |
|---|---:|---:|---:|
| **Contextual Student‑t Top‑N (N=5, k=800)** | **1941.27** | **0.8726%** | **52.2%** |
| Buy & Hold SPY | 1766.08 | 0.7207% | — |
| Cash (RF) | 1066.14 | — | — |

**Relative performance (TEST):**
- vs SPY final wealth: **+9.9%**
- vs Cash final wealth: **+82.1%**

Non-contextual Student-t Model underperformed SPY, contextual Student-t consistently beats SPY, showing that macro features contain real signal
---

## How to Reproduce

### Install dependencies

```bash
pip install numpy pandas scipy matplotlib yfinance
```

### Option A — Run with the prepared panel

1. Place `final_dataset.csv` in `data/`
2. Run:
   - `Non-Contextual_.ipynb` (ablations)
   - `Contextual_studentt (1).ipynb` (final model + grid search)

### Option B — Rebuild the dataset

1. Generate / obtain the raw macro CSVs (FRED-style) used in the build notebook:
   - `TB3MS.csv`, `DGS10.csv`, `BAA.csv`, `VIXCLS.csv`, `UNRATE.csv`, `CPIAUCSL.csv`
2. Run `Copy_of_Tagging (1) (1).ipynb` to produce `tagged_final.csv`
3. Run `Building_dataset (1).ipynb` to merge:
   - ETF monthly returns + risk-free rate
   - lagged macro covariates
   - tags  
   → exports `final_dataset.csv`

> Note: the notebooks were developed in Google Colab and use Drive paths; for GitHub, replace paths with local `./data/...`.

---

## Tech Stack

- Python (NumPy, Pandas)
- SciPy (Student‑t sampling)
- Matplotlib (diagnostic plots)
- yfinance (metadata-based tagging)
- Notebook-first workflow (Colab compatible)

---

## Limitations & Next Steps

- **Transaction costs not modeled** (turnover is tracked to support adding costs)
- **Global volatility** is pooled cross-sectionally (could be ETF- or sector-specific)
- Per‑ETF posteriors are independent beyond the cold‑start prior (could add hierarchical pooling)
- Potential extensions:
  - cost-aware policy layer
  - regime-dependent priors
  - richer contextual models (regularization across ETFs, factor structure)

---

## Contact

If you're a recruiter or hiring manager and want to discuss this work, feel free to reach out via GitHub / LinkedIn.
