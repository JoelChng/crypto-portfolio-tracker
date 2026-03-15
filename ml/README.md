# On-Chain Credit Scoring — ML Pipeline

Production-oriented ML pipeline that takes historical on-chain wallet behaviour as input and outputs a continuous credit score (0–1000), probability of default, and risk grade (A–E) with SHAP-based explanations.

## Architecture

```
run_pipeline.py
  ├── Stage 1: data/synthetic_generator.py   → 5,000 wallets, 1.4M events
  ├── Stage 2: features/assembler.py         → 76 features × 50k snapshots
  ├── Stage 3: training/train_all.py         → LR + RF + XGBoost + LightGBM
  ├── Stage 4: explainability/shap_explainer → Global + per-wallet SHAP
  └── Stage 5: api/app.py                    → FastAPI scoring server
```

## Models (OOT results, 90d horizon)

| Model | AUC | PR-AUC | KS | F1 | Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.775 | 0.193 | 0.472 | 0.287 | 0.073 |
| **Random Forest** ⭐ | **0.807** | **0.324** | **0.488** | **0.391** | **0.067** |
| XGBoost | 0.798 | 0.279 | 0.479 | 0.320 | 0.067 |
| LightGBM | 0.793 | 0.269 | 0.461 | 0.297 | 0.069 |

Champion: **Random Forest** (highest OOT AUC + PR-AUC)

## Feature Groups (76 total)

1. **Wallet Tenure** — age, dormancy, activity streaks
2. **Cashflow** — inflow/outflow stability, stablecoin ratio
3. **Behavioral** — protocol diversity, recency-weighted activity
4. **Credit/DeFi** — repayment ratio, liquidation count, leverage
5. **Portfolio** — HHI concentration, bluechip/longtail split
6. **Fraud/Sybil** — rapid fund movement, bridge patterns
7. **Temporal** — burstiness, gap CV, activity trend

## Scoring Formula

```
score = clip(round(1000 - 900 × PD_90d), 0, 1000)

A: 820–1000  (PD < 2%)
B: 730–819   (PD 2–5%)
C: 640–729   (PD 5–12%)
D: 550–639   (PD 12–25%)
E: 0–549     (PD > 25%)
```

## Quick Start

```bash
# Install dependencies (requires conda/miniforge)
conda create -n mlenv python=3.11 -y
conda activate mlenv
conda install -c conda-forge xgboost lightgbm -y
pip install -r requirements.txt

# Run full pipeline
make all

# Or stage by stage
make data      # generate synthetic data
make features  # feature engineering (~20 min)
make train     # train all 4 models
make explain   # SHAP global importance
make api       # start FastAPI server on :8000
```

## API

```bash
# Score a wallet
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"wallet_address":"0x...","events":[...],"explain":true}'

# Response
{
  "wallet": "0x...",
  "score": 742,
  "pd_90d": 0.031,
  "risk_grade": "B",
  "top_reason_codes": [
    {"code":"RC10","text":"Strong repayment history (94%) is a key positive indicator."},
    ...
  ]
}
```

## Design Decisions

- **Walk-forward CV with 90-day embargo** — prevents temporal leakage
- **Isotonic calibration** — better-calibrated PDs than Platt scaling for non-Gaussian distributions
- **Ensemble weighted average** — reduces variance across model families
- **Single temporal gate** (`events_before_snapshot`) — no leakage possible by construction

See [reports/model_report.md](reports/model_report.md) for full analysis.
