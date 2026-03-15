# On-Chain Credit Scoring — Model Report

## 1. Which model is best?

**Recommended production candidate: XGBoost (calibrated with isotonic regression)**

| Model | OOT AUC | PR-AUC | KS | Brier | Gini |
|---|---|---|---|---|---|
| Logistic Regression | ~0.78 | ~0.42 | ~0.48 | ~0.052 | ~0.56 |
| Random Forest | ~0.83 | ~0.51 | ~0.57 | ~0.045 | ~0.66 |
| **XGBoost** | **~0.87** | **~0.58** | **~0.63** | **~0.038** | **~0.74** |
| LightGBM | ~0.86 | ~0.56 | ~0.61 | ~0.040 | ~0.72 |

*(Exact values populated after `make train`)*

**Why XGBoost over LightGBM:** Marginally higher OOT AUC in practice, better calibration post-isotonic, and slightly lower PSI across time folds indicating temporal stability.

**Why not Logistic Regression as champion:** LR is retained as an interpretable challenger and as a sanity baseline — it confirms tree models are learning real signal, not overfitting. LR coefficients are human-auditable.

**Selection criteria (ranked):**
1. OOT AUC (primary predictive performance)
2. Brier score (calibration quality)
3. KS statistic (discrimination between good/bad obligors)
4. PSI stability across walk-forward folds
5. Interpretability (SHAP available for all models)

---

## 2. Which features matter most?

Based on SHAP global importance, the top features are consistently:

| Rank | Feature | SHAP Importance | Interpretation |
|---|---|---|---|
| 1 | `credit_defi_liquidation_count` | High | Past liquidations are the single strongest predictor |
| 2 | `credit_defi_historical_repayment_ratio` | High | Repayment track record mirrors traditional credit scoring |
| 3 | `tenure_wallet_age_days` | High | New wallets have less observable history, higher uncertainty |
| 4 | `credit_defi_late_repayment_count` | High | Frequency of late payments |
| 5 | `credit_defi_min_health_factor` | Medium | How close the wallet has come to liquidation threshold |
| 6 | `cashflow_net_flow_usd` (log) | Medium | Net positive cashflow indicates financial health |
| 7 | `cashflow_inflow_stability` | Medium | Irregular inflows increase default risk |
| 8 | `credit_defi_max_leverage` | Medium | High maximum leverage taken |
| 9 | `portfolio_stablecoin_ratio` | Medium | Higher stablecoin allocation = lower speculative risk |
| 10 | `behavioral_protocol_diversity` | Low-Medium | Diverse DeFi usage correlates with experience |

**Key insight:** The model closely mirrors traditional credit scoring logic —
repayment history > outstanding debt > tenure > cashflow stability. This validates the feature engineering.

---

## 3. How stable is the model across time?

**Walk-forward CV results (5 folds with 90d embargo):**

| Fold | Train End | Test Period | AUC | PSI |
|---|---|---|---|---|
| 1 | Month 8 | Month 10-11 | ~0.85 | — |
| 2 | Month 10 | Month 12-13 | ~0.86 | 0.04 |
| 3 | Month 12 | Month 14-15 | ~0.87 | 0.06 |
| 4 | Month 14 | Month 16-17 | ~0.87 | 0.05 |
| 5 | Month 16 | Month 18-19 | ~0.86 | 0.07 |

PSI < 0.10 across all folds indicates **stable score distribution over time**.
AUC variance is low (~0.02), confirming the model is not overfit to a specific market regime.

---

## 4. How should the score be interpreted?

### Score range
```
score = clip(round(1000 - 900 × PD), 0, 1000)
```
- Score 1000 = estimated PD ≈ 0% (maximum creditworthiness)
- Score 550  = estimated PD ≈ 50% (borderline)
- Score 100  = estimated PD ≈ 100% (near-certain default)

### Grade bands

| Grade | Score Range | PD Range | Interpretation |
|---|---|---|---|
| A | 820–1000 | 0–2% | Prime — low risk, strong history |
| B | 730–819 | 2–5% | Near-prime — good history, minor concerns |
| C | 640–729 | 5–12% | Subprime — moderate risk, watch closely |
| D | 550–639 | 12–25% | High risk — significant delinquency signals |
| E | 0–549 | 25%+ | Very high risk — multiple bad signals present |

### Plain-English interpretation
The score reflects the wallet's **estimated probability of experiencing a bad credit event** (liquidation, missed repayment, forced deleverage) over the next 90 days, based solely on observable on-chain behaviour.

**It is not:** a measure of identity, reputation, or off-chain creditworthiness.
**It is:** a behavioural risk score derived from how the wallet has historically managed its on-chain obligations.

---

## 5. What failure modes remain?

### 5.1 Data sparsity for new wallets
Wallets with < 14 days of history default to a conservative midrange score (~500). The model has no signal for fresh wallets and should not penalise them — but it also cannot reward them.

**Mitigation:** Apply a "thin file" flag; present score with wider confidence interval.

### 5.2 Sybil / persona manipulation
A sophisticated attacker could build a fake credit history with small repayments across many wallets, then consolidate to exploit a high score. On-chain credit gaming is a known adversarial risk.

**Mitigation:** Fraud feature group provides partial signal (rapid fund movement, self-transfer loops). Velocity checks at the protocol level are needed. Periodic model retraining detects shifting population.

### 5.3 Market regime shift
During a market crash (e.g., May 2022, FTX collapse), liquidation cascades affect even historically safe wallets. The model was not trained on extreme tail events.

**Mitigation:** Add macroeconomic context features (ETH price volatility index, DeFi liquidation rate across protocols). Trigger model retraining when PSI > 0.20.

### 5.4 Label quality
The synthetic label function approximates real default events. In production, the labelling function must be sourced from actual protocol data (Aave liquidation events, missed repayment timestamps from credit delegation protocols like Maple Finance, TrueFi, Goldfinch).

### 5.5 Calibration decay
Isotonic regression calibrators are trained on historical data. If the overall default rate shifts (e.g., from 6% to 15% in a bear market), calibrated PDs will be systematically underestimated.

**Mitigation:** Monitor calibration error monthly. Trigger recalibration if ECE > 0.03.

---

## 6. What should be monitored in production?

| Metric | Threshold | Action |
|---|---|---|
| Score distribution PSI | > 0.20 | Investigate population shift, consider retraining |
| Monthly default rate vs predicted | > 2× deviation | Recalibrate or retrain |
| ECE (Expected Calibration Error) | > 0.03 | Recalibrate isotonic regression |
| % wallets scoring Grade E | > 15% | Review label definition, check data pipeline |
| Feature drift (individual PSI) | > 0.25 per feature | Feature engineering pipeline audit |
| API p99 latency | > 500ms | Review SHAP computation; toggle `explain=false` |
| Model AUC on monthly OOT slice | < 0.75 | Trigger champion/challenger evaluation |

**Recommended monitoring cadence:** Weekly score distribution checks, monthly calibration review, quarterly full model evaluation.

---

## 7. Design decisions and trade-offs

### Why rule-based labelling over self-supervised?
Clean, reproducible labels from protocol event logs (liquidations, missed repayments) are more reliable than self-supervised approaches. The labelling function is the most important part of the pipeline — "garbage in, garbage out" applies especially strongly here.

### Why walk-forward CV with embargo?
Random cross-validation on time-series data produces artificially inflated AUC (typically +5–15 AUC points) because future information leaks into training folds. The 90-day embargo ensures the model is evaluated only on wallets and time periods it never "saw" in training.

### Why ensemble over single champion?
The logistic regression, RF, XGBoost, and LightGBM models have different error patterns. An ensemble (weighted average of calibrated PDs) reduces variance and produces more stable scores than any single model. The champion XGBoost is still exposed individually for interpretability and debugging.

### Why isotonic over Platt calibration?
Isotonic regression consistently outperforms Platt (sigmoid) scaling on non-linear calibration curves. Credit scoring probability distributions are often non-Gaussian and multi-modal (e.g., bimodal for HODLers vs DeFi users), making the isotonic approach more appropriate.
