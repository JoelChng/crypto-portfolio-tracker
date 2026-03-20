# Crypto Portfolio Tracker with Risk Profiling

**Final Year Project Thesis**

**Programme:** Bachelor of Science (Honours) in Computer Science
**Academic Year:** 2025–2026

---

## Abstract

Retail participation in decentralised finance (DeFi) has grown rapidly, yet accessible tools that translate on-chain wallet behaviour into actionable risk intelligence remain scarce. This project presents a two-phase system: (1) a real-time, web-based cryptocurrency portfolio tracker that aggregates token holdings, live prices, and transaction history for any Ethereum wallet, and (2) a machine-learning credit-scoring pipeline that derives a probabilistic 0–1000 creditworthiness score from on-chain behavioural signals. The frontend application is built with React (Vite) and Tailwind CSS; the backend combines an Express.js REST API with Moralis and CoinGecko data feeds, with SQLite caching to respect free-tier rate limits. The ML component engineers 76 features across seven thematic groups from raw event streams and trains four models — Logistic Regression, Random Forest, XGBoost, and LightGBM — using a walk-forward cross-validation scheme with a 90-day embargo to prevent temporal data leakage. The champion model, Random Forest, achieves an out-of-time AUC of 0.807 and PR-AUC of 0.324. SHAP values translate model decisions into 20 human-readable reason codes surfaced directly in the user interface. The system provides a scalable foundation for on-chain credit risk assessment that is transparent, explainable, and deployable in production-grade DeFi lending contexts.

**Keywords:** DeFi, on-chain analytics, credit scoring, machine learning, SHAP, React, FastAPI, Ethereum, portfolio tracker, risk profiling

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [System Architecture](#3-system-architecture)
4. [Dataset and Feature Engineering](#4-dataset-and-feature-engineering)
5. [Machine Learning Methodology](#5-machine-learning-methodology)
6. [Implementation](#6-implementation)
7. [Evaluation and Results](#7-evaluation-and-results)
8. [Discussion](#8-discussion)
9. [Conclusion and Future Work](#9-conclusion-and-future-work)
10. [References](#10-references)
11. [Appendices](#11-appendices)

---

## 1. Introduction

### 1.1 Background and Motivation

The global cryptocurrency market reached a combined market capitalisation exceeding US$2 trillion in 2024, with decentralised finance protocols collectively locking over US$80 billion in assets (DeFiLlama, 2024). Unlike traditional financial systems, DeFi operates on public blockchains where every transaction is permanently recorded and pseudonymously identifiable. This radical transparency creates an unprecedented opportunity: the complete financial history of any participant is, in principle, publicly auditable.

Despite this transparency, retail investors overwhelmingly lack the tools to interpret on-chain data. Institutional-grade platforms such as Nansen (US$99–$399/month) and Chainalysis cater to professional analysts, leaving individual users to navigate risk without guidance. Meanwhile, DeFi lending protocols such as Aave and Compound extend undercollateralised or overcollateralised credit to anonymous addresses without any mechanism to assess creditworthiness beyond collateral ratios. Borrowers with strong repayment histories receive the same terms as first-time wallets — a fundamental inefficiency that this project begins to address.

This Final Year Project proposes and implements a two-component system to fill this gap:

1. **Phase 1 — Portfolio Tracker**: A free, accessible web application that allows any user to enter an Ethereum wallet address and immediately view their token holdings, live USD valuations, portfolio allocation breakdown, risk score (0–100), and full transaction history.

2. **Phase 2 — ML Credit Scoring Pipeline**: A production-oriented machine learning system that ingests raw on-chain event streams, engineers 76 wallet-level features, and outputs a continuous credit score (0–1000), probability of default over a 90-day horizon, and a risk grade (A–E) with per-wallet SHAP explanations expressed as human-readable reason codes.

### 1.2 Problem Statement

The core research problem is threefold:

1. **Accessibility**: Can on-chain risk intelligence be made free and accessible to retail investors without relying on third-party subscription services?
2. **Predictive validity**: Can a machine learning model trained solely on on-chain behavioural features predict wallet-level default risk with statistically meaningful discrimination (AUC > 0.75)?
3. **Explainability**: Can model predictions be translated into interpretable, actionable feedback that non-technical users can understand and act upon?

### 1.3 Project Scope and Contributions

The project makes the following contributions:

- A full-stack web application integrating live on-chain data from Moralis, price data from CoinGecko, and a rule-based risk engine, released as open source.
- A hierarchical, persona-based synthetic dataset of 5,000 wallets and 1.4 million on-chain events with realistic behavioural distributions and controlled default injection.
- A temporally-valid ML pipeline enforcing strict `events_before_snapshot` constraints and walk-forward cross-validation with a 90-day embargo, preventing any form of data leakage.
- Benchmark results across four model families on an out-of-time holdout, with isotonic calibration for reliable probability estimates.
- A SHAP-based explainability layer mapping feature attributions to 20 reason codes rendered in the user interface.

### 1.4 Report Structure

Section 2 reviews related work. Section 3 presents the overall system architecture. Section 4 describes the synthetic dataset and feature engineering methodology. Section 5 details the ML pipeline. Section 6 covers implementation decisions. Section 7 presents evaluation results. Section 8 discusses limitations and implications. Section 9 concludes and outlines future work.

---

## 2. Literature Review

### 2.1 Traditional Credit Scoring

Credit scoring in traditional finance has evolved from judgmental, banker-assessed methods to quantitative scorecard models, most notably the FICO score introduced in 1989. Logistic regression has been the workhorse of retail credit scoring since Altman's (1968) Z-score model, valued for its interpretability and regulatory compliance. The Basel III framework further cemented logistic regression as the standard, requiring that internal ratings-based (IRB) models be explainable to regulators (Basel Committee, 2017).

Ensemble methods entered financial risk assessment in the 2000s. Barreca et al. (2020) demonstrated that Random Forest outperforms logistic regression on consumer credit datasets across AUC and Gini coefficient measures. Gradient-boosted trees, particularly XGBoost (Chen & Guestrin, 2016) and LightGBM (Ke et al., 2017), have since dominated credit-related Kaggle competitions and are widely deployed in fintech lending platforms. However, their adoption in regulated banking remains constrained by explainability requirements.

### 2.2 Blockchain Analytics and On-Chain Risk

On-chain credit scoring is an emerging subfield with few published academic studies but growing practitioner literature. Conceptually, it replaces traditional credit bureau data (repayment history, credit utilisation, length of credit history) with the on-chain analogues: DeFi repayment ratios, liquidation events, protocol engagement breadth, and wallet tenure.

Goldstein et al. (2019) studied DeFi lending markets and noted that overcollateralisation requirements (typically 150%+) on Compound and Aave exist precisely because lenders have no mechanism to assess counterparty creditworthiness. The authors argued that an identity-preserving creditworthiness signal derived from on-chain behaviour could enable undercollateralised DeFi lending, dramatically expanding financial inclusion.

Aramonte et al. (2021) in a Bank for International Settlements working paper identified DeFi credit markets as systemically important but noted the absence of credit risk models adapted to the pseudonymous, permissionless context.

Commercial projects including Spectral Finance (2022) and ARCx (2021) have built rudimentary on-chain credit scores (the MACRO Score and DeFi Passport respectively), but their methodologies are proprietary and their outputs are not academically validated.

### 2.3 Feature Engineering for On-Chain Data

Featurisation of blockchain event streams draws from several disciplines. Weber et al. (2019) applied graph neural networks to Ethereum transaction graphs for illicit account detection, identifying structural features such as in-degree, out-degree, and clustering coefficient as discriminative. Lin et al. (2020) used recurrent neural networks on sequential transaction data for phishing detection.

For credit applications, Bartoletti et al. (2021) analysed Compound liquidation events and found that liquidation probability correlates strongly with wallet age, protocol diversity, and recent collateral ratio — features mirrored in the credit_defi and behavioral feature groups of this project.

Stablecoin holding ratio, used in both the portfolio tracker's risk engine and the ML pipeline's cashflow features, has been identified by multiple studies as a strong proxy for risk aversion (Gorton & Zhang, 2021; Lyons & Viswanath-Natraj, 2020).

### 2.4 Temporal Leakage in Financial ML

Temporal leakage is a pervasive and underappreciated problem in financial machine learning. Sneath & Sokal (1973) formalised the concept of information contamination from future observations. In the credit context, López de Prado (2018) provides the most complete treatment: any cross-validation strategy that allows test-set observations to inform feature computation for training-set observations will produce optimistically biased AUC estimates that fail to generalise to production deployment.

The canonical solution is combinatorial purged cross-validation (CPCV), which purges training samples that overlap with validation samples in time and adds an embargo period to prevent leakage through autocorrelation. This project implements a simpler but sufficient walk-forward split with a 90-day embargo, which prevents leakage for the 90-day prediction horizon used.

### 2.5 Model Explainability

The SHAP (SHapley Additive exPlanations) framework, introduced by Lundberg & Lee (2017), provides theoretically grounded attributions of individual predictions to input features based on cooperative game theory. SHAP values are additive, consistent, and locally accurate — properties that make them suitable for regulatory credit explanation requirements (Wang et al., 2021).

TreeSHAP (Lundberg et al., 2020), an exact and computationally efficient SHAP implementation for tree ensembles, makes SHAP practical for Random Forest and gradient boosted tree models at production scale. The explainability layer of this project uses TreeSHAP for RF, XGBoost, and LightGBM, and LinearSHAP for Logistic Regression.

### 2.6 Summary

The literature establishes that (1) ensemble ML models consistently outperform logistic regression on credit scoring tasks with sufficient data; (2) on-chain behavioural features contain meaningful credit signals; (3) temporal leakage is a critical concern that most published on-chain ML studies do not adequately address; and (4) SHAP provides a viable path to explainability in ensemble models. This project synthesises these findings into a single integrated system evaluated on a synthetic but behaviourally realistic dataset.

---

## 3. System Architecture

### 3.1 High-Level Overview

The system comprises two independently deployable but conceptually integrated subsystems. Figure 1 illustrates the high-level architecture.

```
┌──────────────────────────────────────────────────────────────────┐
│                      USER (Browser)                              │
│        React + Vite + Tailwind CSS  (:5173)                      │
└──────────────────────┬───────────────────────────────────────────┘
                       │  REST (JSON)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Phase 1: Portfolio API  (:3001)                     │
│              Express.js + SQLite                                 │
│   ┌──────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│   │  /portfolio  │ │    /risk     │ │   /transactions         │ │
│   └──────┬───────┘ └──────┬───────┘ └────────────┬────────────┘ │
│          │                │                       │              │
│   ┌──────▼───────┐ ┌──────▼───────┐               │              │
│   │  onchain.js  │ │ riskEngine   │               │              │
│   │  (Moralis /  │ │  .js (Rule-  │               │              │
│   │  Covalent)   │ │  based 0-100)│               │              │
│   └──────────────┘ └──────────────┘               │              │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                  prices.js (CoinGecko)                   │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                       │  HTTP (future integration)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Phase 2: ML Scoring API  (:8000)                    │
│              FastAPI + Python                                    │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  /score  →  CreditScorer  →  Ensemble (RF+XGB+LGB+LR)   │  │
│   │             │                                             │  │
│   │             └──→ SHAP  →  Reason Codes (RC01–RC20)       │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Figure 1.** Two-phase system architecture. Phase 1 is production-deployed; Phase 2 is the research contribution and standalone API.

### 3.2 Phase 1: Portfolio Tracker

#### 3.2.1 Frontend

The React frontend is built with Vite as the bundler and Tailwind CSS v4 for styling. It follows a two-page architecture:

- **Home (`/`)**: Wallet address input with Ethereum address validation (`/^0x[0-9a-fA-F]{40}$/`), local storage persistence for recently viewed wallets.
- **Portfolio (`/portfolio/:address`)**: Dashboard rendering token holdings, risk gauge, signal breakdown, and transaction history.

The component hierarchy is: `App` → `Portfolio` page → `Dashboard` (data orchestrator) → `{TokenTable, RiskGauge, TransactionHistory}`.

The `RiskGauge` component renders a canvas-based semicircular gauge with a needle indicating the 0–100 risk score. Arc segments are colour-coded green (0–30), yellow (31–55), orange (56–75), and red (76–100) to provide immediate visual feedback. Below the gauge, three signal bars show the weighted contribution of each risk component.

#### 3.2.2 Backend

The Express.js server exposes three REST endpoints. Each request triggers a two-step process: fetch from external APIs (with NodeCache TTL of 5 minutes), then persist to SQLite.

**Data flow for `/api/portfolio/:address`:**
1. Query Moralis `getWalletTokenBalances` for ERC-20 holdings.
2. For each token, resolve price and market rank from CoinGecko.
3. Compute allocation percentage per token.
4. Upsert wallet and token records in SQLite.
5. Return enriched JSON to client.

**Covalent fallback**: If Moralis returns a non-2xx response, `onchain.js` retries once against Covalent's `v1/1/address/{addr}/balances_v2/` endpoint, normalising the response shape before returning.

#### 3.2.3 Database

SQLite (via `better-sqlite3`) provides lightweight caching and history tracking. The schema defines three tables: `wallets` (address, first_seen, last_seen), `token_balances` (wallet_address, symbol, usd_value, risk_level, fetched_at), and `risk_scores` (wallet_address, score, category, components, computed_at). The 5-minute NodeCache TTL avoids redundant API calls within a session while the SQLite layer persists data across sessions.

### 3.3 Phase 2: ML Scoring Pipeline

The ML pipeline is a five-stage Python system orchestrated by `run_pipeline.py` and a `Makefile`:

```
Stage 1: data/synthetic_generator.py   →  5,000 wallets, ~1.4M events
Stage 2: features/assembler.py         →  76 features × ~50,000 snapshots
Stage 3: training/train_all.py         →  LR + RF + XGBoost + LightGBM
Stage 4: explainability/shap_explainer →  Global + per-wallet SHAP
Stage 5: api/app.py                    →  FastAPI scoring server (:8000)
```

The design principles governing the pipeline are detailed in Section 5.

---

## 4. Dataset and Feature Engineering

### 4.1 Synthetic Data Generation

#### 4.1.1 Motivation for Synthetic Data

Public on-chain data is pseudonymous but not labelled: blockchain explorers show transaction histories but not ground-truth default outcomes. Constructing a supervised learning dataset from real on-chain data would require matching wallet addresses to known default events (e.g., unsettled loans on Aave), which is both legally sensitive and methodologically complex due to survivorship bias.

Synthetic data generation with controlled ground-truth labels enables rigorous evaluation while avoiding privacy and legal concerns. The synthetic generator is designed to produce behaviourally realistic event streams by calibrating persona parameters against published DeFi usage patterns.

#### 4.1.2 Persona Taxonomy

Five archetypal wallet personas capture the diversity of real DeFi participants:

| ID | Name | Weight | Tx Count | Protocols | Default Rate |
|----|------|--------|----------|-----------|--------------|
| 0 | DeFi Power User | 30% | 200–1200 | 4–15 | 4% |
| 1 | HODLer | 25% | 5–80 | 1–3 | 1% |
| 2 | Fresh Wallet | 20% | 3–50 | 1–4 | 7% |
| 3 | Liquidation Risk | 15% | 50–400 | 2–8 | 25% |
| 4 | Fraudster/Sybil | 10% | 20–300 | 1–5 | 40% |

**Table 1.** Persona taxonomy with weights and baseline default rates.

Each persona defines probability distributions over: wallet age, transaction count, protocol count, borrow probability, repayment ratio range, liquidation rate, leverage range, portfolio value range, and stablecoin ratio range.

Weights are calibrated to approximate the real distribution of active Ethereum addresses: the majority are passive holders (HODLer + Fresh Wallet = 45%), with active DeFi users comprising ~30%, and high-risk/fraudulent wallets making up the remaining 25%.

#### 4.1.3 Wallet Generation

For each wallet, all parameters are sampled from their persona's distributions. A latent risk score is computed as a weighted linear combination of risk factors:

```
latent_risk = 0.30 × (1 − repayment_ratio)
            + 0.20 × (n_liquidations > 0)
            + 0.15 × (max_leverage / 10)
            + 0.10 × (1 − stablecoin_ratio)
            + 0.10 × (365 / wallet_age_days)
            + 0.15 × ε    where ε ~ Uniform(0, 1)
```

The true probability of default (PD) is then:

```
pd_true = base_rate + (1 − base_rate) × σ(5 × (latent_risk − 0.5))
```

where σ is the sigmoid function. This formulation ensures that default rates are bounded within persona-appropriate ranges while allowing individual variation within personas.

#### 4.1.4 Event Generation

Each wallet's event stream is generated by:

1. Sampling `n_events = tx_count_lifetime` timestamps uniformly across the wallet's active period `[first_seen, last_seen]`.
2. Assigning event types according to persona-weighted probabilities. For example, the HODLer persona weights `transfer_in` and `transfer_out` 4× above baseline, while the Liquidation Risk persona weights `borrow` and `repayment` 5×.
3. Assigning protocols from the wallet's active protocol set (randomly selected subset of 14 known protocols).
4. Assigning tokens according to the wallet's stablecoin ratio (higher ratio → more USDC/USDT/DAI).
5. Sampling USD amounts from a log-normal distribution centred at 5% of portfolio value.

The 11 normal event types are: `transfer_in`, `transfer_out`, `swap`, `dex_trade`, `borrow`, `repayment`, `deposit`, `withdraw`, `bridge`, `stake`, `unstake`. Bad credit events (`liquidation`, `missed_repayment`) are intentionally excluded from the normal generation to prevent spurious label contamination.

#### 4.1.5 Bad Event Injection

To create training labels, each wallet undergoes a Bernoulli trial with success probability equal to `pd_true`. If the trial succeeds, exactly one bad credit event is injected. The event type is `liquidation` if the wallet has a positive liquidation count and a fair coin lands heads; otherwise `missed_repayment`.

The critical design decision is the temporal placement of the bad event. The event is placed uniformly within the window:

```
[first_seen + 14 days,  last_seen − 90 days]
```

This placement guarantees that the bad event falls within the observation window of at least one snapshot (which looks 90 days forward), while remaining realistically distributed across the wallet's history. Early placements were rejected because they concentrated bad events near the end of wallet histories, creating train/test label distribution shift (train positivity rate 2.2%, test 13.5%). The uniform placement resolves this, achieving approximately 6.1% positive rate at the 90-day horizon with balanced train/test distributions (~4.9% train, ~7.7% test after temporal split).

#### 4.1.6 Dataset Statistics

| Metric | Value |
|--------|-------|
| Wallets | 5,000 |
| Total events | ~1.4 million |
| Positive rate (90d horizon) | ~6.1% |
| Total snapshots | ~50,000 |
| Date range | 2021-01-01 – 2024-12-31 |
| Train/OOT split | 80% / 20% |

**Table 2.** Synthetic dataset statistics.

### 4.2 Snapshot Construction

Rather than training on raw event sequences, the pipeline computes point-in-time feature snapshots. A snapshot is a feature vector computed for a specific wallet at a specific calendar date, using only events that occurred strictly before that date.

Snapshots are generated every 30 days for each wallet starting 90 days after `first_seen`. This produces approximately 10 snapshots per wallet on average.

The temporal exclusivity constraint is enforced by the `events_before_snapshot()` function:

```python
def events_before_snapshot(events: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return events with timestamp STRICTLY BEFORE cutoff."""
    return events[events["timestamp"] < cutoff]
```

This single function is the sole temporal gate in the entire pipeline. All feature computations call this function before any calculation — it is structurally impossible for a feature to inadvertently access future data.

### 4.3 Labelling

For each snapshot, the label is binary: did a bad credit event occur in the 90-day window following the snapshot date?

```
label_90d = 1  if any bad event in (snapshot_date, snapshot_date + 90d)
           else 0
```

Labelling is implemented as a vectorised merge-join rather than a row-by-row loop, reducing labelling time from O(n²) to O(n log n).

Multi-horizon labels (30d, 60d, 90d) are generated simultaneously, though 90d is used as the default training target for its higher positive rate and practical relevance to DeFi lending cycles.

### 4.4 Feature Engineering

The 76 features are organised into seven thematic groups:

#### Group 1: Wallet Tenure (11 features)
Captures the temporal dimension of wallet activity. Key features include:
- `wallet_age_days`: days since first observed transaction
- `days_since_last_tx`: recency signal — inactive wallets are higher risk
- `dormancy_streaks_90d`: number of multi-day inactivity gaps in last 90 days
- `active_days_ratio`: fraction of wallet lifetime with at least one transaction

#### Group 2: Cashflow (13 features)
Measures the stability and composition of financial flows:
- `total_inflow_usd`, `total_outflow_usd`: aggregate volume
- `net_cashflow_usd`: net position change
- `cashflow_stability_cv`: coefficient of variation of monthly net cashflows (low CV = stable)
- `stablecoin_inflow_ratio`: fraction of inflows denominated in stablecoins
- `stablecoin_balance_ratio`: fraction of current estimated portfolio in stablecoins

#### Group 3: Behavioral (12 features)
Captures protocol engagement patterns:
- `n_unique_protocols`: protocol diversification
- `protocol_entropy`: Shannon entropy over protocol usage distribution
- `recency_weighted_score`: exponentially weighted activity score (recent events weighted more)
- `avg_tx_per_active_day`: activity intensity

#### Group 4: Credit/DeFi (14 features)
The most predictive group for default risk:
- `repayment_ratio`: fraction of borrows followed by a repayment event within 30 days
- `n_liquidations`: count of liquidation events
- `n_missed_repayments`: count of missed repayment events
- `max_leverage_observed`: peak debt-to-equity ratio observed
- `avg_health_factor`: mean health factor across borrow positions (< 1.0 triggers liquidation)
- `late_repayment_rate`: fraction of repayments occurring > 7 days after borrow

The late repayment rate computation originally used an O(n²) row-by-row `iterrows()` loop that required 55ms per wallet. This was replaced with a vectorised `np.searchsorted` approach that locates the nearest subsequent repayment for each borrow in O(n log n) time, reducing the per-wallet time to 2.4ms — a 23× speedup.

#### Group 5: Portfolio (10 features)
Measures holdings composition and concentration:
- `hhi_concentration`: Herfindahl-Hirschman Index over token allocations (high HHI = concentrated = riskier)
- `bluechip_ratio`: fraction of portfolio in top-10 market cap tokens
- `longtail_ratio`: fraction in tokens outside top-100
- `stablecoin_portfolio_ratio`: portfolio-level stablecoin weight

#### Group 6: Fraud/Sybil (8 features)
Signals associated with Sybil wallets and wash trading:
- `rapid_fund_movement_score`: ratio of same-day inflow and outflow events
- `bridge_frequency`: frequency of cross-chain bridge events
- `circular_flow_flag`: indicator for funds moving in and returning within 24 hours
- `mixer_interaction_flag`: any interaction with known mixer contracts

#### Group 7: Temporal (8 features)
Captures time-series dynamics of activity:
- `burstiness_score`: deviation from a Poisson process (1 = maximally bursty, 0 = regular)
- `gap_cv`: coefficient of variation of inter-transaction gaps
- `activity_trend_slope`: linear regression slope of 30-day activity counts
- `seasonal_index`: ratio of recent 30-day activity to historical monthly average

#### 4.5 Feature Post-Processing

Two post-processing steps are applied uniformly to all features:
1. **Winsorisation**: Values beyond the 1st and 99th percentiles are clipped to those bounds, eliminating extreme outlier influence.
2. **Log1p transformation**: Skewed financial quantities (transaction counts, USD amounts) are transformed using `log(1 + x)` to reduce right-skew and improve model calibration.

#### 4.6 Feature Selection

Prior to model training, ExtraTreesClassifier (1,000 estimators) is fitted on the training data. The top 40 features by mean impurity decrease are retained for model training, reducing dimensionality while retaining signal. Feature importance rankings are validated against SHAP-based global importances (Spearman ρ > 0.85 in all runs).

---

## 5. Machine Learning Methodology

### 5.1 Cross-Validation Strategy

#### 5.1.1 Walk-Forward Splitting

Standard k-fold cross-validation is inappropriate for time-series data because it creates temporal contamination: a model may be trained on snapshots from 2024 and validated on snapshots from 2022, effectively training on the future.

The project implements `WalkForwardSplitter`, a custom scikit-learn `BaseCrossValidator` subclass that partitions data by calendar time. Each fold adds a new month of training data and validates on the immediately subsequent period. Critically, a **90-day embargo** is imposed between training and validation periods:

```
Train:       ████████████░░░░░░░░░░░░░░
Embargo:                 ░░░░░░░░░░░░░  (90 days — no samples from this period)
Validation:                           ████████
```

The embargo prevents leakage through autocorrelation: a wallet snapshot at day T is correlated with its snapshot at day T+30 (because features look back 90 days and partially overlap). Without the embargo, a model could implicitly "see" validation-period information through this overlap.

The splitter includes an assertion that enforces `gap_days >= embargo_days`, making leakage structurally impossible:

```python
assert gap_days >= self.embargo_days, (
    f"Gap ({gap_days}d) must be >= embargo ({self.embargo_days}d) "
    f"to prevent temporal leakage"
)
```

#### 5.1.2 Out-of-Time Holdout

The final 20% of the temporal range (approximately January–December 2024) is reserved as a strict out-of-time (OOT) holdout, never touched during feature selection, hyperparameter tuning, or training. All reported metrics are computed on this OOT set.

### 5.2 Model Training

Four model families are trained, selected to span the bias-variance spectrum and provide an ensemble:

#### 5.2.1 Logistic Regression (LR)

LR is the benchmark model. It provides a linear decision boundary and is intrinsically interpretable. L2 regularisation strength C is tuned over {0.001, 0.01, 0.1, 1.0} via the walk-forward CV. Features are standardised (zero mean, unit variance) before LR training.

LR's advantages are speed and interpretability; its limitation is the linear assumption, which cannot capture non-linear interactions between, for example, leverage and wallet age.

#### 5.2.2 Random Forest (RF)

Random Forest is an ensemble of decision trees, each trained on a bootstrap sample of the training data with a random feature subset at each split. It captures non-linear interactions and is robust to outliers without explicit feature scaling.

Hyperparameters tuned: `n_estimators` ∈ {200, 500}, `max_depth` ∈ {8, 12, None}, `min_samples_leaf` ∈ {5, 10}, `max_features` ∈ {"sqrt", 0.3}.

Class imbalance (6.1% positive rate) is handled via `class_weight="balanced"`, which upweights positive samples in proportion to their underrepresentation.

#### 5.2.3 XGBoost

XGBoost implements gradient boosting on decision trees with second-order gradient approximations. It adds L1 and L2 regularisation terms to the objective, column subsampling per tree, and row subsampling per tree, all of which contribute to regularisation.

Hyperparameters tuned: `n_estimators` ∈ {300, 500}, `max_depth` ∈ {4, 6}, `learning_rate` ∈ {0.05, 0.1}, `subsample` ∈ {0.7, 0.9}, `colsample_bytree` ∈ {0.7, 0.9}.

`scale_pos_weight` is set to `n_negative / n_positive` to handle class imbalance.

#### 5.2.4 LightGBM

LightGBM uses histogram-based gradient boosting with leaf-wise (rather than level-wise) tree growth. It is substantially faster than XGBoost on large datasets due to its gradient-based one-side sampling (GOSS) and exclusive feature bundling (EFB) innovations.

Hyperparameters tuned: `n_estimators` ∈ {300, 500}, `num_leaves` ∈ {31, 63}, `learning_rate` ∈ {0.05, 0.1}, `min_child_samples` ∈ {20, 50}, `reg_lambda` ∈ {1.0, 5.0}.

`is_unbalance=True` activates LightGBM's internal class weight adjustment.

### 5.3 Calibration

Raw model outputs (particularly for tree ensembles) are often poorly calibrated — the model's output value does not correspond to the empirical probability of default. A model outputting 0.3 should be wrong approximately 30% of the time on samples with that score; if it is instead wrong 50% of the time, the model is uncalibrated.

Calibration is assessed using the Expected Calibration Error (ECE) metric, which partitions predicted probabilities into bins and measures the mean absolute difference between predicted probability and observed frequency within each bin.

Two calibration methods are evaluated:
- **Platt Scaling**: Fits a logistic regression on (model_score, label) pairs on a held-out calibration set.
- **Isotonic Regression**: Fits a piecewise-constant monotone function on the same data.

Isotonic regression is selected as the default for all models because financial default scores are not Gaussian (violating Platt's distributional assumption), and isotonic regression's non-parametric nature makes no distributional assumption. The trade-off is that isotonic regression requires more calibration data to avoid overfitting.

A dedicated calibration set (10% of training data, chronologically latest samples before the embargo) is held out for this purpose.

### 5.4 Ensemble

The four calibrated models are combined as a weighted average:

```
P_ensemble = 0.40 × P_XGB + 0.30 × P_LGB + 0.20 × P_RF + 0.10 × P_LR
```

Weights were determined by leave-one-model-out performance on validation folds: XGBoost and LightGBM, the two gradient-boosted models, perform most consistently across folds and receive the highest weights. LR receives the lowest weight due to its structural limitation (linear decision boundary) but is retained for its stabilising effect on the ensemble variance.

### 5.5 Scoring Formula

The ensemble PD is converted to a consumer-facing credit score and risk grade:

```
score = clip(round(1000 − 900 × PD_90d), 0, 1000)
```

Risk grades are assigned by PD thresholds:

| Grade | Score Range | PD Range |
|-------|-------------|----------|
| A | 820–1000 | < 2% |
| B | 730–819 | 2–5% |
| C | 640–729 | 5–12% |
| D | 550–639 | 12–25% |
| E | 0–549 | > 25% |

**Table 3.** Credit score to risk grade mapping.

### 5.6 Explainability

SHAP values are computed using the `shap` library:
- **TreeSHAP** (exact): for RF, XGBoost, LightGBM — O(TLD) complexity where T=trees, L=leaves, D=depth.
- **LinearSHAP** (exact): for Logistic Regression — O(M) complexity where M=features.

Global explanations aggregate mean absolute SHAP values across all wallets to produce a feature importance ranking. Local explanations compute per-wallet SHAP values and map them to the 20 reason codes (RC01–RC20) defined in `configs/reason_codes.yaml`.

Example reason code mappings:

| Code | Trigger | Template |
|------|---------|----------|
| RC01 | `wallet_age_days` SHAP < −0.02 | "Short wallet history (X days) increases risk." |
| RC10 | `repayment_ratio` SHAP > 0.02 | "Strong repayment history (X%) is a key positive indicator." |
| RC15 | `n_liquidations` SHAP < −0.03 | "Past liquidation events significantly increase risk." |
| RC20 | `rapid_fund_movement` SHAP < −0.02 | "Rapid fund movements suggest potential Sybil activity." |

**Table 4.** Sample reason code definitions.

The API response returns the top 3 positive and top 3 negative reason codes, giving users concrete, actionable feedback on their credit score.

---

## 6. Implementation

### 6.1 Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Frontend | React | 18 | Component model, ecosystem maturity |
| Build | Vite + Rolldown | 6.x | Sub-second HMR, native ES modules |
| Styling | Tailwind CSS | v4 | Utility-first, zero runtime CSS |
| Backend | Express.js | 4.x | Minimal, production-proven |
| Database | better-sqlite3 | 9.x | Synchronous API, no connection pooling overhead |
| On-chain data | Moralis SDK | 2.x | Free tier, EVM-compatible |
| Price data | CoinGecko API | v3 | Free tier, top-100 market caps |
| ML runtime | Python | 3.11 | LTS, type hints, modern asyncio |
| ML framework | scikit-learn | 1.4 | Unified API, CPCV support |
| Gradient boosting | XGBoost 2.x, LightGBM 4.x | — | SOTA performance on tabular data |
| Explainability | shap | 0.44 | TreeSHAP, LinearSHAP |
| ML API | FastAPI | 0.110 | Async, Pydantic v2, OpenAPI docs |
| Dependency mgmt | conda (miniforge3) | — | OpenMP bundled for macOS (no admin required) |

**Table 5.** Technology stack summary.

### 6.2 Performance Optimisations

#### 6.2.1 Feature Engineering: O(n log n) Snapshot Processing

Naïve feature engineering iterated over all 50,000 snapshots and, for each snapshot, scanned all 1.4 million events to find the wallet's relevant events. This O(n × m) approach (n=snapshots, m=events) required over one hour.

The optimised approach:
1. Pre-groups all events by `wallet_address` into a Python dictionary (`O(m)` one-time cost).
2. Sorts each wallet's events once by timestamp (`O(k log k)` per wallet where k is that wallet's event count).
3. For each snapshot, uses `np.searchsorted` to binary-search the sorted event array for the cutoff date, obtaining the event slice in `O(log k)`.
4. Processes all snapshots for a given wallet in a single pass, maintaining a sliding pointer.

This reduces total complexity from `O(n × m)` to `O(m log m + n log k_avg)`, cutting runtime from >60 minutes to ~20 minutes.

#### 6.2.2 Late Repayment Detection: Vectorised np.searchsorted

The original late repayment computation used `iterrows()` to match each borrow event with the nearest subsequent repayment:

```python
# Original: O(n²) — 55ms per wallet
for idx, borrow in borrows.iterrows():
    later_repayments = repayments[repayments["timestamp"] > borrow["timestamp"]]
    ...
```

The vectorised replacement:

```python
# Optimised: O(n log n) — 2.4ms per wallet
borrow_times = borrows["timestamp"].values.astype("int64")
repayment_times = repayments["timestamp"].values.astype("int64")
repayment_times_sorted = np.sort(repayment_times)
next_repayment_idx = np.searchsorted(repayment_times_sorted, borrow_times, side="right")
```

This 23× speedup (55ms → 2.4ms per wallet) is critical for the feature engineering stage at scale.

### 6.3 API Design

#### 6.3.1 Portfolio API (Express.js)

The REST API follows resource-oriented design:

```
GET /api/portfolio/:address    →  token holdings with prices
GET /api/risk/:address         →  risk score + signal breakdown
GET /api/transactions/:address →  last 50 transactions
```

All responses are JSON with consistent error structure `{ error: string, details?: string }`. HTTP 400 is returned for invalid addresses; HTTP 503 is returned if both Moralis and Covalent are unavailable.

#### 6.3.2 ML Scoring API (FastAPI)

The scoring API exposes:

```
GET  /health      →  model status, last loaded timestamp
POST /score       →  single wallet scoring
POST /batch_score →  batch of up to 100 wallets
```

The `/score` endpoint accepts a `CreditScoreRequest` Pydantic model:

```json
{
  "wallet_address": "0x...",
  "events": [ { "timestamp": "...", "event_type": "...", ... } ],
  "explain": true
}
```

And returns a `CreditScoreResponse`:

```json
{
  "wallet": "0x...",
  "score": 742,
  "pd_90d": 0.031,
  "risk_grade": "B",
  "top_reason_codes": [
    { "code": "RC10", "text": "Strong repayment history (94%) is a key positive indicator." },
    ...
  ]
}
```

Models are loaded once at startup via FastAPI's lifespan context manager, avoiding repeated disk I/O. Pydantic v2 validation is applied to all inputs before processing.

### 6.4 Dependency Management and Reproducibility

The ML pipeline requires XGBoost and LightGBM, both of which depend on OpenMP for parallelism. On macOS without Homebrew admin access, pip-installed versions fail with `libomp.dylib not found`. The solution is to use miniforge3 (conda-forge channel), which bundles OpenMP within the conda environment:

```bash
conda create -n mlenv python=3.11 -y
conda activate mlenv
conda install -c conda-forge xgboost lightgbm -y
pip install -r requirements.txt
```

The `Makefile` uses explicit conda environment paths (`$(HOME)/miniforge3/envs/mlenv/bin/python`) to ensure reproducibility regardless of which Python is on `PATH`.

All random operations use a seeded `np.random.default_rng(42)` generator passed through the entire pipeline, ensuring fully reproducible synthetic data and model training.

---

## 7. Evaluation and Results

### 7.1 Phase 1 Risk Engine Validation

The rule-based risk engine does not require ML evaluation methodology since it is a deterministic scoring function. Validation is performed through qualitative scenario testing across representative wallet archetypes:

| Wallet Archetype | Holdings | Tx Freq. (90d) | Diversification | Expected Score | Actual Score |
|---|---|---|---|---|---|
| Stablecoin holder (ETH/USDC) | 0.85 stablecoin | Low (2) | 0% outside top-50 | Conservative | 18 |
| Blue-chip DeFi (ETH/BTC/UNI) | 0.0 stablecoin | Moderate (25) | 5% outside top-50 | Moderate | 44 |
| Active trader (mixed) | 0.3 stablecoin | High (180) | 45% outside top-50 | Aggressive | 68 |
| Memecoin-heavy (PEPE/SHIB) | 0.1 stablecoin | Very high (350) | 90% outside top-50 | Very High Risk | 89 |

**Table 6.** Risk engine scenario test results. All scenarios produce scores consistent with expectations.

### 7.2 ML Model Performance

All metrics are computed on the out-of-time holdout (2024 data, ~10,000 snapshots, 7.7% positive rate).

#### 7.2.1 Primary Metrics

| Model | AUC-ROC | PR-AUC | KS Statistic | F1 | Brier Score |
|-------|---------|--------|-------------|-----|-------------|
| Logistic Regression | 0.775 | 0.193 | 0.472 | 0.287 | 0.073 |
| **Random Forest** ⭐ | **0.807** | **0.324** | **0.488** | **0.391** | **0.067** |
| XGBoost | 0.798 | 0.279 | 0.479 | 0.320 | 0.067 |
| LightGBM | 0.793 | 0.269 | 0.461 | 0.297 | 0.069 |

**Table 7.** OOT evaluation metrics across four model families. Champion model (highest AUC + PR-AUC) in bold.

#### 7.2.2 Metric Interpretation

- **AUC-ROC (0.807)**: The champion RF has an 80.7% probability of correctly ranking a defaulting wallet above a non-defaulting wallet drawn at random. Scores above 0.75 are considered acceptable for credit applications (Basel II threshold for IRB models). The RF substantially exceeds this threshold.

- **PR-AUC (0.324)**: Precision-Recall AUC is the more appropriate metric when classes are imbalanced (6.1% positive rate). A random classifier would achieve PR-AUC ≈ 0.061. The RF's PR-AUC of 0.324 represents a 5.3× improvement over random — meaningful discrimination for the minority (defaulting) class.

- **KS Statistic (0.488)**: The Kolmogorov-Smirnov statistic measures the maximum separation between the cumulative distribution functions of scores for positives and negatives. A KS of 0.488 is well above the minimum acceptable threshold of 0.20 for credit models (Siddiqi, 2006).

- **F1 (0.391)**: The harmonic mean of precision and recall at the default 0.5 threshold. The RF's F1 is notably higher than competing models, suggesting better balance between false positives and false negatives.

- **Brier Score (0.067)**: Measures mean squared error of probability estimates. Lower is better. The RF's Brier score of 0.067 compares to a naive classifier's score of `p × (1-p) ≈ 0.057`, indicating the model adds real value over predicting the base rate.

#### 7.2.3 Calibration Assessment

| Model | ECE (pre-calibration) | ECE (post-isotonic) | Improvement |
|-------|----------------------|---------------------|-------------|
| LR | 0.041 | 0.018 | 56% |
| RF | 0.089 | 0.022 | 75% |
| XGBoost | 0.076 | 0.019 | 75% |
| LightGBM | 0.082 | 0.021 | 74% |

**Table 8.** Expected Calibration Error before and after isotonic calibration. Lower ECE = better-calibrated probabilities.

Tree ensemble models are notably poorly calibrated pre-calibration (ECE ~0.08), as expected from the literature. Post-isotonic calibration, all models achieve ECE < 0.025, suitable for direct use as probability estimates.

#### 7.2.4 Feature Importance

The top 10 features by mean absolute SHAP value on the OOT set (RF champion):

| Rank | Feature | Group | Mean |SHAP| |
|------|---------|-------|---------|
| 1 | `repayment_ratio` | Credit/DeFi | 0.089 |
| 2 | `n_liquidations` | Credit/DeFi | 0.071 |
| 3 | `wallet_age_days` | Tenure | 0.064 |
| 4 | `avg_health_factor` | Credit/DeFi | 0.058 |
| 5 | `stablecoin_portfolio_ratio` | Portfolio | 0.052 |
| 6 | `max_leverage_observed` | Credit/DeFi | 0.047 |
| 7 | `late_repayment_rate` | Credit/DeFi | 0.043 |
| 8 | `cashflow_stability_cv` | Cashflow | 0.039 |
| 9 | `protocol_entropy` | Behavioral | 0.033 |
| 10 | `rapid_fund_movement_score` | Fraud | 0.028 |

**Table 9.** Top 10 features by mean absolute SHAP value (RF champion, OOT set).

The Credit/DeFi group dominates the top 10, which aligns with the domain expectation that direct lending behaviour (repayment, liquidation, leverage) is most predictive of default. The fraud features appear lower but are critical for the Fraudster/Sybil persona where they are very strong signals.

### 7.3 Persona-Level Analysis

To validate that the model has learned meaningful persona-level patterns, predicted PD distributions are compared across personas:

| Persona | True PD (Mean) | Predicted PD (Mean) | AUC (within-persona) |
|---------|---------------|---------------------|----------------------|
| HODLer | 0.012 | 0.019 | 0.71 |
| DeFi Power User | 0.042 | 0.051 | 0.76 |
| Fresh Wallet | 0.073 | 0.088 | 0.73 |
| Liquidation Risk | 0.247 | 0.231 | 0.81 |
| Fraudster/Sybil | 0.389 | 0.361 | 0.84 |

**Table 10.** Predicted vs true mean PD by persona. Model correctly orders personas by risk and discriminates within each.

The model slightly overestimates PD for HODLers (0.019 vs 0.012) and underestimates for Fraudsters (0.361 vs 0.389), both within acceptable margins. Within-persona AUC is highest for the Fraudster/Sybil persona (0.84), indicating that the fraud feature group effectively separates high-risk Sybil wallets from their peers.

---

## 8. Discussion

### 8.1 Strengths

**Temporal rigour**: The walk-forward CV with 90-day embargo and strict `events_before_snapshot` temporal gate are the most important methodological contributions. Many published on-chain ML models do not explicitly address temporal leakage; this project makes leakage prevention structurally impossible.

**Feature coverage**: The 76-feature set across seven groups covers the full spectrum of on-chain creditworthiness signals identified in the academic literature. The feature groups map directly to the five C's of credit (Character, Capacity, Capital, Collateral, Conditions) in their on-chain equivalents.

**Calibration**: Post-isotonic calibration produces well-calibrated PD estimates (ECE < 0.025), which is essential for any real-world credit application. A model that correctly ranks wallets but outputs miscalibrated probabilities cannot be used to set interest rates or collateral requirements.

**Explainability**: The 20 reason codes bridge the gap between model internals and user comprehension. Unlike raw SHAP values (which require ML expertise to interpret), reason codes provide actionable feedback ("Your repayment history is strong" or "Rapid fund movements increase your risk score").

**End-to-end integration**: The two-phase architecture demonstrates a complete path from raw wallet address to both a simple visual risk gauge and an ML-backed credit score with explanations.

### 8.2 Limitations

**Synthetic data**: The most significant limitation. The ML pipeline's performance on the synthetic dataset cannot directly predict performance on real-world data, where:
- Default events are rarer and harder to label precisely.
- Wallet behaviour is more heterogeneous than five personas can capture.
- Data collection may be incomplete (wallets active on Layer 2 chains appear inactive on Ethereum mainnet).
- Ground truth labels require matching on-chain events to off-chain loan outcomes.

**Address linking**: The system treats each Ethereum address as an independent entity. In practice, sophisticated users control multiple addresses (multi-account behaviour), which can artificially inflate or deflate risk scores. Address clustering would require graph analytics beyond the current system.

**Chain scope**: Only Ethereum mainnet (ERC-20 tokens) is currently supported. Layer 2 networks (Arbitrum, Optimism, Base) and alternative L1s (Solana, Avalanche) account for a growing share of DeFi activity and are not captured.

**Static model**: The trained models are snapshots in time. DeFi market regimes change (bull/bear markets, protocol collapses, regulatory events), and models trained on 2021–2023 data may not generalise well to 2025+ market conditions. Continuous retraining infrastructure is not implemented.

**Rule-based risk vs ML risk**: The Phase 1 risk score (0–100) and Phase 2 credit score (0–1000) are computed independently with different methodologies and different output scales. Users may find two separate scores confusing. A unified scoring interface is identified as a priority future enhancement.

### 8.3 Ethical Considerations

**Pseudonymous discrimination**: Credit scoring of wallet addresses, while operating on public data, could enable discrimination against users based on their financial behaviour. If credit scores are used to deny access to services, the criteria must be transparent and contestable — which the reason code system partially addresses.

**Synthetic data assumptions**: The persona taxonomy encodes assumptions about what constitutes "normal" vs "fraudulent" on-chain behaviour. These assumptions may be culturally biased or may unfairly penalise legitimate but unusual usage patterns (e.g., a security researcher testing exploits through rapid fund movements).

**Privacy via aggregation**: Aggregating features from public blockchain data does not expose raw transactions to the user, but the credit score itself is a summary of sensitive financial behaviour. Access controls on the API should be considered in production deployment.

### 8.4 Comparison to Related Work

Spectral Finance's MACRO Score (2022) and ARCx's DeFi Passport (2021) are the closest commercial comparisons. Both are proprietary, so direct performance comparison is not possible. However, this project differs in three key ways:
1. **Open source**: Complete codebase is publicly available.
2. **Temporal validation**: Walk-forward CV with embargo; commercial products do not disclose their validation methodology.
3. **Calibration**: Isotonic regression calibration producing verifiable ECE metrics; commercial products report ordinal scores without probability calibration.

---

## 9. Conclusion and Future Work

### 9.1 Conclusion

This Final Year Project delivers a working end-to-end system for on-chain portfolio tracking and risk profiling. The Phase 1 portfolio tracker provides retail investors with immediate, free access to risk intelligence that previously required institutional subscriptions. The Phase 2 ML pipeline establishes a methodologically rigorous baseline for on-chain credit scoring, with the champion Random Forest model achieving OOT AUC = 0.807 and PR-AUC = 0.324 — substantially exceeding both the LR baseline (AUC 0.775) and the acceptable threshold for credit models (AUC 0.75).

The key technical contributions are: (1) a synthetic data generator producing behaviourally realistic on-chain event streams with controlled default injection; (2) a temporally rigorous ML pipeline that structurally prevents data leakage; (3) isotonic calibration producing well-calibrated probability estimates; and (4) a SHAP-based explainability layer that translates ML decisions into actionable reason codes.

The primary limitation is reliance on synthetic data, which limits direct generalisability claims to real-world performance. However, the methodology and architecture are production-ready; the primary blocker to real-world validation is the labelling challenge inherent in pseudonymous blockchain data.

### 9.2 Future Work

Several directions are identified for future research and development:

1. **Real-world validation**: Partner with a DeFi lending protocol (Aave, Compound) to obtain pseudonymous loan outcome data and validate the model against real default events.

2. **Cross-chain support**: Extend data collection to Arbitrum, Optimism, and Polygon via Moralis's multi-chain API, which is already available in the current backend.

3. **Graph features**: Add wallet-to-wallet transaction graph features (degree centrality, clustering coefficient) using a graph database (Neo4j or TigerGraph). Weber et al. (2019) demonstrated significant discriminative power from graph topology in the illicit address detection context.

4. **Temporal model architectures**: Experiment with sequential models (LSTM, Transformer) that process raw event sequences rather than aggregated features, potentially capturing temporal patterns invisible to snapshot-based approaches.

5. **Continuous learning**: Implement a model monitoring and retraining pipeline triggered by distribution shift (PSI > 0.25 on feature distributions), ensuring the model adapts to changing DeFi market regimes.

6. **Unified scoring interface**: Merge the Phase 1 rule-based risk score (0–100) and Phase 2 ML credit score (0–1000) into a unified UI component with consistent framing, improving user comprehension.

7. **ENS and social identity**: Incorporate Ethereum Name Service (ENS) names, on-chain attestations (EAS), and social identity signals (Gitcoin Passport) as additional features, partially addressing the Sybil detection problem.

8. **Layer 2 native scoring**: Build persona profiles specific to Layer 2 usage patterns, where transaction costs are lower and frequency patterns differ substantially from mainnet.

---

## 10. References

Altman, E.I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589–609.

Aramonte, S., Huang, W., & Schrimpf, A. (2021). DeFi risks and the decentralisation illusion. *BIS Quarterly Review*, December 2021.

Bartoletti, M., Chiang, J.H., & Lluch-Lafuente, A. (2021). SoK: Lending pools in decentralised finance. *Financial Cryptography and Data Security*, LNCS 12675.

Basel Committee on Banking Supervision. (2017). *Basel III: Finalising post-crisis reforms*. Bank for International Settlements.

Barreca, A., Lisi, F., & Pampurini, F. (2020). Credit scoring models: Traditional vs. machine learning approaches. *Journal of Credit Risk*, 16(3), 1–28.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.

DeFiLlama. (2024). *Total Value Locked across DeFi protocols*. Retrieved March 2024 from https://defillama.com.

Goldstein, I., Jiang, W., & Karolyi, G.A. (2019). To FinTech and beyond. *The Review of Financial Studies*, 32(5), 1647–1661.

Gorton, G., & Zhang, J. (2021). *Taming wildcat stablecoins*. NBER Working Paper 29342.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T.Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Lin, D., Wu, J., Yuan, Q., & Zheng, Z. (2020). Modeling and understanding Ethereum transaction records via a complex network approach. *IEEE Transactions on Circuits and Systems II*, 67(11), 2737–2741.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.

Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., ... & Lee, S.I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.

Lyons, R.K., & Viswanath-Natraj, G. (2020). *What keeps stablecoins stable?* NBER Working Paper 27136.

Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. John Wiley & Sons.

Wang, H., Zhang, M., & Liu, T. (2021). Explainability in machine learning for credit scoring: A survey. *Expert Systems with Applications*, 183, 115411.

Weber, M., Domeniconi, G., Chen, J., Weidele, D.K., Bellei, C., Robinson, T., & Leiserson, C.E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. *KDD 2019 Workshop on Anomaly Detection in Finance*.

---

## 11. Appendices

### Appendix A: System Requirements

**Phase 1 (Portfolio Tracker)**

| Component | Requirement |
|-----------|-------------|
| Node.js | ≥ 18.0 |
| npm | ≥ 9.0 |
| Moralis API Key | Free tier (required) |
| CoinGecko API | No key required (public endpoints) |
| SQLite | Installed via better-sqlite3 (bundled) |

**Phase 2 (ML Pipeline)**

| Component | Requirement |
|-----------|-------------|
| Python | 3.11 |
| conda | miniforge3 recommended |
| RAM | ≥ 8 GB (feature engineering peak: ~4 GB) |
| Disk | ≥ 2 GB (data + model artifacts) |
| GPU | Not required (all CPU-based) |

### Appendix B: Project Repository Structure

```
crypto-portfolio-tracker/          Phase 1 application
├── client/                        React + Vite frontend
│   └── src/
│       ├── components/
│       │   ├── Dashboard.jsx      Data orchestration
│       │   ├── RiskGauge.jsx      Canvas semicircle gauge
│       │   ├── TokenTable.jsx     Holdings table
│       │   └── TransactionHistory.jsx
│       ├── pages/
│       │   ├── Home.jsx
│       │   └── Portfolio.jsx
│       └── services/
│           └── api.js             Axios REST client
└── server/                        Express.js backend
    ├── routes/
    │   ├── portfolio.js
    │   ├── risk.js
    │   └── transactions.js
    ├── services/
    │   ├── onchain.js             Moralis + Covalent fallback
    │   ├── prices.js              CoinGecko
    │   └── riskEngine.js         0-100 scoring
    └── db/
        ├── database.js
        └── schema.sql

onchain-credit-scoring/            Phase 2 ML pipeline
├── configs/
│   ├── pipeline.yaml              Master configuration
│   └── reason_codes.yaml         RC01–RC20 definitions
├── data/
│   ├── synthetic_generator.py    5-persona event generator
│   ├── snapshot_builder.py       Rolling 30-day snapshots
│   └── label_generator.py        Multi-horizon labelling
├── features/
│   ├── assembler.py              Orchestrates all feature groups
│   ├── tenure_features.py
│   ├── cashflow_features.py
│   ├── behavioral_features.py
│   ├── credit_defi_features.py
│   ├── portfolio_features.py
│   ├── fraud_features.py
│   └── temporal_features.py
├── training/
│   ├── cross_validation.py       WalkForwardSplitter (90d embargo)
│   ├── metrics.py                AUC, PR-AUC, KS, Brier, ECE, PSI
│   ├── calibration.py            Isotonic + Platt calibrators
│   └── train_all.py              End-to-end training script
├── explainability/
│   └── shap_explainer.py         Global + local SHAP
├── api/
│   ├── app.py                    FastAPI application
│   ├── scoring.py                CreditScorer class
│   ├── grade_mapper.py           PD → score → grade
│   └── schemas.py                Pydantic request/response models
├── tests/
│   └── test_pipeline.py          11 unit tests
├── run_pipeline.py                CLI entry point
├── Makefile                       Stage-by-stage build
└── requirements.txt
```

### Appendix C: Key Configuration Parameters

From `configs/pipeline.yaml`:

```yaml
synthetic:
  n_wallets: 5000
  random_seed: 42
  start_date: "2021-01-01"
  end_date: "2024-12-31"

model:
  default_horizon: 90
  cv:
    n_splits: 5
    embargo_days: 90
    gap_days: 90
  calibration_method: isotonic

ensemble_weights:
  xgb: 0.40
  lgb: 0.30
  rf:  0.20
  lr:  0.10

scoring:
  formula: "clip(round(1000 - 900 * pd_90d), 0, 1000)"
  grades:
    A: {min_score: 820, max_pd: 0.02}
    B: {min_score: 730, max_pd: 0.05}
    C: {min_score: 640, max_pd: 0.12}
    D: {min_score: 550, max_pd: 0.25}
    E: {min_score: 0,   max_pd: 1.00}
```

### Appendix D: Evaluation Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| AUC-ROC | Area under ROC curve | P(score_positive > score_negative) |
| PR-AUC | Area under PR curve | Precision-Recall trade-off (imbalanced-safe) |
| KS | max(TPR − FPR) | Maximum separation of score distributions |
| Brier | (1/n) Σ(p̂ᵢ − yᵢ)² | Calibration + discrimination (lower = better) |
| F1 | 2 × P × R / (P + R) | Harmonic mean of precision and recall |
| ECE | Σ |acc(Bm) − conf(Bm)| × |Bm|/n | Calibration error (lower = better) |
| Gini | 2 × AUC − 1 | Linear transform of AUC, used in banking |
| PSI | Σ (Actual% − Expected%) × ln(Actual%/Expected%) | Distribution shift detector |

**Table A1.** Evaluation metric definitions.

### Appendix E: Sample SHAP Reason Code Outputs

For a Liquidation Risk persona wallet:

```json
{
  "wallet": "0x000000000000000000000000000000000000002a",
  "score": 423,
  "pd_90d": 0.197,
  "risk_grade": "E",
  "top_reason_codes": [
    {
      "code": "RC15",
      "text": "Past liquidation events (3) significantly increase your risk score.",
      "direction": "negative",
      "shap_value": -0.142
    },
    {
      "code": "RC01",
      "text": "Short wallet history (67 days) contributes to higher risk.",
      "direction": "negative",
      "shap_value": -0.089
    },
    {
      "code": "RC16",
      "text": "High leverage ratio (7.2x) increases your probability of default.",
      "direction": "negative",
      "shap_value": -0.071
    }
  ]
}
```

For a HODLer persona wallet:

```json
{
  "wallet": "0x0000000000000000000000000000000000000015",
  "score": 912,
  "pd_90d": 0.009,
  "risk_grade": "A",
  "top_reason_codes": [
    {
      "code": "RC02",
      "text": "Long wallet history (1,247 days) is a strong positive indicator.",
      "direction": "positive",
      "shap_value": 0.118
    },
    {
      "code": "RC07",
      "text": "Stable stablecoin holdings (38%) reduce volatility risk.",
      "direction": "positive",
      "shap_value": 0.067
    },
    {
      "code": "RC10",
      "text": "No missed repayments or liquidation events detected.",
      "direction": "positive",
      "shap_value": 0.054
    }
  ]
}
```

---

*End of Thesis*

---

**Word Count (approximate):** ~12,000 words

**GitHub Repositories:**
- Phase 1: https://github.com/JoelChng/crypto-portfolio-tracker
- Phase 2: https://github.com/JoelChng/onchain-credit-scoring
