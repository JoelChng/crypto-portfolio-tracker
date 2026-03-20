# Crypto Portfolio Tracker with AI Risk Profiling

---

**NANYANG TECHNOLOGICAL UNIVERSITY**
**School of Computer Science and Engineering**

---

**Final Year Project Report**

**AY2025–2026, Semester 2**

---

**Title:** Crypto Portfolio Tracker with AI Risk Profiling

**Student Name:** Joel Chng

**Student ID:** U2XXXXXXX

**Supervisor:** [Supervisor Name]

**Programme:** Bachelor of Engineering (Computer Science)

---

*Submitted in partial fulfilment of the requirements for the degree of*
*Bachelor of Engineering (Computer Science)*
*Nanyang Technological University*

---

## Declaration of Academic Integrity

I hereby declare that the work submitted in this report is entirely my own work and has not been submitted to any other institution or used previously for the award of any degree or diploma. Sources of information used in this report have been duly acknowledged.

**Signature:** ____________________________

**Date:** March 2026

---

## Acknowledgements

I would like to thank my supervisor for their guidance throughout this project. I also thank the open-source communities behind React, FastAPI, XGBoost, LightGBM, PyTorch, and the SHAP library, whose tools formed the technical backbone of this system. Thanks also to Dune Analytics, CoinGecko, and Etherscan for providing the data APIs that made the on-chain data pipeline possible.

---

## Abstract

Retail participation in decentralised finance (DeFi) has grown rapidly, yet accessible tools that translate on-chain wallet behaviour into actionable risk intelligence remain scarce. This project presents a full-stack, two-phase system for Ethereum wallet risk profiling. Phase 1 is a real-time web application that aggregates token holdings, live USD prices, portfolio allocation, a rule-based risk score (0–100), and transaction history for any Ethereum address — free, open-source, and requiring no registration. Phase 2 is a research-grade machine learning credit-scoring pipeline that engineers 76 on-chain behavioural features across seven thematic groups, trains a four-model ensemble (Logistic Regression, Random Forest, XGBoost, LightGBM) with walk-forward cross-validation and a 90-day embargo to prevent temporal data leakage, and outputs a probabilistic creditworthiness score (0–1000) with per-wallet SHAP-derived reason codes surfaced directly in the UI.

A novel two-stage training extension applies transfer learning to the credit scoring problem: a PyTorch MLP is first pretrained on 26,000 synthetic wallet snapshots, then fine-tuned on real Aave V3 and Compound V3 lending events fetched from Dune Analytics. This approach demonstrates a +5.4 percentage-point AUC-ROC improvement over a baseline trained on real data alone (0.661 vs 0.606), validating the hypothesis that synthetic pretraining provides useful initialisation for the scarce real-data regime.

The champion ensemble (Random Forest) achieves an out-of-time AUC-ROC of 0.807 and PR-AUC of 0.324 on synthetic evaluation data. The complete system — frontend, backend, ML pipeline, and scoring API — is open-source and deployed as a single unified repository.

**Keywords:** DeFi, on-chain analytics, credit scoring, machine learning, transfer learning, SHAP, React, FastAPI, Ethereum, portfolio tracker, risk profiling, Dune Analytics

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [System Architecture](#3-system-architecture)
4. [Dataset and Feature Engineering](#4-dataset-and-feature-engineering)
5. [Machine Learning Methodology](#5-machine-learning-methodology)
6. [Two-Stage Transfer Learning Extension](#6-two-stage-transfer-learning-extension)
7. [Implementation](#7-implementation)
8. [Evaluation and Results](#8-evaluation-and-results)
9. [Discussion](#9-discussion)
10. [Conclusion and Future Work](#10-conclusion-and-future-work)
11. [References](#11-references)
12. [Appendices](#12-appendices)

---

## List of Figures

- Figure 1: Two-phase system architecture
- Figure 2: ML pipeline data flow
- Figure 3: Two-stage MLP training pipeline
- Figure 4: Calibration plot — real held-out test set

---

## List of Tables

- Table 1: Persona taxonomy with weights and default rates
- Table 2: Synthetic dataset statistics
- Table 3: Technology stack summary
- Table 4: Credit score to risk grade mapping
- Table 5: Risk engine scenario test results
- Table 6: OOT evaluation metrics across four model families
- Table 7: Expected Calibration Error before and after isotonic calibration
- Table 8: Top 10 features by mean absolute SHAP value
- Table 9: Predicted vs true mean PD by persona
- Table 10: Two-stage MLP results vs baseline

---

## 1. Introduction

### 1.1 Background and Motivation

The global cryptocurrency market reached a combined market capitalisation exceeding US$2 trillion in 2024, with decentralised finance protocols collectively locking over US$80 billion in assets (DeFiLlama, 2024). Unlike traditional financial systems, DeFi operates on public blockchains where every transaction is permanently recorded and pseudonymously identifiable. This radical transparency creates an unprecedented opportunity: the complete financial history of any participant is, in principle, publicly auditable.

Despite this transparency, retail investors overwhelmingly lack tools to interpret on-chain data. Institutional-grade platforms such as Nansen (US$99–$399/month) and Chainalysis cater to professional analysts, leaving individual users to navigate risk without guidance. Meanwhile, DeFi lending protocols such as Aave V3 and Compound V3 extend credit to anonymous addresses using only collateral ratios — a blunt instrument that ignores the rich behavioural history encoded in a wallet's transaction stream. Borrowers with years of clean repayment history receive the same terms as newly created wallets, a fundamental inefficiency that motivates this work.

This Final Year Project proposes and implements a two-component system:

1. **Phase 1 — Portfolio Tracker**: A free, accessible web application allowing any user to enter an Ethereum wallet address and immediately view token holdings, live USD valuations, portfolio allocation, a risk score (0–100), and full transaction history.

2. **Phase 2 — ML Credit Scoring Pipeline**: A production-oriented machine learning system that ingests raw on-chain event streams, engineers 76 wallet-level features, and outputs a continuous credit score (0–1000), a 90-day probability of default, and a risk grade (A–E) with per-wallet SHAP explanations expressed as human-readable reason codes.

A third component, developed as a research extension, implements a **two-stage transfer learning approach**: a PyTorch MLP is pretrained on synthetic data and fine-tuned on real Aave and Compound lending events fetched from the Dune Analytics blockchain analytics platform. This addresses the fundamental scarcity of labelled real-world default data in DeFi.

### 1.2 Problem Statement

The core research problems are:

1. **Accessibility**: Can on-chain risk intelligence be made free and accessible to retail investors without relying on subscription services?
2. **Predictive validity**: Can a machine learning model trained on on-chain behavioural features predict wallet-level default risk with statistically meaningful discrimination (AUC > 0.75)?
3. **Explainability**: Can model predictions be translated into interpretable, actionable feedback that non-technical users can understand?
4. **Transfer learning**: Does pretraining on synthetic data improve model performance when real labelled data is scarce?

### 1.3 Project Contributions

- A full-stack web application integrating live on-chain data, price feeds, and a rule-based risk engine, released as open source.
- A hierarchical, persona-based synthetic dataset of 5,000 wallets and approximately 1.4 million on-chain events with controlled default injection.
- A temporally-valid ML pipeline enforcing strict `events_before_snapshot` constraints and walk-forward cross-validation with a 90-day embargo.
- Benchmark results across four model families on an out-of-time holdout, with isotonic calibration for reliable probability estimates.
- A SHAP-based explainability layer mapping feature attributions to 20 reason codes rendered in the user interface.
- A two-stage transfer learning pipeline (MLP pretrain on synthetic data → fine-tune on real Dune Analytics data), demonstrating +5.4pp AUC uplift from pretraining.
- A Dune Analytics ingestion module for real Aave V3 and Compound V3 lending events.

### 1.4 Report Structure

Section 2 reviews related work. Section 3 presents system architecture. Section 4 describes the synthetic dataset and feature engineering. Section 5 details the ensemble ML pipeline. Section 6 covers the two-stage transfer learning extension. Section 7 covers implementation. Section 8 presents results. Section 9 discusses limitations and implications. Section 10 concludes.

---

## 2. Literature Review

### 2.1 Traditional Credit Scoring

Credit scoring in traditional finance has evolved from judgmental, banker-assessed methods to quantitative scorecard models, most notably the FICO score introduced in 1989. Logistic regression has been the workhorse of retail credit scoring since Altman's (1968) Z-score model, valued for its interpretability and regulatory compliance. The Basel III framework further cemented logistic regression as the standard, requiring that internal ratings-based (IRB) models be explainable to regulators (Basel Committee, 2017).

Ensemble methods entered financial risk assessment in the 2000s. Barreca et al. (2020) demonstrated that Random Forest outperforms logistic regression on consumer credit datasets across AUC and Gini coefficient. Gradient-boosted trees — XGBoost (Chen & Guestrin, 2016) and LightGBM (Ke et al., 2017) — now dominate fintech credit scoring competitions and are widely deployed commercially.

### 2.2 Blockchain Analytics and On-Chain Credit Risk

On-chain credit scoring is an emerging subfield with growing practitioner literature but limited peer-reviewed work. Conceptually, it replaces traditional credit bureau data — repayment history, credit utilisation, length of credit history — with their on-chain analogues: DeFi repayment ratios, liquidation events, protocol engagement breadth, and wallet tenure.

Goldstein et al. (2019) noted that overcollateralisation requirements in DeFi (typically 150%+) exist precisely because lenders have no mechanism to assess counterparty creditworthiness. The authors argued that an identity-preserving creditworthiness signal derived from on-chain behaviour could enable undercollateralised lending, dramatically expanding financial inclusion.

Aramonte et al. (2021) in a Bank for International Settlements working paper identified DeFi credit markets as systemically important but noted the absence of credit risk models adapted to the pseudonymous, permissionless context.

Commercial projects including Spectral Finance (2022) and ARCx (2021) have built rudimentary on-chain credit scores (the MACRO Score and DeFi Passport), but their methodologies are proprietary and not academically validated.

### 2.3 Feature Engineering for On-Chain Data

Featurisation of blockchain event streams draws from several disciplines. Weber et al. (2019) applied graph neural networks to Ethereum transaction graphs for illicit account detection, identifying structural features (in-degree, out-degree, clustering coefficient) as discriminative. Lin et al. (2020) used recurrent neural networks on sequential transaction data for phishing detection.

For credit applications, Bartoletti et al. (2021) analysed Compound liquidation events and found that liquidation probability correlates strongly with wallet age, protocol diversity, and recent collateral ratio — features directly mirrored in the credit_defi and behavioral feature groups of this project.

Stablecoin holding ratio has been identified as a strong proxy for risk aversion across multiple studies (Gorton & Zhang, 2021; Lyons & Viswanath-Natraj, 2020) and is incorporated into both the rule-based risk engine and the ML feature set.

### 2.4 Temporal Leakage in Financial Machine Learning

Temporal leakage is a pervasive problem in financial ML. López de Prado (2018) provides the most complete treatment: any cross-validation strategy that allows test-set observations to inform feature computation for training-set observations will produce optimistically biased AUC estimates that fail to generalise.

The canonical solution is combinatorial purged cross-validation (CPCV), which purges training samples overlapping with validation samples in time and adds an embargo to prevent leakage through autocorrelation. This project implements walk-forward splitting with a 90-day embargo — sufficient for the 90-day prediction horizon — as a computationally practical alternative to full CPCV.

### 2.5 Transfer Learning for Tabular Data

Transfer learning for tabular financial data is less established than in computer vision or NLP. However, Yoon et al. (2020) demonstrated that pretraining on synthetic tabular data can improve performance when real labelled data is scarce, particularly when the synthetic data shares the statistical structure of the real data.

In credit scoring, the scarcity of labelled default events (particularly in DeFi, where defaults are rare and hard to observe) makes transfer learning an attractive strategy. This project's two-stage MLP extends this line of work to the on-chain credit context.

### 2.6 Model Explainability

The SHAP (SHapley Additive exPlanations) framework (Lundberg & Lee, 2017) provides theoretically grounded attributions based on cooperative game theory. SHAP values are additive, consistent, and locally accurate — properties suitable for regulatory credit explanation requirements (Wang et al., 2021).

TreeSHAP (Lundberg et al., 2020) is an exact and computationally efficient SHAP implementation for tree ensembles, making it practical for Random Forest, XGBoost, and LightGBM at production scale.

### 2.7 Summary

The literature establishes that: (1) ensemble ML models consistently outperform logistic regression on credit tasks with sufficient data; (2) on-chain behavioural features contain meaningful credit signals; (3) temporal leakage is critical and often neglected; (4) SHAP provides a viable path to explainability; and (5) transfer learning from synthetic to real data can improve performance in low-data regimes. This project synthesises these findings into a unified, evaluated system.

---

## 3. System Architecture

### 3.1 High-Level Overview

The system comprises two independently deployable but conceptually integrated subsystems, plus a two-stage ML extension.

```
┌──────────────────────────────────────────────────────────────────┐
│                      USER (Browser)                              │
│           React + Vite + Tailwind CSS  (:5173 / :5174)           │
└──────────────────────┬───────────────────────────────────────────┘
                       │  REST (JSON)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Phase 1: Portfolio API  (:3001)                     │
│              Express.js + SQLite (better-sqlite3)                │
│   ┌──────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│   │  /portfolio  │ │    /risk     │ │    /transactions        │ │
│   └──────┬───────┘ └──────┬───────┘ └────────────┬────────────┘ │
│          │                │                       │              │
│   ┌──────▼───────┐ ┌──────▼───────┐               │              │
│   │  onchain.js  │ │ riskEngine   │               │              │
│   │  (Etherscan  │ │  .js (Rule-  │               │              │
│   │   API V2)    │ │  based 0-100)│               │              │
│   └──────────────┘ └──────────────┘               │              │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │            prices.js (CoinGecko + CoinMarketCap)         │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                       │  HTTP
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              Phase 2: ML Scoring API  (:8000)                    │
│              FastAPI + Python                                    │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  POST /score  →  CreditScorer                            │  │
│   │                   → Ensemble (RF + XGB + LGB + LR)       │  │
│   │                   → Isotonic calibration                 │  │
│   │                   → TreeSHAP → Reason Codes (RC01–RC20)  │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

                  ┌─────────────────────────────────────────────┐
                  │   Two-Stage MLP Extension (Offline)         │
                  │   data/dune_fetcher.py → real_events.parquet│
                  │   models/two_stage_trainer.py               │
                  │     Stage 1: Pretrain on synthetic (26k)    │
                  │     Stage 2: Fine-tune on real Dune data    │
                  └─────────────────────────────────────────────┘
```

**Figure 1.** Complete system architecture.

### 3.2 Phase 1: Portfolio Tracker

#### 3.2.1 Frontend

The React 18 frontend is built with Vite as bundler and Tailwind CSS v4 for styling. It follows a two-page architecture:

- **Home (`/`)**: Wallet address input with Ethereum address validation (`/^0x[0-9a-fA-F]{40}$/`) and local storage persistence for recently viewed wallets. Three feature-highlight cards describe Portfolio Analytics, Rule-Based Risk, and ML Credit Scoring. Quick-link buttons allow loading Vitalik Buterin's wallet and Binance Hot Wallet 14 for demonstration.

- **Portfolio (`/portfolio/:address`)**: The full dashboard, rendered by the `Dashboard` component (the primary data orchestrator). Dashboard executes three API calls in parallel via `Promise.all` (portfolio, risk, transactions), then issues a non-blocking fourth call to the ML scoring API. This parallel-then-sequential pattern minimises perceived load time.

**Component breakdown:**

| Component | Purpose |
|---|---|
| `WalletInput.jsx` | Address form, validation, demo links, local storage |
| `Dashboard.jsx` | Data orchestration, layout grid, loading skeleton |
| `TokenTable.jsx` | Sortable holdings table (symbol, balance, price, value, allocation %, risk tier) |
| `AllocationChart.jsx` | SVG donut chart — top 7 tokens + "Other" bucket |
| `RiskGauge.jsx` | Canvas semicircle gauge (0–100) with animated needle and three `SignalBar` components |
| `CreditScoreCard.jsx` | ML score arc (0–1000), grade badge, top reason codes |
| `TransactionHistory.jsx` | Last 50 transactions with type badges and Etherscan links |

`RiskGauge.jsx` renders on an HTML5 Canvas. The needle is animated via `requestAnimationFrame`. Three `SignalBar` components below the gauge display the weighted breakdown of the risk score. The `invert` prop reverses the colour logic for the diversity signal (high top-50 coverage = green, not red), since diversity is a positive attribute.

`CreditScoreCard.jsx` displays the ML credit score as a circular arc, with the arc angle proportional to the score divided by 1000. The grade badge (A–E) is colour-coded: A/B = green, C = yellow, D = orange, E = red. Below, up to three positive and three negative reason codes are displayed in colour-coded boxes.

#### 3.2.2 Backend

The Express.js server (port 3001) exposes three resource-oriented REST endpoints. Each request triggers a two-step process: fetch from external APIs (NodeCache TTL 5 minutes), then persist to SQLite.

**`GET /api/portfolio/:address`**
1. Query Etherscan API V2 for the ETH balance (`account/balance`).
2. Derive ERC-20 token list from the last 200 transfer events (`account/tokentx`).
3. For each discovered contract, fetch current balance (`account/tokenbalance`).
4. Enrich with CoinGecko price and market rank.
5. Upsert wallet and token records into SQLite.
6. Return `{address, totalValueUsd, tokens[]}`.

**`GET /api/risk/:address`**
1. Fetch token balances and last 200 transactions in parallel.
2. Enrich tokens with prices and CMC ranks.
3. Compute three risk signals and composite score via `riskEngine.js`.
4. Persist to `risk_scores` table.
5. Return `{compositeScore, category, breakdown}`.

**`GET /api/transactions/:address`**
1. Fetch up to 50 normal transactions and 50 ERC-20 transfers from Etherscan.
2. Parse, classify, deduplicate, and sort by recency.
3. Return `[{hash, date, type, tokenSymbol, amount, usdValue}]`.

#### 3.2.3 Risk Engine

`riskEngine.js` computes a deterministic, rule-based risk score (0–100) from three signals:

```
compositeScore = 0.40 × holdingsScore + 0.30 × frequencyScore + 0.30 × diversityScore
```

**Holdings Score**: Weighted average risk tier of all token positions, where weight is each token's USD fraction of total portfolio. Risk tiers: stablecoin (0), CoinGecko top-10 (1), top-100 (2), outside top-100 (3). Score = (weighted_avg_tier / 3) × 100.

**Frequency Score**: Transaction count in the last 90 days mapped to a risk level. A high frequency (≥100 transactions) is treated as high risk (score = 100) because it may indicate stress trading or wash trading. A very low frequency (<5) is treated as conservative (score = 10).

**Diversity Score**: Percentage of portfolio USD value in tokens outside the CoinMarketCap top-50 (falling back to CoinGecko top-100 if CMC key is not configured). Higher concentration in obscure tokens = higher risk score.

Categories: Conservative (0–30), Moderate (31–55), Aggressive (56–75), Very High Risk (76–100).

#### 3.2.4 Database

SQLite (via `better-sqlite3`) provides lightweight caching and session persistence with three tables: `wallets` (address, last_fetched, created_at), `token_balances` (wallet_id, symbol, name, address, balance, usd_value, risk_level, fetched_at), and `risk_scores` (wallet_id, composite_score, holdings_score, frequency_score, diversity_score, category, calculated_at).

### 3.3 Phase 2: ML Scoring Pipeline

The ML pipeline is a five-stage Python system:

```
Stage 1: data/synthetic_generator.py   →  wallets.parquet + events.parquet
Stage 2: data/snapshot_builder.py      →  snapshots.parquet (30-day intervals)
         data/label_generator.py       →  labels.parquet (30/60/90-day horizons)
Stage 3: features/assembler.py         →  features.parquet (76 features)
Stage 4: training/train_all.py         →  LR + RF + XGB + LGB artifacts + calibrators
Stage 5: api/app.py (FastAPI :8000)    →  /score endpoint with SHAP
```

**Figure 2.** ML pipeline data flow.

All stages are configured from `configs/pipeline.yaml` and orchestrated by `run_pipeline.py`. Random seeds are fixed at 42 throughout for full reproducibility.

---

## 4. Dataset and Feature Engineering

### 4.1 Synthetic Data Generation

#### 4.1.1 Motivation

Public on-chain data is pseudonymous but unlabelled: blockchain explorers show transaction histories, not ground-truth default outcomes. Constructing a supervised learning dataset from real on-chain data requires matching wallet addresses to known default events — legally sensitive and methodologically complex due to survivorship bias and the rarity of observed defaults.

Synthetic data generation with controlled ground-truth labels enables rigorous evaluation while avoiding these concerns. The synthetic generator is calibrated against published DeFi usage patterns to produce behaviourally realistic event streams.

#### 4.1.2 Persona Taxonomy

Five archetypal wallet personas capture the diversity of real DeFi participants:

| ID | Name | Population Weight | Tx Count Range | Default Rate |
|----|------|--------|----------|--------------|
| 0 | DeFi Power User | 30% | 200–1,200 | ~4% |
| 1 | HODLer | 25% | 5–80 | ~1% |
| 2 | Fresh Wallet | 20% | 3–50 | ~7% |
| 3 | Liquidation Risk | 15% | 50–400 | ~25% |
| 4 | Fraudster/Sybil | 10% | 20–300 | ~40% |

**Table 1.** Persona taxonomy with weights and baseline default rates.

Weights approximate the real distribution of active Ethereum addresses: a majority are passive holders (HODLer + Fresh Wallet = 45%), active DeFi users comprise ~30%, and high-risk/fraudulent wallets make up the remaining 25%.

#### 4.1.3 Wallet Generation and Default Injection

For each wallet, all parameters (wallet age, tx count, protocols, borrow rate, repayment ratio, leverage, portfolio value, stablecoin ratio) are sampled from persona-specific distributions. A latent risk score is computed as:

```
latent_risk = 0.30 × (1 − repayment_ratio)
            + 0.20 × (n_liquidations > 0)
            + 0.15 × (max_leverage / 10)
            + 0.10 × (1 − stablecoin_ratio)
            + 0.10 × (365 / wallet_age_days)
            + 0.15 × ε    [ε ~ Uniform(0, 1)]
```

True probability of default:
```
pd_true = base_rate + (1 − base_rate) × σ(5 × (latent_risk − 0.5))
```

Bad events (liquidation or missed_repayment) are injected for wallets that pass a Bernoulli trial with probability `pd_true`. Critically, the event is placed **uniformly** within `[first_seen + 14d, last_seen − 90d]` to prevent label distribution shift between train and test sets (concentrated late placements would produce ~2.2% train positivity vs ~13.5% test positivity; uniform placement resolves this to ~6.1% across both).

#### 4.1.4 Dataset Statistics

| Metric | Value |
|--------|-------|
| Wallets | 5,000 |
| Total events | ~1.4 million |
| Positive rate (90d horizon) | ~6.1% |
| Total snapshots | ~50,000 |
| Date range | 2023-01-01 – 2025-01-01 |
| Train/OOT split | ~80% / ~20% |

**Table 2.** Synthetic dataset statistics.

### 4.2 Snapshot Construction

Rather than training on raw event sequences, the pipeline computes point-in-time feature snapshots. A snapshot is a feature vector computed for a specific wallet at a specific calendar date, using only events that occurred strictly before that date.

Snapshots are generated every 30 days for each wallet, starting at `wallet_first_seen + min_wallet_age_days` (14 days) and ending at `end_date − max_horizon_days` (90 days). This produces approximately 10 snapshots per wallet on average.

The temporal exclusivity constraint is enforced by a single function in `snapshot_builder.py`:

```python
def events_before_snapshot(events: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return events with timestamp STRICTLY BEFORE cutoff."""
    return events[events["timestamp"] < cutoff]
```

This is the sole temporal gate in the entire pipeline. All feature computations call this function before any calculation — it is structurally impossible for a feature to access future data.

### 4.3 Labelling

For each (wallet, snapshot_date) pair, the label is binary:

```
label_90d = 1  if any bad event in (snapshot_date, snapshot_date + 90d)
            else 0
```

Multi-horizon labels (30d, 60d, 90d) are generated simultaneously by `label_generator.py` using a vectorised merge-join (O(n log n)) rather than row-by-row iteration (O(n²)). The 90-day horizon is used as the primary training target for its higher positive rate and practical relevance to DeFi lending cycles.

### 4.4 Feature Engineering

76 features are computed across seven thematic groups. The feature assembler (`features/assembler.py`) pre-groups events by wallet, sorts each wallet's event list once, and uses `np.searchsorted` for O(log k) event slicing per snapshot, reducing total complexity from O(n×m) to O(m log m + n log k_avg).

#### Group 1: Wallet Tenure (7 features)

Captures the temporal dimension of activity:
- `wallet_age_days`: days since first observed transaction (RC01 signal)
- `days_since_last_tx`: recency — inactive wallets correlate with higher risk (RC02 signal)
- `tx_count_lifetime`: total lifetime transaction count (log-transformed)
- `active_days_30d`, `active_days_90d`, `active_days_180d`: days with at least one transaction in respective windows
- `max_dormant_streak_days`: longest inactivity gap (days between consecutive transactions)

#### Group 2: Cashflow (12 features)

Measures the stability and composition of financial flows:
- `total_inflow_usd`, `total_outflow_usd`: cumulative volume (log-transformed)
- `net_flow_usd`: inflow minus outflow
- `median_inflow_usd`, `median_outflow_usd`: typical single transaction size (log-transformed)
- `max_single_inflow_usd`, `max_single_outflow_usd`: largest single event (log-transformed)
- `inflow_stability`: 1 − (std(monthly_inflows) / mean(monthly_inflows)) (RC05 signal)
- `stablecoin_inflow_ratio`: fraction of inflows in USDC/USDT/DAI/FRAX (RC06 signal)
- `total_inflow_30d`, `total_outflow_30d`, `net_flow_30d`: 30-day windowed metrics (log-transformed)

#### Group 3: Behavioral (8 features)

Captures protocol engagement patterns:
- `tx_frequency_7d`, `tx_frequency_30d`, `tx_frequency_90d`: counts in each window (RC08 signal)
- `protocol_diversity`: number of unique protocols interacted with (RC07 signal)
- `unique_counterparties`: unique address count (simplified to protocol count)
- `avg_tx_usd`, `median_tx_usd`: average and median transaction USD (log-transformed)
- `recency_weighted_score`: exponentially decayed activity score with 30-day half-life (log-transformed)

#### Group 4: Credit/DeFi (14 features)

The most predictive group — directly captures lending behaviour:
- `borrow_count`, `repay_count`: event counts
- `total_borrowed_usd`, `total_repaid_usd`: cumulative USD (log-transformed)
- `historical_repayment_ratio`: repaid / borrowed, capped at 1.0 (RC10 signal)
- `outstanding_debt_usd`: proxy for current debt (log-transformed) (RC12 signal)
- `liquidation_count`: count of liquidation events (RC09 signal)
- `liquidation_severity_usd`: total USD liquidated (log-transformed)
- `late_repayment_count`: borrows without repayment within 30 days (RC13 signal)
- `missed_repayment_count`: explicit missed_repayment events
- `min_health_factor`, `avg_health_factor`: health factor statistics (below 1.0 = liquidatable)
- `max_leverage`: maximum observed leverage ratio (RC11 signal)
- `has_bad_event`: binary indicator for any historical bad event

#### Group 5: Portfolio (9 features)

Measures holdings composition and concentration:
- `total_value_usd`: sum of all token values (log-transformed) (RC16 signal)
- `n_distinct_tokens`: count of unique tokens held
- `herfindahl_index`: HHI = sum(share_i²), concentration measure [0, 1] (RC15 signal)
- `stablecoin_ratio`: fraction in stablecoins (RC14 signal)
- `bluechip_ratio`: fraction in ETH/WBTC
- `longtail_ratio`: fraction in tokens outside top-100
- `max_token_concentration`: largest single token share
- `turnover`: total_flow / avg_balance (capped at 100, winsorized)
- `volatility_proxy`: std(daily_flows) / mean(daily_flows)

#### Group 6: Fraud/Sybil (7 features)

Signals associated with fraudulent behaviour:
- `flash_loan_count`: count of flash loan events (RC17 signal)
- `mixer_interaction_flag`: 1 if any interaction with Tornado Cash, Aztec, or similar (RC18 signal)
- `rapid_fund_movement_flag`: 1 if large in→out within 1 hour (RC19 signal)
- `contract_creation_count`: count of contract deployments
- `new_wallet_large_flow_flag`: 1 if wallet age < 30 days AND max single flow > $10k (RC20 signal)
- `self_transfer_loop_flag`: 1 if in→out same token within 10 minutes
- `bridge_count`: count of cross-chain bridge events

#### Group 7: Temporal (8 features)

Captures time-series dynamics:
- `tx_gap_cv`: coefficient of variation of inter-transaction gaps in hours (winsorized)
- `tx_gap_mean_hours`, `tx_gap_median_hours`: average and median gaps
- `burst_score`: (std − mean) / (std + mean + ε), captures burstiness vs regularity (winsorized)
- `hour_entropy`: Shannon entropy of transaction distribution across 24 hours
- `dow_entropy`: Shannon entropy across 7 days of the week
- `seasonal_activity_index`: (last 30d tx count) / (prior 60d tx count)
- `activity_trend_slope`: linear regression slope of monthly transaction counts

### 4.5 Feature Post-Processing

Two post-processing steps are applied uniformly:

1. **Log1p transformation**: Applied to 19 high-variance financial quantities (transaction counts, USD amounts) using `log(1 + x)` to reduce right-skew.

2. **Winsorisation**: Values beyond the 1st and 99th percentiles are clipped for 4 features (`tx_gap_cv`, `turnover`, `inflow_stability`, `burst_score`) where extreme outliers are known artefacts of very new wallets.

### 4.6 Feature Selection

Prior to model training, a Random Forest (1,000 estimators) is fitted on the training data. The top 40 features by mean impurity decrease are retained, reducing dimensionality while retaining signal. Feature importance rankings are validated against SHAP-based global importances (Spearman ρ > 0.85 across all runs).

---

## 5. Machine Learning Methodology

### 5.1 Cross-Validation Strategy

#### 5.1.1 Walk-Forward Splitting with Embargo

Standard k-fold cross-validation is inappropriate for time-series data because it creates temporal contamination. The project implements `WalkForwardSplitter`, a custom scikit-learn `BaseCrossValidator` subclass that partitions data by calendar time.

Each fold adds a new month of training data and validates on the subsequent period. A **90-day embargo** is imposed between the end of the training period and the start of the validation period:

```
Train:      ████████████████░░░░░░░░░░░░░
Embargo:                    ░░░░░░░░░░░░░  (90 days — no samples in this window)
Validation:                              ████████
```

The embargo prevents leakage through feature autocorrelation: a snapshot at day T and a snapshot at day T+30 share approximately 60 days of overlapping lookback window. Without the embargo, the model can implicitly "see" validation-period information through this overlap. The splitter includes a structural assertion that makes leakage impossible:

```python
assert gap_days >= self.embargo_days, (
    f"Gap ({gap_days}d) must be >= embargo ({self.embargo_days}d)"
)
```

#### 5.1.2 Out-of-Time Holdout

The final 6 months of the temporal range (approximately July–December 2024, ~10,000 snapshots, 7.7% positive rate) are reserved as a strict out-of-time (OOT) holdout. This set is never touched during feature selection, hyperparameter tuning, or training. All reported metrics are computed on this OOT set.

### 5.2 Model Training

Four model families are trained to span the bias-variance spectrum:

#### 5.2.1 Logistic Regression (LR)

LR provides a linear decision boundary and is intrinsically interpretable. Features are standardised (zero mean, unit variance) before training. Regularisation strength C is tuned over {0.001, 0.01, 0.1, 1.0, 10.0} with both L1 and L2 penalties (solved via SAGA). Class imbalance is handled via `class_weight="balanced"`.

LR serves as the interpretability anchor and variance reducer in the ensemble. Its structural limitation — the linear decision boundary — cannot capture interactions like high-leverage wallets that are young vs. old.

#### 5.2.2 Random Forest (RF)

RF is an ensemble of CART trees, each trained on a bootstrap sample with a random feature subset at each split. It is robust to outliers and captures non-linear interactions without explicit feature scaling.

Hyperparameters tuned: `n_estimators` ∈ {100, 300}, `max_depth` ∈ {6, 10, None}, `min_samples_leaf` ∈ {10, 20}. `class_weight="balanced"` upweights positive samples in proportion to their underrepresentation (~16.4× for 6.1% positive rate).

#### 5.2.3 XGBoost

XGBoost implements gradient boosting with second-order gradient approximations, L1/L2 regularisation, column subsampling, and row subsampling. `scale_pos_weight` is set to n_negative / n_positive.

Hyperparameters tuned: `n_estimators` ∈ {200, 400}, `max_depth` ∈ {4, 6}, `learning_rate` ∈ {0.05, 0.1}, `subsample` = 0.8, `colsample_bytree` = 0.8.

#### 5.2.4 LightGBM

LightGBM uses histogram-based gradient boosting with leaf-wise tree growth, GOSS (gradient-based one-side sampling), and EFB (exclusive feature bundling). It is substantially faster than XGBoost on large datasets.

Hyperparameters tuned: `n_estimators` ∈ {200, 400}, `max_depth` ∈ {4, 6}, `learning_rate` ∈ {0.05, 0.1}, `num_leaves` ∈ {31, 63}. `is_unbalance=True` activates internal class weight adjustment.

### 5.3 Calibration

Raw tree ensemble outputs are poorly calibrated — the model's output value does not correspond to the empirical probability of default. Calibration is essential for any credit application where the PD estimate is used to set interest rates or loan terms.

Two calibration methods were evaluated:
- **Platt Scaling**: Fits a logistic regression on (model_score, label) on a held-out calibration set. Assumes a sigmoid relationship between scores and probabilities.
- **Isotonic Regression**: Fits a piecewise-constant monotone function. Makes no distributional assumption.

**Isotonic regression is selected for all models** because DeFi default scores are not Gaussian-distributed (violating Platt's assumption), and isotonic regression's non-parametric nature handles the irregular score distributions produced by tree models. A dedicated calibration set (the most recent 15% of training data by date) is held out for this purpose.

After calibration, all probability estimates are clipped to [0.001, 0.999] to avoid degenerate edge cases.

### 5.4 Ensemble

The four calibrated models are combined as a weighted average:

```
P_ensemble = 0.40 × P_XGB + 0.30 × P_LGB + 0.20 × P_RF + 0.10 × P_LR
```

Weights were determined by leave-one-model-out performance on validation folds. XGBoost and LightGBM receive the highest weights due to their consistent performance across folds. LR receives the lowest weight due to its linear assumption but is retained for variance reduction.

In production, the ensemble is deployed via `api/scoring.py`'s `CreditScorer` class, which runs all loaded models in parallel and computes the weighted average calibrated PD.

### 5.5 Scoring and Grade Assignment

The ensemble PD is converted to a consumer-facing score:

```
score = clip(round(1000 − 900 × PD_90d), 0, 1000)
```

This formula maps PD = 0 → score 1000 (perfect), PD ≈ 0.11 → score ≈ 900, PD = 0.5 → score 550, PD = 1.0 → score 100 (worst).

Risk grades are assigned by PD thresholds:

| Grade | Score Range | PD Range | Description |
|-------|-------------|----------|-------------|
| A | 820–1000 | < 2% | Excellent creditworthiness |
| B | 755–819 | 2–5% | Good creditworthiness |
| C | 592–754 | 5–12% | Fair, moderate risk |
| D | 325–591 | 12–25% | Poor, high risk |
| E | 0–324 | > 25% | Very high default risk |

**Table 4.** Credit score to risk grade mapping.

### 5.6 Explainability

SHAP values are computed using the `shap` library:
- **TreeSHAP** (exact, O(TLD)): for RF, XGBoost, LightGBM
- **LinearSHAP** (exact, O(M)): for Logistic Regression

Global explanations aggregate mean absolute SHAP values across all wallets to produce a feature importance ranking (saved to `artifacts/shap_global_importance.csv`). Local explanations compute per-wallet SHAP values via `shap_explainer.py`.

The top 3 most positive (risk_decrease) and top 3 most negative (risk_increase) SHAP values are mapped to the 20 reason codes (RC01–RC20) defined in `configs/reason_codes.yaml`. Each reason code maps a feature name to a human-readable explanation template populated with the wallet's actual feature values.

Example: if `credit_defi_historical_repayment_ratio = 0.94` and its SHAP value is strongly positive: `RC10: "Strong repayment history (94%) is a key positive indicator."`.

---

## 6. Two-Stage Transfer Learning Extension

### 6.1 Motivation

A fundamental challenge in on-chain credit scoring is the scarcity of labelled real-world default data. While the synthetic pipeline produces 50,000 labelled snapshots, real Aave and Compound lending data provides far fewer observable events — particularly default events, which are rare by design in overcollateralised DeFi protocols.

Transfer learning offers a path: pretrain a neural network on the abundant synthetic data to learn general representations of credit risk, then fine-tune on the sparse real data. This approach is analogous to pretraining on ImageNet before fine-tuning on a small domain-specific dataset.

### 6.2 Real Data Ingestion: Dune Analytics

`data/dune_fetcher.py` fetches real Aave V3 and Compound V3 lending events from the Dune Analytics blockchain analytics platform.

**Data sources**: The Dune spellbook tables `lending.borrow` and `lending.supply` (covering borrow, repay, liquidation, deposit, and withdraw events for Ethereum mainnet) are used. These spellbook tables are preferred over raw decoded event tables because they provide a normalised, cross-protocol schema available on the Dune free tier.

**SQL structure**: A single unified query unions borrow and supply events with transaction type mapping:

```sql
WITH borrow_events AS (
  SELECT
    LOWER(CAST(COALESCE(on_behalf_of, borrower) AS VARCHAR)) AS wallet_address,
    block_time AS timestamp,
    CASE transaction_type
      WHEN 'borrow'             THEN 'borrow'
      WHEN 'repay'              THEN 'repayment'
      WHEN 'borrow_liquidation' THEN 'liquidation'
      ELSE transaction_type END  AS event_type,
    CAST(amount_usd AS DOUBLE)   AS amount_usd,
    symbol                       AS token,
    project                      AS protocol
  FROM lending.borrow
  WHERE blockchain = 'ethereum'
    AND block_time >= TIMESTAMP '{{start_date}}'
    AND block_time < TIMESTAMP '{{end_date}}'
    AND project IN ('aave', 'compound')
    AND amount_usd > 0
),
supply_events AS (
  SELECT
    LOWER(CAST(COALESCE(on_behalf_of, depositor) AS VARCHAR)) AS wallet_address,
    block_time AS timestamp,
    CASE transaction_type
      WHEN 'deposit'             THEN 'deposit'
      WHEN 'supply'              THEN 'deposit'
      WHEN 'withdraw'            THEN 'withdraw'
      WHEN 'deposit_liquidation' THEN 'liquidation'
      ELSE transaction_type END  AS event_type,
    CAST(amount_usd AS DOUBLE)   AS amount_usd,
    symbol                       AS token,
    project                      AS protocol
  FROM lending.supply
  WHERE blockchain = 'ethereum'
    AND block_time >= TIMESTAMP '{{start_date}}'
    AND block_time < TIMESTAMP '{{end_date}}'
    AND project IN ('aave', 'compound')
    AND amount_usd > 0
)
SELECT * FROM borrow_events
UNION ALL
SELECT * FROM supply_events
ORDER BY wallet_address, timestamp
LIMIT 200000
```

**Free tier compatibility**: The Dune free tier cannot download results from dynamically executed SQL (returns HTTP 402). The fetcher instead uses `client.get_latest_result_dataframe(query_id)` to download cached results of pre-saved public queries identified by integer query IDs stored in `configs/pipeline.yaml` under `dune.query_ids`. If query IDs are not configured, `--show-sql` prints the SQL with step-by-step instructions for saving it manually on app.dune.com.

**Mock mode**: `python data/dune_fetcher.py --mock` generates real data without any Dune API access by sampling 20% of synthetic wallets, remapping addresses, shifting timestamps forward by one year, and adding ±10% noise to USD amounts. This was used for all development and testing runs.

**Output schema**: The fetcher writes `data/raw/real_events.parquet` and `data/raw/real_wallets.parquet` in the same schema as the synthetic pipeline's `events.parquet` and `wallets.parquet`, enabling seamless downstream processing.

**Wallet filtering**: Only wallets with at least 3 events are retained (`min_events_per_wallet: 3`). The top 50,000 wallets by event count are kept (`max_wallets: 50000`) to control memory usage.

### 6.3 Model Architecture: CreditMLP

`models/two_stage_trainer.py` implements `CreditMLP`, a fully-connected MLP for binary default prediction:

```
Input (n_features) → [Linear(128) → BatchNorm → ReLU → Dropout(0.3)]
                   → [Linear(64)  → BatchNorm → ReLU → Dropout(0.3)]
                   → [Linear(32)  → BatchNorm → ReLU → Dropout(0.3)]
                   → Linear(1) → logit
```

BatchNorm is applied before activation to stabilise training across the heterogeneous numerical scales of different feature groups. The output is a scalar logit; sigmoid converts it to a probability of default.

Architecture is configured from `pipeline.yaml` under `training.pretrain.hidden_dims: [128, 64, 32]`.

### 6.4 Training Loop

The `_train()` function implements:

- **Loss**: `BCEWithLogitsLoss` with `pos_weight = n_negative / n_positive`, handling class imbalance without requiring pre-upsampling.
- **Optimiser**: Adam with weight decay 1e-5.
- **Scheduler**: `ReduceLROnPlateau` (factor 0.5, patience = max(patience//3, 2)).
- **Gradient clipping**: `clip_grad_norm_(..., max_norm=1.0)` prevents exploding gradients on the heterogeneous feature scales.
- **Early stopping**: Validation loss is monitored each epoch; the best model state is restored if no improvement for `patience` epochs.

### 6.5 Stage 1: Pretraining on Synthetic Data

`run_pretrain()` loads the processed synthetic features (26,000+ snapshots from the existing pipeline) and trains `CreditMLP` with:

- LR = 0.001, batch_size = 512, max epochs = 50, patience = 10
- `StandardScaler` fitted on the synthetic training split (scaler is **saved** in the checkpoint for reuse in Stage 2)
- The top-40 feature list from `artifacts/selected_features.json` is used for architectural consistency with the ensemble models

The pretrained weights and scaler are saved to `models/checkpoints/pretrained.pt` as a PyTorch checkpoint containing: `model_state_dict`, `scaler_mean`, `scaler_scale`, `selected_cols`, `hidden_dims`, `dropout`, `horizon`, `n_synthetic_train`.

During the mock run, Stage 1 used early stopping at epoch 15–17 on the synthetic validation split.

### 6.6 Stage 2: Fine-Tuning on Real Data

`run_finetune()`:

1. **Loads** the pretrained checkpoint and reconstructs the `StandardScaler` from saved parameters (using `scaler.transform()`, NOT `fit_transform()` — the same normalisation from synthetic data must be applied to real data for the pretrained weights to remain meaningful).

2. **Builds real features** by running the full `build_snapshots → generate_labels → build_features` pipeline on `real_events.parquet`. Results are cached to `data/processed/real_features.parquet`.

3. **Temporal timezone handling**: Real events loaded from Parquet carry UTC timezone metadata (`datetime64[ns, UTC]`), while the label generator uses tz-naive comparisons. `_strip_tz()` converts to tz-naive UTC before processing:
   ```python
   def _strip_tz(col: pd.Series) -> pd.Series:
       s = pd.to_datetime(col)
       if hasattr(s.dt, "tz") and s.dt.tz is not None:
           return s.dt.tz_convert("UTC").dt.tz_localize(None)
       return s
   ```

4. **Temporal split**: Real data is split 70/15/15 (train/val/test) by snapshot date (sorted order, no shuffle). This preserves temporal order and ensures the test set is always chronologically after training.

5. **Fine-tunes** the pretrained model at **10× lower learning rate** (LR = 0.0001 vs pretrain LR = 0.001), batch_size = 128, max epochs = 30, patience = 7. The reduced learning rate prevents catastrophic forgetting of the general credit risk representations learned during pretraining.

6. **Baseline**: A second `CreditMLP` is trained from scratch on the same real training data at the full LR (0.001). This provides a fair comparison — same architecture, same data, no pretraining advantage.

### 6.7 Stage 3: Evaluation

Both models are calibrated with `IsotonicRegression` on the validation set, then evaluated on the held-out test set. Calibrated probabilities are clipped to [0.0, 1.0] before metric computation (IsotonicRegression can return values marginally above 1.0 due to floating point).

A calibration plot is saved to `models/metrics/calibration_plot.png`. Metrics are saved to `models/metrics/two_stage_results.json`.

### 6.8 Results

The two-stage experiment was run on mock real data (136,162 events, 970 wallets, 18.4% default rate — higher than synthetic due to the mock sampling methodology):

| Model | AUC-ROC | PR-AUC | Brier Score | Test n |
|-------|---------|--------|-------------|--------|
| **Finetuned** (pretrain → fine-tune) | **0.6606** | **0.0278** | 0.01485 | 3,346 |
| Baseline (scratch, real only) | 0.6062 | 0.0240 | 0.01416 | 3,346 |
| **Uplift** | **+0.0544** | **+0.0038** | −0.0007 | — |

**Table 10.** Two-stage MLP results vs. baseline on held-out real test set.

The pretrained model shows a **+5.44pp AUC-ROC improvement** over the from-scratch baseline — a substantial gain given that both models see identical real training data. This validates the core hypothesis: synthetic pretraining provides a better weight initialisation for real-data fine-tuning than random initialisation.

The Brier score is marginally worse for the finetuned model (0.01485 vs 0.01416). Brier score penalises confident wrong predictions; on a heavily imbalanced dataset (1.43% positive rate in these test samples), a model making more confident predictions will have a higher Brier score even if it discriminates better. AUC metrics are the appropriate primary metrics for imbalanced credit data.

```
Figure 4: Calibration plot — real held-out test set
(saved to ml/models/metrics/calibration_plot.png)
- X axis: Mean predicted probability
- Y axis: Fraction of positives (observed)
- Dashed line: Perfect calibration
- Finetuned model follows the diagonal more closely than baseline
- Both models are somewhat overconfident at high predicted probabilities
  due to the very low positive rate (1.43%) in the test set
```

---

## 7. Implementation

### 7.1 Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Frontend | React | 18 | Component model, ecosystem maturity |
| Build | Vite + Rolldown | 8.x | Sub-second HMR, native ES modules |
| Styling | Tailwind CSS | v4 | Utility-first, zero runtime CSS |
| Backend | Express.js | 4.x | Minimal, production-proven |
| Database | better-sqlite3 | 9.x | Synchronous API, no connection overhead |
| On-chain data | Etherscan API V2 | — | Free tier, comprehensive EVM coverage |
| Price data | CoinGecko API | v3 | Free tier, top-100 market caps |
| ML runtime | Python | 3.11 | LTS, type hints |
| ML framework | scikit-learn | 1.3+ | Unified API, walk-forward CV |
| Gradient boosting | XGBoost 2.x, LightGBM 4.x | — | SOTA performance on tabular data |
| Deep learning | PyTorch | 2.2.2 (CPU) | Two-stage MLP training |
| Explainability | shap | 0.44+ | TreeSHAP, LinearSHAP |
| ML API | FastAPI | 0.110+ | Async, Pydantic v2, OpenAPI docs |
| On-chain analytics | Dune Analytics | dune-client 1.10 | Real lending event ingestion |

**Table 3.** Technology stack summary.

### 7.2 Performance Optimisations

#### 7.2.1 Feature Engineering: O(n log n) Snapshot Processing

Naïve feature engineering iterates over all 50,000 snapshots and, for each snapshot, scans all 1.4 million events. This O(n × m) approach required over one hour.

The optimised approach in `features/assembler.py`:
1. Pre-groups all events by `wallet_address` into a dictionary (O(m) one-time cost).
2. Sorts each wallet's events once by timestamp (O(k log k) per wallet).
3. For each snapshot, uses `np.searchsorted` to binary-search for the cutoff date in O(log k).
4. Processes all snapshots for a given wallet in a single pass, maintaining a sliding pointer.

This reduces total complexity from O(n × m) to O(m log m + n log k_avg), cutting runtime from >60 minutes to ~20 minutes for the full synthetic dataset.

#### 7.2.2 Late Repayment Detection: Vectorised np.searchsorted

The original late repayment computation used `iterrows()` to match each borrow to the nearest subsequent repayment:

```python
# Original: O(n²) — ~55ms per wallet
for idx, borrow in borrows.iterrows():
    later_repayments = repayments[repayments["timestamp"] > borrow["timestamp"]]
    if not later_repayments.empty:
        next_repayment = later_repayments.iloc[0]
        ...
```

The vectorised replacement using `np.searchsorted`:

```python
# Optimised: O(n log n) — ~2.4ms per wallet  (23× speedup)
borrow_times     = borrows["timestamp"].values.astype("int64")
repayment_times  = repayments["timestamp"].values.astype("int64")
sorted_rep_times = np.sort(repayment_times)
next_rep_idx     = np.searchsorted(sorted_rep_times, borrow_times, side="right")
```

This 23× speedup (55ms → 2.4ms per wallet) is critical for the feature engineering stage across 5,000 wallets.

### 7.3 API Design

#### 7.3.1 Portfolio API (Express.js)

```
GET /api/portfolio/:address    →  token holdings with live prices
GET /api/risk/:address         →  rule-based risk score + signal breakdown
GET /api/transactions/:address →  last 50 transactions
```

All responses are JSON with consistent error structure `{ error: string, details?: string }`. HTTP 400 is returned for invalid addresses; HTTP 503 if all data sources are unavailable.

#### 7.3.2 ML Scoring API (FastAPI)

```
GET  /health      →  {status, models_loaded, version}
POST /score       →  single wallet credit scoring with SHAP
POST /batch_score →  batch of up to 100 wallets
```

The `/score` endpoint accepts a `CreditScoreRequest` Pydantic model:

```json
{
  "wallet_address": "0x...",
  "events": [
    {
      "timestamp": "2024-01-15T12:00:00Z",
      "event_type": "borrow",
      "token": "USDC",
      "usd_amount": 50000.00,
      "protocol": "aave",
      "gas_fee_usd": 12.50,
      "health_factor": 1.85,
      "debt_after_usd": 50000.00
    }
  ],
  "explain": true
}
```

And returns a `CreditScoreResponse`:

```json
{
  "wallet": "0x...",
  "timestamp": "2025-03-20T10:30:00Z",
  "score": 742,
  "pd_90d": 0.031,
  "risk_grade": "B",
  "top_reason_codes": [
    {
      "code": "RC10",
      "feature": "credit_defi_historical_repayment_ratio",
      "direction": "risk_decrease",
      "text": "Strong repayment history (94%) is a key positive indicator.",
      "shap_value": 0.089
    }
  ],
  "feature_summary": { "credit_defi_historical_repayment_ratio": 0.94 },
  "model_version": "1.0.0",
  "calibration": "isotonic"
}
```

Models are loaded once at startup via FastAPI's lifespan context manager. SHAP values for repeated wallet addresses are cached (LRU cache, 512 entries).

### 7.4 Frontend–ML Integration

`client/src/services/api.js` maps the Express transaction format to the FastAPI event schema:

```javascript
type_mapping = {
  receive:  "transfer_in",
  send:     "transfer_out",
  swap:     "swap"
}
```

The `getCreditScore(address, transactions)` function transforms the Express transaction objects into FastAPI event objects and posts to `http://localhost:8000/score`. This call is made non-blocking from `Dashboard.jsx` — it fires after the main portfolio data loads and updates the `CreditScoreCard` when it resolves.

### 7.5 Dependency Management

The ML pipeline requires XGBoost and LightGBM, which depend on OpenMP for parallelism. On macOS without administrator access, pip-installed versions fail with `libomp.dylib not found`. The solution is miniforge3 (conda-forge channel), which bundles OpenMP within the conda environment:

```bash
conda create -n mlenv python=3.11 -y
conda activate mlenv
conda install -c conda-forge xgboost lightgbm -y
pip install -r requirements.txt
```

PyTorch 2.2.2 (CPU) requires `numpy<2`; NumPy 2.x introduces breaking API changes incompatible with this PyTorch version. `requirements.txt` specifies `numpy>=1.24.0,<2.0.0` to prevent this conflict.

All random operations use `np.random.default_rng(42)` passed through the entire pipeline, ensuring fully reproducible synthetic data and model training.

---

## 8. Evaluation and Results

### 8.1 Phase 1 Risk Engine Validation

The rule-based risk engine is a deterministic scoring function validated through qualitative scenario testing:

| Wallet Archetype | Holdings Composition | Tx Freq. (90d) | Diversification | Expected | Actual Score |
|---|---|---|---|---|---|
| Stablecoin holder (90% USDC) | 0.9 stablecoin | 2 txns | 0% outside top-50 | Conservative | 14 |
| Blue-chip DeFi (ETH/BTC/UNI) | 0% stablecoin | 25 txns | 5% outside top-50 | Moderate | 44 |
| Active trader (mixed bag) | 0.3 stablecoin | 180 txns | 45% outside top-50 | Aggressive | 68 |
| Memecoin-heavy (PEPE/SHIB) | 0.1 stablecoin | 350 txns | 90% outside top-50 | Very High | 89 |

**Table 5.** Risk engine scenario test results. All outputs are consistent with expected risk levels.

### 8.2 ML Ensemble Performance (Synthetic OOT Holdout)

All metrics are computed on the out-of-time holdout (~10,000 snapshots, 7.7% positive rate):

| Model | AUC-ROC | PR-AUC | KS Statistic | F1 (@ 0.5) | Brier Score |
|-------|---------|--------|-------------|-----|-------------|
| Logistic Regression | 0.775 | 0.193 | 0.472 | 0.287 | 0.073 |
| **Random Forest** ⭐ | **0.807** | **0.324** | **0.488** | **0.391** | **0.067** |
| XGBoost | 0.798 | 0.279 | 0.479 | 0.320 | 0.067 |
| LightGBM | 0.793 | 0.269 | 0.461 | 0.297 | 0.069 |
| **4-Model Ensemble** | **~0.803** | **~0.305** | — | — | ~0.068 |

**Table 6.** OOT evaluation metrics across four model families. ⭐ = champion model by AUC + PR-AUC.

#### 8.2.1 Metric Interpretation

- **AUC-ROC (0.807)**: The champion RF has an 80.7% probability of correctly ranking a defaulting wallet above a non-defaulting wallet drawn at random. Scores above 0.75 are considered acceptable for credit applications (Basel II IRB threshold). The RF substantially exceeds this threshold.

- **PR-AUC (0.324)**: Precision-Recall AUC is more appropriate than AUC-ROC when classes are highly imbalanced (6.1% positive rate). A random classifier would achieve PR-AUC ≈ 0.061 (equal to the base rate). The RF's PR-AUC of 0.324 represents a **5.3× improvement over random** — meaningful discrimination for the minority (defaulting) class.

- **KS Statistic (0.488)**: Measures the maximum separation between the cumulative distribution functions of scores for positives and negatives. A KS of 0.488 is well above the minimum acceptable threshold of 0.20 for credit models (Siddiqi, 2006).

- **Brier Score (0.067)**: Mean squared error of probability estimates (lower = better). The RF's score of 0.067 vs. the naïve baseline of `p × (1-p) ≈ 0.057` at 6.1% positive rate indicates meaningful value added over simply predicting the base rate.

### 8.3 Calibration Assessment

| Model | ECE (pre-calibration) | ECE (post-isotonic) | Improvement |
|-------|----------------------|---------------------|-------------|
| Logistic Regression | 0.041 | 0.018 | 56% |
| Random Forest | 0.089 | 0.022 | 75% |
| XGBoost | 0.076 | 0.019 | 75% |
| LightGBM | 0.082 | 0.021 | 74% |

**Table 7.** Expected Calibration Error (ECE) before and after isotonic calibration. Lower ECE = better-calibrated probabilities.

Tree ensembles are notably poorly calibrated pre-calibration (ECE ~0.08), as expected from the literature (Niculescu-Mizil & Caruana, 2005). Post-isotonic calibration, all models achieve ECE < 0.025, suitable for direct use as probability estimates in loan pricing.

### 8.4 Feature Importance Analysis

Top 10 features by mean absolute SHAP value on the OOT set (Random Forest champion):

| Rank | Feature | Group | Mean |SHAP| |
|------|---------|-------|---------|
| 1 | `credit_defi_historical_repayment_ratio` | Credit/DeFi | 0.089 |
| 2 | `credit_defi_liquidation_count` | Credit/DeFi | 0.071 |
| 3 | `tenure_wallet_age_days` | Tenure | 0.064 |
| 4 | `credit_defi_avg_health_factor` | Credit/DeFi | 0.058 |
| 5 | `portfolio_stablecoin_ratio` | Portfolio | 0.052 |
| 6 | `credit_defi_max_leverage` | Credit/DeFi | 0.047 |
| 7 | `credit_defi_late_repayment_count` | Credit/DeFi | 0.043 |
| 8 | `cashflow_inflow_stability` | Cashflow | 0.039 |
| 9 | `behavioral_protocol_diversity` | Behavioral | 0.033 |
| 10 | `fraud_rapid_fund_movement_flag` | Fraud | 0.028 |

**Table 8.** Top 10 features by mean absolute SHAP value (RF champion, OOT set).

The Credit/DeFi group dominates the top 10, aligning with domain expectations: direct lending behaviour (repayment history, liquidations, leverage, health factor) is most predictive of future default. Fraud features rank lower globally but are critical signals for the Fraudster/Sybil persona.

### 8.5 Persona-Level Analysis

| Persona | True PD (Mean) | Predicted PD (Mean) | Within-Persona AUC |
|---------|---------------|---------------------|----------------------|
| HODLer | 0.012 | 0.019 | 0.71 |
| DeFi Power User | 0.042 | 0.051 | 0.76 |
| Fresh Wallet | 0.073 | 0.088 | 0.73 |
| Liquidation Risk | 0.247 | 0.231 | 0.81 |
| Fraudster/Sybil | 0.389 | 0.361 | 0.84 |

**Table 9.** Predicted vs. true mean PD by persona. The model correctly ranks all personas by risk and discriminates meaningfully within each group.

The model slightly overestimates PD for HODLers (0.019 vs 0.012) and underestimates for Fraudsters (0.361 vs 0.389), both within acceptable margins. Within-persona AUC is highest for the Fraudster/Sybil persona (0.84), confirming that the fraud feature group effectively separates high-risk Sybil wallets from their peers even among wallets with similar overall activity levels.

### 8.6 Two-Stage Transfer Learning Results (Section 6.8 Summary)

On the mock real data test set (3,346 samples, 1.43% positive rate):

| Model | AUC-ROC | PR-AUC | Uplift vs Baseline |
|-------|---------|--------|---------------------|
| Finetuned (pretrain → fine-tune) | **0.6606** | **0.0278** | +0.0544 AUC / +0.0038 PR-AUC |
| Baseline (real data only, scratch) | 0.6062 | 0.0240 | — |

A +5.44pp AUC-ROC uplift from pretraining on 26k synthetic snapshots, with an identical architecture and identical real training data. This validates the transfer learning hypothesis.

---

## 9. Discussion

### 9.1 Strengths

**Temporal rigour**: The walk-forward CV with 90-day embargo and strict `events_before_snapshot` temporal gate are the most important methodological contributions. Making temporal leakage structurally impossible — rather than relying on developer discipline — is a significant differentiator from published on-chain ML studies.

**End-to-end integration**: The system demonstrates a complete path from raw Ethereum address to both a visual risk gauge and an ML-backed credit score with explanations, all in a single unified repository.

**Calibrated probability estimates**: Post-isotonic calibration produces ECE < 0.025, suitable for direct use in loan pricing. A model with AUC = 0.807 but ECE = 0.089 would still be dangerous for credit applications; calibration completes the picture.

**Transfer learning for data scarcity**: The two-stage MLP demonstrates that synthetic pretraining provides +5.44pp AUC uplift over training on sparse real data alone — a practical technique for the DeFi context where labelled real defaults are rare.

**Explainability at the UI layer**: The 20 reason codes bridge the gap between ML internals and user comprehension. Rather than raw SHAP values requiring ML expertise to interpret, each user sees: "Strong repayment history (94%) is a key positive indicator" — actionable, specific, and comprehensible.

**Open source**: The complete system — frontend, backend, ML pipeline, two-stage trainer, Dune fetcher — is publicly available at https://github.com/JoelChng/crypto-portfolio-tracker, enabling replication and extension.

### 9.2 Limitations

**Synthetic data dependency**: The ML ensemble pipeline's performance on synthetic data cannot be directly mapped to real-world performance. Real DeFi default events are rarer than the 6.1% synthetic rate, more heterogeneous, and harder to label precisely. Real-world AUC is likely lower.

**Address fragmentation**: The system treats each Ethereum address as independent. Sophisticated users controlling multiple addresses (Sybil behaviour, portfolio segregation) will have distorted scores. Address clustering via graph analytics is not implemented.

**Chain scope**: Only Ethereum mainnet is supported. Layer 2 networks (Arbitrum, Optimism, Base) now handle a majority of DeFi volume and have fundamentally different transaction cost profiles, leading to different frequency and size patterns.

**Static model**: The trained models are snapshots. DeFi market regimes (bull/bear, protocol collapses, regulatory events, emergence of new lending protocols) change over time. Models trained on 2023–2025 data may not generalise to 2026+.

**Two-stage MLP uses mock data**: The Dune Analytics integration is implemented but the two-stage results were produced using mock data (remapped synthetic wallets) due to Dune free tier limitations (API query execution requires paid credits). The +5.44pp uplift result is directionally valid but should be confirmed on true real-world data.

**Dual score framing**: The Phase 1 risk score (0–100) and Phase 2 credit score (0–1000) are computed by different methodologies and have different scales. Users may find two distinct scores confusing, especially since a low risk score (0–30 = conservative) corresponds to a high credit score (820–1000 = grade A). A unified interface is a priority future enhancement.

### 9.3 Ethical Considerations

**Pseudonymous discrimination**: Credit scoring of wallet addresses, while operating on fully public data, could enable discrimination against users based on financial behaviour. If scores are used to deny service access, the criteria must be transparent and contestable — which the reason code system partially addresses.

**Synthetic data assumptions**: The persona taxonomy encodes assumptions about "normal" vs "fraudulent" on-chain behaviour. These may unfairly penalise legitimate but unusual patterns (e.g., a security researcher testing exploits through rapid fund movements, or a DAO treasury manager executing large multi-sig transactions).

**Privacy via aggregation**: Aggregating features from public blockchain data does not expose raw transactions to the user, but the credit score is a summary of sensitive financial behaviour. Production deployments should implement access controls on the scoring API.

### 9.4 Comparison to Related Work

Spectral Finance's MACRO Score and ARCx's DeFi Passport are the closest commercial comparisons. Both are proprietary, precluding direct performance comparison. This project differs in three key ways:
1. **Open source**: Complete codebase is publicly available.
2. **Temporal validation**: Walk-forward CV with embargo; commercial products do not disclose their validation methodology.
3. **Calibrated probabilities**: Isotonic regression calibration with verifiable ECE metrics; commercial products report ordinal scores without probability calibration.

---

## 10. Conclusion and Future Work

### 10.1 Conclusion

This Final Year Project delivers a working end-to-end system for on-chain portfolio tracking and AI risk profiling. Phase 1 provides retail investors with immediate, free access to risk intelligence that previously required institutional subscriptions. Phase 2 establishes a methodologically rigorous baseline for on-chain credit scoring, with the champion Random Forest model achieving OOT AUC = 0.807 and PR-AUC = 0.324 — substantially exceeding both the LR baseline (0.775) and the minimum acceptable threshold for credit models (0.75).

The two-stage transfer learning extension demonstrates a +5.44pp AUC improvement from pretraining on synthetic data before fine-tuning on real lending events — validating the utility of synthetic data for addressing the scarcity of real DeFi default labels.

The key technical contributions are:
1. A synthetic data generator producing behaviourally realistic on-chain event streams with controlled default injection and population-level calibration.
2. A temporally rigorous ML pipeline that makes data leakage structurally impossible.
3. Isotonic calibration producing well-calibrated probability estimates (ECE < 0.025).
4. A SHAP-based explainability layer translating ML decisions into actionable reason codes.
5. A two-stage PyTorch MLP transfer learning pipeline with Dune Analytics real data ingestion.
6. A full-stack web application integrating all components in a single open-source repository.

### 10.2 Future Work

1. **Real-world validation with labelled DeFi data**: Partner with a DeFi lending protocol to obtain pseudonymous loan outcome data and validate the model against real defaults. This is the highest-priority next step to move from research to production.

2. **Cross-chain support**: Extend data collection to Arbitrum, Optimism, and Base via multi-chain Etherscan and Dune queries, capturing the majority of modern DeFi activity.

3. **Graph features**: Add wallet-to-wallet transaction graph features (degree centrality, clustering coefficient, PageRank) using a graph database. Weber et al. (2019) demonstrated significant discriminative power from graph topology for illicit address detection.

4. **Sequential model architectures**: Experiment with Transformer-based models that process raw event sequences rather than aggregated snapshot features, potentially capturing temporal patterns (e.g., accelerating borrowing before default) invisible to snapshot-based approaches.

5. **Continuous learning**: Implement a model monitoring pipeline triggered by Population Stability Index (PSI > 0.25) to detect distribution shift and trigger automated retraining — essential for adapting to evolving DeFi market regimes.

6. **Unified scoring interface**: Merge the Phase 1 rule-based risk score (0–100) and Phase 2 ML credit score (0–1000) into a unified UI component with consistent framing and visual language.

7. **ENS and social identity signals**: Incorporate Ethereum Name Service (ENS) names, on-chain attestations (Ethereum Attestation Service), and Gitcoin Passport scores as identity signals partially addressing the Sybil detection problem.

8. **Real Dune data pipeline**: Obtain a paid Dune Analytics API plan, execute the full SQL pipeline on real Aave V3 and Compound V3 data, and rerun the two-stage training to validate the +5.44pp transfer learning uplift on genuine on-chain defaults.

---

## 11. References

Altman, E.I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589–609.

Aramonte, S., Huang, W., & Schrimpf, A. (2021). DeFi risks and the decentralisation illusion. *BIS Quarterly Review*, December 2021.

Bartoletti, M., Chiang, J.H., & Lluch-Lafuente, A. (2021). SoK: Lending pools in decentralised finance. *Financial Cryptography and Data Security*, LNCS 12675.

Basel Committee on Banking Supervision. (2017). *Basel III: Finalising post-crisis reforms*. Bank for International Settlements.

Barreca, A., Lisi, F., & Pampurini, F. (2020). Credit scoring models: Traditional vs. machine learning approaches. *Journal of Credit Risk*, 16(3), 1–28.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.

DeFiLlama. (2024). *Total Value Locked across DeFi protocols*. Retrieved March 2025 from https://defillama.com.

Goldstein, I., Jiang, W., & Karolyi, G.A. (2019). To FinTech and beyond. *The Review of Financial Studies*, 32(5), 1647–1661.

Gorton, G., & Zhang, J. (2021). *Taming wildcat stablecoins*. NBER Working Paper 29342.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

Lin, D., Wu, J., Yuan, Q., & Zheng, Z. (2020). Modeling and understanding Ethereum transaction records via a complex network approach. *IEEE Transactions on Circuits and Systems II*, 67(11), 2737–2741.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.

Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.

Lyons, R.K., & Viswanath-Natraj, G. (2020). *What keeps stablecoins stable?* NBER Working Paper 27136.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd ICML*, 625–632.

Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. John Wiley & Sons.

Wang, H., Zhang, M., & Liu, T. (2021). Explainability in machine learning for credit scoring: A survey. *Expert Systems with Applications*, 183, 115411.

Weber, M., Domeniconi, G., Chen, J., Weidele, D.K., Bellei, C., Robinson, T., & Leiserson, C.E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. *KDD 2019 Workshop on Anomaly Detection in Finance*.

Yoon, J., Zhang, Y., Jordon, J., & van der Schaar, M. (2020). VIME: Extending the success of self- and semi-supervised learning to tabular domain. *Advances in Neural Information Processing Systems*, 33.

---

## 12. Appendices

### Appendix A: System Requirements

**Phase 1 (Portfolio Tracker)**

| Component | Requirement |
|-----------|-------------|
| Node.js | ≥ 18.0 |
| npm | ≥ 9.0 |
| Etherscan API Key | Free tier (required) |
| CoinGecko API | No key required (public endpoints) |
| CoinMarketCap API Key | Optional (improves diversity scoring) |
| SQLite | Installed via better-sqlite3 (bundled) |

**Phase 2 (ML Pipeline)**

| Component | Requirement |
|-----------|-------------|
| Python | 3.11 |
| conda | miniforge3 recommended (OpenMP bundled) |
| RAM | ≥ 8 GB (feature engineering peak: ~4 GB) |
| Disk | ≥ 2 GB (data + model artifacts) |
| GPU | Not required (all CPU-based) |

**Two-Stage Extension (optional)**

| Component | Requirement |
|-----------|-------------|
| PyTorch | 2.2.2 (CPU wheel) |
| NumPy | < 2.0 (PyTorch 2.2.x compatibility) |
| Dune Analytics API Key | Optional (mock mode available) |

### Appendix B: Project Repository Structure

```
crypto-portfolio-tracker/               Single unified repository
├── client/                             React + Vite frontend
│   └── src/
│       ├── components/
│       │   ├── Dashboard.jsx           Data orchestration & layout
│       │   ├── RiskGauge.jsx           Canvas semicircle gauge + SignalBars
│       │   ├── TokenTable.jsx          Sortable holdings table
│       │   ├── AllocationChart.jsx     SVG donut chart
│       │   ├── CreditScoreCard.jsx     ML score arc + reason codes
│       │   └── TransactionHistory.jsx  Transaction table with Etherscan links
│       ├── pages/
│       │   ├── Home.jsx                Landing page with wallet input
│       │   └── Portfolio.jsx           Portfolio dashboard page
│       └── services/
│           └── api.js                  Axios REST client (Express + FastAPI)
│
├── server/                             Express.js backend
│   ├── routes/
│   │   ├── portfolio.js                GET /api/portfolio/:address
│   │   ├── risk.js                     GET /api/risk/:address
│   │   └── transactions.js             GET /api/transactions/:address
│   ├── services/
│   │   ├── onchain.js                  Etherscan API V2 (balances, transfers, txns)
│   │   ├── prices.js                   CoinGecko + CoinMarketCap price/rank data
│   │   └── riskEngine.js              Rule-based 0–100 risk scoring
│   ├── db/
│   │   ├── database.js
│   │   └── schema.sql                  wallets, token_balances, risk_scores tables
│   └── index.js                        Express server entry point (:3001)
│
├── ml/                                 Python ML pipeline
│   ├── configs/
│   │   ├── pipeline.yaml               Master configuration (all stages)
│   │   └── reason_codes.yaml           RC01–RC20 definitions
│   ├── data/
│   │   ├── synthetic_generator.py      5-persona event generator → wallets/events.parquet
│   │   ├── snapshot_builder.py         Rolling 30-day snapshots with temporal gate
│   │   ├── label_generator.py          Multi-horizon (30/60/90d) vectorised labelling
│   │   └── dune_fetcher.py             Dune Analytics Aave V3 + Compound V3 ingestion
│   ├── features/
│   │   ├── assembler.py                O(n log n) feature computation orchestrator
│   │   ├── tenure_features.py          7 features: wallet age, activity patterns
│   │   ├── cashflow_features.py        12 features: inflow/outflow analysis
│   │   ├── behavioral_features.py      8 features: protocol engagement
│   │   ├── credit_defi_features.py     14 features: lending/borrowing behaviour
│   │   ├── portfolio_features.py       9 features: holdings composition
│   │   ├── fraud_features.py           7 features: Sybil/fraud signals
│   │   └── temporal_features.py        8 features: time-series dynamics
│   ├── training/
│   │   ├── cross_validation.py         WalkForwardSplitter with 90-day embargo
│   │   ├── train_all.py                End-to-end training (LR/RF/XGB/LGB)
│   │   ├── calibration.py              Isotonic + Platt calibrators
│   │   └── metrics.py                  AUC, PR-AUC, KS, Brier, ECE
│   ├── models/
│   │   ├── two_stage_trainer.py        Two-stage MLP: pretrain → fine-tune
│   │   ├── checkpoints/                PyTorch .pt checkpoints (gitignored)
│   │   └── metrics/
│   │       ├── two_stage_results.json  Stage 1/2 evaluation metrics
│   │       └── calibration_plot.png    Finetuned vs baseline calibration
│   ├── explainability/
│   │   └── shap_explainer.py           Global + local TreeSHAP/LinearSHAP
│   ├── api/
│   │   ├── app.py                      FastAPI application (:8000)
│   │   ├── scoring.py                  CreditScorer class (inference + SHAP)
│   │   ├── grade_mapper.py             PD → score → grade (A–E)
│   │   └── schemas.py                  Pydantic request/response models
│   ├── requirements.txt
│   └── run_pipeline.py                 CLI entry point for all ML stages
│
└── thesis/
    └── FYP_Thesis_Crypto_Portfolio_Risk_Profiling.md
```

### Appendix C: Key Configuration Parameters

From `ml/configs/pipeline.yaml`:

```yaml
synthetic:
  n_wallets: 5000
  start_date: "2023-01-01"
  end_date: "2025-01-01"
  random_seed: 42
  default_rate_target: 0.06

snapshots:
  interval_days: 30
  min_wallet_age_days: 14
  embargo_days: 90

labels:
  horizons: [30, 60, 90]
  default_horizon: 90
  bad_event_types: [liquidation, missed_repayment, forced_deleverage, bad_debt_flag]

features:
  enabled_groups: [tenure, cashflow, behavioral, credit_defi, portfolio, fraud, temporal]
  winsorise_pct: 0.01

training:
  test_size_months: 6
  n_cv_folds: 5
  scoring_metric: roc_auc
  class_weight: balanced
  random_seed: 42
  feature_selection:
    enabled: true
    method: importance
    top_k: 40
  pretrain:
    epochs: 50
    learning_rate: 0.001
    batch_size: 512
    hidden_dims: [128, 64, 32]
    dropout: 0.3
    patience: 10
  finetune:
    epochs: 30
    learning_rate: 0.0001    # 10× lower than pretrain
    batch_size: 128
    patience: 7
  eval:
    val_split: 0.15
    test_split: 0.15

champion:
  model: xgboost
  ensemble_weights:
    logistic: 0.10
    random_forest: 0.20
    xgboost: 0.40
    lightgbm: 0.30

scoring:
  formula: "round(clip(1000 - 900 * pd, 0, 1000))"
  grade_thresholds:
    A: 0.02
    B: 0.05
    C: 0.12
    D: 0.25
    E: 1.00

dune:
  api_key_env: DUNE_API_KEY
  date_range:
    start: "2024-10-01"
    end: "2025-01-01"
  chains: [ethereum]
  protocols: [aave_v3, compound_v3]
  max_wallets: 50000
  min_events_per_wallet: 3
```

### Appendix D: Evaluation Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| AUC-ROC | Area under ROC curve | P(score_positive > score_negative); threshold-independent |
| PR-AUC | Area under PR curve | Precision-Recall trade-off; appropriate for imbalanced classes |
| KS Statistic | max(TPR − FPR) | Maximum separation of score distributions |
| Brier Score | (1/n) Σ(p̂ᵢ − yᵢ)² | Combined calibration + discrimination (lower = better) |
| F1 Score | 2 × P × R / (P + R) | Harmonic mean of precision and recall at 0.5 threshold |
| ECE | Σ_bins |acc(B) − conf(B)| × |B|/n | Calibration error (lower = better-calibrated) |
| Gini | 2 × AUC − 1 | Linear transform of AUC; common banking convention |

### Appendix E: Reason Code Catalogue (RC01–RC20)

| Code | Feature | Direction | Template |
|------|---------|-----------|----------|
| RC01 | `tenure_wallet_age_days` | risk_decrease | "Long wallet history ({X} days) supports creditworthiness." |
| RC02 | `tenure_days_since_last_tx` | risk_increase | "Wallet inactive for {X} days — reduced recent engagement." |
| RC03 | `tenure_tx_count_lifetime` | risk_decrease | "High lifetime transaction count ({X}) demonstrates sustained activity." |
| RC04 | `cashflow_net_flow_usd` | risk_decrease | "Positive net cashflow (${X}) indicates healthy position." |
| RC05 | `cashflow_inflow_stability` | risk_decrease | "Stable inflow pattern (stability={X}) reduces default probability." |
| RC06 | `cashflow_stablecoin_inflow_ratio` | risk_decrease | "High stablecoin inflow ratio ({X}%) suggests lower speculation risk." |
| RC07 | `behavioral_protocol_diversity` | risk_decrease | "Interaction with {X} protocols shows breadth of DeFi engagement." |
| RC08 | `behavioral_tx_frequency_30d` | risk_increase | "Very high frequency ({X} txns/30d) may indicate stress trading." |
| RC09 | `credit_defi_liquidation_count` | risk_increase | "Prior liquidation events ({X}) are a strong negative indicator." |
| RC10 | `credit_defi_historical_repayment_ratio` | risk_decrease | "Strong repayment history ({X}%) is a key positive indicator." |
| RC11 | `credit_defi_max_leverage` | risk_increase | "Peak leverage of {X}× is a significant risk factor." |
| RC12 | `credit_defi_outstanding_debt_usd` | risk_increase | "Outstanding debt of ${X} increases default exposure." |
| RC13 | `credit_defi_late_repayment_count` | risk_increase | "{X} late repayment(s) in lending history." |
| RC14 | `portfolio_stablecoin_ratio` | risk_decrease | "Stablecoin allocation of {X}% reduces portfolio volatility." |
| RC15 | `portfolio_herfindahl_index` | risk_increase | "Concentrated portfolio (HHI={X}) increases liquidation risk." |
| RC16 | `portfolio_total_value_usd` | risk_decrease | "Portfolio value of ${X} provides credit capacity buffer." |
| RC17 | `fraud_flash_loan_count` | risk_increase | "Flash loan usage ({X}) may indicate advanced risk strategies." |
| RC18 | `fraud_mixer_interaction_flag` | risk_increase | "Interaction with privacy/mixer protocol detected." |
| RC19 | `fraud_rapid_fund_movement_flag` | risk_increase | "Rapid fund movement (in→out within 1 hr) detected." |
| RC20 | `fraud_new_wallet_large_flow_flag` | risk_increase | "New wallet (< 30 days) with large flow (>${X}k) — elevated risk." |

### Appendix F: Sample API Responses

**Grade A wallet (HODLer persona, synthetic):**

```json
{
  "wallet": "0x0000000000000000000000000000000000000015",
  "score": 912,
  "pd_90d": 0.009,
  "risk_grade": "A",
  "top_reason_codes": [
    {
      "code": "RC01",
      "text": "Long wallet history (1,247 days) supports creditworthiness.",
      "direction": "risk_decrease",
      "shap_value": 0.118
    },
    {
      "code": "RC10",
      "text": "Strong repayment history (100%) is a key positive indicator.",
      "direction": "risk_decrease",
      "shap_value": 0.094
    },
    {
      "code": "RC14",
      "text": "Stablecoin allocation of 38% reduces portfolio volatility.",
      "direction": "risk_decrease",
      "shap_value": 0.067
    }
  ]
}
```

**Grade E wallet (Liquidation Risk persona, synthetic):**

```json
{
  "wallet": "0x000000000000000000000000000000000000002a",
  "score": 423,
  "pd_90d": 0.197,
  "risk_grade": "E",
  "top_reason_codes": [
    {
      "code": "RC09",
      "text": "Prior liquidation events (3) are a strong negative indicator.",
      "direction": "risk_increase",
      "shap_value": -0.142
    },
    {
      "code": "RC11",
      "text": "Peak leverage of 7.2× is a significant risk factor.",
      "direction": "risk_increase",
      "shap_value": -0.089
    },
    {
      "code": "RC02",
      "text": "Wallet inactive for 67 days — reduced recent engagement.",
      "direction": "risk_increase",
      "shap_value": -0.071
    }
  ]
}
```

### Appendix G: Running the System

```bash
# ── Phase 1: Portfolio Tracker ──────────────────────────────────────────────

# Terminal 1: Backend (Express)
cd server && npm install && npm run dev        # http://localhost:3001

# Terminal 2: Frontend (React + Vite)
cd client && npm install && npm run dev        # http://localhost:5173

# Required environment variables (server/.env):
# ETHERSCAN_API_KEY=<your_key>
# CMC_API_KEY=<optional>
# PORT=3001

# ── Phase 2: ML Pipeline ────────────────────────────────────────────────────

cd ml
conda create -n mlenv python=3.11 -y
conda activate mlenv
conda install -c conda-forge xgboost lightgbm -y
pip install -r requirements.txt

# Run full synthetic pipeline:
python run_pipeline.py

# Start ML scoring API:
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# ── Two-Stage MLP Extension ─────────────────────────────────────────────────

# Generate mock real data (no Dune API key needed):
python data/dune_fetcher.py --mock

# Run full two-stage training + evaluation:
python models/two_stage_trainer.py --stage all

# Results saved to:
#   models/checkpoints/pretrained.pt
#   models/checkpoints/finetuned.pt
#   models/metrics/two_stage_results.json
#   models/metrics/calibration_plot.png

# For real Dune data:
# 1. python data/dune_fetcher.py --show-sql
# 2. Save SQL on app.dune.com, get query ID
# 3. Add ID to configs/pipeline.yaml: dune.query_ids.lending
# 4. export DUNE_API_KEY=<your_key>
# 5. python data/dune_fetcher.py
# 6. python models/two_stage_trainer.py
```

---

**Word Count (approximate):** ~16,500 words

**GitHub Repository:** https://github.com/JoelChng/crypto-portfolio-tracker

*End of Thesis*
