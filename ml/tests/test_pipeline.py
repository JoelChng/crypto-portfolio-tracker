"""
Smoke tests for the core pipeline modules.
Run from ml/ directory: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


# ─── Synthetic Generator ──────────────────────────────────────────────────────

def test_synthetic_generator_schema():
    from data.synthetic_generator import SyntheticGenerator
    cfg = {
        "n_wallets": 50,
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "random_seed": 42,
    }
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SyntheticGenerator(cfg)
        wallets, events = gen.generate(tmpdir)
        assert len(wallets) == 50
        assert "wallet_address" in wallets.columns
        assert "pd_true" in wallets.columns
        assert wallets["pd_true"].between(0, 1).all()
        assert len(events) > 0
        assert "timestamp" in events.columns
        assert "event_type" in events.columns


def test_persona_distribution():
    from data.synthetic_generator import SyntheticGenerator
    cfg = {"n_wallets": 1000, "start_date": "2023-01-01",
           "end_date": "2024-06-01", "random_seed": 0}
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SyntheticGenerator(cfg)
        wallets, _ = gen.generate(tmpdir)
        counts = wallets["persona_id"].value_counts(normalize=True)
        assert len(counts) == 5
        for pid, expected_weight in {0: 0.30, 1: 0.25, 2: 0.20, 3: 0.15, 4: 0.10}.items():
            assert abs(counts.get(pid, 0) - expected_weight) < 0.08


# ─── Snapshot Builder ─────────────────────────────────────────────────────────

def test_snapshot_builder_no_leakage():
    from data.snapshot_builder import build_snapshots, events_before_snapshot
    wallets = pd.DataFrame([{
        "wallet_address": "0xABC",
        "first_seen": pd.Timestamp("2023-01-01"),
        "last_seen": pd.Timestamp("2024-01-01"),
    }])
    snaps = build_snapshots(wallets, end_date=pd.Timestamp("2024-06-01"),
                            interval_days=30, max_horizon_days=90)
    assert len(snaps) > 0
    # All snapshots must be before end_date - 90d
    cutoff = pd.Timestamp("2024-06-01") - pd.Timedelta(days=90)
    assert (pd.to_datetime(snaps["snapshot_date"]) <= cutoff).all()


def test_events_before_snapshot_exclusive():
    from data.snapshot_builder import events_before_snapshot
    events = pd.DataFrame([
        {"wallet_address": "0xA", "timestamp": pd.Timestamp("2023-06-01"), "usd_amount": 100},
        {"wallet_address": "0xA", "timestamp": pd.Timestamp("2023-07-01"), "usd_amount": 200},
        {"wallet_address": "0xA", "timestamp": pd.Timestamp("2023-08-01"), "usd_amount": 300},
    ])
    snap_date = pd.Timestamp("2023-07-01")
    result = events_before_snapshot(events, "0xA", snap_date)
    assert len(result) == 1
    assert result.iloc[0]["usd_amount"] == 100   # only June event, July is exclusive


# ─── Feature Groups ───────────────────────────────────────────────────────────

def _make_events(n=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="5D")
    return pd.DataFrame({
        "wallet_address": "0xTEST",
        "timestamp": dates,
        "event_type": rng.choice(["transfer_in", "transfer_out", "borrow", "repayment",
                                   "swap", "deposit"], size=n),
        "token": rng.choice(["ETH", "USDC", "USDT", "WBTC"], size=n),
        "usd_amount": np.abs(rng.lognormal(5, 1, n)),
        "protocol": rng.choice(["aave", "uniswap_v3", "compound"], size=n),
        "gas_fee_usd": np.abs(rng.lognormal(2, 0.5, n)),
        "health_factor": rng.uniform(1.0, 3.0, n),
        "debt_after_usd": rng.uniform(0, 1000, n),
    })


def test_tenure_features_empty():
    from features.tenure_features import compute, _empty
    empty_ev = pd.DataFrame(columns=["wallet_address", "timestamp"])
    result = compute(empty_ev, pd.Timestamp("2023-06-01"), "0xA")
    assert "tenure_wallet_age_days" in result
    assert result["tenure_wallet_age_days"] == 0


def test_cashflow_features():
    from features.cashflow_features import compute
    ev = _make_events()
    snapshot = pd.Timestamp("2023-08-01")
    result = compute(ev, snapshot, "0xTEST")
    assert "cashflow_total_inflow_usd" in result
    assert "cashflow_stablecoin_inflow_ratio" in result
    assert 0 <= result["cashflow_stablecoin_inflow_ratio"] <= 1


def test_credit_defi_features_repayment_ratio():
    from features.credit_defi_features import compute
    ev = pd.DataFrame([
        {"wallet_address": "0xA", "timestamp": pd.Timestamp("2023-01-10"),
         "event_type": "borrow", "usd_amount": 1000, "health_factor": 1.5,
         "debt_after_usd": 1000, "token": "ETH", "protocol": "aave", "gas_fee_usd": 5},
        {"wallet_address": "0xA", "timestamp": pd.Timestamp("2023-02-01"),
         "event_type": "repayment", "usd_amount": 800, "health_factor": 2.0,
         "debt_after_usd": 200, "token": "USDC", "protocol": "aave", "gas_fee_usd": 5},
    ])
    result = compute(ev, pd.Timestamp("2023-03-01"), "0xA")
    assert abs(result["credit_defi_historical_repayment_ratio"] - 0.8) < 0.01
    assert result["credit_defi_borrow_count"] == 1


def test_fraud_features_rapid_movement():
    from features.fraud_features import compute
    ev = pd.DataFrame([
        {"wallet_address": "0xF", "timestamp": pd.Timestamp("2023-01-01 10:00"),
         "event_type": "transfer_in", "usd_amount": 50000, "token": "ETH",
         "protocol": "none", "gas_fee_usd": 10, "health_factor": 2.0, "debt_after_usd": 0},
        {"wallet_address": "0xF", "timestamp": pd.Timestamp("2023-01-01 10:30"),
         "event_type": "transfer_out", "usd_amount": 45000, "token": "ETH",
         "protocol": "none", "gas_fee_usd": 10, "health_factor": 2.0, "debt_after_usd": 0},
    ])
    result = compute(ev, pd.Timestamp("2023-02-01"), "0xF")
    assert result["fraud_rapid_fund_movement_flag"] == 1


# ─── Walk-forward CV ──────────────────────────────────────────────────────────

def test_walk_forward_no_embargo_violation():
    from training.cross_validation import WalkForwardSplitter
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="3D")
    df = pd.DataFrame({"snapshot_date": dates})
    splitter = WalkForwardSplitter(n_splits=3, embargo_days=90, min_train_months=4)
    for train_idx, test_idx in splitter.split(df):
        train_dates = df.iloc[train_idx]["snapshot_date"]
        test_dates  = df.iloc[test_idx]["snapshot_date"]
        gap = (test_dates.min() - train_dates.max()).days
        assert gap >= 90, f"Embargo violated: gap={gap}d"


# ─── Grade Mapper ─────────────────────────────────────────────────────────────

def test_grade_mapper():
    from api.grade_mapper import pd_to_score, pd_to_grade
    assert pd_to_score(0.0) == 1000
    assert pd_to_score(1.0) == 100
    assert pd_to_score(0.5) == 550
    assert pd_to_grade(0.01) == "A"
    assert pd_to_grade(0.04) == "B"
    assert pd_to_grade(0.10) == "C"
    assert pd_to_grade(0.20) == "D"
    assert pd_to_grade(0.50) == "E"


# ─── Metrics ─────────────────────────────────────────────────────────────────

def test_metrics_balanced():
    from training.metrics import evaluate
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 2, 500)
    # Good model: probs correlated with labels
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.2, 500), 0, 1)
    m = evaluate(y_true, y_prob)
    assert m["roc_auc"] > 0.7
    assert 0 < m["opt_threshold"] < 1
    assert 0 <= m["brier_score"] <= 1
