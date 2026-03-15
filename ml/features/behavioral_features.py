"""Group 3 — Behavioral Features"""

import pandas as pd
import numpy as np


def compute(events_df: pd.DataFrame, snapshot_date: pd.Timestamp, wallet_address: str) -> dict:
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])

    if ev.empty:
        return _empty()

    # Transaction frequency per window
    freq = {}
    for days in [7, 30, 90]:
        cutoff = snapshot_date - pd.Timedelta(days=days)
        freq[f"behavioral_tx_frequency_{days}d"] = int(ev[ev["timestamp"] >= cutoff].shape[0])

    # Protocol diversity
    n_protocols = int(ev["protocol"].nunique()) if "protocol" in ev.columns else 0

    # Unique counterparties (use protocol as proxy since we don't have from/to)
    n_counterparties = n_protocols  # simplified

    # Average transaction size
    avg_tx_usd = float(ev["usd_amount"].mean()) if len(ev) else 0.0
    median_tx_usd = float(ev["usd_amount"].median()) if len(ev) else 0.0

    # Recency-weighted activity score (exponential decay, half-life = 30d)
    age_days = (snapshot_date - ev["timestamp"]).dt.days.clip(lower=0)
    weights = np.exp(-age_days / 30.0)
    recency_score = float(weights.sum())

    # Hour-of-day entropy (proxy for human vs bot behaviour)
    hours = ev["timestamp"].dt.hour
    hour_counts = hours.value_counts(normalize=True)
    hour_entropy = float(-np.sum(hour_counts * np.log(hour_counts + 1e-10)))

    # Day-of-week entropy
    dow = ev["timestamp"].dt.dayofweek
    dow_counts = dow.value_counts(normalize=True)
    dow_entropy = float(-np.sum(dow_counts * np.log(dow_counts + 1e-10)))

    return {
        **freq,
        "behavioral_protocol_diversity": n_protocols,
        "behavioral_unique_counterparties": n_counterparties,
        "behavioral_avg_tx_usd": avg_tx_usd,
        "behavioral_median_tx_usd": median_tx_usd,
        "behavioral_recency_weighted_score": recency_score,
        "behavioral_hour_entropy": hour_entropy,
        "behavioral_dow_entropy": dow_entropy,
    }


def _empty() -> dict:
    d = {f"behavioral_tx_frequency_{d}d": 0 for d in [7, 30, 90]}
    d.update({
        "behavioral_protocol_diversity": 0,
        "behavioral_unique_counterparties": 0,
        "behavioral_avg_tx_usd": 0.0,
        "behavioral_median_tx_usd": 0.0,
        "behavioral_recency_weighted_score": 0.0,
        "behavioral_hour_entropy": 0.0,
        "behavioral_dow_entropy": 0.0,
    })
    return d
