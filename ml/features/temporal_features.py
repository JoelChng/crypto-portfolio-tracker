"""Group 7 — Temporal / Timing Pattern Features"""

import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy


def compute(events_df: pd.DataFrame, snapshot_date: pd.Timestamp, wallet_address: str) -> dict:
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])

    if len(ev) < 2:
        return _empty()

    ev_sorted = ev.sort_values("timestamp")

    # Inter-transaction gap statistics
    gaps = ev_sorted["timestamp"].diff().dropna().dt.total_seconds() / 3600  # hours
    gap_cv = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
    gap_mean = float(gaps.mean())
    gap_median = float(gaps.median())

    # Burstiness score (coefficient of variation of gap)
    burst_score = float((gaps.std() - gaps.mean()) / (gaps.std() + gaps.mean() + 1e-10))

    # Hour-of-day entropy
    hours = ev["timestamp"].dt.hour
    hour_dist = np.bincount(hours, minlength=24) / len(hours)
    hour_entropy = float(scipy_entropy(hour_dist + 1e-10))

    # Day-of-week entropy
    dow = ev["timestamp"].dt.dayofweek
    dow_dist = np.bincount(dow, minlength=7) / len(dow)
    dow_entropy = float(scipy_entropy(dow_dist + 1e-10))

    # Seasonal activity — ratio of last 30d to prior 60d
    cutoff_30 = snapshot_date - pd.Timedelta(days=30)
    cutoff_90 = snapshot_date - pd.Timedelta(days=90)
    recent_count = len(ev[ev["timestamp"] >= cutoff_30])
    prior_count  = len(ev[(ev["timestamp"] >= cutoff_90) & (ev["timestamp"] < cutoff_30)])
    seasonal_idx = recent_count / (prior_count + 1e-10)

    # Trend — is activity accelerating or decelerating?
    # Slope of monthly tx count (simple linear regression)
    ev["month_num"] = (ev["timestamp"].dt.year * 12 + ev["timestamp"].dt.month)
    monthly = ev.groupby("month_num").size()
    if len(monthly) >= 3:
        x = np.arange(len(monthly))
        slope = float(np.polyfit(x, monthly.values, 1)[0])
    else:
        slope = 0.0

    return {
        "temporal_tx_gap_cv": gap_cv,
        "temporal_tx_gap_mean_hours": gap_mean,
        "temporal_tx_gap_median_hours": gap_median,
        "temporal_burst_score": burst_score,
        "temporal_hour_entropy": hour_entropy,
        "temporal_dow_entropy": dow_entropy,
        "temporal_seasonal_activity_index": float(seasonal_idx),
        "temporal_activity_trend_slope": slope,
    }


def _empty() -> dict:
    return {
        "temporal_tx_gap_cv": 0.0,
        "temporal_tx_gap_mean_hours": 0.0,
        "temporal_tx_gap_median_hours": 0.0,
        "temporal_burst_score": 0.0,
        "temporal_hour_entropy": 0.0,
        "temporal_dow_entropy": 0.0,
        "temporal_seasonal_activity_index": 0.0,
        "temporal_activity_trend_slope": 0.0,
    }
