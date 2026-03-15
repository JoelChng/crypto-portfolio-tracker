"""Group 1 — Wallet Tenure Features"""

import pandas as pd
import numpy as np


def compute(events_df: pd.DataFrame, snapshot_date: pd.Timestamp, wallet_address: str) -> dict:
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])

    if ev.empty:
        return _empty()

    first_tx = ev["timestamp"].min()
    last_tx  = ev["timestamp"].max()
    wallet_age = (snapshot_date - first_tx).days
    days_since_last = (snapshot_date - last_tx).days

    # Active days (days with at least one tx)
    tx_days = ev["timestamp"].dt.date.nunique()
    spans = [(30, "30d"), (90, "90d"), (180, "180d")]
    active_days = {}
    for window, label in spans:
        cutoff = snapshot_date - pd.Timedelta(days=window)
        active_days[f"tenure_active_days_{label}"] = int(
            ev[ev["timestamp"] >= cutoff]["timestamp"].dt.date.nunique()
        )

    # Dormant streaks — max gap between consecutive tx days
    day_series = pd.Series(sorted(ev["timestamp"].dt.date.unique()))
    if len(day_series) >= 2:
        gaps = (day_series.shift(-1) - day_series).dropna().apply(lambda x: x.days)
        max_dormant = int(gaps.max())
        avg_dormant = float(gaps.mean())
    else:
        max_dormant = wallet_age
        avg_dormant = float(wallet_age)

    return {
        "tenure_wallet_age_days": max(wallet_age, 0),
        "tenure_days_since_first_tx": max(wallet_age, 0),
        "tenure_days_since_last_tx": max(days_since_last, 0),
        "tenure_tx_count_lifetime": len(ev),
        "tenure_active_tx_days": tx_days,
        "tenure_max_dormant_streak_days": max_dormant,
        "tenure_avg_dormant_streak_days": avg_dormant,
        **active_days,
    }


def _empty() -> dict:
    return {
        "tenure_wallet_age_days": 0,
        "tenure_days_since_first_tx": 0,
        "tenure_days_since_last_tx": 9999,
        "tenure_tx_count_lifetime": 0,
        "tenure_active_tx_days": 0,
        "tenure_max_dormant_streak_days": 9999,
        "tenure_avg_dormant_streak_days": 9999,
        "tenure_active_days_30d": 0,
        "tenure_active_days_90d": 0,
        "tenure_active_days_180d": 0,
    }
