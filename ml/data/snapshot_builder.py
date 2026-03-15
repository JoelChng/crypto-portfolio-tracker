"""
Builds (wallet_address, snapshot_date) pairs for training.

Rules:
- First snapshot >= wallet first_seen + min_wallet_age_days
- Snapshots every interval_days until data end_date
- No snapshot after (end_date - max_horizon) to avoid label truncation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


def build_snapshots(
    wallets_df: pd.DataFrame,
    end_date: pd.Timestamp,
    interval_days: int = 30,
    min_wallet_age_days: int = 14,
    max_horizon_days: int = 90,
) -> pd.DataFrame:
    """Returns DataFrame with columns: wallet_address, snapshot_date"""
    rows = []
    cutoff = end_date - pd.Timedelta(days=max_horizon_days)

    for _, w in wallets_df.iterrows():
        first_snap = pd.to_datetime(w["first_seen"]) + pd.Timedelta(days=min_wallet_age_days)
        if first_snap > cutoff:
            continue
        dates = pd.date_range(start=first_snap, end=cutoff, freq=f"{interval_days}D")
        for d in dates:
            rows.append({"wallet_address": w["wallet_address"], "snapshot_date": d})

    df = pd.DataFrame(rows)
    logger.info(f"Built {len(df):,} snapshots for {df['wallet_address'].nunique():,} wallets")
    return df


def events_before_snapshot(
    events_df: pd.DataFrame,
    wallet_address: str,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Authoritative gating function — always use this, never reimplement inline.
    Returns events for a wallet strictly BEFORE snapshot_date (exclusive).
    """
    mask = (
        (events_df["wallet_address"] == wallet_address) &
        (pd.to_datetime(events_df["timestamp"]) < snapshot_date)
    )
    return events_df[mask].copy()


def run(config_path: str = "configs/pipeline.yaml") -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir       = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    wallets_df = pd.read_parquet(raw_dir / "wallets.parquet")
    end_date   = pd.Timestamp(cfg["synthetic"]["end_date"])
    max_horizon = max(cfg["labels"]["horizons"])

    snaps = build_snapshots(
        wallets_df,
        end_date=end_date,
        interval_days=cfg["snapshots"]["interval_days"],
        min_wallet_age_days=cfg["snapshots"]["min_wallet_age_days"],
        max_horizon_days=max_horizon,
    )

    out = processed_dir / "snapshots.parquet"
    snaps.to_parquet(out, index=False)
    logger.info(f"Saved snapshots → {out}")
    return snaps


if __name__ == "__main__":
    run()
