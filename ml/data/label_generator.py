"""
Label generation with strict temporal cutoff.

For each (wallet, snapshot_date) pair, labels whether a bad credit event
occurred in (snapshot_date, snapshot_date + horizon_days].

BAD CREDIT EVENTS: liquidation, missed_repayment, forced_deleverage, bad_debt_flag
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BAD_EVENT_TYPES = {"liquidation", "missed_repayment", "forced_deleverage", "bad_debt_flag"}


def generate_labels(
    events_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    horizons: list[int] = (30, 60, 90),
) -> pd.DataFrame:
    """
    Returns snapshots_df with added columns:
      default_30d, default_60d, default_90d  (bool)
    """
    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])

    # Only bad events matter for labeling
    bad_events = events_df[events_df["event_type"].isin(BAD_EVENT_TYPES)][
        ["wallet_address", "timestamp"]
    ].copy()

    labels = snapshots_df.copy()
    labels["snapshot_date"] = pd.to_datetime(labels["snapshot_date"])

    for h in horizons:
        col = f"default_{h}d"
        labels[col] = False

    if bad_events.empty:
        logger.warning("No bad credit events found in event data.")
        return labels

    # Merge-based vectorised approach
    merged = labels.merge(bad_events, on="wallet_address", how="left")
    merged["event_ts"] = pd.to_datetime(merged["timestamp"])

    for h in horizons:
        col = f"default_{h}d"
        horizon_end = merged["snapshot_date"] + pd.Timedelta(days=h)
        in_window = (
            (merged["event_ts"] > merged["snapshot_date"]) &
            (merged["event_ts"] <= horizon_end)
        )
        flags = merged[in_window][["wallet_address", "snapshot_date"]].drop_duplicates()
        flags[col] = True
        labels = labels.merge(flags, on=["wallet_address", "snapshot_date"], how="left")
        if f"{col}_y" in labels.columns:
            labels[col] = labels[f"{col}_y"].fillna(False)
            labels = labels.drop(columns=[f"{col}_x", f"{col}_y"])
        else:
            labels[col] = labels[col].fillna(False)

    # Log class balance
    for h in horizons:
        col = f"default_{h}d"
        rate = labels[col].mean()
        logger.info(f"  Default rate {h}d horizon: {rate:.3%} ({labels[col].sum():,} positives)")

    return labels


def run(config_path: str = "configs/pipeline.yaml") -> pd.DataFrame:
    import yaml
    logging.basicConfig(level=logging.INFO)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["data"]["processed_dir"])
    raw_dir = Path(cfg["data"]["raw_dir"])

    events_df   = pd.read_parquet(raw_dir / "events.parquet")
    snapshots_df = pd.read_parquet(processed_dir / "snapshots.parquet")
    horizons    = cfg["labels"]["horizons"]

    logger.info("Generating labels...")
    labels = generate_labels(events_df, snapshots_df, horizons)

    out_path = processed_dir / "labels.parquet"
    labels.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(labels):,} labelled snapshots → {out_path}")
    return labels


if __name__ == "__main__":
    run()
