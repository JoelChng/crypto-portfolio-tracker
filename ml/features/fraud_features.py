"""Group 6 — Fraud / Sybil Detection Features"""

import pandas as pd
import numpy as np


SUSPICIOUS_PROTOCOLS = {"tornado", "mixer", "tumbler"}


def compute(events_df: pd.DataFrame, snapshot_date: pd.Timestamp, wallet_address: str) -> dict:
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])

    if ev.empty:
        return _empty()

    # Flash loan events
    flash_loans = ev[ev["event_type"].isin({"flash_loan", "flashloan"})]
    flash_loan_count = len(flash_loans)

    # Mixer / suspicious protocol interaction
    if "protocol" in ev.columns:
        mixer_flag = int(ev["protocol"].str.lower().isin(SUSPICIOUS_PROTOCOLS).any())
    else:
        mixer_flag = 0

    # Rapid fund movement — large transfer in followed by transfer out within 1 hour
    inflows  = ev[ev["event_type"] == "transfer_in"].sort_values("timestamp")
    outflows = ev[ev["event_type"] == "transfer_out"].sort_values("timestamp")
    rapid_movement = 0
    if not inflows.empty and not outflows.empty:
        for _, inf in inflows.iterrows():
            window_end = inf["timestamp"] + pd.Timedelta(hours=1)
            quick_out = outflows[
                (outflows["timestamp"] > inf["timestamp"]) &
                (outflows["timestamp"] <= window_end) &
                (outflows["usd_amount"] >= inf["usd_amount"] * 0.8)
            ]
            if not quick_out.empty:
                rapid_movement = 1
                break

    # Contract creation count (approximated by event type)
    contract_create_count = len(ev[ev["event_type"] == "contract_create"])

    # New wallet with large flow — wallet age < 30 days but flow > $10k
    wallet_age = (snapshot_date - ev["timestamp"].min()).days
    max_single_flow = float(ev["usd_amount"].max())
    new_wallet_large_flow = int(wallet_age < 30 and max_single_flow > 10_000)

    # Self-transfer loop detection (simplified: same token in and out within 10 min)
    self_loop = 0
    if not inflows.empty and not outflows.empty:
        for _, inf in inflows.iterrows():
            window_end = inf["timestamp"] + pd.Timedelta(minutes=10)
            same = outflows[
                (outflows["timestamp"] > inf["timestamp"]) &
                (outflows["timestamp"] <= window_end) &
                (outflows["token"] == inf["token"])
            ]
            if not same.empty:
                self_loop = 1
                break

    # Bridge activity (can indicate chain-hopping to obscure funds)
    bridge_count = len(ev[ev["event_type"] == "bridge"])

    return {
        "fraud_flash_loan_count": flash_loan_count,
        "fraud_mixer_interaction_flag": mixer_flag,
        "fraud_rapid_fund_movement_flag": rapid_movement,
        "fraud_contract_creation_count": contract_create_count,
        "fraud_new_wallet_large_flow_flag": new_wallet_large_flow,
        "fraud_self_transfer_loop_flag": self_loop,
        "fraud_bridge_count": bridge_count,
    }


def _empty() -> dict:
    return {
        "fraud_flash_loan_count": 0,
        "fraud_mixer_interaction_flag": 0,
        "fraud_rapid_fund_movement_flag": 0,
        "fraud_contract_creation_count": 0,
        "fraud_new_wallet_large_flow_flag": 0,
        "fraud_self_transfer_loop_flag": 0,
        "fraud_bridge_count": 0,
    }
