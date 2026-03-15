"""Group 2 — Cashflow Features"""

import pandas as pd
import numpy as np

INFLOW_EVENTS  = {"transfer_in",  "repayment", "deposit", "unstake"}
OUTFLOW_EVENTS = {"transfer_out", "borrow",    "withdraw", "stake"}
STABLE_TOKENS  = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "FRAX"}


def compute(events_df: pd.DataFrame, snapshot_date: pd.Timestamp, wallet_address: str) -> dict:
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])

    if ev.empty:
        return _empty()

    inflow  = ev[ev["event_type"].isin(INFLOW_EVENTS)]["usd_amount"]
    outflow = ev[ev["event_type"].isin(OUTFLOW_EVENTS)]["usd_amount"]

    total_in  = float(inflow.sum())
    total_out = float(outflow.sum())
    net_flow  = total_in - total_out

    # Monthly aggregates for stability metric
    ev["month"] = ev["timestamp"].dt.to_period("M")
    monthly_in = ev[ev["event_type"].isin(INFLOW_EVENTS)].groupby("month")["usd_amount"].sum()
    inflow_cv  = float(monthly_in.std() / monthly_in.mean()) if len(monthly_in) > 1 and monthly_in.mean() > 0 else 0.0
    inflow_stability = max(0.0, 1.0 - inflow_cv)

    # Stablecoin ratios
    stable_in_amt = float(ev[ev["event_type"].isin(INFLOW_EVENTS) & ev["token"].isin(STABLE_TOKENS)]["usd_amount"].sum())
    stable_in_ratio = stable_in_amt / total_in if total_in > 0 else 0.0

    # Windowed cashflow
    windows = [(30, "30d"), (90, "90d")]
    windowed = {}
    for days, label in windows:
        cutoff = snapshot_date - pd.Timedelta(days=days)
        w = ev[ev["timestamp"] >= cutoff]
        wi = w[w["event_type"].isin(INFLOW_EVENTS)]["usd_amount"].sum()
        wo = w[w["event_type"].isin(OUTFLOW_EVENTS)]["usd_amount"].sum()
        windowed[f"cashflow_total_inflow_{label}"] = float(wi)
        windowed[f"cashflow_total_outflow_{label}"] = float(wo)
        windowed[f"cashflow_net_flow_{label}"] = float(wi - wo)

    return {
        "cashflow_total_inflow_usd": total_in,
        "cashflow_total_outflow_usd": total_out,
        "cashflow_net_flow_usd": net_flow,
        "cashflow_median_inflow_usd": float(inflow.median()) if len(inflow) else 0.0,
        "cashflow_median_outflow_usd": float(outflow.median()) if len(outflow) else 0.0,
        "cashflow_max_single_inflow_usd": float(inflow.max()) if len(inflow) else 0.0,
        "cashflow_max_single_outflow_usd": float(outflow.max()) if len(outflow) else 0.0,
        "cashflow_inflow_stability": inflow_stability,
        "cashflow_stablecoin_inflow_ratio": stable_in_ratio,
        "cashflow_non_stablecoin_inflow_ratio": 1.0 - stable_in_ratio,
        **windowed,
    }


def _empty() -> dict:
    d = {k: 0.0 for k in [
        "cashflow_total_inflow_usd", "cashflow_total_outflow_usd",
        "cashflow_net_flow_usd", "cashflow_median_inflow_usd",
        "cashflow_median_outflow_usd", "cashflow_max_single_inflow_usd",
        "cashflow_max_single_outflow_usd", "cashflow_inflow_stability",
        "cashflow_stablecoin_inflow_ratio", "cashflow_non_stablecoin_inflow_ratio",
        "cashflow_total_inflow_30d", "cashflow_total_outflow_30d", "cashflow_net_flow_30d",
        "cashflow_total_inflow_90d", "cashflow_total_outflow_90d", "cashflow_net_flow_90d",
    ]}
    return d
