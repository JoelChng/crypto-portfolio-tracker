"""Group 4 — Credit/DeFi Risk Features"""

import pandas as pd
import numpy as np

BAD_EVENTS = {"liquidation", "missed_repayment", "forced_deleverage", "bad_debt_flag"}


def compute(events_df: pd.DataFrame, snapshot_date: pd.Timestamp, wallet_address: str) -> dict:
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])

    if ev.empty:
        return _empty()

    borrows    = ev[ev["event_type"] == "borrow"]
    repayments = ev[ev["event_type"] == "repayment"]
    liq_events = ev[ev["event_type"] == "liquidation"]

    borrow_count  = len(borrows)
    repay_count   = len(repayments)
    total_borrowed = float(borrows["usd_amount"].sum())
    total_repaid   = float(repayments["usd_amount"].sum())
    repayment_ratio = min(total_repaid / total_borrowed, 1.0) if total_borrowed > 0 else 1.0

    n_liquidations = len(liq_events)
    liq_severity   = float(liq_events["usd_amount"].sum()) if n_liquidations > 0 else 0.0

    # Late repayments — vectorised: borrows without a repayment within 30 days
    late_count = 0
    if not borrows.empty and not repayments.empty:
        bts = borrows["timestamp"].values.astype("datetime64[ns]")
        rts = repayments["timestamp"].sort_values().values.astype("datetime64[ns]")
        window = np.timedelta64(30, "D")
        # For each borrow, find first repayment after it
        lo = np.searchsorted(rts, bts, side="right")
        hi = np.searchsorted(rts, bts + window, side="right")
        late_count = int(np.sum(lo >= hi))
    elif not borrows.empty:
        late_count = len(borrows)

    # Outstanding debt proxy (last known debt_after_usd)
    if "debt_after_usd" in ev.columns:
        debt_series = ev[ev["debt_after_usd"] > 0]["debt_after_usd"]
        outstanding_debt = float(debt_series.iloc[-1]) if len(debt_series) else 0.0
    else:
        outstanding_debt = max(0.0, total_borrowed - total_repaid)

    # Health factor statistics
    if "health_factor" in ev.columns:
        hf = ev["health_factor"].replace(0, np.nan).dropna()
        min_hf  = float(hf.min()) if len(hf) else 2.0
        avg_hf  = float(hf.mean()) if len(hf) else 2.0
    else:
        min_hf = avg_hf = 2.0

    # Leverage proxies
    max_leverage = float(ev.get("max_leverage", pd.Series([1.0])).max()) if "max_leverage" in ev.columns else 1.0

    # Missed repayments count
    missed = ev[ev["event_type"] == "missed_repayment"]

    return {
        "credit_defi_borrow_count": borrow_count,
        "credit_defi_repay_count": repay_count,
        "credit_defi_total_borrowed_usd": total_borrowed,
        "credit_defi_total_repaid_usd": total_repaid,
        "credit_defi_historical_repayment_ratio": repayment_ratio,
        "credit_defi_outstanding_debt_usd": outstanding_debt,
        "credit_defi_liquidation_count": n_liquidations,
        "credit_defi_liquidation_severity_usd": liq_severity,
        "credit_defi_late_repayment_count": late_count,
        "credit_defi_missed_repayment_count": len(missed),
        "credit_defi_min_health_factor": min_hf,
        "credit_defi_avg_health_factor": avg_hf,
        "credit_defi_max_leverage": max_leverage,
        "credit_defi_has_any_bad_event": int(len(ev[ev["event_type"].isin(BAD_EVENTS)]) > 0),
    }


def _empty() -> dict:
    return {
        "credit_defi_borrow_count": 0,
        "credit_defi_repay_count": 0,
        "credit_defi_total_borrowed_usd": 0.0,
        "credit_defi_total_repaid_usd": 0.0,
        "credit_defi_historical_repayment_ratio": 1.0,
        "credit_defi_outstanding_debt_usd": 0.0,
        "credit_defi_liquidation_count": 0,
        "credit_defi_liquidation_severity_usd": 0.0,
        "credit_defi_late_repayment_count": 0,
        "credit_defi_missed_repayment_count": 0,
        "credit_defi_min_health_factor": 2.0,
        "credit_defi_avg_health_factor": 2.0,
        "credit_defi_max_leverage": 1.0,
        "credit_defi_has_any_bad_event": 0,
    }
