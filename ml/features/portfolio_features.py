"""Group 5 — Portfolio Quality Features"""

import pandas as pd
import numpy as np

STABLE_TOKENS  = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "FRAX"}
BLUECHIP_TOKENS = {"ETH", "WBTC", "BTC"}


def compute(events_df: pd.DataFrame, snapshot_date: pd.Timestamp, wallet_address: str) -> dict:
    ev = events_df.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])

    if ev.empty:
        return _empty()

    # Token value distribution from recent transfer/balance events
    token_vals = ev.groupby("token")["usd_amount"].sum()
    total_val = float(token_vals.sum())

    if total_val <= 0:
        return _empty()

    token_share = token_vals / total_val

    # Herfindahl-Hirschman Index (concentration)
    hhi = float((token_share ** 2).sum())

    # Stablecoin share
    stable_tokens_present = [t for t in STABLE_TOKENS if t in token_share.index]
    stablecoin_share = float(token_share[stable_tokens_present].sum()) if stable_tokens_present else 0.0

    # Bluechip share
    blue_tokens = [t for t in BLUECHIP_TOKENS if t in token_share.index]
    bluechip_share = float(token_share[blue_tokens].sum()) if blue_tokens else 0.0

    # Long-tail share
    longtail_share = max(0.0, 1.0 - stablecoin_share - bluechip_share)

    # Number of distinct tokens
    n_tokens = int(token_vals.shape[0])

    # Portfolio turnover — sum of all flows relative to average balance
    total_flow = float(ev["usd_amount"].sum())
    turnover = total_flow / total_val if total_val > 0 else 0.0

    # Simple volatility proxy — std of daily USD flows
    ev["date"] = ev["timestamp"].dt.date
    daily_flow = ev.groupby("date")["usd_amount"].sum()
    vol_proxy = float(daily_flow.std() / daily_flow.mean()) if len(daily_flow) > 1 and daily_flow.mean() > 0 else 0.0

    # Max single-token concentration
    max_concentration = float(token_share.max()) if len(token_share) else 1.0

    return {
        "portfolio_total_value_usd": total_val,
        "portfolio_n_distinct_tokens": n_tokens,
        "portfolio_herfindahl_index": hhi,
        "portfolio_stablecoin_ratio": stablecoin_share,
        "portfolio_bluechip_ratio": bluechip_share,
        "portfolio_longtail_ratio": longtail_share,
        "portfolio_max_token_concentration": max_concentration,
        "portfolio_turnover": min(turnover, 100.0),
        "portfolio_volatility_proxy": vol_proxy,
    }


def _empty() -> dict:
    return {
        "portfolio_total_value_usd": 0.0,
        "portfolio_n_distinct_tokens": 0,
        "portfolio_herfindahl_index": 1.0,
        "portfolio_stablecoin_ratio": 0.0,
        "portfolio_bluechip_ratio": 0.0,
        "portfolio_longtail_ratio": 0.0,
        "portfolio_max_token_concentration": 1.0,
        "portfolio_turnover": 0.0,
        "portfolio_volatility_proxy": 0.0,
    }
