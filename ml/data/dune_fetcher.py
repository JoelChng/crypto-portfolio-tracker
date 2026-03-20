"""
Dune Analytics on-chain data fetcher.

Pulls real lending events from Aave V3 and Compound V3 on Ethereum,
maps them to the same schema as synthetic events.parquet, and builds
a wallet-level summary file matching wallets.parquet.

Output:
    data/raw/real_events.parquet   — event-level (matches events.parquet schema)
    data/raw/real_wallets.parquet  — wallet-level (matches wallets.parquet schema)

Usage:
    export DUNE_API_KEY=your_key_here
    python data/dune_fetcher.py [--config configs/pipeline.yaml]

    # Or with pre-saved Dune query IDs (faster — avoids re-executing SQL):
    # Set dune.query_ids.aave_v3 / compound_v3 in configs/pipeline.yaml

Query IDs:
    If you have saved these queries on Dune Analytics, put their integer IDs
    in pipeline.yaml under dune.query_ids.  If those fields are null/absent,
    the fetcher creates and runs temporary inline queries automatically.
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  SQL queries
#  These target the standard Dune Analytics spellbook tables.
#  Verify table names in your Dune workspace if queries return errors.
# ─────────────────────────────────────────────────────────────────────────────


# Single unified SQL using Dune spellbook tables (lending.borrow + lending.supply).
# These cover Aave v2/v3, Compound v2/v3 on Ethereum with pre-computed USD amounts.
LENDING_SQL = """
-- Ethereum lending events: Aave + Compound, all event types.
-- Uses Dune spellbook tables lending.borrow and lending.supply.
-- Parameterise with {{start_date}} and {{end_date}} (YYYY-MM-DD strings).

WITH borrow_events AS (
    SELECT
        LOWER(CAST(COALESCE(on_behalf_of, borrower) AS VARCHAR)) AS wallet_address,
        block_time                                                AS timestamp,
        CASE transaction_type
            WHEN 'borrow'            THEN 'borrow'
            WHEN 'repay'             THEN 'repayment'
            WHEN 'repay_with_atokens' THEN 'repayment'
            WHEN 'borrow_liquidation' THEN 'liquidation'
            ELSE transaction_type
        END                                                       AS event_type,
        UPPER(COALESCE(symbol, 'UNKNOWN'))                        AS token,
        COALESCE(TRY(CAST(amount_usd AS DOUBLE)), 0)              AS usd_amount,
        LOWER(project)                                            AS protocol,
        0.0                                                       AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                                      AS health_factor,
        CASE WHEN transaction_type = 'borrow'
             THEN COALESCE(TRY(CAST(amount_usd AS DOUBLE)), 0)
             ELSE 0.0
        END                                                       AS debt_after_usd
    FROM lending.borrow
    WHERE blockchain = 'ethereum'
      AND block_time >= TIMESTAMP '{{start_date}}'
      AND block_time <  TIMESTAMP '{{end_date}}'
      AND project IN ('aave', 'compound')
      AND amount_usd > 0
),

supply_events AS (
    SELECT
        LOWER(CAST(COALESCE(on_behalf_of, depositor) AS VARCHAR)) AS wallet_address,
        block_time                                                 AS timestamp,
        CASE transaction_type
            WHEN 'deposit'            THEN 'deposit'
            WHEN 'supply'             THEN 'deposit'
            WHEN 'withdraw'           THEN 'withdraw'
            WHEN 'deposit_liquidation' THEN 'liquidation'
            WHEN 'supply_liquidation'  THEN 'liquidation'
            ELSE transaction_type
        END                                                        AS event_type,
        UPPER(COALESCE(symbol, 'UNKNOWN'))                         AS token,
        COALESCE(TRY(CAST(amount_usd AS DOUBLE)), 0)               AS usd_amount,
        LOWER(project)                                             AS protocol,
        0.0                                                        AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                                       AS health_factor,
        0.0                                                        AS debt_after_usd
    FROM lending.supply
    WHERE blockchain = 'ethereum'
      AND block_time >= TIMESTAMP '{{start_date}}'
      AND block_time <  TIMESTAMP '{{end_date}}'
      AND project IN ('aave', 'compound')
      AND amount_usd > 0
)

SELECT * FROM borrow_events
UNION ALL
SELECT * FROM supply_events
ORDER BY wallet_address, timestamp
LIMIT 200000
"""

# Keep these as aliases so the rest of the code can still reference them
AAVE_V3_SQL = LENDING_SQL       # fetched together in one query now
COMPOUND_V3_SQL = None          # merged into LENDING_SQL

_PLACEHOLDER = """
-- placeholder — not used (merged into LENDING_SQL)
WITH aave_borrows AS (
    SELECT
        LOWER(CAST(b."onBehalfOf" AS VARCHAR)) AS wallet_address,
        b.evt_block_time                        AS timestamp,
        'borrow'                                AS event_type,
        COALESCE(tk.symbol, 'UNKNOWN')          AS token,
        TRY(CAST(b.amount AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 18))
            * COALESCE(p.price, 0))             AS usd_amount,
        'aave_v3'                               AS protocol,
        0.0                                     AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                    AS health_factor,
        TRY(CAST(b.amount AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 18))
            * COALESCE(p.price, 0))             AS debt_after_usd
    FROM aave_v3_ethereum.evt_Borrow b
    LEFT JOIN tokens.erc20 tk
        ON tk.contract_address = b.reserve
        AND tk.blockchain = 'ethereum'
    LEFT JOIN prices.usd p
        ON p.contract_address = b.reserve
        AND p.blockchain = 'ethereum'
        AND p.minute = date_trunc('minute', b.evt_block_time)
    WHERE b.evt_block_time >= TIMESTAMP '{{start_date}}'
      AND b.evt_block_time <  TIMESTAMP '{{end_date}}'
),

aave_repays AS (
    SELECT
        LOWER(CAST(r."user" AS VARCHAR))       AS wallet_address,
        r.evt_block_time                        AS timestamp,
        'repayment'                             AS event_type,
        COALESCE(tk.symbol, 'UNKNOWN')          AS token,
        TRY(CAST(r.amount AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 18))
            * COALESCE(p.price, 0))             AS usd_amount,
        'aave_v3'                               AS protocol,
        0.0                                     AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                    AS health_factor,
        0.0                                     AS debt_after_usd
    FROM aave_v3_ethereum.evt_Repay r
    LEFT JOIN tokens.erc20 tk
        ON tk.contract_address = r.reserve
        AND tk.blockchain = 'ethereum'
    LEFT JOIN prices.usd p
        ON p.contract_address = r.reserve
        AND p.blockchain = 'ethereum'
        AND p.minute = date_trunc('minute', r.evt_block_time)
    WHERE r.evt_block_time >= TIMESTAMP '{{start_date}}'
      AND r.evt_block_time <  TIMESTAMP '{{end_date}}'
),

aave_liquidations AS (
    SELECT
        LOWER(CAST(l."user" AS VARCHAR))       AS wallet_address,
        l.evt_block_time                        AS timestamp,
        'liquidation'                           AS event_type,
        COALESCE(tk.symbol, 'UNKNOWN')          AS token,
        TRY(CAST(l."debtToCover" AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 18))
            * COALESCE(p.price, 0))             AS usd_amount,
        'aave_v3'                               AS protocol,
        0.0                                     AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                    AS health_factor,
        0.0                                     AS debt_after_usd
    FROM aave_v3_ethereum.evt_LiquidationCall l
    LEFT JOIN tokens.erc20 tk
        ON tk.contract_address = l."debtAsset"
        AND tk.blockchain = 'ethereum'
    LEFT JOIN prices.usd p
        ON p.contract_address = l."debtAsset"
        AND p.blockchain = 'ethereum'
        AND p.minute = date_trunc('minute', l.evt_block_time)
    WHERE l.evt_block_time >= TIMESTAMP '{{start_date}}'
      AND l.evt_block_time <  TIMESTAMP '{{end_date}}'
),

aave_deposits AS (
    SELECT
        LOWER(CAST(s."onBehalfOf" AS VARCHAR))  AS wallet_address,
        s.evt_block_time                         AS timestamp,
        'deposit'                                AS event_type,
        COALESCE(tk.symbol, 'UNKNOWN')           AS token,
        TRY(CAST(s.amount AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 18))
            * COALESCE(p.price, 0))              AS usd_amount,
        'aave_v3'                                AS protocol,
        0.0                                      AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                     AS health_factor,
        0.0                                      AS debt_after_usd
    FROM aave_v3_ethereum.evt_Supply s
    LEFT JOIN tokens.erc20 tk
        ON tk.contract_address = s.reserve
        AND tk.blockchain = 'ethereum'
    LEFT JOIN prices.usd p
        ON p.contract_address = s.reserve
        AND p.blockchain = 'ethereum'
        AND p.minute = date_trunc('minute', s.evt_block_time)
    WHERE s.evt_block_time >= TIMESTAMP '{{start_date}}'
      AND s.evt_block_time <  TIMESTAMP '{{end_date}}'
),

aave_withdraws AS (
    SELECT
        LOWER(CAST(w."user" AS VARCHAR))        AS wallet_address,
        w.evt_block_time                         AS timestamp,
        'withdraw'                               AS event_type,
        COALESCE(tk.symbol, 'UNKNOWN')           AS token,
        TRY(CAST(w.amount AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 18))
            * COALESCE(p.price, 0))              AS usd_amount,
        'aave_v3'                                AS protocol,
        0.0                                      AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                     AS health_factor,
        0.0                                      AS debt_after_usd
    FROM aave_v3_ethereum.evt_Withdraw w
    LEFT JOIN tokens.erc20 tk
        ON tk.contract_address = w.reserve
        AND tk.blockchain = 'ethereum'
    LEFT JOIN prices.usd p
        ON p.contract_address = w.reserve
        AND p.blockchain = 'ethereum'
        AND p.minute = date_trunc('minute', w.evt_block_time)
    WHERE w.evt_block_time >= TIMESTAMP '{{start_date}}'
      AND w.evt_block_time <  TIMESTAMP '{{end_date}}'
)

SELECT * FROM aave_borrows
UNION ALL SELECT * FROM aave_repays
UNION ALL SELECT * FROM aave_liquidations
UNION ALL SELECT * FROM aave_deposits
UNION ALL SELECT * FROM aave_withdraws
ORDER BY wallet_address, timestamp
"""

# Compound V3 (Comet) — USDC market (0xc3d688B66703497DAA19211EEdff47f25384cdc3)
# and ETH market (0xA17581A9E3195B1b55f914b34B0e2E940B8F8a7A).
# Supply = lend; Withdraw = borrow (base asset); AbsorbCollateral = liquidation.
COMPOUND_V3_SQL = """
-- Compound V3 (Comet) Ethereum events.
-- Note: In Compound V3, "Withdraw" of the base asset = borrowing,
--       "Supply" of the base asset = lending/repaying.
-- We infer the direction from whether the wallet has an open borrow position.
-- For simplicity: track all Supply / Withdraw / AbsorbCollateral events.

WITH compound_supply AS (
    SELECT
        LOWER(CAST(s.dst AS VARCHAR))           AS wallet_address,
        s.evt_block_time                         AS timestamp,
        'deposit'                                AS event_type,
        COALESCE(tk.symbol, 'USDC')             AS token,
        TRY(CAST(s.amount AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 6))
            * COALESCE(p.price, 1))              AS usd_amount,
        'compound_v3'                            AS protocol,
        0.0                                      AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                     AS health_factor,
        0.0                                      AS debt_after_usd
    FROM compound_v3_ethereum.evt_Supply s
    LEFT JOIN tokens.erc20 tk
        ON tk.contract_address = s.contract_address
        AND tk.blockchain = 'ethereum'
    LEFT JOIN prices.usd p
        ON p.symbol = 'USDC'
        AND p.blockchain = 'ethereum'
        AND p.minute = date_trunc('minute', s.evt_block_time)
    WHERE s.evt_block_time >= TIMESTAMP '{{start_date}}'
      AND s.evt_block_time <  TIMESTAMP '{{end_date}}'
),

compound_withdraw AS (
    SELECT
        LOWER(CAST(w.src AS VARCHAR))           AS wallet_address,
        w.evt_block_time                         AS timestamp,
        'borrow'                                 AS event_type,
        COALESCE(tk.symbol, 'USDC')             AS token,
        TRY(CAST(w.amount AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 6))
            * COALESCE(p.price, 1))              AS usd_amount,
        'compound_v3'                            AS protocol,
        0.0                                      AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                     AS health_factor,
        TRY(CAST(w.amount AS DOUBLE)
            / POWER(10, COALESCE(tk.decimals, 6))
            * COALESCE(p.price, 1))              AS debt_after_usd
    FROM compound_v3_ethereum.evt_Withdraw w
    LEFT JOIN tokens.erc20 tk
        ON tk.contract_address = w.contract_address
        AND tk.blockchain = 'ethereum'
    LEFT JOIN prices.usd p
        ON p.symbol = 'USDC'
        AND p.blockchain = 'ethereum'
        AND p.minute = date_trunc('minute', w.evt_block_time)
    WHERE w.evt_block_time >= TIMESTAMP '{{start_date}}'
      AND w.evt_block_time <  TIMESTAMP '{{end_date}}'
),

compound_liquidations AS (
    SELECT
        LOWER(CAST(a.borrower AS VARCHAR))      AS wallet_address,
        a.evt_block_time                         AS timestamp,
        'liquidation'                            AS event_type,
        COALESCE(tk.symbol, 'UNKNOWN')           AS token,
        TRY(CAST(a."usdValue" AS DOUBLE) / 1e8) AS usd_amount,
        'compound_v3'                            AS protocol,
        0.0                                      AS gas_fee_usd,
        CAST(NULL AS DOUBLE)                     AS health_factor,
        0.0                                      AS debt_after_usd
    FROM compound_v3_ethereum.evt_AbsorbCollateral a
    LEFT JOIN tokens.erc20 tk
        ON tk.contract_address = a.asset
        AND tk.blockchain = 'ethereum'
    WHERE a.evt_block_time >= TIMESTAMP '{{start_date}}'
      AND a.evt_block_time <  TIMESTAMP '{{end_date}}'
)

SELECT * FROM compound_supply
UNION ALL SELECT * FROM compound_withdraw
UNION ALL SELECT * FROM compound_liquidations
ORDER BY wallet_address, timestamp
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Dune client helpers
# ─────────────────────────────────────────────────────────────────────────────

EVENTS_SCHEMA = [
    "wallet_address", "timestamp", "event_type", "token",
    "usd_amount", "protocol", "gas_fee_usd", "health_factor", "debt_after_usd",
]


def _get_client():
    """Return a DuneClient instance, raising clearly if package/key missing."""
    try:
        from dune_client.client import DuneClient
    except ImportError:
        raise ImportError(
            "dune-client is not installed. Run: pip install dune-client"
        )
    api_key = os.environ.get("DUNE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "DUNE_API_KEY environment variable is not set. "
            "Get a key at https://dune.com/settings/api"
        )
    return DuneClient(api_key=api_key)


def _fetch_saved_query(client, query_id: int, query_name: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Download latest cached results for a saved Dune query (FREE on all plans).
    Falls back to re-running the query if no cached results exist.
    """
    from dune_client.query import QueryBase

    # Try cached results first (0 credits)
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching cached results for query {query_id} ({query_name})...")
            df = client.get_latest_result_dataframe(query_id)
            logger.info(f"  → {len(df):,} rows from cache for {query_name}")
            return df
        except Exception as exc:
            err_str = str(exc).lower()
            if "404" in err_str or "not found" in err_str:
                # No cached results — need to run the query first
                logger.info(f"  No cache found, executing query {query_id}...")
                break
            if any(kw in err_str for kw in ("rate", "429", "quota", "timeout", "busy")):
                wait = 30.0 * (2 ** attempt)
                logger.warning(f"  Rate limit, waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise

    # Execute the saved query (costs 1 credit)
    for attempt in range(max_retries):
        try:
            df = client.run_query_dataframe(QueryBase(query_id=query_id))
            logger.info(f"  → {len(df):,} rows for {query_name}")
            return df
        except Exception as exc:
            err_str = str(exc).lower()
            if "402" in err_str or "payment required" in err_str:
                raise RuntimeError(
                    f"Dune 402 for query {query_id}: result download requires a paid plan.\n"
                    "Please upgrade your Dune plan or run the query on the website first\n"
                    "so the results are cached, then re-run the fetcher."
                ) from exc
            if any(kw in err_str for kw in ("rate", "429", "quota", "timeout", "busy")):
                wait = 30.0 * (2 ** attempt)
                logger.warning(f"  Rate limit, waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                logger.error(f"Dune query {query_id} failed: {exc}")
                raise

    raise RuntimeError(f"Failed to fetch Dune query {query_id} after retries.")


# ─────────────────────────────────────────────────────────────────────────────
#  Schema normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    """Cast/rename columns to match events.parquet schema exactly."""
    if df.empty:
        return pd.DataFrame(columns=EVENTS_SCHEMA)

    df = df.copy()

    # Ensure all required columns exist
    for col in EVENTS_SCHEMA:
        if col not in df.columns:
            df[col] = None

    df = df[EVENTS_SCHEMA].copy()

    df["wallet_address"] = df["wallet_address"].astype(str).str.lower().str.strip()
    df["timestamp"]      = pd.to_datetime(df["timestamp"], utc=True)
    df["event_type"]     = df["event_type"].astype(str)
    df["token"]          = df["token"].fillna("UNKNOWN").astype(str).str.upper()
    df["usd_amount"]     = pd.to_numeric(df["usd_amount"],    errors="coerce").fillna(0.0)
    df["protocol"]       = df["protocol"].astype(str)
    df["gas_fee_usd"]    = pd.to_numeric(df["gas_fee_usd"],   errors="coerce").fillna(0.0)
    df["health_factor"]  = pd.to_numeric(df["health_factor"], errors="coerce")   # keep NaN
    df["debt_after_usd"] = pd.to_numeric(df["debt_after_usd"], errors="coerce").fillna(0.0)

    # Drop rows with obviously invalid wallet addresses
    df = df[df["wallet_address"].str.startswith("0x")]
    df = df[df["usd_amount"] >= 0]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Wallet-level feature aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _build_wallet_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event-level data into per-wallet summary rows.

    Output schema matches wallets.parquet:
        wallet_address, first_seen, last_seen, wallet_age_days,
        tx_count_lifetime, n_protocols, portfolio_value_usd,
        stablecoin_ratio, max_leverage, has_borrow,
        repayment_ratio, n_liquidations, label
    """
    STABLECOINS = {"USDC", "USDT", "DAI", "BUSD", "FRAX", "LUSD", "TUSD", "USDP"}

    rows = []
    for wallet, grp in events_df.groupby("wallet_address"):
        grp = grp.sort_values("timestamp")

        first_seen = grp["timestamp"].min()
        last_seen  = grp["timestamp"].max()
        age_days   = max((last_seen - first_seen).days, 1)

        borrows    = grp[grp["event_type"] == "borrow"]
        repays     = grp[grp["event_type"] == "repayment"]
        liqs       = grp[grp["event_type"] == "liquidation"]

        total_borrowed = borrows["usd_amount"].sum()
        total_repaid   = repays["usd_amount"].sum()
        repayment_ratio = (
            min(total_repaid / total_borrowed, 1.0) if total_borrowed > 0 else 1.0
        )

        # Portfolio value proxy: max single-day deposit or borrow USD
        deposits = grp[grp["event_type"].isin(["deposit", "borrow"])]
        portfolio_value = deposits["usd_amount"].sum() if not deposits.empty else 0.0

        # Stablecoin ratio
        stable_usd = grp[grp["token"].isin(STABLECOINS)]["usd_amount"].sum()
        total_usd  = grp["usd_amount"].sum()
        stablecoin_ratio = stable_usd / total_usd if total_usd > 0 else 0.0

        # Max leverage proxy: if debt_after_usd exists
        max_leverage = grp["debt_after_usd"].max()
        if pd.isna(max_leverage) or max_leverage <= 0:
            max_leverage = 1.0
        else:
            max_leverage = max(max_leverage / max(portfolio_value, 1.0), 1.0)
            max_leverage = min(max_leverage, 20.0)  # cap

        # Default label: any liquidation = 1
        label = 1 if len(liqs) > 0 else 0

        rows.append({
            "wallet_address":    wallet,
            "first_seen":        first_seen,
            "last_seen":         last_seen,
            "wallet_age_days":   age_days,
            "tx_count_lifetime": len(grp),
            "n_protocols":       grp["protocol"].nunique(),
            "portfolio_value_usd": float(portfolio_value),
            "stablecoin_ratio":  float(stablecoin_ratio),
            "max_leverage":      float(max_leverage),
            "has_borrow":        int(len(borrows) > 0),
            "repayment_ratio":   float(repayment_ratio),
            "n_liquidations":    int(len(liqs)),
            "label":             label,
        })

    wallets_df = pd.DataFrame(rows)
    logger.info(
        f"Built wallet features for {len(wallets_df):,} wallets. "
        f"Default rate: {wallets_df['label'].mean():.3%}"
    )
    return wallets_df


# ─────────────────────────────────────────────────────────────────────────────
#  Mock data generator (no Dune API required — for local testing)
# ─────────────────────────────────────────────────────────────────────────────

def make_mock_real_data(config_path: str = "configs/pipeline.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create mock 'real' data by sampling and lightly perturbing the existing
    synthetic events.parquet.  Useful for testing the two-stage training
    pipeline without a Dune API key.

    Saves to data/raw/real_events.parquet and data/raw/real_wallets.parquet.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    synt_events  = pd.read_parquet(raw_dir / "events.parquet")
    synt_wallets = pd.read_parquet(raw_dir / "wallets.parquet")

    rng = np.random.default_rng(seed=99)

    # Sample ~20% of wallets as the "real" cohort
    n_real = max(200, int(len(synt_wallets) * 0.20))
    sampled_wallets = synt_wallets.sample(n=n_real, random_state=99)

    # Remap wallet addresses to look like fresh on-chain addresses
    addr_map = {
        old: "0x" + "".join(f"{b:02x}" for b in rng.integers(0, 256, size=20))
        for old in sampled_wallets["wallet_address"]
    }

    events_df = synt_events[synt_events["wallet_address"].isin(addr_map)].copy()
    events_df["wallet_address"] = events_df["wallet_address"].map(addr_map)

    # Shift timestamps forward by ~1 year to avoid date overlap with synthetic
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"]) + pd.DateOffset(years=1)

    # Add small noise to USD amounts (±10%) to simulate real data distribution shift
    mask = events_df["usd_amount"] > 0
    noise = rng.uniform(0.90, 1.10, size=mask.sum())
    events_df.loc[mask, "usd_amount"] = (events_df.loc[mask, "usd_amount"] * noise).round(2)

    # Map protocol names to real protocol names
    proto_map = {"aave_v3": "aave", "compound_v3": "compound",
                 "uniswap_v3": "aave", "curve": "compound"}
    events_df["protocol"] = events_df["protocol"].map(
        lambda p: proto_map.get(p, "aave")
    )

    # Only keep lending-relevant event types
    keep_types = {"borrow", "repayment", "liquidation", "deposit", "withdraw"}
    events_df = events_df[events_df["event_type"].isin(keep_types)]

    # Build wallet features
    wallets_df = sampled_wallets.copy()
    wallets_df["wallet_address"] = wallets_df["wallet_address"].map(addr_map)
    wallets_df = wallets_df[[
        "wallet_address", "first_seen", "last_seen", "wallet_age_days",
        "tx_count_lifetime", "n_protocols", "portfolio_value_usd",
        "stablecoin_ratio", "max_leverage", "has_borrow",
        "repayment_ratio", "n_liquidations",
    ]].copy()
    wallets_df["label"] = (wallets_df["n_liquidations"] > 0).astype(int)

    events_out  = raw_dir / "real_events.parquet"
    wallets_out = raw_dir / "real_wallets.parquet"
    events_df.to_parquet(events_out,  index=False)
    wallets_df.to_parquet(wallets_out, index=False)

    logger.info(
        f"Mock real data: {len(events_df):,} events, "
        f"{events_df['wallet_address'].nunique():,} wallets, "
        f"default_rate={wallets_df['label'].mean():.2%}"
    )
    logger.info(f"Saved → {events_out}")
    logger.info(f"Saved → {wallets_out}")
    return events_df, wallets_df


def _print_dune_instructions(start_date: str, end_date: str):
    sql_with_dates = LENDING_SQL.replace("{{start_date}}", start_date).replace("{{end_date}}", end_date)
    print("\n" + "="*70)
    print("DUNE SETUP INSTRUCTIONS")
    print("="*70)
    print(f"""
The free Dune API tier requires you to save queries on the website first.

Steps:
  1. Go to  https://dune.com/queries
  2. Click  New Query
  3. Paste the SQL below into the editor
  4. Click  Save  (give it any name, e.g. 'lending_events')
  5. Click  Run   (runs once in the UI; free, no API credits used)
  6. Copy the query ID from the URL:
       https://dune.com/queries/XXXXXXX  ← this number
  7. Open  configs/pipeline.yaml  and set:
       dune:
         query_ids:
           lending: XXXXXXX   ← paste the number here
  8. Re-run:  python data/dune_fetcher.py

SQL to save on Dune
(uses lending.borrow + lending.supply spellbook, {start_date} – {end_date}):
""")
    print(sql_with_dates)
    print("="*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main fetch routine
# ─────────────────────────────────────────────────────────────────────────────

def fetch(config_path: str = "configs/pipeline.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch Aave V3 + Compound V3 events from Dune, save parquet files.

    Returns:
        (events_df, wallets_df)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dune_cfg  = cfg.get("dune", {})
    raw_dir   = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start_date = dune_cfg.get("date_range", {}).get("start", "2023-01-01")
    end_date   = dune_cfg.get("date_range", {}).get("end",   "2025-01-01")

    # Substitute date placeholders into inline SQL
    def _sql(template: str) -> str:
        return template.replace("{{start_date}}", start_date).replace("{{end_date}}", end_date)

    query_ids   = dune_cfg.get("query_ids", {}) or {}
    lending_qid = query_ids.get("lending")

    if lending_qid is None:
        # Print the SQL and guide the user to save it on Dune
        _print_dune_instructions(start_date, end_date)
        raise SystemExit(
            "\nNo saved query ID found.\n"
            "Please follow the instructions above to save the SQL on Dune, "
            "then add the query ID to configs/pipeline.yaml under dune.query_ids.lending.\n"
            "To test the pipeline without Dune, run:\n"
            "  python data/dune_fetcher.py --mock"
        )

    client = _get_client()

    # ── Fetch using saved query (free-tier compatible) ────────────────────────
    raw       = _fetch_saved_query(client, lending_qid, "lending_events")
    events_df = _normalise_events(raw)

    if events_df.empty:
        logger.warning(
            "No events returned from Dune. "
            "Check your API key, query IDs, and date range."
        )
        return events_df, pd.DataFrame()

    # Optional: cap at max_wallets to avoid memory issues with huge datasets
    max_wallets = dune_cfg.get("max_wallets", 50_000)
    min_events  = dune_cfg.get("min_events_per_wallet", 3)

    # Filter wallets with too few events (noise)
    event_counts = events_df.groupby("wallet_address").size()
    active_wallets = event_counts[event_counts >= min_events].index
    events_df = events_df[events_df["wallet_address"].isin(active_wallets)]
    logger.info(f"After min_events filter ({min_events}): {events_df['wallet_address'].nunique():,} wallets")

    # Cap wallet count (largest by event volume, to keep most interesting wallets)
    if events_df["wallet_address"].nunique() > max_wallets:
        top_wallets = (
            event_counts[active_wallets]
            .nlargest(max_wallets)
            .index
        )
        events_df = events_df[events_df["wallet_address"].isin(top_wallets)]
        logger.info(f"Capped to top {max_wallets:,} wallets by event count")

    logger.info(
        f"Final events_df: {len(events_df):,} rows, "
        f"{events_df['wallet_address'].nunique():,} wallets"
    )

    # ── Build wallet-level summary ───────────────────────────────────────────
    wallets_df = _build_wallet_features(events_df)

    # ── Save ─────────────────────────────────────────────────────────────────
    events_out  = raw_dir / "real_events.parquet"
    wallets_out = raw_dir / "real_wallets.parquet"

    events_df.to_parquet(events_out,  index=False)
    wallets_df.to_parquet(wallets_out, index=False)

    logger.info(f"Saved real events  → {events_out}")
    logger.info(f"Saved real wallets → {wallets_out}")

    return events_df, wallets_df


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch real on-chain lending data from Dune Analytics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch real data (requires saved Dune query ID in pipeline.yaml):
  python data/dune_fetcher.py

  # Generate mock real data from synthetic data (no Dune API required):
  python data/dune_fetcher.py --mock

  # Print the Dune SQL and setup instructions:
  python data/dune_fetcher.py --show-sql
""",
    )
    parser.add_argument("--config",   default="configs/pipeline.yaml")
    parser.add_argument("--mock",     action="store_true",
                        help="Generate mock real data from synthetic data (no Dune API needed)")
    parser.add_argument("--show-sql", action="store_true",
                        help="Print the Dune SQL and setup instructions, then exit")
    args = parser.parse_args()

    if args.show_sql:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        dc = cfg.get("dune", {})
        _print_dune_instructions(
            dc.get("date_range", {}).get("start", "2024-10-01"),
            dc.get("date_range", {}).get("end",   "2025-01-01"),
        )
    elif args.mock:
        make_mock_real_data(args.config)
    else:
        fetch(args.config)
