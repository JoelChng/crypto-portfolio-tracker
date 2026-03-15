"""
Hierarchical persona-based synthetic on-chain event generator.

Five personas with distinct behavioral profiles:
  0 - DeFi Power User   : high activity, diversified, moderate risk
  1 - HODLer            : low activity, concentrated, low risk
  2 - Fresh Wallet       : short history, small balances, moderate risk
  3 - Liquidation Risk   : high leverage, borrow-heavy, high risk
  4 - Fraudster / Sybil  : rapid flows, circular patterns, very high risk

Generates two parquet files:
  data/raw/events.parquet   - one row per on-chain event
  data/raw/wallets.parquet  - one row per wallet (metadata + persona)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# Persona parameter tables
# ──────────────────────────────────────────────────
PERSONAS = {
    0: dict(
        name="defi_power_user",
        weight=0.30,
        wallet_age_days=(180, 900),          # uniform (min, max)
        tx_count=(200, 1200),
        n_protocols=(4, 15),
        borrow_prob=0.70,
        repayment_ratio=(0.80, 1.0),
        liquidation_rate=0.04,
        max_leverage=(1.2, 4.0),
        portfolio_value=(5_000, 200_000),
        stablecoin_ratio=(0.20, 0.50),
        default_base_rate=0.04,
    ),
    1: dict(
        name="hodler",
        weight=0.25,
        wallet_age_days=(365, 1800),
        tx_count=(5, 80),
        n_protocols=(1, 3),
        borrow_prob=0.05,
        repayment_ratio=(0.95, 1.0),
        liquidation_rate=0.001,
        max_leverage=(1.0, 1.2),
        portfolio_value=(10_000, 500_000),
        stablecoin_ratio=(0.05, 0.25),
        default_base_rate=0.01,
    ),
    2: dict(
        name="fresh_wallet",
        weight=0.20,
        wallet_age_days=(7, 120),
        tx_count=(3, 50),
        n_protocols=(1, 4),
        borrow_prob=0.20,
        repayment_ratio=(0.60, 1.0),
        liquidation_rate=0.05,
        max_leverage=(1.0, 2.5),
        portfolio_value=(100, 5_000),
        stablecoin_ratio=(0.30, 0.70),
        default_base_rate=0.07,
    ),
    3: dict(
        name="liquidation_risk",
        weight=0.15,
        wallet_age_days=(30, 400),
        tx_count=(50, 400),
        n_protocols=(2, 8),
        borrow_prob=0.95,
        repayment_ratio=(0.40, 0.85),
        liquidation_rate=0.30,
        max_leverage=(3.0, 10.0),
        portfolio_value=(2_000, 50_000),
        stablecoin_ratio=(0.05, 0.20),
        default_base_rate=0.25,
    ),
    4: dict(
        name="fraudster_sybil",
        weight=0.10,
        wallet_age_days=(1, 60),
        tx_count=(20, 300),
        n_protocols=(1, 5),
        borrow_prob=0.40,
        repayment_ratio=(0.20, 0.60),
        liquidation_rate=0.20,
        max_leverage=(1.0, 5.0),
        portfolio_value=(100, 10_000),
        stablecoin_ratio=(0.50, 0.90),
        default_base_rate=0.40,
    ),
}

# Normal event types — bad credit events are injected separately via _inject_bad_events
EVENT_TYPES = [
    "transfer_in", "transfer_out",
    "swap", "dex_trade",
    "borrow", "repayment",
    "deposit", "withdraw",
    "bridge", "stake", "unstake",
]

PROTOCOLS = [
    "aave", "compound", "uniswap_v3", "uniswap_v2",
    "curve", "balancer", "lido", "maker", "yearn",
    "1inch", "sushi", "stargate", "hop", "across",
]

TOKENS = [
    "ETH", "WBTC", "USDC", "USDT", "DAI",
    "LINK", "UNI", "AAVE", "CRV", "MKR",
    "PEPE", "SHIB", "ARB", "OP", "MATIC",
]


def _uniform(lo, hi, rng):
    return rng.uniform(lo, hi)


def _randint(lo, hi, rng):
    return int(rng.integers(lo, hi + 1))


class SyntheticGenerator:
    def __init__(self, config: dict, rng: np.random.Generator = None):
        self.cfg = config
        self.rng = rng or np.random.default_rng(config.get("random_seed", 42))
        self.start = pd.Timestamp(config["start_date"])
        self.end   = pd.Timestamp(config["end_date"])
        self.n_wallets = config["n_wallets"]

    def _assign_personas(self):
        weights = [PERSONAS[i]["weight"] for i in range(5)]
        persona_ids = self.rng.choice(5, size=self.n_wallets, p=weights)
        return persona_ids

    def _generate_wallet(self, wallet_id: int, persona_id: int) -> dict:
        p = PERSONAS[persona_id]
        rng = self.rng

        age_days = _uniform(*p["wallet_age_days"], rng)
        first_seen = self.end - pd.Timedelta(days=age_days)
        first_seen = max(first_seen, self.start)

        tx_count = _randint(*p["tx_count"], rng)
        n_protocols = _randint(*p["n_protocols"], rng)
        portfolio_value = float(np.exp(rng.uniform(
            np.log(p["portfolio_value"][0]),
            np.log(p["portfolio_value"][1])
        )))
        stablecoin_ratio = _uniform(*p["stablecoin_ratio"], rng)
        max_leverage = _uniform(*p["max_leverage"], rng)
        has_borrow = rng.random() < p["borrow_prob"]
        repayment_ratio = _uniform(*p["repayment_ratio"], rng) if has_borrow else 1.0
        n_liquidations = int(rng.poisson(p["liquidation_rate"] * (5 if has_borrow else 0)))

        # PD is a sigmoid of latent risk score
        latent_risk = (
            0.3 * (1 - repayment_ratio)
            + 0.2 * (n_liquidations > 0)
            + 0.15 * (max_leverage / 10.0)
            + 0.10 * (1 - stablecoin_ratio)
            + 0.10 * (1 / max(age_days, 1) * 365)
            + 0.15 * rng.random()
        )
        pd_true = p["default_base_rate"] + (1 - p["default_base_rate"]) * (1 / (1 + np.exp(-5 * (latent_risk - 0.5))))
        pd_true = float(np.clip(pd_true, 0.001, 0.999))

        return {
            "wallet_id": wallet_id,
            "wallet_address": f"0x{wallet_id:040x}",
            "persona_id": persona_id,
            "persona_name": PERSONAS[persona_id]["name"],
            "first_seen": first_seen,
            "last_seen": self.end - pd.Timedelta(days=_uniform(0, 30, rng)),
            "wallet_age_days": age_days,
            "tx_count_lifetime": tx_count,
            "n_protocols": n_protocols,
            "portfolio_value_usd": portfolio_value,
            "stablecoin_ratio": stablecoin_ratio,
            "max_leverage": max_leverage,
            "has_borrow": has_borrow,
            "repayment_ratio": repayment_ratio,
            "n_liquidations": n_liquidations,
            "pd_true": pd_true,
        }

    def _generate_events(self, wallet: dict) -> pd.DataFrame:
        rng = self.rng
        persona = PERSONAS[wallet["persona_id"]]
        n_events = wallet["tx_count_lifetime"]

        first_seen = wallet["first_seen"]
        last_seen  = wallet["last_seen"]
        span_secs = max((last_seen - first_seen).total_seconds(), 1)

        # Random timestamps spread across wallet lifetime
        offsets = np.sort(rng.uniform(0, span_secs, n_events))
        timestamps = [first_seen + pd.Timedelta(seconds=float(s)) for s in offsets]

        # Event type distribution per persona
        event_weights = _event_weights_for_persona(wallet["persona_id"])
        event_types = rng.choice(EVENT_TYPES, size=n_events, p=event_weights)

        # Protocol distribution
        n_proto = wallet["n_protocols"]
        active_protocols = rng.choice(PROTOCOLS, size=min(n_proto, len(PROTOCOLS)), replace=False)
        protocols = rng.choice(active_protocols, size=n_events)

        # Token distribution
        tokens = rng.choice(TOKENS, size=n_events,
                            p=_token_weights_for_persona(wallet["persona_id"], wallet["stablecoin_ratio"]))

        # USD amounts — log-normal, scaled by portfolio value
        base_amount = wallet["portfolio_value_usd"] * 0.05
        amounts = np.abs(rng.lognormal(np.log(max(base_amount, 1)), 1.2, size=n_events))

        # Gas fees
        gas_fees = np.abs(rng.lognormal(2.0, 0.8, size=n_events))

        # Health factor — relevant for borrow events
        health_factors = rng.uniform(1.0, 3.0, size=n_events)
        for i, et in enumerate(event_types):
            if et == "liquidation":
                health_factors[i] = rng.uniform(0.5, 1.0)

        # Debt tracking (simplified)
        debt_after = np.where(
            np.isin(event_types, ["borrow"]),
            amounts * wallet["max_leverage"] * rng.uniform(0.3, 1.0, n_events),
            0.0
        )

        events = pd.DataFrame({
            "wallet_address": wallet["wallet_address"],
            "timestamp": timestamps,
            "event_type": event_types,
            "token": tokens,
            "usd_amount": amounts,
            "protocol": protocols,
            "gas_fee_usd": gas_fees,
            "health_factor": health_factors,
            "debt_after_usd": debt_after,
            "persona_id": wallet["persona_id"],
        })

        # Inject bad credit events based on true PD
        events = self._inject_bad_events(events, wallet)
        return events

    def _inject_bad_events(self, events: pd.DataFrame, wallet: dict) -> pd.DataFrame:
        """
        Inject at most ONE bad credit event per wallet.
        - Injection probability = pd_true
        - Event is placed in the window [50% of wallet life, last_seen - 30d]
          so that it falls within the observable snapshot label windows.
        """
        rng = self.rng
        pd_true = wallet["pd_true"]

        if rng.random() > pd_true:
            return events
        if len(events) == 0:
            return events

        first_ts = pd.to_datetime(wallet["first_seen"])
        last_ts  = pd.to_datetime(wallet["last_seen"])
        span_days = max((last_ts - first_ts).days, 1)

        # Place bad event uniformly across the wallet lifetime minus the last 90 days
        # (so it falls within the snapshot label windows for some snapshots)
        max_offset = max(span_days - 90, span_days // 2)
        bad_ts = first_ts + pd.Timedelta(days=int(rng.integers(14, max(max_offset, 15))))

        event_type = "liquidation" if wallet["n_liquidations"] > 0 and rng.random() < 0.5 else "missed_repayment"

        extra = {
            "wallet_address": wallet["wallet_address"],
            "timestamp": bad_ts,
            "event_type": event_type,
            "token": "USDC",
            "usd_amount": float(rng.lognormal(6, 1)),
            "protocol": "aave",
            "gas_fee_usd": 0.0,
            "health_factor": float(rng.uniform(0.5, 0.99)) if event_type == "liquidation" else 1.05,
            "debt_after_usd": float(rng.lognormal(7, 1)),
            "persona_id": wallet["persona_id"],
        }
        return pd.concat([events, pd.DataFrame([extra])], ignore_index=True)

    def generate(self, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {self.n_wallets} wallets...")
        persona_ids = self._assign_personas()

        wallets_list = []
        events_list = []

        for i in range(self.n_wallets):
            wallet = self._generate_wallet(i, persona_ids[i])
            wallets_list.append(wallet)
            events = self._generate_events(wallet)
            events_list.append(events)

            if (i + 1) % 500 == 0:
                logger.info(f"  {i+1}/{self.n_wallets} wallets generated")

        wallets_df = pd.DataFrame(wallets_list)
        events_df  = pd.concat(events_list, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        wallets_path = output_path / "wallets.parquet"
        events_path  = output_path / "events.parquet"
        wallets_df.to_parquet(wallets_path, index=False)
        events_df.to_parquet(events_path,  index=False)

        logger.info(f"Saved {len(wallets_df):,} wallets → {wallets_path}")
        logger.info(f"Saved {len(events_df):,} events  → {events_path}")
        return wallets_df, events_df


# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

def _event_weights_for_persona(persona_id: int) -> np.ndarray:
    # EVENT_TYPES: 0=transfer_in, 1=transfer_out, 2=swap, 3=dex_trade,
    #              4=borrow, 5=repayment, 6=deposit, 7=withdraw,
    #              8=bridge, 9=stake, 10=unstake
    base = np.ones(len(EVENT_TYPES))
    if persona_id == 0:   # DeFi power user
        base[[2,3,4,5,6,7]] *= 3.0
    elif persona_id == 1: # HODLer
        base[[0,1]] *= 4.0
        base[[4,5,6,7]] *= 0.3
    elif persona_id == 2: # Fresh
        base[[0,1,2]] *= 2.0
    elif persona_id == 3: # Liquidation risk (heavy borrowing)
        base[[4,5]] *= 5.0
    elif persona_id == 4: # Fraudster (rapid fund movement, bridging)
        base[[0,1,8]] *= 5.0
        base[[4]] *= 2.0
    base /= base.sum()
    return base


def _token_weights_for_persona(persona_id: int, stablecoin_ratio: float) -> np.ndarray:
    n = len(TOKENS)
    weights = np.ones(n)
    stablecoin_idx = [2, 3, 4]   # USDC, USDT, DAI
    for idx in stablecoin_idx:
        weights[idx] = stablecoin_ratio * n / len(stablecoin_idx)
    non_stable = [i for i in range(n) if i not in stablecoin_idx]
    remaining = 1.0 - stablecoin_ratio
    for idx in non_stable:
        weights[idx] = remaining * n / len(non_stable)
    weights /= weights.sum()
    return weights


def run(config_path: str = "configs/pipeline.yaml"):
    logging.basicConfig(level=logging.INFO)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    syn_cfg = {**cfg["synthetic"], **cfg["data"]}
    gen = SyntheticGenerator(syn_cfg)
    return gen.generate(cfg["data"]["raw_dir"])


if __name__ == "__main__":
    run()
