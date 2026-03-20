"""
Two-stage pretrain → fine-tune MLP trainer.

Stage 1 — Pretrain on synthetic data:
    Reads data/processed/features.parquet + labels.parquet (produced by the
    existing pipeline) and trains a PyTorch MLP credit-scoring model.
    Saves checkpoint to models/checkpoints/pretrained.pt.

Stage 2 — Fine-tune on real data:
    Runs the full snapshot→label→feature pipeline on data/raw/real_events.parquet
    (produced by data/dune_fetcher.py), loads the pretrained weights, and
    fine-tunes with a 10× lower learning rate.
    Saves checkpoint to models/checkpoints/finetuned.pt.

Evaluation:
    Evaluates ONLY on the held-out real test set.  Compares fine-tuned model
    against a same-architecture baseline trained on real data only (no
    pretraining).  Reports AUC-ROC, PR-AUC, Brier score, and a calibration
    plot.  Saves metrics to models/metrics/two_stage_results.json.

Usage:
    # Run full pipeline (requires pretrained synthetic features + real events):
    python models/two_stage_trainer.py

    # Pretrain only (synthetic data must already be processed):
    python models/two_stage_trainer.py --stage pretrain

    # Fine-tune only (pretrained.pt must already exist):
    python models/two_stage_trainer.py --stage finetune

    # Evaluate only (finetuned.pt must already exist):
    python models/two_stage_trainer.py --stage eval
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("models/checkpoints")
METRICS_DIR    = Path("models/metrics")


# ─────────────────────────────────────────────────────────────────────────────
#  Model architecture
# ─────────────────────────────────────────────────────────────────────────────

class CreditMLP(nn.Module):
    """
    Fully-connected MLP for binary default prediction.

    Input : feature vector (n_features,)
    Output: scalar logit (sigmoid → probability of default)
    """

    def __init__(self, n_features: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)   # raw logits, shape (batch,)


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_processed_dataset(
    features_df: pd.DataFrame,
    labels_df:   pd.DataFrame,
    selected_cols: list[str],
    horizon: int = 90,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Merge features + labels, align to selected_cols. Returns X, y, dates."""
    label_col = f"default_{horizon}d"
    merged = features_df.merge(
        labels_df[["wallet_address", "snapshot_date", label_col]],
        on=["wallet_address", "snapshot_date"],
        how="inner",
    )
    merged["snapshot_date"] = pd.to_datetime(merged["snapshot_date"])

    # Align to the expected feature set; fill any missing columns with 0
    for col in selected_cols:
        if col not in merged.columns:
            merged[col] = 0.0

    X = merged[selected_cols].values.astype(np.float32)
    y = merged[label_col].astype(int).values.astype(np.float32)
    return X, y, merged["snapshot_date"].reset_index(drop=True)


def _temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    val_frac:  float = 0.15,
    test_frac: float = 0.15,
) -> tuple:
    """70/15/15 temporal split by snapshot date (no shuffle)."""
    order = np.argsort(dates.values)
    n = len(order)
    val_start  = int(n * (1 - val_frac - test_frac))
    test_start = int(n * (1 - test_frac))

    train_idx = order[:val_start]
    val_idx   = order[val_start:test_start]
    test_idx  = order[test_start:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


def _build_real_features(
    cfg: dict,
    horizon: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run snapshot → label → feature pipeline on real_events.parquet.
    Caches result to data/processed/real_features.parquet to avoid recomputing.
    """
    from data.snapshot_builder import build_snapshots
    from data.label_generator  import generate_labels
    from features.assembler    import build_features

    raw_dir  = Path(cfg["data"]["raw_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])

    events_path  = raw_dir / "real_events.parquet"
    wallets_path = raw_dir / "real_wallets.parquet"

    if not events_path.exists():
        raise FileNotFoundError(
            f"Real events not found at {events_path}. "
            "Run  python data/dune_fetcher.py  first."
        )

    events_df  = pd.read_parquet(events_path)
    wallets_df = pd.read_parquet(wallets_path)

    # Normalise all datetime columns to tz-naive UTC to avoid tz comparison errors
    # in label_generator.py (which compares event_ts with tz-naive snapshot_date).
    def _strip_tz(col: pd.Series) -> pd.Series:
        s = pd.to_datetime(col)
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s

    events_df["timestamp"] = _strip_tz(events_df["timestamp"])
    for col in ("first_seen", "last_seen"):
        if col in wallets_df.columns:
            wallets_df[col] = _strip_tz(wallets_df[col])

    # Cache check
    feat_cache   = proc_dir / "real_features.parquet"
    labels_cache = proc_dir / "real_labels.parquet"
    if feat_cache.exists() and labels_cache.exists():
        logger.info("Loading cached real features/labels from data/processed/")
        return pd.read_parquet(feat_cache), pd.read_parquet(labels_cache)

    proc_dir.mkdir(parents=True, exist_ok=True)

    # Derive date bounds from events (timestamps already tz-naive after _strip_tz above)
    end_date    = pd.to_datetime(events_df["timestamp"]).max()
    max_horizon = max(cfg["labels"]["horizons"])

    logger.info(f"Building snapshots for {wallets_df['wallet_address'].nunique():,} real wallets...")
    snaps_df = build_snapshots(
        wallets_df,
        end_date=end_date,
        interval_days=cfg["snapshots"]["interval_days"],
        min_wallet_age_days=cfg["snapshots"]["min_wallet_age_days"],
        max_horizon_days=max_horizon,
    )

    if snaps_df.empty:
        raise ValueError(
            "No valid snapshots generated for real wallets. "
            "The wallets may be too new (< min_wallet_age_days) or the date "
            "range too short to create label windows."
        )

    logger.info("Generating labels for real data...")
    labels_df = generate_labels(events_df, snaps_df, cfg["labels"]["horizons"])

    logger.info("Engineering features for real data...")
    features_df = build_features(events_df, snaps_df, cfg)

    features_df.to_parquet(feat_cache,   index=False)
    labels_df.to_parquet(labels_cache,   index=False)
    logger.info(f"Cached real features → {feat_cache}")

    return features_df, labels_df


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train(
    model:      CreditMLP,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    epochs:     int,
    lr:         float,
    batch_size: int,
    patience:   int = 10,
    device:     str = "cpu",
) -> list[float]:
    """
    Train with BCEWithLogitsLoss + class weighting + early stopping.
    Returns validation loss history.
    """
    model.to(device)

    # Class weight to handle imbalance
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=max(patience // 3, 2), factor=0.5, verbose=False
    )

    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train).to(device),
            torch.FloatTensor(y_train).to(device),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(X_train) > batch_size),
    )

    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    val_losses    = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.binary_cross_entropy_with_logits(
                model(X_v), y_v, pos_weight=pos_weight
            ).item()

        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"    epoch {epoch+1:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

        if no_improve >= patience:
            logger.info(f"    Early stopping at epoch {epoch + 1} (no val improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return val_losses


def _get_probs(model: CreditMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    model.to(device)
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.FloatTensor(X).to(device))
        ).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration & metrics
# ─────────────────────────────────────────────────────────────────────────────

def _fit_calibrator(raw_probs: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_probs, y)
    return cal


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, label: str) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    m = {
        "label":       label,
        "n":           int(len(y_true)),
        "pos_rate":    float(y_true.mean()),
        "roc_auc":     float(roc_auc_score(y_true, probs)),
        "pr_auc":      float(average_precision_score(y_true, probs)),
        "brier_score": float(brier_score_loss(y_true, probs)),
    }
    logger.info(
        f"  [{label:40s}]  "
        f"AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
        f"Brier={m['brier_score']:.4f}  n={m['n']:,}  pos={m['pos_rate']:.2%}"
    )
    return m


def _save_calibration_plot(
    y_true:      np.ndarray,
    probs_dict:  dict[str, np.ndarray],
    out_path:    Path,
):
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect", linewidth=1)
    for label, probs in probs_dict.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label, linewidth=1.5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration plot — real test set")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved calibration plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Stage runners
# ─────────────────────────────────────────────────────────────────────────────

def run_pretrain(cfg: dict, device: str, horizon: int = 90) -> tuple[CreditMLP, StandardScaler, list[str]]:
    """
    Stage 1: pretrain on synthetic data.
    Returns (model, scaler, selected_cols).
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1 — Pretraining on synthetic data")
    logger.info("=" * 60)

    proc_dir   = Path(cfg["data"]["processed_dir"])
    art_dir    = Path(cfg["data"]["artifacts_dir"])
    pt_cfg     = cfg.get("training", {}).get("pretrain", {})
    eval_cfg   = cfg.get("training", {}).get("eval", {})

    # ── Load processed synthetic data ──────────────────────────────────────
    features_df = pd.read_parquet(proc_dir / "features.parquet")
    labels_df   = pd.read_parquet(proc_dir / "labels.parquet")

    # Selected feature list from prior train_all.py run, or derive from data
    feat_list_path = art_dir / "selected_features.json"
    if feat_list_path.exists():
        selected_cols = json.loads(feat_list_path.read_text())
        logger.info(f"Using {len(selected_cols)} features from {feat_list_path}")
    else:
        exclude = {"wallet_address", "snapshot_date", "default_30d", "default_60d", "default_90d"}
        selected_cols = [c for c in features_df.columns if c not in exclude]
        logger.info(f"Derived {len(selected_cols)} features from features.parquet")

    X, y, dates = _load_processed_dataset(features_df, labels_df, selected_cols, horizon)

    # OOT split (consistent with train_all.py)
    from training.cross_validation import make_oot_split

    merged = features_df.merge(
        labels_df[["wallet_address", "snapshot_date", f"default_{horizon}d"]],
        on=["wallet_address", "snapshot_date"],
        how="inner",
    )
    merged["snapshot_date"] = pd.to_datetime(merged["snapshot_date"])
    for col in selected_cols:
        if col not in merged.columns:
            merged[col] = 0.0

    train_df, _ = make_oot_split(merged, test_months=cfg["training"]["test_size_months"])

    X_train = train_df[selected_cols].values.astype(np.float32)
    y_train = train_df[f"default_{horizon}d"].astype(int).values.astype(np.float32)

    # Fit scaler on training data (will be re-used for real data)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)

    # Hold out last val_frac% of training data for early stopping
    val_frac = eval_cfg.get("val_split", 0.15)
    n_val    = max(1, int(len(X_train_sc) * val_frac))
    X_pt, y_pt = X_train_sc[:-n_val], y_train[:-n_val]
    X_pv, y_pv = X_train_sc[-n_val:], y_train[-n_val:]

    logger.info(
        f"Synthetic  train={len(y_pt):,}  val={len(y_pv):,}  "
        f"pos={y_pt.mean():.3%}"
    )

    # ── Build + train model ────────────────────────────────────────────────
    n_features  = len(selected_cols)
    hidden_dims = pt_cfg.get("hidden_dims", [128, 64, 32])
    dropout     = pt_cfg.get("dropout", 0.3)

    model = CreditMLP(n_features, hidden_dims, dropout)
    logger.info(f"Architecture: {n_features} → {hidden_dims} → 1  dropout={dropout}")

    _train(
        model, X_pt, y_pt, X_pv, y_pv,
        epochs     = pt_cfg.get("epochs", 50),
        lr         = pt_cfg.get("learning_rate", 0.001),
        batch_size = pt_cfg.get("batch_size", 512),
        patience   = pt_cfg.get("patience", 10),
        device     = device,
    )

    # ── Save checkpoint ────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "pretrained.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean":      scaler.mean_.tolist(),
        "scaler_scale":     scaler.scale_.tolist(),
        "selected_cols":    selected_cols,
        "hidden_dims":      hidden_dims,
        "dropout":          dropout,
        "horizon":          horizon,
        "n_synthetic_train": int(len(y_pt)),
    }, ckpt_path)
    logger.info(f"Saved pretrained checkpoint → {ckpt_path}")

    return model, scaler, selected_cols


def run_finetune(cfg: dict, device: str, horizon: int = 90) -> tuple[CreditMLP, CreditMLP, np.ndarray, np.ndarray]:
    """
    Stage 2: fine-tune on real data + train baseline.
    Returns (finetuned_model, baseline_model, X_test_sc, y_test).
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 — Fine-tuning on real data")
    logger.info("=" * 60)

    ft_cfg   = cfg.get("training", {}).get("finetune", {})
    pt_cfg   = cfg.get("training", {}).get("pretrain", {})
    eval_cfg = cfg.get("training", {}).get("eval", {})

    # ── Load pretrained checkpoint ─────────────────────────────────────────
    ckpt_path = CHECKPOINT_DIR / "pretrained.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {ckpt_path}. "
            "Run  python models/two_stage_trainer.py --stage pretrain  first."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    selected_cols = ckpt["selected_cols"]
    hidden_dims   = ckpt["hidden_dims"]
    dropout       = ckpt["dropout"]
    n_features    = len(selected_cols)

    scaler = StandardScaler()
    scaler.mean_  = np.array(ckpt["scaler_mean"],  dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = n_features

    logger.info(f"Loaded pretrained checkpoint  ({n_features} features, hidden={hidden_dims})")

    # ── Build real features ────────────────────────────────────────────────
    real_features, real_labels = _build_real_features(cfg, horizon)
    X_real, y_real, dates_real = _load_processed_dataset(
        real_features, real_labels, selected_cols, horizon
    )

    if len(X_real) == 0:
        raise ValueError(
            "No labelled snapshots found for real data. "
            "Check that real_events.parquet covers a long enough time window "
            "to generate snapshots and non-empty label windows."
        )

    # Scale using same scaler from pretraining
    X_real_sc = scaler.transform(X_real).astype(np.float32)

    # 70/15/15 temporal split
    val_frac  = eval_cfg.get("val_split",  0.15)
    test_frac = eval_cfg.get("test_split", 0.15)
    X_tr, y_tr, X_va, y_va, X_te, y_te = _temporal_split(
        X_real_sc, y_real, dates_real, val_frac, test_frac
    )

    logger.info(
        f"Real data split  train={len(y_tr):,}  val={len(y_va):,}  "
        f"test={len(y_te):,}  test_pos={y_te.mean():.3%}"
    )

    if len(y_tr) < 10:
        raise ValueError(
            f"Only {len(y_tr)} training samples in real data. "
            "Fetch more data or widen the date range."
        )

    # ── Fine-tune from pretrained weights ──────────────────────────────────
    logger.info("\nFine-tuning from pretrained weights (10× lower LR)...")
    finetuned = CreditMLP(n_features, hidden_dims, dropout)
    finetuned.load_state_dict(ckpt["model_state_dict"])

    _train(
        finetuned, X_tr, y_tr, X_va, y_va,
        epochs     = ft_cfg.get("epochs", 30),
        lr         = ft_cfg.get("learning_rate", 0.0001),   # 10× lower
        batch_size = ft_cfg.get("batch_size", 128),
        patience   = ft_cfg.get("patience", 7),
        device     = device,
    )

    ft_ckpt = CHECKPOINT_DIR / "finetuned.pt"
    torch.save({
        "model_state_dict": finetuned.state_dict(),
        "scaler_mean":      scaler.mean_.tolist(),
        "scaler_scale":     scaler.scale_.tolist(),
        "selected_cols":    selected_cols,
        "hidden_dims":      hidden_dims,
        "dropout":          dropout,
        "horizon":          horizon,
        "n_real_train":     int(len(y_tr)),
    }, ft_ckpt)
    logger.info(f"Saved fine-tuned checkpoint → {ft_ckpt}")

    # ── Baseline: train from scratch on real data only ─────────────────────
    logger.info("\nTraining baseline (scratch, real data only)...")
    baseline = CreditMLP(n_features, hidden_dims, dropout)

    _train(
        baseline, X_tr, y_tr, X_va, y_va,
        epochs     = ft_cfg.get("epochs", 30),
        lr         = pt_cfg.get("learning_rate", 0.001),    # same as pretrain LR
        batch_size = ft_cfg.get("batch_size", 128),
        patience   = ft_cfg.get("patience", 7),
        device     = device,
    )

    # Store calibrators and val probs for eval stage
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "baseline_state_dict": baseline.state_dict(),
        "X_val_sc":            X_va,
        "y_val":               y_va,
        "X_test_sc":           X_te,
        "y_test":              y_te,
    }, CHECKPOINT_DIR / "_eval_cache.pt")

    return finetuned, baseline, X_te, y_te


def run_eval(cfg: dict, device: str, horizon: int = 90) -> dict:
    """
    Evaluate finetuned vs baseline on held-out real test set.
    Requires finetuned.pt and _eval_cache.pt to exist.
    """
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION — real held-out test set")
    logger.info("=" * 60)

    ft_ckpt     = CHECKPOINT_DIR / "finetuned.pt"
    cache_ckpt  = CHECKPOINT_DIR / "_eval_cache.pt"

    if not ft_ckpt.exists() or not cache_ckpt.exists():
        raise FileNotFoundError(
            "Missing checkpoint files. Run --stage finetune first."
        )

    ft_data    = torch.load(ft_ckpt,    map_location="cpu")
    cache_data = torch.load(cache_ckpt, map_location="cpu")

    selected_cols = ft_data["selected_cols"]
    hidden_dims   = ft_data["hidden_dims"]
    dropout       = ft_data["dropout"]
    n_features    = len(selected_cols)

    X_val     = cache_data["X_val_sc"]
    y_val     = cache_data["y_val"]
    X_test    = cache_data["X_test_sc"]
    y_test    = cache_data["y_test"]

    finetuned = CreditMLP(n_features, hidden_dims, dropout)
    finetuned.load_state_dict(ft_data["model_state_dict"])

    baseline  = CreditMLP(n_features, hidden_dims, dropout)
    baseline.load_state_dict(cache_data["baseline_state_dict"])

    # Calibrate on validation set
    cal_ft = _fit_calibrator(_get_probs(finetuned, X_val, device), y_val)
    cal_bl = _fit_calibrator(_get_probs(baseline,  X_val, device), y_val)

    probs_ft = np.clip(cal_ft.predict(_get_probs(finetuned, X_test, device)), 0.0, 1.0)
    probs_bl = np.clip(cal_bl.predict(_get_probs(baseline,  X_test, device)), 0.0, 1.0)

    logger.info("\nResults on held-out real test set:")
    m_ft = _compute_metrics(y_test, probs_ft, "finetuned  (pretrain → fine-tune)")
    m_bl = _compute_metrics(y_test, probs_bl, "baseline   (real data only, scratch)")

    uplift = {
        "roc_auc_uplift":     round(m_ft["roc_auc"]    - m_bl["roc_auc"],     4),
        "pr_auc_uplift":      round(m_ft["pr_auc"]      - m_bl["pr_auc"],      4),
        "brier_score_delta":  round(m_bl["brier_score"] - m_ft["brier_score"], 4),  # + = better
    }

    logger.info(f"\n  Pretraining uplift: {uplift}")
    if uplift["roc_auc_uplift"] > 0:
        logger.info("  ✓ Pretraining IMPROVED the model on real data.")
    else:
        logger.info("  ✗ Pretraining did NOT improve the model (baseline won).")
        logger.info("    Consider: more synthetic data, longer finetune epochs, or a different LR ratio.")

    results = {
        "finetuned": m_ft,
        "baseline":  m_bl,
        "uplift":    uplift,
        "horizon_days": horizon,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "two_stage_results.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info(f"\nSaved metrics → {out}")

    _save_calibration_plot(
        y_test,
        {"finetuned": probs_ft, "baseline": probs_bl},
        METRICS_DIR / "calibration_plot.png",
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str = "configs/pipeline.yaml", stage: str = "all", horizon: int = 90):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch device: {device}")

    if stage in ("all", "pretrain"):
        run_pretrain(cfg, device, horizon)

    if stage in ("all", "finetune"):
        run_finetune(cfg, device, horizon)

    if stage in ("all", "eval"):
        results = run_eval(cfg, device, horizon)
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage pretrain → fine-tune MLP trainer.")
    parser.add_argument(
        "--config", default="configs/pipeline.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "pretrain", "finetune", "eval"],
        default="all",
        help=(
            "all      = run all three stages (default)\n"
            "pretrain = Stage 1 only (synthetic data)\n"
            "finetune = Stage 2 only (real data, requires pretrained.pt)\n"
            "eval     = Evaluate only (requires finetuned.pt + _eval_cache.pt)"
        ),
    )
    parser.add_argument("--horizon", type=int, default=90, help="Default horizon in days (30/60/90)")
    args = parser.parse_args()
    run(args.config, args.stage, args.horizon)
