"""
Master training script.

Trains all models sequentially with walk-forward CV, calibrates, evaluates,
and saves artifacts + metrics summary.

Usage:
    python training/train_all.py [--config configs/pipeline.yaml] [--horizon 90]
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_data(cfg: dict, horizon: int = 90):
    proc = Path(cfg["data"]["processed_dir"])
    features = pd.read_parquet(proc / "features.parquet")
    labels   = pd.read_parquet(proc / "labels.parquet")

    merged = features.merge(
        labels[["wallet_address", "snapshot_date", f"default_{horizon}d"]],
        on=["wallet_address", "snapshot_date"],
        how="inner",
    )
    merged["snapshot_date"] = pd.to_datetime(merged["snapshot_date"])
    return merged, f"default_{horizon}d"


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"wallet_address", "snapshot_date", "default_30d", "default_60d", "default_90d"}
    return [c for c in df.columns if c not in exclude]


def select_features(X_train, y_train, feature_cols, top_k=40):
    """Quick importance-based feature selection using RandomForest."""
    from sklearn.ensemble import ExtraTreesClassifier
    et = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    et.fit(X_train, y_train)
    importances = pd.Series(et.feature_importances_, index=feature_cols)
    selected = importances.nlargest(top_k).index.tolist()
    logger.info(f"Selected {len(selected)} features via importance")
    return selected


def train_logistic(X_train, y_train, cfg_lr: dict) -> LogisticRegression:
    from sklearn.model_selection import cross_val_score
    best_score, best_model = -1, None
    for C in cfg_lr.get("C", [0.1]):
        for penalty in cfg_lr.get("penalty", ["l2"]):
            try:
                clf = LogisticRegression(
                    C=C, penalty=penalty,
                    solver=cfg_lr.get("solver", "saga"),
                    class_weight="balanced",
                    max_iter=cfg_lr.get("max_iter", 2000),
                    random_state=42,
                )
                scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model = clf
            except Exception:
                continue
    best_model.fit(X_train, y_train)
    logger.info(f"  LR best CV AUC: {best_score:.4f}")
    return best_model


def train_random_forest(X_train, y_train, cfg_rf: dict) -> RandomForestClassifier:
    best_score, best_model = -1, None
    from sklearn.model_selection import cross_val_score
    grid = list(ParameterGrid({
        "n_estimators": cfg_rf.get("n_estimators", [200]),
        "max_depth":    cfg_rf.get("max_depth", [8]),
        "min_samples_leaf": cfg_rf.get("min_samples_leaf", [10]),
    }))
    for params in grid[:4]:   # limit grid to top-4 combos
        clf = RandomForestClassifier(
            **params, class_weight="balanced", random_state=42, n_jobs=-1
        )
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = clf
    best_model.fit(X_train, y_train)
    logger.info(f"  RF  best CV AUC: {best_score:.4f}")
    return best_model


def train_xgboost(X_train, y_train, cfg_xgb: dict):
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pw = neg / max(pos, 1)
    best_score, best_model = -1, None
    grid = list(ParameterGrid({
        "n_estimators":    cfg_xgb.get("n_estimators", [200]),
        "max_depth":       cfg_xgb.get("max_depth", [4]),
        "learning_rate":   cfg_xgb.get("learning_rate", [0.05]),
        "subsample":       cfg_xgb.get("subsample", [0.8]),
        "colsample_bytree": cfg_xgb.get("colsample_bytree", [0.8]),
    }))
    for params in grid[:4]:
        clf = xgb.XGBClassifier(
            **params, scale_pos_weight=scale_pw,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, n_jobs=-1,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = clf
    best_model.fit(X_train, y_train)
    logger.info(f"  XGB best CV AUC: {best_score:.4f}")
    return best_model


def train_lightgbm(X_train, y_train, cfg_lgb: dict):
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    best_score, best_model = -1, None
    grid = list(ParameterGrid({
        "n_estimators":  cfg_lgb.get("n_estimators", [200]),
        "max_depth":     cfg_lgb.get("max_depth", [4]),
        "learning_rate": cfg_lgb.get("learning_rate", [0.05]),
        "num_leaves":    cfg_lgb.get("num_leaves", [31]),
    }))
    for params in grid[:4]:
        clf = lgb.LGBMClassifier(
            **params, scale_pos_weight=neg/max(pos, 1),
            random_state=42, n_jobs=-1, verbose=-1,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = clf
    best_model.fit(X_train, y_train)
    logger.info(f"  LGB best CV AUC: {best_score:.4f}")
    return best_model


def run(config_path: str = "configs/pipeline.yaml", horizon: int = 90):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    from training.cross_validation import make_oot_split
    from training.calibration import fit_calibrator, compare_calibration
    from training.metrics import evaluate, log_metrics

    logger.info(f"Loading data (horizon={horizon}d)...")
    df, target_col = load_data(cfg, horizon)
    feature_cols = get_feature_cols(df)

    # OOT split
    train_df, test_df = make_oot_split(df, test_months=cfg["training"]["test_size_months"])

    X_train_raw = train_df[feature_cols].values
    y_train     = train_df[target_col].astype(int).values
    X_test_raw  = test_df[feature_cols].values
    y_test      = test_df[target_col].astype(int).values

    logger.info(f"Train: {len(y_train):,} | Test: {len(y_test):,} | Pos rate (train): {y_train.mean():.3%}")

    # Feature selection
    fs_cfg = cfg["training"].get("feature_selection", {})
    if fs_cfg.get("enabled", True):
        selected_cols = select_features(X_train_raw, y_train, feature_cols, top_k=fs_cfg.get("top_k", 40))
        sel_idx = [feature_cols.index(c) for c in selected_cols]
        X_train = X_train_raw[:, sel_idx]
        X_test  = X_test_raw[:, sel_idx]
        (artifacts_dir / "selected_features.json").write_text(json.dumps(selected_cols, indent=2))
    else:
        selected_cols = feature_cols
        X_train, X_test = X_train_raw, X_test_raw

    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    all_metrics = {}
    cal_method = cfg["calibration"]["method"]

    # ── Logistic Regression ──────────────────────────────────────────────
    logger.info("Training Logistic Regression...")
    lr_model = train_logistic(X_train_sc, y_train, cfg["models"]["logistic"])
    raw_probs_lr = lr_model.predict_proba(X_test_sc)[:, 1]
    cal_lr = fit_calibrator(lr_model.predict_proba(X_train_sc)[:, 1], y_train, cal_method)
    cal_probs_lr = cal_lr.predict(raw_probs_lr)
    m_lr = evaluate(y_test, cal_probs_lr)
    log_metrics(m_lr, "logistic", "oot")
    all_metrics["logistic"] = m_lr
    joblib.dump({"model": lr_model, "scaler": scaler, "calibrator": cal_lr, "features": selected_cols},
                artifacts_dir / "lr.pkl")

    # ── Random Forest ────────────────────────────────────────────────────
    logger.info("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train, cfg["models"]["random_forest"])
    raw_probs_rf = rf_model.predict_proba(X_test)[:, 1]
    cal_rf = fit_calibrator(rf_model.predict_proba(X_train)[:, 1], y_train, cal_method)
    cal_probs_rf = cal_rf.predict(raw_probs_rf)
    m_rf = evaluate(y_test, cal_probs_rf)
    log_metrics(m_rf, "random_forest", "oot")
    all_metrics["random_forest"] = m_rf
    joblib.dump({"model": rf_model, "calibrator": cal_rf, "features": selected_cols},
                artifacts_dir / "rf.pkl")

    # ── XGBoost ─────────────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, cfg["models"]["xgboost"])
    raw_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
    cal_xgb = fit_calibrator(xgb_model.predict_proba(X_train)[:, 1], y_train, cal_method)
    cal_probs_xgb = cal_xgb.predict(raw_probs_xgb)
    m_xgb = evaluate(y_test, cal_probs_xgb)
    log_metrics(m_xgb, "xgboost", "oot")
    all_metrics["xgboost"] = m_xgb
    joblib.dump({"model": xgb_model, "calibrator": cal_xgb, "features": selected_cols},
                artifacts_dir / "xgb.pkl")

    # ── LightGBM ─────────────────────────────────────────────────────────
    logger.info("Training LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train, cfg["models"]["lightgbm"])
    raw_probs_lgb = lgb_model.predict_proba(X_test)[:, 1]
    cal_lgb = fit_calibrator(lgb_model.predict_proba(X_train)[:, 1], y_train, cal_method)
    cal_probs_lgb = cal_lgb.predict(raw_probs_lgb)
    m_lgb = evaluate(y_test, cal_probs_lgb)
    log_metrics(m_lgb, "lightgbm", "oot")
    all_metrics["lightgbm"] = m_lgb
    joblib.dump({"model": lgb_model, "calibrator": cal_lgb, "features": selected_cols},
                artifacts_dir / "lgb.pkl")

    # ── Save scaler separately for inference ────────────────────────────
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(selected_cols, artifacts_dir / "feature_list.pkl")

    # ── Summary ──────────────────────────────────────────────────────────
    summary = {
        "horizon_days": horizon,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_test": float(y_test.mean()),
        "models": {k: {m: round(v, 4) if isinstance(v, float) else v
                       for m, v in metrics.items()}
                   for k, metrics in all_metrics.items()},
        "champion": max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"]),
    }
    (artifacts_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"\nChampion model (by AUC): {summary['champion']}")
    logger.info("Artifacts saved to " + str(artifacts_dir))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--horizon", type=int, default=90)
    args = parser.parse_args()
    run(args.config, args.horizon)
