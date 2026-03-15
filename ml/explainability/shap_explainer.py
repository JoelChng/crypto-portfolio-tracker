"""
SHAP explainability — global feature importance and per-wallet reason codes.
"""

import numpy as np
import pandas as pd
import logging
import yaml
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — explainability disabled")


def build_global_shap(model, X: np.ndarray, feature_names: list, artifacts_dir: Path):
    """Compute and save global SHAP values for tree models."""
    if not SHAP_AVAILABLE:
        return None

    model_type = type(model).__name__.lower()

    if "xgb" in model_type or "lgbm" in model_type or "randomforest" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    elif "logistic" in model_type:
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X[:100])
        shap_values = explainer.shap_values(X[:500])

    # For binary classifiers, shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    out_path = artifacts_dir / "shap_global_importance.csv"
    importance_df.to_csv(out_path, index=False)
    logger.info(f"Saved global SHAP importance → {out_path}")

    # Save background dataset (200 representative samples)
    bg_size = min(200, len(X))
    idx = np.random.choice(len(X), bg_size, replace=False)
    bg_df = pd.DataFrame(X[idx], columns=feature_names)
    bg_path = artifacts_dir / "shap_background.parquet"
    bg_df.to_parquet(bg_path, index=False)

    return importance_df


def local_shap(model, x_row: np.ndarray, feature_names: list,
               background: np.ndarray = None) -> dict:
    """
    Compute per-wallet SHAP values.
    Returns dict: {feature_name: shap_value}
    """
    if not SHAP_AVAILABLE:
        return {}

    model_type = type(model).__name__.lower()
    x_row = np.asarray(x_row).reshape(1, -1)

    try:
        if "xgb" in model_type or "lgbm" in model_type or "randomforest" in model_type:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x_row)
        elif "logistic" in model_type:
            bg = background if background is not None else x_row
            explainer = shap.LinearExplainer(model, bg)
            sv = explainer.shap_values(x_row)
        else:
            bg = background[:50] if background is not None else x_row
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            sv = explainer.shap_values(x_row)

        if isinstance(sv, list):
            sv = sv[1]

        return dict(zip(feature_names, sv.flatten()))

    except Exception as e:
        logger.error(f"SHAP local error: {e}")
        return {}


def top_reason_codes(
    shap_vals: dict,
    reason_codes: dict,
    feature_values: dict,
    top_n: int = 3,
) -> list:
    """
    Maps SHAP values to human-readable reason codes.

    Returns list of dicts: {code, direction, text, shap_value}
    """
    if not shap_vals:
        return []

    sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    results = []

    for feat_name, sv in sorted_feats[:top_n * 2]:   # take extra, filter below
        if feat_name not in reason_codes:
            continue

        rc = reason_codes[feat_name]
        val = feature_values.get(feat_name, 0)
        direction = "risk_increase" if sv > 0 else "risk_decrease"
        is_risk_increase = sv > 0

        # Pick text template
        if is_risk_increase and "text_high" in rc:
            try:
                text = rc["text_high"].format(val=val)
            except (KeyError, ValueError):
                text = rc.get("text_high", "")
        elif not is_risk_increase and "text_low" in rc:
            try:
                text = rc["text_low"].format(val=val)
            except (KeyError, ValueError):
                text = rc.get("text_low", "")
        else:
            continue

        results.append({
            "code": rc["code"],
            "feature": feat_name,
            "direction": direction,
            "text": text,
            "shap_value": round(float(sv), 4),
        })

        if len(results) >= top_n:
            break

    return results


def run_global_shap(config_path: str = "configs/pipeline.yaml", model_name: str = "xgb"):
    import joblib
    logging.basicConfig(level=logging.INFO)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    proc_dir = Path(cfg["data"]["processed_dir"])

    artifact_map = {"xgb": "xgb.pkl", "lgb": "lgb.pkl", "rf": "rf.pkl", "lr": "lr.pkl"}
    artifact_path = artifacts_dir / artifact_map.get(model_name, "xgb.pkl")

    bundle = joblib.load(artifact_path)
    model = bundle["model"]
    features = bundle["features"]

    features_df = pd.read_parquet(proc_dir / "features.parquet")
    X = features_df[features].values[:2000]  # use subset for speed

    importance_df = build_global_shap(model, X, features, artifacts_dir)
    logger.info("\nTop 15 features by mean |SHAP|:")
    print(importance_df.head(15).to_string(index=False))
    return importance_df


if __name__ == "__main__":
    run_global_shap()
