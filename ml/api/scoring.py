"""
Core scoring logic — feature computation → ensemble inference → calibration → score.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
from pathlib import Path

from api.grade_mapper import pd_to_score, pd_to_grade
from api.schemas import CreditScoreResponse, ReasonCode

logger = logging.getLogger(__name__)


class CreditScorer:
    """
    Loads trained model artifacts and scores wallets.

    Ensemble weights are applied to calibrated PD outputs from each model.
    Champion model is determined by highest OOT AUC.
    """

    VERSION = "v1.0.0"

    def __init__(self, artifacts_dir: str, config: dict, reason_codes: dict):
        self.artifacts_dir = Path(artifacts_dir)
        self.cfg = config
        self.reason_codes = reason_codes
        self._models: dict = {}
        self._scaler = None
        self._feature_list: list = []
        self._cal_method = config.get("calibration", {}).get("method", "isotonic")

    def load(self):
        import joblib
        art = self.artifacts_dir

        model_files = {
            "xgboost":       "xgb.pkl",
            "lightgbm":      "lgb.pkl",
            "random_forest": "rf.pkl",
            "logistic":      "lr.pkl",
        }
        for name, fname in model_files.items():
            path = art / fname
            if path.exists():
                self._models[name] = joblib.load(path)
                logger.info(f"Loaded model: {name}")
            else:
                logger.warning(f"Artifact not found: {path}")

        scaler_path = art / "scaler.pkl"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)

        feat_path = art / "feature_list.pkl"
        if feat_path.exists():
            self._feature_list = joblib.load(feat_path)

        if not self._models:
            raise RuntimeError("No model artifacts found. Run training/train_all.py first.")
        logger.info(f"Loaded {len(self._models)} models: {list(self._models.keys())}")

    def _build_feature_row(self, events_df: pd.DataFrame, wallet_address: str,
                           snapshot_date: pd.Timestamp) -> tuple[np.ndarray, dict]:
        from data.snapshot_builder import events_before_snapshot
        from features.assembler import build_features
        from features import (
            tenure_features, cashflow_features, behavioral_features,
            credit_defi_features, portfolio_features, fraud_features, temporal_features,
        )

        wallet_events = events_before_snapshot(events_df, wallet_address, snapshot_date)

        row = {}
        for module in [tenure_features, cashflow_features, behavioral_features,
                       credit_defi_features, portfolio_features, fraud_features, temporal_features]:
            try:
                row.update(module.compute(wallet_events, snapshot_date, wallet_address))
            except Exception:
                row.update(module._empty())

        feature_values = {k: row.get(k, 0) for k in self._feature_list}
        x = np.array([feature_values.get(f, 0) for f in self._feature_list], dtype=np.float64)
        return x, feature_values

    def _predict_pd(self, x_raw: np.ndarray) -> dict:
        """Run all loaded models and return dict of calibrated PDs."""
        pds = {}
        weights_cfg = self.cfg.get("champion", {}).get("ensemble_weights", {})
        for name, bundle in self._models.items():
            try:
                model = bundle["model"]
                cal   = bundle.get("calibrator")
                feats = bundle.get("features", self._feature_list)

                # Align features to model's expected feature list
                if feats != self._feature_list:
                    x = np.array([x_raw[self._feature_list.index(f)]
                                  if f in self._feature_list else 0.0
                                  for f in feats]).reshape(1, -1)
                else:
                    x = x_raw.reshape(1, -1)

                # Scale for logistic regression
                if name == "logistic" and self._scaler is not None:
                    x = self._scaler.transform(x)

                raw_pd = model.predict_proba(x)[0, 1]
                cal_pd = float(cal.predict(np.array([raw_pd]))[0]) if cal else raw_pd
                pds[name] = float(np.clip(cal_pd, 0.001, 0.999))
            except Exception as e:
                logger.error(f"Inference error for {name}: {e}")

        return pds

    def _ensemble_pd(self, pds: dict) -> float:
        """Weighted average ensemble."""
        weights_cfg = self.cfg.get("champion", {}).get("ensemble_weights", {})
        total_w, weighted_sum = 0.0, 0.0
        for name, pd_val in pds.items():
            w = weights_cfg.get(name, 1.0)
            weighted_sum += w * pd_val
            total_w += w
        return float(weighted_sum / max(total_w, 1e-10))

    def score_wallet(
        self,
        events_df: pd.DataFrame,
        wallet_address: str,
        snapshot_date: pd.Timestamp = None,
        explain: bool = True,
    ) -> CreditScoreResponse:
        if snapshot_date is None:
            snapshot_date = pd.Timestamp.now()

        x_raw, feature_values = self._build_feature_row(events_df, wallet_address, snapshot_date)
        pds = self._predict_pd(x_raw)

        if not pds:
            raise RuntimeError("All models failed to produce predictions")

        ensemble_pd = self._ensemble_pd(pds)
        score = pd_to_score(ensemble_pd)
        grade = pd_to_grade(ensemble_pd)

        # Reason codes via SHAP (optional)
        reason_codes_out = []
        if explain and self._models:
            from explainability.shap_explainer import local_shap, top_reason_codes
            champion_name = self.cfg.get("champion", {}).get("model", list(self._models.keys())[0])
            bundle = self._models.get(champion_name, list(self._models.values())[0])
            feats = bundle.get("features", self._feature_list)
            x_aligned = np.array([x_raw[self._feature_list.index(f)]
                                   if f in self._feature_list else 0.0
                                   for f in feats])
            shap_vals = local_shap(bundle["model"], x_aligned, feats)
            raw_codes = top_reason_codes(shap_vals, self.reason_codes, feature_values, top_n=3)
            reason_codes_out = [ReasonCode(**rc) for rc in raw_codes]

        return CreditScoreResponse(
            wallet=wallet_address,
            timestamp=datetime.now(timezone.utc).isoformat(),
            score=score,
            pd_90d=round(ensemble_pd, 6),
            pd_30d=None,   # can be added with multi-horizon models
            pd_60d=None,
            risk_grade=grade,
            top_reason_codes=reason_codes_out,
            feature_summary={k: round(float(v), 4) for k, v in list(feature_values.items())[:20]},
            model_version=self.VERSION,
            calibration=self._cal_method,
        )
