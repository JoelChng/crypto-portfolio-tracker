"""Probability calibration — Platt scaling and isotonic regression."""

import numpy as np
import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Sigmoid (Platt) scaling on held-out fold probabilities."""

    def __init__(self):
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        probs = np.asarray(probs).reshape(-1, 1)
        self._lr.fit(probs, y_true)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs).reshape(-1, 1)
        return self._lr.predict_proba(probs)[:, 1]

    def save(self, path: str):
        joblib.dump(self._lr, path)

    def load(self, path: str):
        self._lr = joblib.load(path)
        return self


class IsotonicCalibrator:
    """Isotonic regression calibration."""

    def __init__(self):
        self._ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        self._ir.fit(np.asarray(probs), np.asarray(y_true))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self._ir.predict(np.asarray(probs))

    def save(self, path: str):
        joblib.dump(self._ir, path)

    def load(self, path: str):
        self._ir = joblib.load(path)
        return self


def fit_calibrator(probs_train: np.ndarray, y_train: np.ndarray, method: str = "isotonic"):
    """
    Fit a calibrator on held-out fold outputs.
    method: 'platt' | 'isotonic'
    """
    if method == "platt":
        cal = PlattCalibrator().fit(probs_train, y_train)
    else:
        cal = IsotonicCalibrator().fit(probs_train, y_train)
    return cal


def compare_calibration(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
) -> dict:
    """Compare ECE before and after calibration."""
    from training.metrics import calibration_error
    ece_raw = calibration_error(y_true, raw_probs)
    ece_cal = calibration_error(y_true, cal_probs)
    return {"ece_raw": ece_raw, "ece_calibrated": ece_cal, "improvement": ece_raw - ece_cal}
