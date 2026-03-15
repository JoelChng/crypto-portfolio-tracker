"""Maps calibrated PD to score 0-1000 and letter grade A-E."""

import numpy as np


GRADE_THRESHOLDS = {
    "A": 0.02,
    "B": 0.05,
    "C": 0.12,
    "D": 0.25,
    "E": 1.00,
}


def pd_to_score(pd_value: float) -> int:
    """Monotonic transform: higher PD → lower score."""
    pd_value = float(np.clip(pd_value, 0.0, 1.0))
    score = 1000 - 900 * pd_value
    return int(np.clip(round(score), 0, 1000))


def pd_to_grade(pd_value: float, thresholds: dict = None) -> str:
    thresholds = thresholds or GRADE_THRESHOLDS
    for grade, upper in thresholds.items():
        if pd_value <= upper:
            return grade
    return "E"


def score_to_grade(score: int) -> str:
    """Convenience: derive grade from score (A=800-1000, etc.)."""
    if score >= 800: return "A"
    if score >= 700: return "B"
    if score >= 600: return "C"
    if score >= 500: return "D"
    return "E"
