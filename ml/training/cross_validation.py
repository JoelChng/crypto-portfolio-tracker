"""
Walk-forward cross-validation with mandatory embargo gap.

Structure:
  |--- train ---|-- embargo --|--- test ---|
                 ^ 90d gap

No data point whose label window overlaps the test snapshot period is
allowed in the training fold.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)


class WalkForwardSplitter(BaseCrossValidator):
    """
    Time-series walk-forward splitter.

    Parameters
    ----------
    n_splits        : number of test folds
    test_months     : size of each test fold in months
    embargo_days    : gap between train end and test start (default 90d)
    min_train_months: minimum training data before first fold
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_months: int = 2,
        embargo_days: int = 90,
        min_train_months: int = 6,
    ):
        self.n_splits = n_splits
        self.test_months = test_months
        self.embargo_days = embargo_days
        self.min_train_months = min_train_months

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        X must have a 'snapshot_date' column or be a DatetimeIndex.
        Returns (train_indices, test_indices) pairs.
        """
        if isinstance(X, pd.DataFrame) and "snapshot_date" in X.columns:
            dates = pd.to_datetime(X["snapshot_date"])
        elif isinstance(X, pd.DatetimeIndex):
            dates = X
        else:
            raise ValueError("X must be a DataFrame with 'snapshot_date' column or DatetimeIndex")

        dates = dates.reset_index(drop=True)
        min_date = dates.min()
        max_date = dates.max()
        total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)

        required_months = self.min_train_months + self.n_splits * self.test_months + self.embargo_days // 30
        if total_months < required_months:
            logger.warning(
                f"Dataset spans {total_months} months but {required_months} required. "
                f"Reducing n_splits."
            )

        # Build fold boundaries
        folds_generated = 0
        # Step backwards from max_date to find test windows
        test_end = max_date
        fold_boundaries = []
        for _ in range(self.n_splits):
            test_start = test_end - pd.DateOffset(months=self.test_months)
            train_end  = test_start - pd.Timedelta(days=self.embargo_days)
            if (train_end - min_date).days < self.min_train_months * 30:
                break
            fold_boundaries.append((train_end, test_start, test_end))
            test_end = test_start - pd.Timedelta(days=1)

        fold_boundaries = list(reversed(fold_boundaries))
        logger.info(f"Generated {len(fold_boundaries)} walk-forward folds")

        for train_end, test_start, test_end in fold_boundaries:
            train_idx = np.where(dates <= train_end)[0]
            test_idx  = np.where((dates >= test_start) & (dates <= test_end))[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            # Verify no overlap
            max_train_date = dates.iloc[train_idx].max()
            min_test_date  = dates.iloc[test_idx].min()
            gap = (min_test_date - max_train_date).days
            assert gap >= self.embargo_days, (
                f"Embargo violation: gap={gap}d < {self.embargo_days}d"
            )
            logger.debug(f"  Fold: train_end={train_end.date()} test={test_start.date()}→{test_end.date()} "
                         f"gap={gap}d train_n={len(train_idx)} test_n={len(test_idx)}")
            folds_generated += 1
            yield train_idx, test_idx

        logger.info(f"Walk-forward CV: {folds_generated} valid folds generated")

    def _iter_test_masks(self, X=None, y=None, groups=None):
        for train_idx, test_idx in self.split(X, y, groups):
            mask = np.zeros(len(X), dtype=bool)
            mask[test_idx] = True
            yield mask


def make_oot_split(
    df: pd.DataFrame,
    test_months: int = 6,
    date_col: str = "snapshot_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple OOT (out-of-time) split: last `test_months` months held out.
    """
    dates = pd.to_datetime(df[date_col])
    cutoff = dates.max() - pd.DateOffset(months=test_months)
    train = df[dates <= cutoff].copy()
    test  = df[dates > cutoff].copy()
    logger.info(f"OOT split: train={len(train):,} test={len(test):,} cutoff={cutoff.date()}")
    return train, test
