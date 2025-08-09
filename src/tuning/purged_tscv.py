"""Purged/Embargoed Time Series Cross-Validation splitter.

Motivation:
- Our targets look ahead up to 72 hours. Standard TimeSeriesSplit can leak future
  information through labels. This splitter enforces an "embargo" gap between
  the end of the training set and the start of the test set to mitigate leakage.

Usage:
    splitter = PurgedTimeSeriesSplit(n_splits=5, embargo_hours=72)
    for train_idx, test_idx in splitter.split(X, timestamps=ts):
        ...

Notes:
- Provide timestamps aligned to X rows for accurate conversion of embargo_hours
  to sample counts. If timestamps are not provided, embargo is treated as 0.
- The splitter keeps temporal order and uses contiguous folds similar to
  sklearn.model_selection.TimeSeriesSplit but adds a purge gap.
"""

from __future__ import annotations

from typing import Generator, Iterable, Optional, Tuple
import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, embargo_hours: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = int(n_splits)
        self.embargo_hours = int(embargo_hours)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _infer_samples_per_hour(self, timestamps: pd.Series) -> float:
        """Infer samples per hour from median delta of timestamps."""
        ts = pd.to_datetime(timestamps)
        if len(ts) < 2:
            return 1.0
        deltas = ts.diff().dropna().dt.total_seconds()
        if deltas.empty:
            return 1.0
        median_seconds = np.median(deltas.values)
        if median_seconds <= 0:
            return 1.0
        samples_per_hour = 3600.0 / median_seconds
        return max(samples_per_hour, 1e-6)

    def _embargo_samples(self, timestamps: Optional[Iterable]) -> int:
        if self.embargo_hours <= 0 or timestamps is None:
            return 0
        ts = pd.to_datetime(pd.Series(timestamps))
        sph = self._infer_samples_per_hour(ts)
        embargo = int(round(self.embargo_hours * sph))
        return max(embargo, 0)

    def split(
        self,
        X: Iterable,
        y: Optional[Iterable] = None,
        groups: Optional[Iterable] = None,
        timestamps: Optional[Iterable] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        X = np.asarray(X)
        n_samples = X.shape[0]
        if self.n_splits + 1 > n_samples:
            raise ValueError(
                f"Not enough samples ({n_samples}) for n_splits={self.n_splits}"
            )

        embargo = self._embargo_samples(timestamps)

        # Build splits similar to sklearn TimeSeriesSplit: increasing folds.
        test_size = n_samples // (self.n_splits + 1)
        if test_size == 0:
            test_size = 1  # minimal

        indices = np.arange(n_samples)
        for i in range(1, self.n_splits + 1):
            test_start = i * test_size
            test_end = test_start + test_size
            if i == self.n_splits:
                # last fold takes the remainder
                test_end = n_samples

            # training is all before (test_start - embargo)
            train_end = max(test_start - embargo, 0)
            if train_end <= 0:
                # nothing to train on, skip this fold
                continue

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            # ensure non-empty
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx
