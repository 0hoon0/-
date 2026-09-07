import numpy as np
import pandas as pd

from audit_anomaly.outliers import detect_outliers_3sigma


def test_flags_obvious_outlier_global_mode():
    rng = np.random.default_rng(0)
    values = list(rng.normal(loc=8.0, scale=0.1, size=19))
    values.insert(10, 0.5)  # a single, clearly anomalous value
    series = pd.Series(values, index=pd.RangeIndex(len(values)))

    result = detect_outliers_3sigma(series, sigma=3.0, mode="global")

    assert result.is_outlier.iloc[10]
    assert result.is_outlier.sum() == 1


def test_stable_series_has_no_outliers():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(loc=1.8, scale=0.05, size=100))

    result = detect_outliers_3sigma(series, sigma=3.0, mode="global")

    assert result.is_outlier.sum() == 0


def test_rolling_mode_runs_and_returns_bounds():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(loc=10, scale=1.0, size=24))

    result = detect_outliers_3sigma(series, sigma=3.0, mode="rolling", window=6)

    assert len(result.lower_bound) == len(series)
    assert len(result.upper_bound) == len(series)
    assert result.mode == "rolling"
