"""Statistical outlier screening for financial ratio trends.

Implements the classic control-chart approach (mean +/- k*sigma) used in
analytical review procedures (ISA 520) to flag periods where a financial
ratio (e.g. 매출채권회전율, 유동비율, 매출총이익률) moves outside the range
that would be expected from normal business fluctuation, and therefore
warrants further audit inquiry.

Two modes are supported:
  * "global"  - mean/std computed over the whole series (best for a
                 short or stable series where a single baseline applies)
  * "rolling" - a rolling mean/std window, which adapts the baseline as
                 the series trends over time (better for longer series
                 with seasonality or gradual drift)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OutlierResult:
    series: pd.Series
    mean: pd.Series
    std: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    z_score: pd.Series
    is_outlier: pd.Series
    sigma: float
    mode: str

    @property
    def outliers(self) -> pd.Series:
        return self.series[self.is_outlier]

    def summary(self) -> str:
        n_out = int(self.is_outlier.sum())
        lines = [
            f"Outlier screening ({self.mode}, {self.sigma}-sigma): "
            f"{n_out} of {len(self.series)} observations flagged.",
        ]
        for idx, val in self.outliers.items():
            lines.append(f"  - {idx}: value={val:.4f}  z={self.z_score[idx]:.2f}")
        return "\n".join(lines)


def detect_outliers_3sigma(
    series: pd.Series,
    sigma: float = 3.0,
    mode: str = "global",
    window: int = 6,
    min_periods: int = 3,
) -> OutlierResult:
    """Flag values in ``series`` that fall outside a mean +/- sigma*std band.

    Parameters
    ----------
    series:
        A time-indexed series of a single financial ratio.
    sigma:
        Number of standard deviations defining the acceptable band
        (3.0 is the conventional statistical control-chart threshold).
    mode:
        "global" uses one mean/std for the whole series; "rolling" uses a
        rolling window so the baseline adapts over time.
    window, min_periods:
        Only used when mode="rolling"; passed to pandas.Series.rolling.
    """
    series = series.astype(float)

    if mode == "global":
        mean_val = series.mean()
        std_val = series.std(ddof=0)
        mean = pd.Series(mean_val, index=series.index)
        std = pd.Series(std_val, index=series.index)
    elif mode == "rolling":
        mean = series.rolling(window=window, min_periods=min_periods, center=True).mean()
        std = series.rolling(window=window, min_periods=min_periods, center=True).std(ddof=0)
        mean = mean.fillna(series.mean())
        std = std.fillna(series.std(ddof=0))
    else:
        raise ValueError("mode must be 'global' or 'rolling'")

    std_safe = std.replace(0, np.nan)
    z_score = (series - mean) / std_safe
    z_score = z_score.fillna(0.0)

    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    is_outlier = (series < lower_bound) | (series > upper_bound)

    return OutlierResult(
        series=series,
        mean=mean,
        std=std,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        z_score=z_score,
        is_outlier=is_outlier,
        sigma=sigma,
        mode=mode,
    )
