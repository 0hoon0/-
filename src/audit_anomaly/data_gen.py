"""Synthetic sample data generator.

No real audit client data is used in this project. This module generates
a realistic-looking journal ledger and a financial-ratio time series with
a small number of deliberately injected anomalies, so the detection
methods in ``benford.py`` and ``outliers.py`` have something to catch and
so the results are reproducible for anyone running this repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ACCOUNTS = ["매출", "매출원가", "매입", "판매비와관리비", "미수금", "미지급금"]
_VENDORS = [f"거래처-{i:03d}" for i in range(1, 41)]
_PREPARERS = ["김회계", "이감사", "박전표", "최담당"]


def generate_journal_entries(
    n_normal: int = 4000,
    n_manipulated: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic journal ledger (분개장).

    ``n_normal`` entries are drawn from a log-normal distribution, which
    naturally conforms to Benford's Law (as most real transaction amounts
    do). ``n_manipulated`` entries are generated to mimic fabricated or
    plugged figures: round numbers and amounts clustered just under a
    common approval threshold, which distorts the leading-digit
    distribution and should be flagged by the Benford's Law test.
    """
    rng = np.random.default_rng(seed)

    normal_amounts = rng.lognormal(mean=13.5, sigma=1.6, size=n_normal)

    round_amounts = rng.choice(
        [1_000_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000], size=n_manipulated // 2
    ).astype(float)
    threshold_amounts = 9_999_000 - rng.integers(0, 5000, size=n_manipulated - len(round_amounts))
    manipulated_amounts = np.concatenate([round_amounts, threshold_amounts]).astype(float)

    amounts = np.concatenate([normal_amounts, manipulated_amounts])
    is_manipulated = np.concatenate(
        [np.zeros(n_normal, dtype=bool), np.ones(len(manipulated_amounts), dtype=bool)]
    )

    n_total = len(amounts)
    order = rng.permutation(n_total)
    amounts = amounts[order]
    is_manipulated = is_manipulated[order]

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=n_total), unit="D"
    )

    df = pd.DataFrame({
        "entry_id": [f"JE-{i:06d}" for i in range(1, n_total + 1)],
        "date": dates,
        "account": rng.choice(_ACCOUNTS, size=n_total),
        "vendor": rng.choice(_VENDORS, size=n_total),
        "amount": np.round(amounts).astype(int),
        "preparer": rng.choice(_PREPARERS, size=n_total),
        "is_manipulated": is_manipulated,  # ground truth label, for demo/validation only
    })
    return df.sort_values("date").reset_index(drop=True)


def generate_financial_ratios(
    n_months: int = 36,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic monthly financial-ratio time series with a few
    injected outliers (e.g. a receivables-turnover collapse consistent with
    channel stuffing, or a spike in the current ratio from a one-off
    financing event)."""
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2023-01", periods=n_months, freq="M")

    receivable_turnover = rng.normal(loc=8.0, scale=0.6, size=n_months)
    current_ratio = rng.normal(loc=1.8, scale=0.12, size=n_months)
    gross_margin = rng.normal(loc=0.32, scale=0.015, size=n_months)

    outlier_idx = rng.choice(n_months, size=3, replace=False)
    receivable_turnover[outlier_idx[0]] -= 4.5
    current_ratio[outlier_idx[1]] += 0.9
    gross_margin[outlier_idx[2]] -= 0.12

    df = pd.DataFrame({
        "period": periods.astype(str),
        "receivable_turnover": receivable_turnover,
        "current_ratio": current_ratio,
        "gross_margin": gross_margin,
    })
    return df


if __name__ == "__main__":
    generate_journal_entries().to_csv("data/sample_journal_entries.csv", index=False)
    generate_financial_ratios().to_csv("data/sample_financial_ratios.csv", index=False)
    print("Sample data written to data/")
