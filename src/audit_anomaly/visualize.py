"""Seaborn/matplotlib visualizations for the audit analytics results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from audit_anomaly.benford import BenfordResult
from audit_anomaly.outliers import OutlierResult

sns.set_theme(style="whitegrid", context="talk")


def plot_benford(result: BenfordResult, title: str | None = None, save_path: str | None = None):
    """Bar chart of observed vs. expected Benford digit frequencies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    digits = result.expected_freq.index
    x = range(len(digits))

    bars = ax.bar(x, result.observed_freq.values, width=0.6, color="#4C72B0",
                  label="Observed", zorder=2)
    ax.plot(x, result.expected_freq.values, color="#C44E52", marker="o",
            linewidth=2, label="Benford expected", zorder=3)

    for i in x:
        if digits[i] in result.flagged_digits:
            bars[i].set_color("#DD8452")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(d) for d in digits], rotation=0 if result.digit_test == "first" else 90)
    ax.set_xlabel("Leading digit" if result.digit_test == "first" else "Leading two digits")
    ax.set_ylabel("Relative frequency")
    ax.set_title(title or f"Benford's Law test ({result.digit_test}, n={result.n})\n"
                 f"chi2 p={result.chi_square_p_value:.4f}, MAD={result.mad:.5f} -> {result.conformity}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_outliers(result: OutlierResult, ylabel: str = "Ratio", title: str | None = None,
                   save_path: str | None = None):
    """Line chart of a financial ratio trend with a +/-sigma band and flagged outliers."""
    fig, ax = plt.subplots(figsize=(13, 6))

    x = result.series.index

    ax.fill_between(x, result.lower_bound, result.upper_bound,
                     color="#4C72B0", alpha=0.15, label=f"+/-{result.sigma}sigma band", zorder=1)
    ax.plot(x, result.series.values, color="#4C72B0", marker="o", linewidth=2,
             label=ylabel, zorder=2)

    outliers = result.outliers
    if len(outliers):
        ax.scatter(outliers.index, outliers.values, color="#C44E52", s=120,
                   zorder=3, label="Flagged outlier", edgecolor="black")

    ax.set_xlabel("Period")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{ylabel} trend with {result.sigma}-sigma outlier screening "
                 f"({result.mode})")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
