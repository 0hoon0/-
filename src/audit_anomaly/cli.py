"""End-to-end demo: run the audit risk-screening procedures on the sample
data and write result charts to output/.

Usage:
    python -m audit_anomaly.cli
"""

from __future__ import annotations

import os

import pandas as pd

from audit_anomaly.benford import analyze_benford
from audit_anomaly.data_gen import generate_financial_ratios, generate_journal_entries
from audit_anomaly.outliers import detect_outliers_3sigma
from audit_anomaly.visualize import plot_benford, plot_outliers

OUTPUT_DIR = "output"
DATA_DIR = "data"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    journal_path = os.path.join(DATA_DIR, "sample_journal_entries.csv")
    ratios_path = os.path.join(DATA_DIR, "sample_financial_ratios.csv")

    if os.path.exists(journal_path):
        journal = pd.read_csv(journal_path, parse_dates=["date"])
    else:
        journal = generate_journal_entries()
        journal.to_csv(journal_path, index=False)

    if os.path.exists(ratios_path):
        ratios = pd.read_csv(ratios_path)
    else:
        ratios = generate_financial_ratios()
        ratios.to_csv(ratios_path, index=False)

    print("=" * 70)
    print("1) Benford's Law screening of journal entry amounts")
    print("=" * 70)
    first_digit_result = analyze_benford(journal["amount"], digit_test="first")
    print(first_digit_result.summary())
    plot_benford(first_digit_result, save_path=os.path.join(OUTPUT_DIR, "benford_first_digit.png"))

    first_two_result = analyze_benford(journal["amount"], digit_test="first_two")
    print()
    print(first_two_result.summary())
    plot_benford(first_two_result, save_path=os.path.join(OUTPUT_DIR, "benford_first_two_digits.png"))

    print()
    print("=" * 70)
    print("2) 3-sigma outlier screening of financial ratio trends")
    print("=" * 70)
    # Chart labels are kept in English since the default matplotlib fonts
    # do not ship Hangul glyphs; Korean names are used in the printed
    # summary and in the README instead.
    ratio_labels = {
        "receivable_turnover": ("Receivable Turnover", "매출채권회전율"),
        "current_ratio": ("Current Ratio", "유동비율"),
        "gross_margin": ("Gross Margin", "매출총이익률"),
    }
    ratios_indexed = ratios.set_index("period")
    for column, (label_en, label_ko) in ratio_labels.items():
        result = detect_outliers_3sigma(ratios_indexed[column], sigma=3.0, mode="global")
        print(f"[{label_ko} / {label_en}]")
        print(result.summary())
        print()
        plot_outliers(
            result,
            ylabel=label_en,
            save_path=os.path.join(OUTPUT_DIR, f"outliers_{column}.png"),
        )

    print(f"Charts written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
