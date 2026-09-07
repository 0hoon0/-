"""Benford's Law conformity testing for journal entry / ledger amounts.

Benford's Law predicts the distribution of leading digits in naturally
occurring numerical data (transaction amounts, invoice totals, etc.).
Deviation from the expected distribution is a well-established red flag
in audit data analytics (AICPA AU-C 240 / ISA 240 fraud risk procedures,
ISA 315 risk-of-material-misstatement assessment) and is used here to
probabilistically screen a ledger for manipulated or fabricated figures.

Two digit tests are supported:
  * "first"      - leading digit (1-9), the classic Benford test
  * "first_two"  - leading two digits (10-99), a finer-grained test that
                    is more sensitive to small, deliberate manipulations
                    (e.g. amounts clustered just under an approval limit)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# Nigrini's Mean Absolute Deviation conformity thresholds.
# See Nigrini, M. (2012), "Benford's Law: Applications for Forensic
# Accounting, Auditing, and Fraud Detection".
_MAD_THRESHOLDS = {
    "first": [(0.006, "Close conformity"), (0.012, "Acceptable conformity"),
              (0.015, "Marginally acceptable conformity")],
    "first_two": [(0.0012, "Close conformity"), (0.0018, "Acceptable conformity"),
                  (0.0022, "Marginally acceptable conformity")],
}


@dataclass
class BenfordResult:
    digit_test: str
    observed_freq: pd.Series
    expected_freq: pd.Series
    counts: pd.Series
    n: int
    chi_square: float
    chi_square_p_value: float
    mad: float
    conformity: str
    z_scores: pd.Series = field(repr=False)
    flagged_digits: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Benford's Law test: {self.digit_test} digit(s), n={self.n}",
            f"Chi-square = {self.chi_square:.3f} (p = {self.chi_square_p_value:.4f})",
            f"MAD = {self.mad:.5f}  -> {self.conformity}",
        ]
        if self.flagged_digits:
            lines.append(f"Digits flagged by |z| > 1.96: {self.flagged_digits}")
        else:
            lines.append("No individual digits show significant deviation (|z| <= 1.96).")
        return "\n".join(lines)


def _leading_digits(amounts: pd.Series, digit_test: str) -> pd.Series:
    values = amounts.abs()
    values = values[values > 0]
    values = values[np.isfinite(values)]

    exponent = np.floor(np.log10(values))
    if digit_test == "first":
        leading = (values / (10 ** exponent)).astype(int)
    elif digit_test == "first_two":
        leading = (values / (10 ** (exponent - 1))).astype(int)
    else:
        raise ValueError("digit_test must be 'first' or 'first_two'")
    return leading


def _expected_distribution(digit_test: str) -> pd.Series:
    if digit_test == "first":
        digits = np.arange(1, 10)
    else:
        digits = np.arange(10, 100)
    expected = np.log10(1 + 1 / digits)
    return pd.Series(expected, index=digits)


def _mad_conformity(mad: float, digit_test: str) -> str:
    for threshold, label in _MAD_THRESHOLDS[digit_test]:
        if mad <= threshold:
            return label
    return "Nonconformity (investigate further)"


def analyze_benford(amounts: pd.Series, digit_test: str = "first") -> BenfordResult:
    """Run a Benford's Law conformity test on a series of amounts.

    Parameters
    ----------
    amounts:
        Raw transaction amounts (e.g. a journal entry / ledger 금액 column).
        Zeros, negatives and non-finite values are dropped before testing
        since Benford's Law applies to the magnitude of naturally occurring
        numbers.
    digit_test:
        "first" for the leading-digit test (1-9) or "first_two" for the
        leading-two-digit test (10-99).

    Returns
    -------
    BenfordResult with the observed/expected distributions, a chi-square
    goodness-of-fit test, Nigrini's MAD conformity metric and a per-digit
    z-score used to flag specific digits worth substantive testing.
    """
    digits = _leading_digits(amounts, digit_test)
    n = len(digits)
    if n == 0:
        raise ValueError("No valid positive amounts to test.")

    expected_freq = _expected_distribution(digit_test)
    counts = digits.value_counts().reindex(expected_freq.index, fill_value=0)
    observed_freq = counts / n

    expected_counts = expected_freq * n
    chi_square, p_value = stats.chisquare(f_obs=counts.values, f_exp=expected_counts.values)

    mad = float((observed_freq - expected_freq).abs().mean())
    conformity = _mad_conformity(mad, digit_test)

    # Per-digit z-score (proportion test), per Nigrini (2012).
    z_scores = (observed_freq - expected_freq).abs() / np.sqrt(
        expected_freq * (1 - expected_freq) / n
    )
    flagged_digits = z_scores[z_scores > 1.96].index.tolist()

    return BenfordResult(
        digit_test=digit_test,
        observed_freq=observed_freq,
        expected_freq=expected_freq,
        counts=counts,
        n=n,
        chi_square=float(chi_square),
        chi_square_p_value=float(p_value),
        mad=mad,
        conformity=conformity,
        z_scores=z_scores,
        flagged_digits=flagged_digits,
    )
