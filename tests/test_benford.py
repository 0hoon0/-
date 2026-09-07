import numpy as np
import pandas as pd
import pytest

from audit_anomaly.benford import analyze_benford


def test_benford_conforms_for_lognormal_data():
    rng = np.random.default_rng(0)
    amounts = pd.Series(rng.lognormal(mean=10, sigma=2, size=20000))

    result = analyze_benford(amounts, digit_test="first")

    assert result.n == 20000
    assert result.mad < 0.015
    assert result.chi_square_p_value > 0.01


def test_benford_flags_round_number_manipulation():
    manipulated = pd.Series([5_000_000, 5_000_000, 5_000_000, 5_000_000] * 50)

    result = analyze_benford(manipulated, digit_test="first")

    assert 5 in result.flagged_digits
    assert result.conformity == "Nonconformity (investigate further)"


def test_analyze_benford_rejects_empty_input():
    with pytest.raises(ValueError):
        analyze_benford(pd.Series([0, 0, 0], dtype=float))


def test_first_two_digit_test_runs():
    rng = np.random.default_rng(1)
    amounts = pd.Series(rng.lognormal(mean=10, sigma=2, size=5000))

    result = analyze_benford(amounts, digit_test="first_two")

    assert len(result.expected_freq) == 90
    assert result.digit_test == "first_two"
