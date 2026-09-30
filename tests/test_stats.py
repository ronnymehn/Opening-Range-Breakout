import math

import numpy as np
import pandas as pd
import pytest

import orb_stats as st


def test_psr_at_threshold_is_half():
    assert st.probabilistic_sharpe(0.05, 1000, 0.0, 3.0, sr0=0.05) == pytest.approx(0.5)


def test_dsr_with_one_trial_equals_psr_vs_zero():
    assert st.deflated_sharpe(0.05, 1000, 0.1, 4.0, 1, 0.001) == pytest.approx(
        st.probabilistic_sharpe(0.05, 1000, 0.1, 4.0))


def test_dsr_falls_as_trials_grow():
    values = [st.deflated_sharpe(0.05, 1000, 0.0, 3.0, n, 0.0004) for n in (2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_t_stat_closed_form():
    x = np.array([1.0, -1.0, 2.0, 0.0, 3.0])
    ann, t = st.sharpe_t_stat(x)
    assert t == pytest.approx(x.mean() / x.std(ddof=1) * math.sqrt(5))
    assert ann == pytest.approx(x.mean() / x.std(ddof=1) * math.sqrt(252))


def test_moments_of_normal_sample():
    skew, kurt = st.moments(np.random.default_rng(0).normal(size=200_000))
    assert abs(skew) < 0.02 and kurt == pytest.approx(3.0, abs=0.05)


@pytest.mark.parametrize("block", [1.0, 5.0])
def test_bootstrap_ci_contains_estimate_and_is_reproducible(block):
    x = np.random.default_rng(1).normal(0.1, 1.0, 500)
    lo, hi = st.bootstrap_ci(x, stat="mean", n_boot=2_000, mean_block=block, seed=4)
    assert lo < x.mean() < hi
    assert (lo, hi) == st.bootstrap_ci(x, stat="mean", n_boot=2_000, mean_block=block, seed=4)


def test_stationary_indices_form_contiguous_blocks():
    idx = st.stationary_bootstrap_indices(100, 50, 10.0, np.random.default_rng(2))
    steps = np.diff(idx, axis=1) % 100
    assert idx.min() >= 0 and idx.max() < 100
    assert (steps == 1).mean() > 0.8  # mostly consecutive days


def test_pbo_near_half_for_pure_noise():
    m = np.random.default_rng(5).normal(size=(1600, 20))
    assert 0.3 <= st.pbo_cscv(m) <= 0.7


def test_pbo_near_zero_for_dominant_strategy():
    m = np.random.default_rng(6).normal(size=(1600, 20))
    m[:, 3] += 1.0
    assert st.pbo_cscv(m) < 0.05


def test_daily_pnl_fills_flat_sessions():
    sessions = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    trades = pd.DataFrame({"session_date": pd.to_datetime(["2024-01-03"]), "net_pnl": [50.0]})
    assert list(st.daily_pnl(trades, sessions)) == [0.0, 50.0, 0.0]


def test_significance_table_columns():
    rng = np.random.default_rng(7)
    daily = {f"c{i}": pd.Series(rng.normal(0.05, 1, 300)) for i in range(3)}
    table = st.significance_table(daily, trial_counts=(1, 3), n_boot=500)
    assert {"t_stat", "psr_vs_0", "dsr_N1", "dsr_N3", "sharpe_ci_iid"} <= set(table.columns)
    assert (table["dsr_N3"] <= table["dsr_N1"]).all()
