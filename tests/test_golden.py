"""The refactored engine must reproduce the original scripts' trades on clean
synthetic data (full sessions, no gaps, no FOMC days). Fixtures in
tests/fixtures were produced by the pre-refactor run_reference_orb_sweep.py and
run_reference_orb_nq.py. Settings that intentionally differ from the old
behaviour (size_from, roll masking) are pinned to the old values here."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import orb_engine as eng

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NUMERIC = ["entry", "exit", "stop", "target", "qty", "net_pnl"]


@pytest.fixture(scope="module")
def bars():
    df = pd.read_csv(FIXTURES / "golden_bars.csv")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), utc=True)).tz_convert(eng.ET)
    return df


def _compare(expected: pd.DataFrame, actual: pd.DataFrame):
    assert len(actual) == len(expected)
    for column in NUMERIC:
        np.testing.assert_allclose(actual[column].to_numpy(float), expected[column].to_numpy(float),
                                   rtol=0, atol=1e-9, err_msg=column)
    assert list(actual["exit_reason"]) == list(expected["exit_reason"])
    assert [str(t) for t in actual["entry_time"]] == list(expected["entry_time"])
    assert [str(t) for t in actual["exit_time"]] == list(expected["exit_time"])


MNQ_CASES = {
    "fixed_0.5": dict(sizing="fixed", target_r=0.5),
    "fixed_1.0": dict(sizing="fixed", target_r=1.0),
    "fixed_2.0": dict(sizing="fixed", target_r=2.0),
    "fixed_1.4_2x": dict(sizing="fixed", target_r=1.4, slippage_ticks=2),
    "orb_risk_1.0": dict(sizing="orb_risk_target", target_r=1.0, size_from="fill"),
    "harvey_1.0": dict(sizing="harvey_target_vol", target_r=1.0),
    "moreira_1.0": dict(sizing="moreira_muir", target_r=1.0),
    "warmup_1.0": dict(sizing="fixed_matched_warmup", target_r=1.0),
}


@pytest.mark.parametrize("name", MNQ_CASES)
def test_mnq_matches_legacy(bars, name):
    expected = pd.read_csv(FIXTURES / "golden_mnq.csv").query("config == @name")
    cfg = eng.StrategyConfig(fixed_qty=5, risk_budget=1_000.0, **MNQ_CASES[name])
    vol = eng.realized_volatility(bars, 20, roll_dates=())
    actual = eng.simulate(bars, eng.MNQ, cfg, annual_vol=vol)
    _compare(expected, actual)
    np.testing.assert_allclose(actual["estimated_annual_vol"].to_numpy(float),
                               expected["estimated_annual_vol"].to_numpy(float), atol=1e-12)


NQ_CASES = {
    "fixed_1.0": dict(sizing="fixed", target_r=1.0),
    "fixed_2.0": dict(sizing="fixed", target_r=2.0),
    "equal_risk_1.0": dict(sizing="orb_risk_target", target_r=1.0, size_from="fill"),
}


@pytest.mark.parametrize("name", NQ_CASES)
def test_nq_matches_legacy(bars, name):
    expected = pd.read_csv(FIXTURES / "golden_nq.csv").query("config == @name")
    cfg = eng.StrategyConfig(fixed_qty=1, risk_budget=2_000.0, **NQ_CASES[name])
    _compare(expected, eng.simulate(bars, eng.NQ, cfg))
