"""Sweep target_r x stop_r for the fade (mean-reversion) strategy on the
2021-2024 IN-SAMPLE period, to check whether the naive 1:1 fade's loss is a
sizing/target-calibration problem rather than "no edge at all". Only a config
that looks genuinely good here should go on to a 2025-2026 holdout check.
"""
from __future__ import annotations

import pandas as pd

from backtest import compute_metrics, load_sessions_and_vix, run_backtest

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"

TARGET_R_GRID = [0.25, 0.5, 0.75, 1.0]
STOP_R_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]


def main() -> None:
    sessions, vix_series = load_sessions_and_vix()

    rows = []
    for target_r in TARGET_R_GRID:
        for stop_r in STOP_R_GRID:
            df = run_backtest(
                start=IN_SAMPLE_START, end=IN_SAMPLE_END,
                target_r=target_r, stop_r=stop_r, vix_cap=None, mode="fade",
                sizing="fixed_1_nq", sessions=sessions, vix_series=vix_series, save=False,
            )
            if df.empty:
                continue
            m = compute_metrics(df)
            rows.append(dict(target_r=target_r, stop_r=stop_r, trades=m["n"], win_rate=m["win_rate"],
                              total_net=m["total_net"], profit_factor=m["profit_factor"],
                              max_dd=m["max_dd"], sharpe=m["sharpe"]))

    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"FADE strategy sweep, 2021-2024 in-sample, fixed 1 NQ contract")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
