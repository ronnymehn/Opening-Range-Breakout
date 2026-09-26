"""Sweep ib_minutes x target_r for the Initial Balance breakout strategy
(idea #5) on 2021-2024 IN-SAMPLE data, compare against the pure VIX-band
breakout baseline, then validate the best config once, unmodified, on the
2025-2026 holdout.
"""
from __future__ import annotations

import pandas as pd

from backtest import compute_metrics, load_sessions_and_vix, run_backtest, run_backtest_ib

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-12-31"

IB_MINUTES_GRID = [15, 30, 45, 60, 90]
TARGET_R_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]


def main() -> None:
    sessions, vix_series = load_sessions_and_vix()

    baseline = run_backtest(start=IN_SAMPLE_START, end=IN_SAMPLE_END, target_r=1.0, stop_r=1.0,
                             vix_cap=None, mode="breakout", sizing="fixed_1_nq",
                             sessions=sessions, vix_series=vix_series, save=False)
    bm = compute_metrics(baseline)
    print(f"Pure VIX-band breakout baseline, {IN_SAMPLE_START}->{IN_SAMPLE_END}: "
          f"net=${bm['total_net']:,.0f}  max_dd=${bm['max_dd']:,.0f}  sharpe={bm['sharpe']:.2f}  trades={bm['n']}")
    print()

    rows = []
    for ib_min in IB_MINUTES_GRID:
        for target_r in TARGET_R_GRID:
            df = run_backtest_ib(start=IN_SAMPLE_START, end=IN_SAMPLE_END, ib_minutes=ib_min,
                                  target_r=target_r, sizing="fixed_1_nq",
                                  sessions=sessions, vix_series=vix_series, save=False)
            if df.empty:
                continue
            m = compute_metrics(df)
            rows.append(dict(ib_minutes=ib_min, target_r=target_r, trades=m["n"], win_rate=m["win_rate"],
                              total_net=m["total_net"], profit_factor=m["profit_factor"],
                              max_dd=m["max_dd"], sharpe=m["sharpe"]))

    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("IB breakout sweep, 2021-2024 in-sample (top 15 by Sharpe):")
    print(result.head(15).to_string(index=False))

    best = result.iloc[0]
    print(f"\nBest by Sharpe: ib_minutes={best['ib_minutes']:.0f}, target_r={best['target_r']}")

    print(f"\n### HOLDOUT check (no re-tuning), {HOLDOUT_START}->{HOLDOUT_END} ###")
    holdout = run_backtest_ib(start=HOLDOUT_START, end=HOLDOUT_END, ib_minutes=int(best["ib_minutes"]),
                               target_r=best["target_r"], sizing="fixed_1_nq",
                               sessions=sessions, vix_series=vix_series, save=False)
    hm = compute_metrics(holdout)
    print(f"net=${hm['total_net']:,.0f}  max_dd=${hm['max_dd']:,.0f}  sharpe={hm['sharpe']:.2f}  "
          f"win_rate={hm['win_rate']:.1%}  trades={hm['n']}")

    baseline_holdout = run_backtest(start=HOLDOUT_START, end=HOLDOUT_END, target_r=1.0, stop_r=1.0,
                                     vix_cap=None, mode="breakout", sizing="fixed_1_nq",
                                     sessions=sessions, vix_series=vix_series, save=False)
    bhm = compute_metrics(baseline_holdout)
    print(f"(pure VIX-band breakout baseline on same holdout: net=${bhm['total_net']:,.0f}  "
          f"max_dd=${bhm['max_dd']:,.0f}  sharpe={bhm['sharpe']:.2f})")


if __name__ == "__main__":
    main()
