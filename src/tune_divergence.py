"""Sweep the implied-vs-realized divergence signal's lookback/thresholds on
2021-2024 IN-SAMPLE data, compare against the pure-breakout baseline, then
validate the best config once, unmodified, on the 2025-2026 holdout.

default_mode="breakout" means: outside the low-ratio "fade" regime, this
reduces to the already-validated breakout strategy. So this sweep is really
asking "does fading during a trailing-realized/implied-undershoot regime add
anything on top of the breakout baseline" -- not re-deriving breakout itself.
"""
from __future__ import annotations

import pandas as pd

from backtest import compute_metrics, load_sessions_and_vix, run_backtest, run_backtest_switching

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-12-31"

LOOKBACK_GRID = [5, 10, 20, 40]
LOW_THRESH_GRID = [0.6, 0.7, 0.8, 0.9]
HIGH_THRESH_GRID = [1.1, 1.2, 1.3, 1.4]


def main() -> None:
    sessions, vix_series = load_sessions_and_vix()

    baseline = run_backtest(start=IN_SAMPLE_START, end=IN_SAMPLE_END, target_r=1.0, stop_r=1.0,
                             vix_cap=None, mode="breakout", sizing="fixed_1_nq",
                             sessions=sessions, vix_series=vix_series, save=False)
    bm = compute_metrics(baseline)
    print(f"Pure breakout baseline, {IN_SAMPLE_START}->{IN_SAMPLE_END}: "
          f"net=${bm['total_net']:,.0f}  max_dd=${bm['max_dd']:,.0f}  sharpe={bm['sharpe']:.2f}  trades={bm['n']}")
    print()

    rows = []
    for lookback in LOOKBACK_GRID:
        for low_t in LOW_THRESH_GRID:
            for high_t in HIGH_THRESH_GRID:
                df = run_backtest_switching(
                    start=IN_SAMPLE_START, end=IN_SAMPLE_END, lookback=lookback,
                    low_thresh=low_t, high_thresh=high_t, default_mode="breakout",
                    target_r=1.0, stop_r=1.0, sizing="fixed_1_nq",
                    sessions=sessions, vix_series=vix_series, save=False,
                )
                if df.empty:
                    continue
                m = compute_metrics(df)
                n_fade = (df["mode"] == "fade").sum()
                rows.append(dict(lookback=lookback, low_thresh=low_t, high_thresh=high_t,
                                  trades=m["n"], n_fade=n_fade, total_net=m["total_net"],
                                  profit_factor=m["profit_factor"], max_dd=m["max_dd"], sharpe=m["sharpe"]))

    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"Divergence-switching sweep, 2021-2024 in-sample (top 15 by Sharpe):")
    print(result.head(15).to_string(index=False))

    best = result.iloc[0]
    print(f"\nBest by Sharpe: lookback={best['lookback']:.0f}, low_thresh={best['low_thresh']}, "
          f"high_thresh={best['high_thresh']}  (n_fade={best['n_fade']:.0f}/{best['trades']:.0f} trades)")

    print(f"\n### HOLDOUT check (no re-tuning), {HOLDOUT_START}->{HOLDOUT_END} ###")
    holdout = run_backtest_switching(
        start=HOLDOUT_START, end=HOLDOUT_END, lookback=int(best["lookback"]),
        low_thresh=best["low_thresh"], high_thresh=best["high_thresh"], default_mode="breakout",
        target_r=1.0, stop_r=1.0, sizing="fixed_1_nq",
        sessions=sessions, vix_series=vix_series, save=False,
    )
    hm = compute_metrics(holdout)
    n_fade_h = (holdout["mode"] == "fade").sum() if not holdout.empty else 0
    print(f"net=${hm['total_net']:,.0f}  max_dd=${hm['max_dd']:,.0f}  sharpe={hm['sharpe']:.2f}  "
          f"trades={hm['n']}  n_fade={n_fade_h}")

    baseline_holdout = run_backtest(start=HOLDOUT_START, end=HOLDOUT_END, target_r=1.0, stop_r=1.0,
                                     vix_cap=None, mode="breakout", sizing="fixed_1_nq",
                                     sessions=sessions, vix_series=vix_series, save=False)
    bhm = compute_metrics(baseline_holdout)
    print(f"(pure breakout baseline on same holdout: net=${bhm['total_net']:,.0f}  "
          f"max_dd=${bhm['max_dd']:,.0f}  sharpe={bhm['sharpe']:.2f})")


if __name__ == "__main__":
    main()
