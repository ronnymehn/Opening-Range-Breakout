"""Holdout check (no re-tuning): the fixed best in-sample config
(target_r=0.75, trail_r=0.25) for VIX-target+runner, both unfiltered and
combined with the above-midpoint overnight filter, run on a given period
(default 2025-2026). Reuses the exact same fixed parameters chosen on
2021-2024 -- nothing is re-swept here.
"""
from __future__ import annotations

import pandas as pd

from orb_vix_runner import fetch_mnq_5m, simulate, metrics
from orb_overnight_filter import compute_overnight_midpoint_side
from backtest import prev_vix_close_lookup
from fetch_data import fetch_vix_daily

TARGET_R = 0.75
TRAIL_R = 0.25


def main(start: str, end: str) -> None:
    bars = fetch_mnq_5m(start, end)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)

    print("Computing overnight midpoint side per session...", flush=True)
    side = compute_overnight_midpoint_side(bars)

    trades = simulate(bars, vix_series, sizing="fixed_5_micros", use_vix_target=True,
                       use_runner=True, target_r=TARGET_R, trail_r=TRAIL_R)
    trades["session_date_only"] = trades["session_date"].dt.date
    trades["overnight_side"] = trades["session_date_only"].map(side)
    above_only = trades[trades["overnight_side"] == "above"]
    below_only = trades[trades["overnight_side"] == "below"]

    for label, sub in [("VIX-target+runner, UNFILTERED", trades),
                        ("VIX-target+runner + above-midpoint filter", above_only),
                        ("VIX-target+runner, below-midpoint only (placebo)", below_only)]:
        m = metrics(sub)
        print(f"--- {label} ({start} -> {end}) ---")
        if m.get("trades", 0) == 0:
            print("  no trades\n")
            continue
        print(f"  trades: {m['trades']}")
        print(f"  win_rate: {m['win_rate_pct']:.1f}%")
        print(f"  net_pnl: ${m['net_pnl']:,.0f}")
        print(f"  profit_factor: {m['profit_factor']:.2f}")
        print(f"  max_dd: ${m['max_drawdown']:,.0f}")
        print(f"  sharpe: {m['sharpe']:.2f}")
        print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-12-31")
    args = p.parse_args()
    main(args.start, args.end)
