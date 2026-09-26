"""Sweep target_r for the full VIX-target + runner ORB strategy, restricted
to sessions where RTH opened ABOVE the prior overnight-range midpoint (the
directional filter validated in orb_overnight_filter.py). Checks whether
stacking the filter with the already-validated VIX-target/runner mechanism
compounds the improvement, and whether the best target_r shifts once the
lower-quality "below" trades are removed. trail_r held at 0.25 (already
established best) to keep this a focused one-variable sweep.

2021-2024 IN-SAMPLE ONLY.
"""
from __future__ import annotations

import pandas as pd

from orb_vix_runner import fetch_mnq_5m, simulate, metrics
from orb_overnight_filter import compute_overnight_midpoint_side
from backtest import prev_vix_close_lookup
from fetch_data import fetch_vix_daily

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"

TARGET_R_GRID = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
TRAIL_R = 0.25


def main() -> None:
    bars = fetch_mnq_5m(IN_SAMPLE_START, IN_SAMPLE_END)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)

    print("Computing overnight midpoint side per session...", flush=True)
    side = compute_overnight_midpoint_side(bars)

    rows = []
    for target_r in TARGET_R_GRID:
        trades = simulate(bars, vix_series, sizing="fixed_5_micros", use_vix_target=True,
                           use_runner=True, target_r=target_r, trail_r=TRAIL_R)
        trades["session_date_only"] = trades["session_date"].dt.date
        trades["overnight_side"] = trades["session_date_only"].map(side)
        above_only = trades[trades["overnight_side"] == "above"]

        m_all = metrics(trades)
        m_above = metrics(above_only)
        if m_above.get("trades", 0) == 0:
            continue
        rows.append(dict(
            target_r=target_r,
            all_trades=m_all["trades"], all_sharpe=m_all["sharpe"], all_net=m_all["net_pnl"],
            above_trades=m_above["trades"], above_win_rate=m_above["win_rate_pct"],
            above_net=m_above["net_pnl"], above_pf=m_above["profit_factor"],
            above_max_dd=m_above["max_drawdown"], above_sharpe=m_above["sharpe"],
        ))

    result = pd.DataFrame(rows).sort_values("above_sharpe", ascending=False)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"\nVIX-target+runner sweep, ABOVE-midpoint-only, {IN_SAMPLE_START} -> {IN_SAMPLE_END} (trail_r={TRAIL_R})")
    print(result.to_string(index=False))
    print(f"\n(reference) unfiltered VIX-target+runner best: target_r=0.75 -> Sharpe 0.99, net $42,587")
    print(f"(reference) plain-ORB above-midpoint-only (no VIX/runner): Sharpe 0.85, net $24,646")


if __name__ == "__main__":
    main()
