"""Sweep target_r x trail_r for the ORB + VIX-target + runner strategy on
2021-2024 IN-SAMPLE data only. 2025-2026 stays untouched per project
discipline -- do not add a holdout call here.

Prior finding at the defaults (target_r=1.0, trail_r=0.5): only 19/610
trades (3%) ever reached the VIX-sized target, so the runner mechanic barely
ever activated. This sweep checks whether a smaller target_r makes the
target reachable often enough to give the runner a fair test, and whether
that actually helps performance.
"""
from __future__ import annotations

import pandas as pd

from orb_vix_runner import fetch_mnq_5m, simulate, metrics, realized_volatility, VOL_LOOKBACK_SESSIONS
from backtest import prev_vix_close_lookup
from fetch_data import fetch_vix_daily

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"

TARGET_R_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
TRAIL_R_GRID = [0.25, 0.5]


def main() -> None:
    bars = fetch_mnq_5m(IN_SAMPLE_START, IN_SAMPLE_END)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)
    annual_vol = realized_volatility(bars, VOL_LOOKBACK_SESSIONS)

    rows = []
    for target_r in TARGET_R_GRID:
        for trail_r in TRAIL_R_GRID:
            trades = simulate(bars, vix_series, sizing="fixed_5_micros", annual_vol=annual_vol,
                               target_r=target_r, trail_r=trail_r)
            m = metrics(trades)
            if m.get("trades", 0) == 0:
                continue
            resolved = m["target_or_trail_exits"] + m["stop_exits"]
            rows.append(dict(
                target_r=target_r, trail_r=trail_r, trades=m["trades"],
                win_rate=m["win_rate_pct"], net_pnl=m["net_pnl"], profit_factor=m["profit_factor"],
                max_dd=m["max_drawdown"], sharpe=m["sharpe"],
                target_trail_exits=m["target_or_trail_exits"], stop_exits=m["stop_exits"],
                time_exits=m["time_exits"], resolved_pct=resolved / m["trades"] * 100,
            ))

    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"ORB+VIX-target+runner sweep, {IN_SAMPLE_START} -> {IN_SAMPLE_END} (2025-2026 NOT touched)")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
