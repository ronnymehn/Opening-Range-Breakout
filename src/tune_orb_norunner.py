"""Stress test: sweep target_r (VIX-implied-move multiple) WITHOUT the
runner (single full exit at target/stop/time, no partial), to check whether
the earlier finding that target_r=0.75 beats target_r=1.0 is specific to the
runner mechanic, or holds even in a simpler no-runner setup -- comparable to
what a different backtest (e.g. run elsewhere) might have tested if it
didn't implement partial-exit/trailing logic. 2021-2024 IN-SAMPLE ONLY,
2025-2026 not touched.
"""
from __future__ import annotations

import pandas as pd

from orb_vix_runner import fetch_mnq_5m, simulate, metrics, realized_volatility, VOL_LOOKBACK_SESSIONS
from backtest import prev_vix_close_lookup
from fetch_data import fetch_vix_daily

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"

TARGET_R_GRID = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0]


def main() -> None:
    bars = fetch_mnq_5m(IN_SAMPLE_START, IN_SAMPLE_END)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)
    annual_vol = realized_volatility(bars, VOL_LOOKBACK_SESSIONS)

    rows = []
    for target_r in TARGET_R_GRID:
        trades = simulate(bars, vix_series, sizing="fixed_5_micros", annual_vol=annual_vol,
                           target_r=target_r, use_vix_target=True, use_runner=False)
        m = metrics(trades)
        if m.get("trades", 0) == 0:
            continue
        resolved = m["target_or_trail_exits"] + m["stop_exits"]
        rows.append(dict(
            target_r=target_r, trades=m["trades"], win_rate=m["win_rate_pct"],
            net_pnl=m["net_pnl"], profit_factor=m["profit_factor"],
            max_dd=m["max_drawdown"], sharpe=m["sharpe"],
            target_exits=m["target_or_trail_exits"], stop_exits=m["stop_exits"],
            time_exits=m["time_exits"], resolved_pct=resolved / m["trades"] * 100,
        ))

    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"NO-RUNNER target_r sweep (single full exit), {IN_SAMPLE_START} -> {IN_SAMPLE_END}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
