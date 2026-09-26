"""Run the ORB strategy with an ATR-sized target/trail instead of VIX-sized,
at a matched reachability level (atr_k=0.5 corresponds to ~9-10% reach rate,
same ballpark as target_r=0.75 * em_points -- see target_achievability.py),
to test realized vol (ATR) vs implied vol (VIX) as the sizing input.
2021-2024 IN-SAMPLE ONLY.
"""
from __future__ import annotations

import pandas as pd

from orb_vix_runner import fetch_mnq_5m, simulate, metrics
from target_achievability import fetch_mnq_daily_atr
from backtest import prev_vix_close_lookup
from fetch_data import fetch_vix_daily

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"

ATR_K_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
TRAIL_R_GRID = [0.25, 0.5]


def main() -> None:
    bars = fetch_mnq_5m(IN_SAMPLE_START, IN_SAMPLE_END)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)
    atr = fetch_mnq_daily_atr(IN_SAMPLE_START, IN_SAMPLE_END)
    atr_by_date = {ts.date(): v for ts, v in atr.items()}
    atr_series = pd.Series(atr_by_date)

    rows = []
    for atr_k in ATR_K_GRID:
        for trail_r in TRAIL_R_GRID:
            trades = simulate(bars, vix_series, sizing="fixed_5_micros",
                               use_vix_target=False, use_atr_target=True,
                               atr_series=atr_series, atr_k=atr_k, trail_r=trail_r, use_runner=True)
            m = metrics(trades)
            if m.get("trades", 0) == 0:
                continue
            resolved = m["target_or_trail_exits"] + m["stop_exits"]
            rows.append(dict(atr_k=atr_k, trail_r=trail_r, trades=m["trades"], win_rate=m["win_rate_pct"],
                              net_pnl=m["net_pnl"], profit_factor=m["profit_factor"],
                              max_dd=m["max_drawdown"], sharpe=m["sharpe"],
                              resolved_pct=resolved / m["trades"] * 100))

    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"ATR-target ORB sweep, {IN_SAMPLE_START} -> {IN_SAMPLE_END}")
    print(result.to_string(index=False))
    print(f"\n(reference) VIX-target best: target_r=0.75, trail_r=0.25 -> Sharpe 0.99, net $42,587")


if __name__ == "__main__":
    main()
