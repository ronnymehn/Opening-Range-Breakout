"""Test the overnight-range-midpoint directional filter on the PLAIN ORB
strategy (fixed 1R off the ORB range, no VIX target, no runner -- isolating
this filter's effect from the VIX-sizing question already tested
separately).

Idea (user-supplied stat): mark the overnight session's (18:00-09:30 ET)
high, low, and midpoint. If RTH opens above the midpoint, the overnight high
tends to break first (76.2% historically); if below, the overnight low
tends to break first (75.6%). Since our ORB strategy is long-only, this is
used as a pre-trade filter: only take the long ORB breakout when the RTH
open sat above the prior overnight midpoint (the side the flow already
leans toward), and check the inverse (open below midpoint) as a placebo --
if the filter carries real signal, "above" trades should outperform
"below" trades; if it's noise, they should look similar.

2021-2024 IN-SAMPLE ONLY.
"""
from __future__ import annotations

import pandas as pd

from orb_vix_runner import fetch_mnq_5m, simulate, metrics
from backtest import prev_vix_close_lookup
from fetch_data import fetch_vix_daily

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"


def compute_overnight_midpoint_side(bars: pd.DataFrame) -> pd.Series:
    """For each RTH session date present in `bars`, whether the 09:30 open
    sat 'above' or 'below' the prior overnight (18:00-09:30 ET) range
    midpoint. Returns a Series indexed by session date (python date)."""
    session_dates = sorted(set(bars.index.date))
    sides = {}
    for d in session_dates:
        day_start = pd.Timestamp(d, tz="America/New_York")
        overnight_start = day_start - pd.Timedelta(hours=15, minutes=30)  # prior day 18:00
        overnight_end = day_start + pd.Timedelta(hours=9, minutes=30)     # today 09:30
        window = bars.loc[(bars.index >= overnight_start) & (bars.index < overnight_end)]
        if window.empty:
            continue
        ovn_high, ovn_low = float(window["high"].max()), float(window["low"].min())
        if ovn_high <= ovn_low:
            continue
        midpoint = (ovn_high + ovn_low) / 2
        rth_open_bar = bars.loc[(bars.index >= overnight_end) & (bars.index < overnight_end + pd.Timedelta(minutes=5))]
        if rth_open_bar.empty:
            continue
        rth_open = float(rth_open_bar.iloc[0]["open"])
        sides[d] = "above" if rth_open > midpoint else "below"
    return pd.Series(sides)


def main(start: str = IN_SAMPLE_START, end: str = IN_SAMPLE_END) -> None:
    bars = fetch_mnq_5m(start, end)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)

    print("Computing overnight midpoint side per session...", flush=True)
    side = compute_overnight_midpoint_side(bars)
    print(f"  {len(side)} sessions classified: {(side == 'above').sum()} above, {(side == 'below').sum()} below\n")

    trades = simulate(bars, vix_series, sizing="fixed_5_micros", use_vix_target=False, use_runner=False)
    trades["session_date_only"] = trades["session_date"].dt.date
    trades["overnight_side"] = trades["session_date_only"].map(side)

    unfiltered = trades
    above_only = trades[trades["overnight_side"] == "above"]
    below_only = trades[trades["overnight_side"] == "below"]

    for label, sub in [("UNFILTERED (all ORB trades)", unfiltered),
                        ("Opened ABOVE overnight midpoint (filter says: take it)", above_only),
                        ("Opened BELOW overnight midpoint (filter says: skip it -- placebo check)", below_only)]:
        m = metrics(sub)
        print(f"--- {label} ---")
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
    p.add_argument("--start", default=IN_SAMPLE_START)
    p.add_argument("--end", default=IN_SAMPLE_END)
    args = p.parse_args()
    main(args.start, args.end)
