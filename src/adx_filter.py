"""ADX(14) trend-strength filter, computed from NQ daily bars (shared market
regime signal), applied to both the VIX-band breakout strategy and the
ORB+VIX-target+runner strategy: only take a trade if the PRIOR day's ADX(14)
> 20 (the classic "market is trending" threshold; below it is considered
range-bound/choppy). Standard Wilder's smoothing.

2021-2024 IN-SAMPLE ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import quantpad_data as qpd

from orb_vix_runner import fetch_mnq_5m, simulate as orb_simulate, metrics as orb_metrics
from backtest import (
    load_sessions_and_vix, run_backtest, compute_metrics as vix_metrics,
    prev_vix_close_lookup,
)
from fetch_data import fetch_vix_daily, _to_ms

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"
NQ_SYMBOL = "NQ.C.0"
ADX_PERIOD = 14
ADX_THRESHOLD = 20.0


def fetch_nq_daily(start: str, end: str) -> pd.DataFrame:
    s, e = _to_ms(start), _to_ms(end)
    bars = qpd.get_bars(NQ_SYMBOL, "1d", s, e)
    bars = bars.reset_index().rename(columns={"t": "ts"})
    # Daily bar timestamps are already UTC-midnight anchored to the correct
    # trading date -- do NOT tz_convert (that shifts the date back a day).
    bars["date"] = bars["ts"].dt.normalize().dt.tz_localize(None)
    return bars.set_index("date")[["open", "high", "low", "close"]].sort_index()


def compute_adx(daily: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """Standard Wilder's ADX. Returns ADX shifted by 1 day (only prior-day
    values used -- no lookahead into today's own developing range)."""
    high, low, close = daily["high"], daily["low"], daily["close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=daily.index)
    minus_dm = pd.Series(minus_dm, index=daily.index)

    # Wilder's smoothing (equivalent to an EMA with alpha=1/period, seeded by a simple rolling sum)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return adx.shift(1)  # only prior-day ADX used to filter today's trade


def main() -> None:
    nq_daily = fetch_nq_daily(IN_SAMPLE_START, IN_SAMPLE_END)
    adx = compute_adx(nq_daily)
    adx_by_date = {ts.date(): v for ts, v in adx.items()}
    print(f"ADX(14) computed for {adx.notna().sum()} sessions, mean={adx.mean():.1f}, "
          f"median={adx.median():.1f}, %>20={100*(adx>20).mean():.1f}%\n")

    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)

    # --- VIX-band breakout strategy (the main validated baseline) ---
    print("=" * 70)
    print("VIX-band breakout (NQ, fixed 1 contract, target_r=stop_r=1.0)")
    print("=" * 70)
    vix_trades = run_backtest(start=IN_SAMPLE_START, end=IN_SAMPLE_END, target_r=1.0, stop_r=1.0,
                               vix_cap=None, mode="breakout", sizing="fixed_1_nq", save=False)
    vix_trades["adx"] = vix_trades["date"].dt.date.map(adx_by_date)
    report_filtered(vix_trades, adx_by_date, vix_metrics, pnl_col="pnl_usd_net")

    # --- ORB + VIX-target + runner strategy ---
    print("\n" + "=" * 70)
    print("ORB + VIX-target + runner (MNQ, target_r=0.75, trail_r=0.25)")
    print("=" * 70)
    bars = fetch_mnq_5m(IN_SAMPLE_START, IN_SAMPLE_END)
    orb_trades = orb_simulate(bars, vix_series, sizing="fixed_5_micros", use_vix_target=True,
                               use_runner=True, target_r=0.75, trail_r=0.25)
    orb_trades["adx"] = orb_trades["session_date"].dt.date.map(adx_by_date)
    report_filtered(orb_trades, adx_by_date, orb_metrics, pnl_col="net_pnl")


def report_filtered(trades: pd.DataFrame, adx_by_date: dict, metrics_fn, pnl_col: str) -> None:
    unfiltered = trades
    trending = trades[trades["adx"] > ADX_THRESHOLD]
    choppy = trades[trades["adx"] <= ADX_THRESHOLD]

    for label, sub in [("UNFILTERED (all trades)", unfiltered),
                        (f"ADX > {ADX_THRESHOLD:.0f} (trending -- filter says: take it)", trending),
                        (f"ADX <= {ADX_THRESHOLD:.0f} (choppy -- filter says: skip it)", choppy)]:
        m = metrics_fn(sub) if not sub.empty else {}
        n = m.get("n", m.get("trades", 0))
        print(f"--- {label} ---")
        if not n:
            print("  no trades\n")
            continue
        total_net = sub[pnl_col].sum()
        win_rate = (sub[pnl_col] > 0).mean() * 100
        print(f"  trades: {n}")
        print(f"  win_rate: {win_rate:.1f}%")
        print(f"  net_pnl: ${total_net:,.0f}")
        print(f"  sharpe: {m.get('sharpe', float('nan')):.2f}")
        print()


if __name__ == "__main__":
    main()
