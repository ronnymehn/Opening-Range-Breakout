"""Empirical target-achievability analysis for the ORB strategy, inspired by
the gap-fill touch-probability methodology: instead of asking "does price
touch yesterday's close" (a different reference point), ask the directly
relevant question -- for every ORB-triggered session, how far did price
travel FROM ENTRY, in the breakout direction, by the 14:00 ET cutoff?
Normalize that distance two ways (ATR(14) and VIX em_points) to see which
one the market's actual reachable-distance distribution lines up with
better, and where target_r=0.75 * em_points falls on this curve.

2021-2024 IN-SAMPLE ONLY, matching this project's discipline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import quantpad_data as qpd

from orb_vix_runner import fetch_mnq_5m, adverse_fill, BASE_SLIPPAGE_TICKS, MNQ_SYMBOL
from backtest import prev_vix_close_lookup, get_prev_vix
from fetch_data import fetch_vix_daily, _to_ms
from strategy import expected_move_points

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"


def fetch_mnq_daily_atr(start: str, end: str, lookback: int = 14) -> pd.Series:
    """ATR(14) on RTH daily bars (True Range, rolling mean, shifted by 1 so
    only PRIOR sessions are used -- no lookahead into today's own range)."""
    s, e = _to_ms(start), _to_ms(end)
    bars = qpd.get_bars(MNQ_SYMBOL, "1d", s, e)
    bars = bars.reset_index().rename(columns={"t": "ts"})
    bars["date"] = bars["ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    bars = bars.set_index("date")[["open", "high", "low", "close"]].sort_index()

    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback).mean().shift(1)
    return atr


def collect_mfe_by_session(bars: pd.DataFrame, vix_series: pd.Series, atr: pd.Series) -> pd.DataFrame:
    """For every ORB-triggered session: entry, direction (long-only, matching
    the reference script), em_points, atr_14, and MFE (max favorable
    excursion in points, from entry to the 14:00 cutoff)."""
    rows = []
    for session_date, day in bars.groupby(bars.index.date):
        opening = day.between_time("09:30", "09:55")
        trade_window = day.between_time("10:00", "14:00")
        if len(opening) != 6 or trade_window.empty:
            continue
        range_high = float(opening["high"].max())
        range_low = float(opening["low"].min())
        trigger_positions = np.flatnonzero(trade_window["close"].to_numpy() > range_high)
        if trigger_positions.size == 0:
            continue
        trigger_time = trade_window.index[trigger_positions[0]]
        day_times = day.index
        trigger_idx = day_times.get_loc(trigger_time)
        if trigger_idx + 1 >= len(day):
            continue
        entry_bar_time = day.index[trigger_idx + 1]
        if entry_bar_time.time() >= pd.Timestamp("14:00").time():
            continue
        entry_bar = day.loc[entry_bar_time]
        entry = adverse_fill(float(entry_bar["open"]), "entry", BASE_SLIPPAGE_TICKS)
        stop = range_low
        if entry <= stop:
            continue

        pv = get_prev_vix(vix_series, pd.Timestamp(session_date))
        em = expected_move_points(entry, pv) if not pd.isna(pv) else np.nan
        atr_14 = atr.get(pd.Timestamp(session_date), np.nan)

        holding = day.loc[entry_bar_time:].between_time("00:00", "13:59")
        if holding.empty:
            continue
        mfe = max(0.0, float(holding["high"].max()) - entry)

        rows.append(dict(session_date=pd.Timestamp(session_date), entry=entry, stop=stop,
                          orb_range=entry - stop, em_points=em, atr_14=atr_14, mfe=mfe))
    return pd.DataFrame(rows)


def main() -> None:
    bars = fetch_mnq_5m(IN_SAMPLE_START, IN_SAMPLE_END)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)
    atr = fetch_mnq_daily_atr(IN_SAMPLE_START, IN_SAMPLE_END)

    df = collect_mfe_by_session(bars, vix_series, atr)
    df = df.dropna(subset=["em_points", "atr_14"])
    print(f"{len(df)} ORB-triggered sessions with valid VIX + ATR data\n")

    # How does em_points compare to ATR(14) in typical magnitude?
    ratio = df["em_points"] / df["atr_14"]
    print(f"em_points / ATR(14) ratio: mean={ratio.mean():.2f}  median={ratio.median():.2f}  "
          f"std={ratio.std():.2f}")
    print(f"  -> target_r=0.75 * em_points corresponds to ~{0.75 * ratio.median():.2f}x ATR(14) on a typical day\n")

    # Empirical achievability curve: normalized by ATR(14)
    df["mfe_atr"] = df["mfe"] / df["atr_14"]
    df["mfe_em"] = df["mfe"] / df["em_points"]

    print("Empirical 'reached >= X' rate by 14:00, normalized by ATR(14):")
    for x in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]:
        pct = (df["mfe_atr"] >= x).mean() * 100
        print(f"  >= {x:.1f} ATR: {pct:5.1f}%  ({(df['mfe_atr'] >= x).sum()}/{len(df)})")

    print("\nEmpirical 'reached >= X' rate by 14:00, normalized by VIX em_points:")
    for x in [0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5]:
        pct = (df["mfe_em"] >= x).mean() * 100
        print(f"  >= {x:.2f}x em: {pct:5.1f}%  ({(df['mfe_em'] >= x).sum()}/{len(df)})")

    print(f"\nAt target_r=0.75 (the sweep-found peak): reached in {(df['mfe_em'] >= 0.75).mean()*100:.1f}% of sessions")
    print(f"At target_r=1.00 (default): reached in {(df['mfe_em'] >= 1.00).mean()*100:.1f}% of sessions")

    df.to_csv("data/orb_vix_runner/target_achievability.csv", index=False)


if __name__ == "__main__":
    main()
