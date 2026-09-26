"""Implied-vs-realized divergence signal (roadmap idea #3, docs/strategy_ideas.md).

Each session has a VIX-implied expected move (em_points). We also know what
the session's realized range actually was. ratio = realized_range / em_points
tells us, after the fact, whether that day's VIX pricing over- or
under-stated the actual move.

The signal: a TRAILING average of that ratio (using only prior days, no
lookahead) as a regime indicator for TODAY's trade:
  - trailing ratio persistently < 1 -> VIX has been overpricing moves lately
    (a "calm" regime) -> today's band breach is more likely an overreaction
    -> fade it.
  - trailing ratio persistently > 1 -> VIX has been underpricing moves
    lately (a "trending/expanding" regime) -> trust continuation -> trade
    the breakout.
  - in between -> default_mode (breakout, since that's the validated
    baseline) or None to skip.

This adapts across years automatically (unlike the earlier fixed VIX>35
cutoff, which was calibrated to one historical episode and failed to
generalize), since it's always relative to the market's own recent behavior.
"""
from __future__ import annotations

import pandas as pd

from strategy import expected_move_points


def build_ratio_series(
    sessions: dict[pd.Timestamp, pd.DataFrame],
    vix_series: pd.Series,
    get_prev_vix,
) -> pd.DataFrame:
    """One row per session: open, em_points, realized_range, ratio.
    Computed for ALL available sessions (not just the backtest window) so a
    trailing lookback has history at the start of any window."""
    rows = []
    for session_date in sorted(sessions):
        bars = sessions[session_date]
        if len(bars) < 5:
            continue
        pv = get_prev_vix(vix_series, session_date)
        if pd.isna(pv):
            continue
        open_price = float(bars.iloc[0]["open"])
        em = expected_move_points(open_price, pv)
        if em <= 0:
            continue
        realized_range = float(bars["high"].max() - bars["low"].min())
        rows.append(dict(date=session_date, open=open_price, em_points=em,
                          realized_range=realized_range, ratio=realized_range / em))

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def add_trailing_signal(ratio_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing mean ratio using only PRIOR days (shift(1) before rolling,
    so today's own realized_range never leaks into today's signal)."""
    df = ratio_df.copy()
    df["trailing_ratio"] = df["ratio"].shift(1).rolling(lookback, min_periods=lookback).mean()
    return df


def decide_mode(trailing_ratio: float, low_thresh: float, high_thresh: float, default_mode: str | None) -> str | None:
    if pd.isna(trailing_ratio):
        return default_mode
    if trailing_ratio < low_thresh:
        return "fade"
    if trailing_ratio > high_thresh:
        return "breakout"
    return default_mode
