"""VIX-implied expected-move calculation and the NQ band-breakout/momentum strategy.

Idea: yesterday's VIX close estimates the S&P's annualized expected volatility.
Scaling it to one day (divide by sqrt(252)) and applying it to today's NQ open
gives a volatility-based "expected move" for the session. This module marks
that distance above/below the open as upper/lower bands, then trades the
breakout of those bands as a momentum signal (continuation, not mean reversion).

These bands are NOT reversal/take-profit levels on their own -- they describe
how big a move is typical for the day's vol regime. Here we operationalize
them as a breakout trigger + R-multiple target/stop, per the chosen strategy
design.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DAILY_VOL_SCALER = sqrt(TRADING_DAYS_PER_YEAR)  # ~15.87


def expected_move_points(open_price: float, prev_vix_close: float) -> float:
    """Open x (prevVIX / 100) / sqrt(252) -> expected 1-day move in points."""
    return open_price * (prev_vix_close / 100.0) / DAILY_VOL_SCALER


def expected_move_bands(open_price: float, prev_vix_close: float) -> tuple[float, float, float]:
    """Returns (em_points, upper_band, lower_band)."""
    em = expected_move_points(open_price, prev_vix_close)
    return em, open_price + em, open_price - em


@dataclass
class Trade:
    date: pd.Timestamp
    direction: str          # "long" or "short"
    mode: str                # "breakout" or "fade"
    open_price: float
    prev_vix_close: float
    em_points: float
    upper_band: float
    lower_band: float
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str        # "target", "stop", "eod"
    pnl_points: float


def simulate_session(
    session_date: pd.Timestamp,
    bars: pd.DataFrame,
    prev_vix_close: float,
    target_r: float = 1.0,
    stop_r: float = 1.0,
    vix_cap: float | None = None,
    mode: str = "breakout",
) -> Trade | None:
    """mode="breakout" (momentum, default): first close beyond a band enters
    in that direction. Stop distance = stop_r * (entry_price - open_price) --
    stop_r=1.0 places the stop at the session open (the original "give back
    the whole move" risk); stop_r<1.0 tightens it proportionally. Target =
    entry +/- target_r * em_points beyond entry (further from the open).

    mode="fade" (mean reversion): first close beyond a band enters AGAINST
    that direction, betting on reversion toward the open. Stop is placed
    stop_r * em_points further beyond the band (giving room for the move to
    extend before admitting the fade is wrong); target is target_r * em_points
    back toward the open (target_r=1.0 lands ~at the open, since entry is
    ~1 em_points away from it).

    Both modes: exit at stop/target intrabar (stop checked first if both hit
    in the same bar), else flat at session close.

    vix_cap: if set, skip the session entirely (no trade) when prev_vix_close
    exceeds it -- a simple regime filter for extreme-vol whipsaw periods
    (e.g. the March 2020 COVID crash) where realized intraday moves blow
    through the vol-implied band in both directions.

    `bars` must be this session's 1-min OHLC bars only, sorted ascending,
    with columns open/high/low/close and a `ts` timestamp column.
    """
    if mode not in ("breakout", "fade"):
        raise ValueError(f"unknown mode: {mode}")
    if bars.empty or pd.isna(prev_vix_close):
        return None
    if vix_cap is not None and prev_vix_close > vix_cap:
        return None

    open_price = float(bars.iloc[0]["open"])
    em, upper, lower = expected_move_bands(open_price, prev_vix_close)
    if em <= 0:
        return None

    direction = None
    entry_time = entry_price = None
    stop = target = None

    for i, row in bars.iterrows():
        if direction is None:
            upper_breach = row["close"] > upper
            lower_breach = row["close"] < lower
            if not (upper_breach or lower_breach):
                continue

            if mode == "breakout":
                direction = "long" if upper_breach else "short"
            else:  # fade: trade against the breach, toward the open
                direction = "short" if upper_breach else "long"

            entry_time, entry_price = row["ts"], float(row["close"])
            if mode == "breakout":
                if direction == "long":
                    stop = entry_price - stop_r * (entry_price - open_price)
                    target = entry_price + target_r * em
                else:
                    stop = entry_price + stop_r * (open_price - entry_price)
                    target = entry_price - target_r * em
            else:  # fade
                if direction == "long":
                    stop = entry_price - stop_r * em
                    target = entry_price + target_r * em
                else:
                    stop = entry_price + stop_r * em
                    target = entry_price - target_r * em
            continue

        # position open: check this and all subsequent bars for stop/target
        if direction == "long":
            hit_stop = row["low"] <= stop
            hit_target = row["high"] >= target
        else:
            hit_stop = row["high"] >= stop
            hit_target = row["low"] <= target

        if hit_stop:
            exit_time, exit_price, reason = row["ts"], stop, "stop"
        elif hit_target:
            exit_time, exit_price, reason = row["ts"], target, "target"
        else:
            continue

        pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        return Trade(session_date, direction, mode, open_price, prev_vix_close, em, upper, lower,
                     entry_time, entry_price, exit_time, exit_price, reason, pnl)

    if direction is not None:
        last = bars.iloc[-1]
        exit_time, exit_price = last["ts"], float(last["close"])
        pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        return Trade(session_date, direction, mode, open_price, prev_vix_close, em, upper, lower,
                     entry_time, entry_price, exit_time, exit_price, "eod", pnl)

    return None


def simulate_session_ib(
    session_date: pd.Timestamp,
    bars: pd.DataFrame,
    prev_vix_close: float,
    ib_minutes: int = 30,
    target_r: float = 1.0,
) -> "Trade | None":
    """Idea #5: Initial Balance breakout, VIX-implied expected move used only
    to SIZE the profit target (never as the entry trigger). Entry: first
    close beyond the IB high/low (the first `ib_minutes` of the session)
    after the IB window closes. Stop: the opposite IB bound -- the classic
    IB-failure level, not a fitted parameter. Target: entry +/- target_r *
    em_points. `upper_band`/`lower_band` on the returned Trade hold
    IB high/low (reusing the same record shape as the band-breakout trades).

    `bars` must be this session's 1-min RTH bars only, sorted ascending.
    """
    if bars.empty or pd.isna(prev_vix_close) or len(bars) <= ib_minutes:
        return None

    open_price = float(bars.iloc[0]["open"])
    em = expected_move_points(open_price, prev_vix_close)
    if em <= 0:
        return None

    ib_bars = bars.iloc[:ib_minutes]
    post_ib_bars = bars.iloc[ib_minutes:]
    ib_high = float(ib_bars["high"].max())
    ib_low = float(ib_bars["low"].min())
    if ib_high <= ib_low:
        return None

    direction = None
    entry_time = entry_price = None
    stop = target = None

    for i, row in post_ib_bars.iterrows():
        if direction is None:
            if row["close"] > ib_high:
                direction = "long"
                entry_time, entry_price = row["ts"], float(row["close"])
                stop = ib_low
                target = entry_price + target_r * em
            elif row["close"] < ib_low:
                direction = "short"
                entry_time, entry_price = row["ts"], float(row["close"])
                stop = ib_high
                target = entry_price - target_r * em
            continue

        if direction == "long":
            hit_stop = row["low"] <= stop
            hit_target = row["high"] >= target
        else:
            hit_stop = row["high"] >= stop
            hit_target = row["low"] <= target

        if hit_stop:
            exit_time, exit_price, reason = row["ts"], stop, "stop"
        elif hit_target:
            exit_time, exit_price, reason = row["ts"], target, "target"
        else:
            continue

        pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        return Trade(session_date, direction, "ib_breakout", open_price, prev_vix_close, em,
                     ib_high, ib_low, entry_time, entry_price, exit_time, exit_price, reason, pnl)

    if direction is not None:
        last = bars.iloc[-1]
        exit_time, exit_price = last["ts"], float(last["close"])
        pnl = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        return Trade(session_date, direction, "ib_breakout", open_price, prev_vix_close, em,
                     ib_high, ib_low, entry_time, entry_price, exit_time, exit_price, "eod", pnl)

    return None
