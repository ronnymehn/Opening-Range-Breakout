"""MNQ opening-range breakout, combining three techniques from this project's
roadmap and a supplied reference script:

1. ORB entry (from the reference script): 09:30-09:55 America/New_York
   opening range on 5-minute bars. A completed 5-minute close above the
   range high enters long at the FOLLOWING bar's open (no lookahead).
   Long-only, matching the reference script.

2. VIX targeting (this project's core idea): the profit target is sized off
   the VIX-implied expected move (em_points = open * prevVIX/100 / sqrt(252))
   instead of a fixed 1R. Stop stays at the ORB low, per the reference
   script -- that's a session-structure level, not VIX-derived.

3. Runner: at the VIX-sized target, take a PARTIAL profit (half the
   position) instead of exiting fully. Move the stop on the remainder to
   breakeven and trail it (highest high since entry minus trail_r *
   em_points, ratcheting up only) for the rest of the session, exiting at
   the trailing stop or the 14:00 ET time-exit, whichever comes first.

Position sizing schemes (fixed micros / ORB-risk-budget / Harvey target-vol
/ Moreira-Muir inverse-variance) are ported from the reference script
essentially unchanged -- they're generic on entry/stop and don't depend on
how the target is defined.

Data: MNQ.V.0 (volume-rolled continuous Micro E-mini Nasdaq-100) 1-minute
bars via the same qpd.get_bars() pathway used throughout this project
(cached to data/mnq_1m.parquet), resampled to 5-minute bars client-side.
VIX daily closes reuse the existing fetch_data.fetch_vix_daily() cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import quantpad_data as qpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_data import fetch_vix_daily, _to_ms  # noqa: E402
from backtest import prev_vix_close_lookup, get_prev_vix  # noqa: E402
from strategy import expected_move_points  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "orb_vix_runner"
MNQ_PARQUET = DATA_DIR / "mnq_1m.parquet"
MNQ_SYMBOL = "MNQ.V.0"

POINT_VALUE = 2.0
TICK_SIZE = 0.25
COMMISSION_PER_SIDE = 0.62
BASE_SLIPPAGE_TICKS = 1
INITIAL_CAPITAL = 100_000.0
VOL_TARGET_CAPITAL = 100_000.0
RISK_BUDGET = 1_000.0
FIXED_MICRO_CONTRACTS = 5
VOL_LOOKBACK_SESSIONS = 20
TARGET_ANNUAL_VOL = 0.20
MIN_MICRO_CONTRACTS = 1
MAX_MICRO_CONTRACTS = 10

TARGET_R = 1.0      # VIX target = entry + TARGET_R * em_points
TRAIL_R = 0.5        # runner trailing distance = TRAIL_R * em_points

CANONICAL_HEADERS = [
    "Trade #", "Type", "Date/Time", "Signal", "Price USD",
    "Position size (qty)", "Position size (value)", "Net P&L USD",
    "Net P&L %", "Run-up USD", "Run-up %", "Drawdown USD",
    "Drawdown %", "Cumulative P&L USD", "Cumulative P&L %",
]


# ---------------------------------------------------------------- data ----

def fetch_mnq_5m(start: str, end: str) -> pd.DataFrame:
    """1-minute MNQ bars via the proven get_bars pathway (cached), resampled
    to 5-minute bars client-side. Cache covers the widest range requested
    across runs; re-fetches if the request falls outside it."""
    start_ms, end_ms = _to_ms(start), _to_ms(end)

    if MNQ_PARQUET.exists():
        cached = pd.read_parquet(MNQ_PARQUET)
        cached_start_ms = int(cached["ts"].min().timestamp() * 1000)
        cached_end_ms = int(cached["ts"].max().timestamp() * 1000)
        if cached_start_ms <= start_ms and cached_end_ms >= end_ms - 1:
            bars = cached
        else:
            bars = None
    else:
        bars = None

    if bars is None:
        print(f"[mnq] downloading {MNQ_SYMBOL} 1m bars {start}..{end} from QuantPad ...", flush=True)
        chunks = []
        years = pd.date_range(start, end, freq="YS").tolist()
        if not years or years[0] > pd.Timestamp(start):
            years = [pd.Timestamp(start)] + years
        boundaries = years + [pd.Timestamp(end)]
        for i in range(len(boundaries) - 1):
            s = int(boundaries[i].tz_localize("UTC").timestamp() * 1000)
            e = int(boundaries[i + 1].tz_localize("UTC").timestamp() * 1000)
            if e <= start_ms or s >= end_ms:
                continue
            s, e = max(s, start_ms), min(e, end_ms)
            print(f"  fetching {boundaries[i].date()}..{boundaries[i + 1].date()}", flush=True)
            df = qpd.get_bars(MNQ_SYMBOL, "1m", s, e)
            df = df.reset_index().rename(columns={"t": "ts"})
            df = df[["ts", "open", "high", "low", "close", "volume"]]
            chunks.append(df)
        bars = pd.concat(chunks, ignore_index=True)
        if bars["ts"].dt.tz is None:
            bars["ts"] = bars["ts"].dt.tz_localize("UTC")
        bars = bars.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(MNQ_PARQUET, index=False)
        print(f"[mnq] cached {len(bars):,} bars -> {MNQ_PARQUET}", flush=True)

    local = bars.set_index("ts")[["open", "high", "low", "close", "volume"]]
    local.index = local.index.tz_convert("America/New_York")
    local = local[(local.index >= pd.Timestamp(start, tz="America/New_York"))
                  & (local.index < pd.Timestamp(end, tz="America/New_York"))]

    ohlc = local.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])
    return ohlc


# ------------------------------------------------------------ helpers -----

def adverse_fill(price: float, action: str, slippage_ticks: int) -> float:
    slip = slippage_ticks * TICK_SIZE
    return price + slip if action == "entry" else price - slip


def realized_volatility(bars: pd.DataFrame, lookback: int) -> pd.Series:
    closes = bars.at_time("16:00")["close"].groupby(lambda ts: ts.date()).last()
    if closes.empty:
        # 16:00 exact bar may not exist for 5m bars ending at 15:55/16:00 boundary quirks;
        # fall back to each day's last available close.
        closes = bars["close"].groupby(lambda ts: ts.date()).last()
    returns = closes.pct_change()
    annual_vol = returns.rolling(lookback).std(ddof=1) * np.sqrt(252)
    return annual_vol.shift(1)


def stop_loss_per_contract(entry: float, stop: float, slippage_ticks: int) -> float:
    exit_slippage = slippage_ticks * TICK_SIZE
    return ((entry - stop) + exit_slippage) * POINT_VALUE + 2 * COMMISSION_PER_SIDE


def contracts_for_trade(
    entry: float, stop: float, sizing: str, slippage_ticks: int,
    risk_budget: float, annual_vol: "float | None",
) -> int:
    if sizing == "fixed_5_micros":
        return FIXED_MICRO_CONTRACTS
    if sizing == "orb_risk_target":
        per_contract_risk = stop_loss_per_contract(entry, stop, slippage_ticks)
        return max(0, int(np.floor(risk_budget / per_contract_risk)))
    if annual_vol is None or not np.isfinite(annual_vol) or annual_vol <= 0:
        return 0
    exponent = 1 if sizing == "harvey_target_vol" else 2
    scale = (TARGET_ANNUAL_VOL / annual_vol) ** exponent
    return int(np.clip(np.floor(FIXED_MICRO_CONTRACTS * scale), MIN_MICRO_CONTRACTS, MAX_MICRO_CONTRACTS))


# ------------------------------------------------------------ engine ------

def simulate(
    bars: pd.DataFrame,
    vix_series: pd.Series,
    sizing: str = "fixed_5_micros",
    slippage_ticks: int = BASE_SLIPPAGE_TICKS,
    risk_budget: float = RISK_BUDGET,
    annual_vol: "pd.Series | None" = None,
    target_r: float = TARGET_R,
    trail_r: float = TRAIL_R,
    use_vix_target: bool = True,
    use_runner: bool = True,
    use_atr_target: bool = False,
    atr_series: "pd.Series | None" = None,
    atr_k: float = 0.5,
) -> pd.DataFrame:
    """use_atr_target=True overrides use_vix_target: both the target and the
    runner's trailing distance are sized off ATR(14) (atr_k * atr_14)
    instead of VIX em_points -- a clean, single-variable test of realized
    vol (ATR) vs implied vol (VIX) as the sizing input, at a matched
    reachability level (see src/target_achievability.py)."""
    records: list[dict] = []
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
        entry = adverse_fill(float(entry_bar["open"]), "entry", slippage_ticks)
        stop = range_low
        if entry <= stop:
            continue

        pv = get_prev_vix(vix_series, pd.Timestamp(session_date))
        em = expected_move_points(entry, pv) if not pd.isna(pv) else float("nan")
        atr_14 = atr_series.get(session_date, np.nan) if atr_series is not None else np.nan

        if use_atr_target:
            if pd.isna(atr_14) or atr_14 <= 0:
                continue
            vol_unit = atr_14
            vix_target = entry + atr_k * atr_14
        elif use_vix_target:
            if pd.isna(pv) or em <= 0:
                continue
            vol_unit = em
            vix_target = entry + target_r * em
        else:
            vol_unit = em  # only used by the runner trail; irrelevant here since use_runner is typically False in this mode
            vix_target = entry + (entry - stop)  # fixed 1R off the ORB range, matches reference script

        vol_for_day = annual_vol.get(session_date, np.nan) if annual_vol is not None else np.nan
        qty = contracts_for_trade(entry, stop, sizing, slippage_ticks, risk_budget, vol_for_day)
        if qty == 0:
            continue
        if use_runner and qty >= 2:
            leg1_qty = qty // 2
            leg2_qty = qty - leg1_qty
        else:
            leg1_qty = qty
            leg2_qty = 0

        holding = day.loc[entry_bar_time:]
        phase = "combined"          # "combined" -> both legs share the ORB stop; "runner" -> leg1 closed, leg2 trailing
        runner_stop = None
        highest_since_entry = float(entry_bar["high"])
        legs: list[dict] = []       # each: {qty, exit_time, exit_price, reason}
        held_through = None

        for ts, bar in holding.iterrows():
            if ts.time() >= pd.Timestamp("14:00").time():
                remaining_qty = qty if phase == "combined" else leg2_qty
                legs.append(dict(qty=remaining_qty, exit_time=ts,
                                  exit_price=adverse_fill(float(bar["open"]), "exit", slippage_ticks),
                                  reason="time"))
                held_through = ts
                break

            if phase == "combined":
                hit_stop = float(bar["low"]) <= stop
                hit_target = float(bar["high"]) >= vix_target
                if hit_stop:
                    legs.append(dict(qty=qty, exit_time=ts,
                                      exit_price=adverse_fill(stop, "exit", slippage_ticks), reason="stop"))
                    held_through = ts
                    break
                if hit_target:
                    legs.append(dict(qty=leg1_qty, exit_time=ts,
                                      exit_price=adverse_fill(vix_target, "exit", slippage_ticks), reason="target"))
                    if leg2_qty == 0:
                        held_through = ts
                        break
                    phase = "runner"
                    runner_stop = entry  # breakeven
                    highest_since_entry = max(highest_since_entry, float(bar["high"]))
                    # fall through to check the runner's stop against the SAME bar's low
                    # (bar already made its high -> target; still check low -> breakeven stop)
                    if float(bar["low"]) <= runner_stop:
                        legs.append(dict(qty=leg2_qty, exit_time=ts,
                                          exit_price=adverse_fill(runner_stop, "exit", slippage_ticks),
                                          reason="trail_stop"))
                        held_through = ts
                        break
                    continue
                continue

            # phase == "runner"
            highest_since_entry = max(highest_since_entry, float(bar["high"]))
            new_trail = highest_since_entry - trail_r * vol_unit
            runner_stop = max(runner_stop, new_trail)
            if float(bar["low"]) <= runner_stop:
                legs.append(dict(qty=leg2_qty, exit_time=ts,
                                  exit_price=adverse_fill(runner_stop, "exit", slippage_ticks), reason="trail_stop"))
                held_through = ts
                break

        if held_through is None:
            continue  # never resolved within the session -- drop, matching reference script's convention

        gross_pnl = sum((leg["exit_price"] - entry) * POINT_VALUE * leg["qty"] for leg in legs)
        net_pnl = gross_pnl - 2 * COMMISSION_PER_SIDE * qty
        planned_risk_per_contract = stop_loss_per_contract(entry, stop, slippage_ticks)
        held = holding.loc[:held_through]
        mfe_points = max(0.0, float(held["high"].max()) - entry)
        mae_points = max(0.0, entry - float(held["low"].min()))
        exit_reasons = "+".join(leg["reason"] for leg in legs)

        records.append({
            "session_date": pd.Timestamp(session_date), "entry_time": entry_bar_time,
            "trigger_time": trigger_time, "entry": entry, "stop": stop, "vix_target": vix_target,
            "em_points": em, "atr_14": atr_14, "qty": qty, "leg1_qty": leg1_qty, "leg2_qty": leg2_qty,
            "estimated_annual_vol": vol_for_day, "range_points": range_high - range_low,
            "risk_points": entry - stop, "planned_risk_per_contract": planned_risk_per_contract,
            "planned_risk_usd": planned_risk_per_contract * qty,
            "gross_pnl": gross_pnl, "net_pnl": net_pnl, "exit_reason": exit_reasons,
            "exit_time": held_through, "mfe_usd": mfe_points * POINT_VALUE * qty,
            "mae_usd": mae_points * POINT_VALUE * qty,
        })

    return pd.DataFrame(records)


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return dict(trades=0)
    pnl = trades["net_pnl"]
    daily = trades.groupby("session_date")["net_pnl"].sum()
    full_days = pd.date_range(trades["session_date"].min(), trades["session_date"].max(), freq="B")
    daily = daily.reindex(full_days, fill_value=0.0)
    equity = INITIAL_CAPITAL + pnl.cumsum()
    drawdown = equity - equity.cummax()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_loss = abs(losses.sum())
    return {
        "trades": len(trades), "net_pnl": pnl.sum(), "win_rate_pct": (pnl > 0).mean() * 100,
        "profit_factor": wins.sum() / gross_loss if gross_loss else np.nan,
        "expectancy": pnl.mean(),
        "avg_win": wins.mean() if len(wins) else np.nan,
        "avg_loss": losses.mean() if len(losses) else np.nan,
        "max_drawdown": drawdown.min(),
        "sharpe": daily.mean() / daily.std(ddof=1) * np.sqrt(252) if daily.std(ddof=1) else np.nan,
        "avg_contracts": trades["qty"].mean(),
        "runner_activated_pct": (trades["leg2_qty"] > 0).mean() * 100,
        "target_or_trail_exits": int(trades["exit_reason"].str.contains("target|trail").sum()),
        "stop_exits": int((trades["exit_reason"] == "stop").sum()),
        "time_exits": int(trades["exit_reason"].str.endswith("time").sum()),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--sizing", default="fixed_5_micros",
                    choices=["fixed_5_micros", "orb_risk_target", "harvey_target_vol", "moreira_muir"])
    p.add_argument("--target-r", type=float, default=TARGET_R)
    p.add_argument("--trail-r", type=float, default=TRAIL_R)
    p.add_argument("--no-vix-target", action="store_true", help="use fixed 1R (ORB range) target instead of VIX-sized")
    p.add_argument("--no-runner", action="store_true", help="disable partial exit/trailing runner, single full exit")
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    bars = fetch_mnq_5m(args.start, args.end)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)
    annual_vol = realized_volatility(bars, VOL_LOOKBACK_SESSIONS)

    trades = simulate(bars, vix_series, sizing=args.sizing, annual_vol=annual_vol,
                       target_r=args.target_r, trail_r=args.trail_r,
                       use_vix_target=not args.no_vix_target, use_runner=not args.no_runner)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUT_DIR / f"trades_{args.sizing}{args.out_suffix}.csv", index=False)

    m = metrics(trades)
    print(f"ORB strategy  |  {args.start} -> {args.end}  |  sizing={args.sizing}  target_r={args.target_r}  "
          f"trail_r={args.trail_r}  use_vix_target={not args.no_vix_target}  use_runner={not args.no_runner}")
    for k, v in m.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
