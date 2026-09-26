"""Reference ORB script (target_r sweep) run on NQ (full-size E-mini, not
MNQ micro), 2020-2026. Same entry/exit logic as run_reference_orb_sweep.py,
adjusted for NQ's contract specs. Self-contained: fetches its own NQ 1-minute
bars (cached locally to data/nq_1m.parquet), no dependency on anything
outside this folder.
  - POINT_VALUE: $20/pt (vs MNQ's $2/pt)
  - COMMISSION_PER_SIDE: $2.30 (round-turn $4.60) vs MNQ's $0.62
  - Fixed sizing: 1 NQ contract instead of 5 MNQ micros -- NOT a
    risk-equivalent comparison, roughly 2x the notional exposure of the
    "5 micros" MNQ tests (1 NQ ~= 10 micros).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import quantpad_data as qpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fomc_dates import FOMC_DATES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "orb_reference_nq"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NQ_SYMBOL = "NQ.C.0"
POINT_VALUE = 20.0
TICK_SIZE = 0.25
COMMISSION_PER_SIDE = 2.30
BASE_SLIPPAGE_TICKS = 1
INITIAL_CAPITAL = 100_000.0
FIXED_CONTRACTS = 1
RISK_BUDGET_USD = 2_000.0  # equal-dollar-risk sizing: $ risked per trade (assumption -- adjust to your account)
BACKTEST_START = "2020-01-01"
BACKTEST_END = "2026-12-31"
TARGET_R_VALUES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)


def _to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_nq_5m(start: str, end: str) -> pd.DataFrame:
    nq_parquet = DATA_DIR / "nq_1m.parquet"
    start_ms, end_ms = _to_ms(start), _to_ms(end)

    bars = None
    if nq_parquet.exists():
        cached = pd.read_parquet(nq_parquet)
        cached_start_ms = int(cached["ts"].min().timestamp() * 1000)
        cached_end_ms = int(cached["ts"].max().timestamp() * 1000)
        if cached_start_ms <= start_ms and cached_end_ms >= end_ms - 86400000:
            bars = cached

    if bars is None:
        print(f"[nq] downloading {NQ_SYMBOL} 1m bars {start}..{end} from QuantPad ...", flush=True)
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
            df = qpd.get_bars(NQ_SYMBOL, "1m", s, e)
            df = df.reset_index().rename(columns={"t": "ts"})
            df = df[["ts", "open", "high", "low", "close", "volume"]]
            chunks.append(df)
        bars = pd.concat(chunks, ignore_index=True)
        if bars["ts"].dt.tz is None:
            bars["ts"] = bars["ts"].dt.tz_localize("UTC")
        bars = bars.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
        bars.to_parquet(nq_parquet, index=False)
        print(f"[nq] cached {len(bars):,} bars -> {nq_parquet}", flush=True)

    local = bars.set_index("ts")[["open", "high", "low", "close", "volume"]]
    local.index = local.index.tz_convert("America/New_York")
    local = local[(local.index >= pd.Timestamp(start, tz="America/New_York"))
                  & (local.index < pd.Timestamp(end, tz="America/New_York"))]
    ohlc = local.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return ohlc.dropna(subset=["open", "high", "low", "close"])


def adverse_fill(price: float, action: str, slippage_ticks: int) -> float:
    slip = slippage_ticks * TICK_SIZE
    return price + slip if action == "entry" else price - slip


def stop_loss_per_contract(entry: float, stop: float) -> float:
    exit_slippage = BASE_SLIPPAGE_TICKS * TICK_SIZE
    return ((entry - stop) + exit_slippage) * POINT_VALUE + 2 * COMMISSION_PER_SIDE


def simulate(bars: pd.DataFrame, target_r: float, sizing: str = "fixed_1", risk_budget: float = RISK_BUDGET_USD) -> pd.DataFrame:
    records: list[dict] = []
    for session_date, day in bars.groupby(bars.index.date):
        if session_date in FOMC_DATES:
            continue  # skip: FOMC statement at 14:00 ET sits right at our time-exit
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
        entry_bar = day.iloc[trigger_idx + 1]
        entry_time = day.index[trigger_idx + 1]
        if entry_time.time() >= pd.Timestamp("14:00").time():
            continue
        entry = adverse_fill(float(entry_bar["open"]), "entry", BASE_SLIPPAGE_TICKS)
        stop = range_low
        if entry <= stop:
            continue
        target = entry + target_r * (entry - stop)
        if sizing == "fixed_1":
            qty = FIXED_CONTRACTS
        else:  # equal-dollar-risk: fewer contracts on wide-range days, more on quiet days
            per_contract_risk = stop_loss_per_contract(entry, stop)
            qty = max(0, int(risk_budget // per_contract_risk))
            if qty == 0:
                continue

        holding = day.loc[entry_time:]
        exit_time = exit_price = exit_reason = None
        for ts, bar in holding.iterrows():
            if ts.time() >= pd.Timestamp("14:00").time():
                exit_time, exit_price, exit_reason = ts, float(bar["open"]), "time"
                break
            hit_stop = float(bar["low"]) <= stop
            hit_target = float(bar["high"]) >= target
            if hit_stop or hit_target:
                raw_exit = stop if hit_stop else target
                exit_time, exit_price, exit_reason = ts, raw_exit, ("stop" if hit_stop else "target")
                break
        if exit_time is None:
            continue
        filled_exit = adverse_fill(exit_price, "exit", BASE_SLIPPAGE_TICKS)
        gross_pnl = (filled_exit - entry) * POINT_VALUE * qty
        net_pnl = gross_pnl - 2 * COMMISSION_PER_SIDE * qty
        records.append(dict(session_date=pd.Timestamp(session_date), entry_time=entry_time,
                             exit_time=exit_time, entry=entry, exit=filled_exit, stop=stop,
                             target=target, qty=qty, net_pnl=net_pnl, exit_reason=exit_reason))
    return pd.DataFrame(records)


def metrics(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> dict:
    if trades.empty:
        return dict(trades=0)
    pnl = trades["net_pnl"]
    daily = trades.groupby("session_date")["net_pnl"].sum().reindex(session_dates, fill_value=0.0)
    equity = INITIAL_CAPITAL + pnl.cumsum()
    drawdown = equity - equity.cummax()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_loss = abs(losses.sum())
    return dict(
        trades=len(trades), net_pnl=pnl.sum(), win_rate_pct=(pnl > 0).mean() * 100,
        profit_factor=wins.sum() / gross_loss if gross_loss else np.nan,
        max_drawdown=drawdown.min(),
        sharpe=daily.mean() / daily.std(ddof=1) * np.sqrt(252) if daily.std(ddof=1) else np.nan,
        target_exits=int((trades["exit_reason"] == "target").sum()),
        stop_exits=int((trades["exit_reason"] == "stop").sum()),
        time_exits=int((trades["exit_reason"] == "time").sum()),
    )


def main(start: str = BACKTEST_START, end: str = BACKTEST_END, out_suffix: str = "",
         sizing: str = "fixed_1", target_r_values=TARGET_R_VALUES) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = fetch_nq_5m(start, end)
    session_dates = pd.DatetimeIndex(pd.to_datetime(bars.between_time("09:30", "14:00").index.date).unique())

    results = {}
    for target_r in target_r_values:
        trades = simulate(bars, target_r, sizing=sizing)
        trades.to_csv(OUT_DIR / f"trades_target_{target_r:.1f}r{out_suffix}.csv", index=False)
        results[f"target_{target_r:.1f}r"] = metrics(trades, session_dates)

    report = pd.DataFrame(results).T
    report.index.name = "configuration"
    report.to_csv(OUT_DIR / f"summary{out_suffix}.csv")
    pd.set_option("display.width", 160)
    print(f"NQ ORB sweep, {start} -> {end}, sizing={sizing}")
    print(report.round(2).to_string())
    print(f"\nWrote results to {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=BACKTEST_START)
    p.add_argument("--end", default=BACKTEST_END)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--sizing", default="fixed_1", choices=["fixed_1", "equal_risk"])
    p.add_argument("--target-r", type=float, default=None, help="run a single target_r instead of the full sweep")
    args = p.parse_args()
    target_r_values = [args.target_r] if args.target_r is not None else TARGET_R_VALUES
    main(args.start, args.end, args.out_suffix, sizing=args.sizing, target_r_values=target_r_values)
