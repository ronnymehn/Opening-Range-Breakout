"""Opening-range-breakout backtest engine shared by the MNQ and NQ runs.

The default StrategyConfig reproduces the original reference ORB: a
09:30-10:00 America/New_York opening range built from six left-labelled
5-minute bars; the first completed bar closing beyond the range enters at the
following bar's open; stop at the opposite side of the range, target
= entry + target_r * risk, time exit at the 14:00 bar's open, stop-first when
stop and target fall in the same bar, FOMC days skipped.

Differences from the original scripts (all covered by tests/test_engine.py):
  - Sessions without a bar stamped exactly at `time_exit` (exchange holidays
    and early closes, which halt around 13:00 ET) are not traded. Previously
    their "time" exit filled at the 18:00 ET reopen, and early-close trades
    that missed the target were silently dropped.
  - A stop that gaps through fills at the bar open, not the stop price.
  - Equal-risk (orb_risk_target) contracts are sized from the trigger-bar
    close, which is known when the order is placed, not from the next bar's
    open (`size_from="fill"` restores the old behaviour).
  - Realized volatility ignores returns around quarterly contract rolls,
    because the continuous series is (very likely) not back-adjusted.
Everything else (short side, alternative stops, filters, later exits) is
opt-in and off by default.
"""
from __future__ import annotations

import datetime
import json
import time as _time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fomc_dates import FOMC_DATES

ET = "America/New_York"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BAR_MINUTES = 5


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    symbol: str
    point_value: float
    tick_size: float
    commission_per_side: float
    cache_file: str


MNQ = InstrumentSpec("MNQ", "MNQ.V.0", 2.0, 0.25, 0.62, "mnq_1m.parquet")
NQ = InstrumentSpec("NQ", "NQ.C.0", 20.0, 0.25, 2.30, "nq_1m.parquet")
INSTRUMENTS = {"MNQ": MNQ, "NQ": NQ}

SIZINGS = ("fixed", "fixed_matched_warmup", "orb_risk_target", "harvey_target_vol", "moreira_muir")
VOL_SIZINGS = ("fixed_matched_warmup", "harvey_target_vol", "moreira_muir")


@dataclass(frozen=True)
class StrategyConfig:
    target_r: float | None = 1.0          # None = no target: stop or time exit only
    sizing: str = "fixed"
    fixed_qty: int = 5                    # also the base size for the vol-target overlays
    risk_budget: float = 1_000.0          # $ per trade for orb_risk_target
    size_from: str = "trigger_close"      # orb_risk_target: "trigger_close" or "fill"
    slippage_ticks: int = 1               # entry, target and time exits
    stop_slippage_ticks: int | None = None  # None -> slippage_ticks
    commission_per_side: float | None = None  # None -> instrument default
    or_start: str = "09:30"
    or_bars: int = 6
    time_exit: str = "14:00"
    last_entry: str | None = None         # entry bar must open before this; None -> time_exit
    skip_dates: frozenset = field(default=FOMC_DATES)
    require_full_session: bool = True
    side: str = "long"                    # "long", "short" or "both" (first trigger of the day)
    stop_mode: str = "or_opposite"        # "or_opposite", "or_mid" or "atr_frac"
    stop_atr_frac: float = 0.5
    atr_lookback: int = 14
    or_width_atr_range: tuple[float, float] | None = None
    exits_active_on_entry_bar: str = "both"  # "both", "stop_only" or "none"
    vol_lookback: int = 20
    target_annual_vol: float = 0.20
    min_qty: int = 1
    max_qty: int = 10

    def __post_init__(self):
        checks = {
            "sizing": (self.sizing, SIZINGS),
            "size_from": (self.size_from, ("trigger_close", "fill")),
            "side": (self.side, ("long", "short", "both")),
            "stop_mode": (self.stop_mode, ("or_opposite", "or_mid", "atr_frac")),
            "exits_active_on_entry_bar": (self.exits_active_on_entry_bar, ("both", "stop_only", "none")),
        }
        for name, (value, allowed) in checks.items():
            if value not in allowed:
                raise ValueError(f"{name}={value!r}; expected one of {allowed}")

    def stop_slip(self) -> int:
        return self.slippage_ticks if self.stop_slippage_ticks is None else self.stop_slippage_ticks

    def commission(self, spec: InstrumentSpec) -> float:
        return spec.commission_per_side if self.commission_per_side is None else self.commission_per_side

    def with_(self, **changes) -> "StrategyConfig":
        return replace(self, **changes)


TRADE_COLUMNS = [
    "session_date", "side", "entry_time", "exit_time", "trigger_time", "trigger_close",
    "entry", "exit", "stop", "target", "qty", "target_r", "estimated_annual_vol",
    "range_points", "risk_points", "planned_risk_per_contract", "planned_risk_usd",
    "gross_pnl", "net_pnl", "exit_reason", "mfe_usd", "mae_usd",
]


# --------------------------------------------------------------------------- data

def _et_midnight_ms(date) -> int:
    return int(pd.Timestamp(pd.Timestamp(date).date(), tz=ET).tz_convert("UTC").timestamp() * 1000)


def _quantpad_fetcher():
    import quantpad_data as qpd  # private dependency, only needed to download bars

    return qpd.get_bars


def _normalize_1m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index().rename(columns={"t": "ts", "index": "ts"})
    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    df["ts"] = df["ts"].dt.tz_convert("UTC")
    return df


def _fetch_range(fetcher, symbol: str, start_ms: int, end_ms: int) -> list[pd.DataFrame]:
    chunk_ms = 366 * 86_400_000
    chunks = []
    for s in range(start_ms, end_ms, chunk_ms):
        e = min(s + chunk_ms, end_ms)
        print(f"  fetching {symbol} {pd.Timestamp(s, unit='ms', tz='UTC')} .. "
              f"{pd.Timestamp(e, unit='ms', tz='UTC')}", flush=True)
        chunks.append(_normalize_1m(fetcher(symbol, "1m", s, e)))
    return chunks


def load_1m_bars(spec: InstrumentSpec, start, end, *, fetcher=None, cache_dir: Path = DATA_DIR,
                 now_ms: int | None = None) -> pd.DataFrame:
    """1-minute bars for [start, end) (ET dates), cached in `cache_dir`.

    The covered range is recorded in a sidecar `<cache>.meta.json`, so a
    request for data that cannot exist yet (an end date in the future) is
    clamped to now instead of forcing a re-download. Only the missing leading
    or trailing piece is fetched, and the cache keeps the union of every
    range downloaded."""
    now_ms = int(_time.time() * 1000) if now_ms is None else now_ms
    start_ms, end_ms = _et_midnight_ms(start), min(_et_midnight_ms(end), now_ms)
    path = Path(cache_dir) / spec.cache_file
    meta_path = path.with_name(path.name + ".meta.json")

    cached, coverage = None, None
    if path.exists():
        cached = pd.read_parquet(path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if meta and meta.get("symbol") == spec.symbol:
            coverage = (meta["start_ms"], meta["end_ms"])
        elif meta is None and not cached.empty:  # cache written by the old scripts
            coverage = (int(cached["ts"].min().timestamp() * 1000),
                        int(cached["ts"].max().timestamp() * 1000) + 60_000)
        else:
            cached = None

    if coverage is None:
        missing = [(start_ms, end_ms)]
    else:
        missing = [(start_ms, coverage[0])] if start_ms < coverage[0] else []
        if end_ms > coverage[1]:
            missing.append((coverage[1], end_ms))

    if missing:
        fetcher = fetcher or _quantpad_fetcher()
        frames = [] if cached is None else [cached]
        for s, e in missing:
            frames.extend(_fetch_range(fetcher, spec.symbol, s, e))
        cached = (pd.concat(frames, ignore_index=True).drop_duplicates(subset="ts")
                  .sort_values("ts").reset_index(drop=True))
        coverage = (start_ms, end_ms) if coverage is None else (min(start_ms, coverage[0]),
                                                                max(end_ms, coverage[1]))
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(path, index=False)
        meta_path.write_text(json.dumps({"symbol": spec.symbol, "start_ms": coverage[0],
                                         "end_ms": coverage[1]}))
        print(f"[{spec.name}] cached {len(cached):,} bars -> {path}", flush=True)

    lo, hi = pd.Timestamp(start_ms, unit="ms", tz="UTC"), pd.Timestamp(end_ms, unit="ms", tz="UTC")
    return cached[(cached["ts"] >= lo) & (cached["ts"] < hi)].reset_index(drop=True)


def resample_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    local = bars_1m.set_index("ts")[["open", "high", "low", "close", "volume"]]
    local.index = local.index.tz_convert(ET)
    ohlc = local.resample(f"{BAR_MINUTES}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return ohlc.dropna(subset=["open", "high", "low", "close"])


def fetch_5m(spec: InstrumentSpec, start, end, *, fetcher=None, cache_dir: Path = DATA_DIR) -> pd.DataFrame:
    bars = resample_5m(load_1m_bars(spec, start, end, fetcher=fetcher, cache_dir=cache_dir))
    if bars.empty:
        raise RuntimeError(f"No {spec.name} bars returned for {start}..{end}.")
    return bars


# ------------------------------------------------------------------ calendars/vol

def cme_roll_dates(years) -> list[datetime.date]:
    """CME equity-index roll convention: the Thursday eight days before the
    third Friday of March, June, September and December."""
    dates = []
    for year in sorted(set(years)):
        for month in (3, 6, 9, 12):
            first = datetime.date(year, month, 1)
            third_friday = first + datetime.timedelta(days=(4 - first.weekday()) % 7 + 14)
            dates.append(third_friday - datetime.timedelta(days=8))
    return dates


def _roll_mask(dates: pd.Index, roll_dates, roll_window: int) -> np.ndarray:
    """True for sessions within `roll_window` sessions of a roll date. A volume
    roll does not land exactly on the CME rule date, hence the window."""
    mask = np.zeros(len(dates), dtype=bool)
    ordered = np.array(sorted(dates))
    for roll in roll_dates:
        pos = int(np.searchsorted(ordered, roll))
        if pos >= len(ordered) or (pos == 0 and ordered[0] > roll + datetime.timedelta(days=7)):
            continue
        mask[max(0, pos - roll_window): pos + roll_window + 1] = True
    return mask


def realized_volatility(bars: pd.DataFrame, lookback: int = 20, roll_dates=None,
                        roll_window: int = 2) -> pd.Series:
    """Annualized close-to-close vol from the 16:00 ET bar, known before the
    next session (shifted one session). Returns around contract rolls are
    dropped; pass roll_dates=() to disable that (the original behaviour)."""
    closes = bars.at_time("16:00")["close"].groupby(lambda ts: ts.date()).last()
    returns = closes.pct_change()
    if roll_dates is None:
        roll_dates = cme_roll_dates(d.year for d in closes.index)
    roll_dates = list(roll_dates)
    min_periods = lookback
    if roll_dates:
        returns[_roll_mask(closes.index, roll_dates, roll_window)] = np.nan
        min_periods = max(2, lookback - (2 * roll_window + 1))
    annual_vol = returns.rolling(lookback, min_periods=min_periods).std(ddof=1) * np.sqrt(252)
    return annual_vol.shift(1)


def daily_atr(bars: pd.DataFrame, lookback: int = 14, roll_dates=None, roll_window: int = 2) -> pd.Series:
    """ATR in points from RTH (09:30-16:00 ET) bars, shifted so each session
    only sees prior sessions. Around rolls the true range falls back to
    high - low so the contract switch is not counted as a move."""
    rth = bars.between_time("09:30", "15:55")
    by_day = rth.groupby(lambda ts: ts.date())
    daily = pd.DataFrame({"high": by_day["high"].max(), "low": by_day["low"].min(),
                          "close": by_day["close"].last()})
    prev_close = daily["close"].shift(1)
    true_range = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev_close).abs(),
                            (daily["low"] - prev_close).abs()], axis=1).max(axis=1)
    if roll_dates is None:
        roll_dates = cme_roll_dates(d.year for d in daily.index)
    roll_dates = list(roll_dates)
    if roll_dates:
        rolled = _roll_mask(daily.index, roll_dates, roll_window)
        true_range[rolled] = (daily["high"] - daily["low"])[rolled]
    return true_range.rolling(lookback).mean().shift(1)


def session_dates(bars: pd.DataFrame, cfg: StrategyConfig) -> pd.DatetimeIndex:
    """Denominator for daily statistics: sessions with both the opening-range
    start bar and the time-exit bar. Skipped days (FOMC etc.) stay in as flat
    days; holidays and early closes drop out."""
    times = pd.Series(bars.index.time, index=bars.index)
    dates = pd.Series(bars.index.date, index=bars.index)
    has_exit = set(dates[times == _time_of(cfg.time_exit)])
    has_open = set(dates[times == _time_of(cfg.or_start)])
    if not cfg.require_full_session:
        has_exit = set(dates[(times >= _time_of(cfg.or_start)) & (times <= _time_of(cfg.time_exit))])
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in has_exit & has_open))


# ---------------------------------------------------------------------- simulation

def _time_of(hhmm: str) -> datetime.time:
    return datetime.datetime.strptime(hhmm, "%H:%M").time()


def _add_minutes(hhmm: str, minutes: int) -> str:
    base = datetime.datetime.combine(datetime.date(2000, 1, 3), _time_of(hhmm))
    return (base + datetime.timedelta(minutes=minutes)).strftime("%H:%M")


def adverse_fill(price: float, direction: int, action: str, slippage_ticks: int, tick_size: float) -> float:
    """Slippage always costs: buys fill higher, sells fill lower."""
    slip = slippage_ticks * tick_size
    buying = (direction == 1) == (action == "entry")
    return price + slip if buying else price - slip


def risk_per_contract(ref_price: float, stop: float, direction: int, spec: InstrumentSpec,
                      cfg: StrategyConfig) -> float:
    """Dollar loss per contract if stopped from `ref_price`, incl. stop
    slippage and round-trip commission."""
    return ((direction * (ref_price - stop) + cfg.stop_slip() * spec.tick_size) * spec.point_value
            + 2 * cfg.commission(spec))


def contracts_for_trade(cfg: StrategyConfig, per_contract_risk: float, annual_vol: float) -> int:
    if cfg.sizing == "fixed":
        return cfg.fixed_qty
    if cfg.sizing == "fixed_matched_warmup":
        return cfg.fixed_qty if annual_vol is not None and np.isfinite(annual_vol) else 0
    if cfg.sizing == "orb_risk_target":
        return max(0, int(np.floor(cfg.risk_budget / per_contract_risk))) if per_contract_risk > 0 else 0
    if annual_vol is None or not np.isfinite(annual_vol) or annual_vol <= 0:
        return 0
    exponent = 1 if cfg.sizing == "harvey_target_vol" else 2
    scale = (cfg.target_annual_vol / annual_vol) ** exponent
    return int(np.clip(np.floor(cfg.fixed_qty * scale), cfg.min_qty, cfg.max_qty))


def _round_stop(price: float, direction: int, tick: float) -> float:
    """Round a computed stop to the tick, away from the entry."""
    return (np.floor(price / tick) if direction == 1 else np.ceil(price / tick)) * tick


def simulate_day(day: pd.DataFrame, spec: InstrumentSpec, cfg: StrategyConfig,
                 annual_vol: float = np.nan, atr: float = np.nan) -> dict | None:
    """Trade one calendar date of 5-minute bars; None when no trade is taken."""
    t_exit = _time_of(cfg.time_exit)
    last_entry = _time_of(cfg.last_entry or cfg.time_exit)
    times = day.index.time
    if cfg.require_full_session and not (times == t_exit).any():
        return None
    opening = day.between_time(cfg.or_start, _add_minutes(cfg.or_start, BAR_MINUTES * (cfg.or_bars - 1)))
    trade_window = day.between_time(_add_minutes(cfg.or_start, BAR_MINUTES * cfg.or_bars), cfg.time_exit)
    if len(opening) != cfg.or_bars or trade_window.empty:
        return None
    range_high, range_low = float(opening["high"].max()), float(opening["low"].min())

    if cfg.or_width_atr_range is not None:
        if not np.isfinite(atr) or atr <= 0:
            return None
        lo, hi = cfg.or_width_atr_range
        if not lo <= (range_high - range_low) / atr <= hi:
            return None

    closes = trade_window["close"].to_numpy()
    candidates = []
    if cfg.side in ("long", "both"):
        hits = np.flatnonzero(closes > range_high)
        if hits.size:
            candidates.append((hits[0], 1))
    if cfg.side in ("short", "both"):
        hits = np.flatnonzero(closes < range_low)
        if hits.size:
            candidates.append((hits[0], -1))
    if not candidates:
        return None
    trigger_pos, d = min(candidates)
    trigger_time = trade_window.index[trigger_pos]
    trigger_close = float(closes[trigger_pos])
    trigger_idx = day.index.get_loc(trigger_time)
    if trigger_idx + 1 >= len(day):
        return None
    entry_time = day.index[trigger_idx + 1]
    if entry_time.time() >= last_entry:
        return None
    entry = adverse_fill(float(day.iloc[trigger_idx + 1]["open"]), d, "entry", cfg.slippage_ticks, spec.tick_size)

    if cfg.stop_mode == "or_opposite":
        stop = range_low if d == 1 else range_high
    elif cfg.stop_mode == "or_mid":
        stop = _round_stop((range_high + range_low) / 2, d, spec.tick_size)
    else:
        if not np.isfinite(atr) or atr <= 0:
            return None
        stop = _round_stop(trigger_close - d * cfg.stop_atr_frac * atr, d, spec.tick_size)
    risk_points = d * (entry - stop)
    if risk_points <= 0:
        return None
    target = entry + d * cfg.target_r * risk_points if cfg.target_r is not None else np.nan

    size_ref = entry if cfg.size_from == "fill" else adverse_fill(
        trigger_close, d, "entry", cfg.slippage_ticks, spec.tick_size)
    qty = contracts_for_trade(cfg, risk_per_contract(size_ref, stop, d, spec, cfg), annual_vol)
    if qty <= 0:
        return None

    holding = day.loc[entry_time:]
    holding = holding[holding.index.time <= t_exit]
    exit_time = exit_price = exit_reason = None
    exit_slip = cfg.slippage_ticks
    for i, (ts, bar) in enumerate(holding.iterrows()):
        if ts.time() >= t_exit:
            exit_time, exit_price, exit_reason = ts, float(bar["open"]), "time"
            break
        on_entry_bar = i == 0
        check_stop = not on_entry_bar or cfg.exits_active_on_entry_bar in ("both", "stop_only")
        check_target = cfg.target_r is not None and (not on_entry_bar or cfg.exits_active_on_entry_bar == "both")
        hit_stop = check_stop and (bar["low"] <= stop if d == 1 else bar["high"] >= stop)
        hit_target = check_target and (bar["high"] >= target if d == 1 else bar["low"] <= target)
        if hit_stop:  # stop-first when both fall in the same bar
            bar_open = float(bar["open"])
            exit_price = min(bar_open, stop) if d == 1 else max(bar_open, stop)
            exit_time, exit_reason, exit_slip = ts, "stop", cfg.stop_slip()
            break
        if hit_target:
            exit_time, exit_price, exit_reason = ts, target, "target"
            break
    if exit_time is None:  # only reachable with require_full_session=False (data gap / halt)
        exit_time, exit_price, exit_reason = holding.index[-1], float(holding.iloc[-1]["close"]), "session_end"

    filled_exit = adverse_fill(exit_price, d, "exit", exit_slip, spec.tick_size)
    gross_pnl = d * (filled_exit - entry) * spec.point_value * qty
    net_pnl = gross_pnl - 2 * cfg.commission(spec) * qty
    planned_risk = risk_per_contract(entry, stop, d, spec, cfg)
    held = holding.loc[:exit_time]
    if d == 1:
        mfe, mae = float(held["high"].max()) - entry, entry - float(held["low"].min())
    else:
        mfe, mae = entry - float(held["low"].min()), float(held["high"].max()) - entry
    return {
        "session_date": pd.Timestamp(day.index[0].date()), "side": "long" if d == 1 else "short",
        "entry_time": entry_time, "exit_time": exit_time, "trigger_time": trigger_time,
        "trigger_close": trigger_close, "entry": entry, "exit": filled_exit, "stop": stop,
        "target": target, "qty": qty, "target_r": cfg.target_r, "estimated_annual_vol": annual_vol,
        "range_points": range_high - range_low, "risk_points": risk_points,
        "planned_risk_per_contract": planned_risk, "planned_risk_usd": planned_risk * qty,
        "gross_pnl": gross_pnl, "net_pnl": net_pnl, "exit_reason": exit_reason,
        "mfe_usd": max(0.0, mfe) * spec.point_value * qty, "mae_usd": max(0.0, mae) * spec.point_value * qty,
    }


def simulate(bars: pd.DataFrame, spec: InstrumentSpec, cfg: StrategyConfig,
             annual_vol: pd.Series | None = None, atr: pd.Series | None = None) -> pd.DataFrame:
    if cfg.sizing in VOL_SIZINGS and annual_vol is None:
        annual_vol = realized_volatility(bars, cfg.vol_lookback)
    if atr is None and (cfg.stop_mode == "atr_frac" or cfg.or_width_atr_range is not None):
        atr = daily_atr(bars, cfg.atr_lookback)
    records = []
    for session_date, day in bars.groupby(bars.index.date):
        if session_date in cfg.skip_dates:
            continue
        vol = annual_vol.get(session_date, np.nan) if annual_vol is not None else np.nan
        day_atr = atr.get(session_date, np.nan) if atr is not None else np.nan
        record = simulate_day(day, spec, cfg, vol, day_atr)
        if record is not None:
            records.append(record)
    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def simulate_benchmark(bars: pd.DataFrame, spec: InstrumentSpec, cfg: StrategyConfig,
                       entry_time: str = "10:05") -> pd.DataFrame:
    """Always-long control: buy the `entry_time` open and sell the `time_exit`
    open on every valid, non-skipped session, with the same quantity and
    costs. If the ORB cannot beat this, the breakout adds nothing beyond the
    market's intraday drift."""
    valid = set(session_dates(bars, cfg).date)
    records = []
    for session_date, day in bars.groupby(bars.index.date):
        if session_date not in valid or session_date in cfg.skip_dates:
            continue
        entry_bar = day[day.index.time == _time_of(entry_time)]
        exit_bar = day[day.index.time == _time_of(cfg.time_exit)]
        if entry_bar.empty or exit_bar.empty:
            continue
        records.append(_control_trade(day, spec, cfg, entry_bar.index[0], float(entry_bar["open"].iloc[0]),
                                      exit_bar, 1, cfg.fixed_qty))
    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def matched_control(trades: pd.DataFrame, bars: pd.DataFrame, spec: InstrumentSpec,
                    cfg: StrategyConfig) -> pd.DataFrame:
    """Each ORB trade held from its own entry to the time exit with no stop or
    target: isolates what the stop/target add versus the entry signal."""
    days = dict(tuple(bars.groupby(bars.index.date)))
    records = []
    for trade in trades.itertuples():
        day = days.get(trade.session_date.date())
        if day is None:
            continue
        exit_bar = day[day.index.time == _time_of(cfg.time_exit)]
        if exit_bar.empty:
            continue
        d = 1 if trade.side == "long" else -1
        records.append(_control_trade(day, spec, cfg, trade.entry_time, trade.entry, exit_bar, d,
                                      int(trade.qty), entry_is_filled=True))
    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def _control_trade(day, spec, cfg, entry_time, entry_price, exit_bar, d, qty, entry_is_filled=False) -> dict:
    entry = entry_price if entry_is_filled else adverse_fill(entry_price, d, "entry", cfg.slippage_ticks,
                                                              spec.tick_size)
    exit_fill = adverse_fill(float(exit_bar["open"].iloc[0]), d, "exit", cfg.slippage_ticks, spec.tick_size)
    gross = d * (exit_fill - entry) * spec.point_value * qty
    return {
        "session_date": pd.Timestamp(day.index[0].date()), "side": "long" if d == 1 else "short",
        "entry_time": entry_time, "exit_time": exit_bar.index[0], "entry": entry, "exit": exit_fill,
        "qty": qty, "gross_pnl": gross, "net_pnl": gross - 2 * cfg.commission(spec) * qty,
        "exit_reason": "time", "planned_risk_usd": np.nan,
    }


# ------------------------------------------------------------------------ reports

def metrics(trades: pd.DataFrame, sessions: pd.DatetimeIndex, initial_capital: float = 100_000.0) -> dict:
    if trades.empty:
        return {"trades": 0}
    pnl = trades["net_pnl"]
    daily = trades.groupby("session_date")["net_pnl"].sum().reindex(sessions, fill_value=0.0)
    equity = initial_capital + pnl.cumsum()
    drawdown = equity - equity.cummax()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_loss = abs(losses.sum())
    daily_sd = daily.std(ddof=1)
    reasons = trades["exit_reason"].value_counts()
    return {
        "trades": len(trades), "net_pnl": pnl.sum(), "win_rate_pct": (pnl > 0).mean() * 100,
        "profit_factor": wins.sum() / gross_loss if gross_loss else np.nan,
        "expectancy": pnl.mean(), "avg_win": wins.mean(), "avg_loss": losses.mean(),
        "max_drawdown": drawdown.min(),
        "sharpe": daily.mean() / daily_sd * np.sqrt(252) if daily_sd else np.nan,
        "sessions": len(sessions),
        "avg_contracts": trades["qty"].mean(), "median_contracts": trades["qty"].median(),
        "avg_planned_risk": trades["planned_risk_usd"].mean(),
        "min_planned_risk": trades["planned_risk_usd"].min(),
        "max_planned_risk": trades["planned_risk_usd"].max(),
        "target_exits": int(reasons.get("target", 0)), "stop_exits": int(reasons.get("stop", 0)),
        "time_exits": int(reasons.get("time", 0)), "session_end_exits": int(reasons.get("session_end", 0)),
    }


CANONICAL_HEADERS = [
    "Trade #", "Type", "Date/Time", "Signal", "Price USD",
    "Position size (qty)", "Position size (value)", "Net P&L USD",
    "Net P&L %", "Run-up USD", "Run-up %", "Drawdown USD",
    "Drawdown %", "Cumulative P&L USD", "Cumulative P&L %",
]


def export_trade_log(trades: pd.DataFrame, path: Path, spec: InstrumentSpec,
                     initial_capital: float = 100_000.0) -> None:
    """TradingView-style trade list. The % columns use raw continuous-contract
    prices for notional."""
    rows, cumulative = [], 0.0
    for number, trade in enumerate(trades.itertuples(), start=1):
        cumulative += trade.net_pnl
        notional = trade.entry * spec.point_value * trade.qty
        signal = "ORB close above 30m high" if trade.side == "long" else "ORB close below 30m low"
        base = {column: "" for column in CANONICAL_HEADERS}
        entry = base | {
            "Trade #": number, "Type": f"Entry {trade.side}", "Date/Time": str(trade.entry_time),
            "Signal": signal, "Price USD": round(trade.entry, 2),
            "Position size (qty)": trade.qty, "Position size (value)": round(notional, 2),
        }
        exit_row = base | {
            "Trade #": number, "Type": f"Exit {trade.side}", "Date/Time": str(trade.exit_time),
            "Signal": trade.exit_reason, "Price USD": round(trade.exit, 2),
            "Position size (qty)": trade.qty, "Position size (value)": round(notional, 2),
            "Net P&L USD": round(trade.net_pnl, 2),
            "Net P&L %": round(trade.net_pnl / notional * 100, 4),
            "Run-up USD": round(trade.mfe_usd, 2), "Run-up %": round(trade.mfe_usd / notional * 100, 4),
            "Drawdown USD": round(trade.mae_usd, 2), "Drawdown %": round(trade.mae_usd / notional * 100, 4),
            "Cumulative P&L USD": round(cumulative, 2),
            "Cumulative P&L %": round(cumulative / initial_capital * 100, 4),
        }
        rows.extend([entry, exit_row])
    pd.DataFrame(rows, columns=CANONICAL_HEADERS).to_csv(path, index=False)
