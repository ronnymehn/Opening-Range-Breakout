"""Five-year MNQ opening-range breakout backtest -- user-supplied reference
script, run as-is with two adaptations only:
  1. fetch_bars() uses this project's already-proven get_bars-based MNQ
     fetch (orb_vix_runner.fetch_mnq_5m) instead of the untested
     qpd.Historical().timeseries.get_range(resample=...) call.
  2. OUT_DIR moved from a root-level "backtests/" folder to data/, per this
     project's convention of not writing working files to root.

Everything else (simulate/metrics/sizing/target_r sweep logic) is the
user's script, unchanged.

The signal follows a 09:30-10:00 America/New_York opening range. A completed
five-minute bar closing above the range enters long at the following bar's open.
Stops and targets are evaluated from five-minute OHLC, with stop-first handling
when both levels occur in the same bar. Target = entry + target_r * (entry - stop)
-- i.e. target_r is a multiple of the ORB RANGE itself, not VIX em_points.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))  # main project src/, for orb_vix_runner
from orb_vix_runner import fetch_mnq_5m  # noqa: E402
from fomc_dates import FOMC_DATES  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mnq_orb_reference"
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
BACKTEST_START = pd.Timestamp("2021-01-01", tz="UTC")
BACKTEST_END = pd.Timestamp("2025-01-01", tz="UTC")
TARGET_R_VALUES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
CANONICAL_HEADERS = [
    "Trade #", "Type", "Date/Time", "Signal", "Price USD",
    "Position size (qty)", "Position size (value)", "Net P&L USD",
    "Net P&L %", "Run-up USD", "Run-up %", "Drawdown USD",
    "Drawdown %", "Cumulative P&L USD", "Cumulative P&L %",
]


def fetch_bars(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    bars = fetch_mnq_5m(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if bars.empty:
        raise RuntimeError("No MNQ bars returned for the requested period.")
    return bars


def adverse_fill(price: float, side: str, action: str, slippage_ticks: int) -> float:
    slip = slippage_ticks * TICK_SIZE
    if side != "long":
        raise ValueError("This implementation is long-only.")
    return price + slip if action == "entry" else price - slip


def realized_volatility(bars: pd.DataFrame, lookback: int) -> pd.Series:
    closes = bars.at_time("16:00")["close"].groupby(lambda timestamp: timestamp.date()).last()
    returns = closes.pct_change()
    annual_vol = returns.rolling(lookback).std(ddof=1) * np.sqrt(252)
    return annual_vol.shift(1)


def stop_loss_per_contract(entry: float, stop: float, slippage_ticks: int) -> float:
    exit_slippage = slippage_ticks * TICK_SIZE
    return ((entry - stop) + exit_slippage) * POINT_VALUE + 2 * COMMISSION_PER_SIDE


def contracts_for_trade(
    entry: float,
    stop: float,
    sizing: str,
    slippage_ticks: int,
    risk_budget: float,
    annual_vol: "float | None",
) -> int:
    if sizing == "fixed_5_micros":
        return FIXED_MICRO_CONTRACTS
    if sizing == "fixed_5_micros_matched_warmup":
        return FIXED_MICRO_CONTRACTS if annual_vol is not None and np.isfinite(annual_vol) else 0
    if sizing == "orb_risk_target":
        per_contract_risk = stop_loss_per_contract(entry, stop, slippage_ticks)
        return max(0, int(np.floor(risk_budget / per_contract_risk)))
    if annual_vol is None or not np.isfinite(annual_vol) or annual_vol <= 0:
        return 0
    exponent = 1 if sizing == "harvey_target_vol" else 2
    scale = (TARGET_ANNUAL_VOL / annual_vol) ** exponent
    return int(np.clip(np.floor(FIXED_MICRO_CONTRACTS * scale), MIN_MICRO_CONTRACTS, MAX_MICRO_CONTRACTS))


def simulate(
    bars: pd.DataFrame,
    sizing: str,
    slippage_ticks: int,
    risk_budget: float,
    annual_vol: pd.Series,
    target_r: float,
) -> pd.DataFrame:
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
        entry = adverse_fill(float(entry_bar["open"]), "long", "entry", slippage_ticks)
        stop = range_low
        target = entry + target_r * (entry - stop)
        estimated_vol = annual_vol.get(session_date, np.nan)
        qty = contracts_for_trade(entry, stop, sizing, slippage_ticks, risk_budget, estimated_vol)
        if qty == 0:
            continue

        exit_time = None
        exit_price = None
        exit_reason = None
        holding = day.loc[entry_time:]
        for timestamp, bar in holding.iterrows():
            if timestamp.time() >= pd.Timestamp("14:00").time():
                exit_time, exit_price, exit_reason = timestamp, float(bar["open"]), "time"
                break
            hit_stop = float(bar["low"]) <= stop
            hit_target = float(bar["high"]) >= target
            if hit_stop or hit_target:
                raw_exit = stop if hit_stop else target
                exit_time = timestamp
                exit_price = raw_exit
                exit_reason = "stop" if hit_stop else "target"
                break
        if exit_time is None:
            continue
        filled_exit = adverse_fill(exit_price, "long", "exit", slippage_ticks)
        gross_pnl = (filled_exit - entry) * POINT_VALUE * qty
        net_pnl = gross_pnl - 2 * COMMISSION_PER_SIDE * qty
        planned_risk_per_contract = stop_loss_per_contract(entry, stop, slippage_ticks)
        planned_risk_usd = planned_risk_per_contract * qty
        held = holding.loc[:exit_time]
        mfe_points = max(0.0, float(held["high"].max()) - entry)
        mae_points = max(0.0, entry - float(held["low"].min()))
        records.append({
            "session_date": pd.Timestamp(session_date), "entry_time": entry_time,
            "exit_time": exit_time, "trigger_time": trigger_time, "entry": entry,
            "exit": filled_exit, "stop": stop, "target": target, "qty": qty,
            "target_r": target_r,
            "estimated_annual_vol": estimated_vol,
            "range_points": range_high - range_low, "risk_points": entry - stop,
            "planned_risk_per_contract": planned_risk_per_contract,
            "planned_risk_usd": planned_risk_usd,
            "gross_pnl": gross_pnl, "net_pnl": net_pnl, "exit_reason": exit_reason,
            "mfe_usd": mfe_points * POINT_VALUE * qty, "mae_usd": mae_points * POINT_VALUE * qty,
        })
    return pd.DataFrame(records)


def metrics(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> dict:
    if trades.empty:
        raise RuntimeError("The simulation produced no trades.")
    pnl = trades["net_pnl"]
    daily = trades.groupby("session_date")["net_pnl"].sum()
    daily = daily.reindex(session_dates, fill_value=0.0)
    equity = INITIAL_CAPITAL + pnl.cumsum()
    drawdown = equity - equity.cummax()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_loss = abs(losses.sum())
    return {
        "trades": len(trades), "net_pnl": pnl.sum(), "win_rate_pct": (pnl > 0).mean() * 100,
        "profit_factor": wins.sum() / gross_loss if gross_loss else np.nan,
        "expectancy": pnl.mean(), "avg_win": wins.mean(), "avg_loss": losses.mean(),
        "max_drawdown": drawdown.min(),
        "sharpe": daily.mean() / daily.std(ddof=1) * np.sqrt(252) if daily.std(ddof=1) else np.nan,
        "avg_contracts": trades["qty"].mean(), "median_contracts": trades["qty"].median(),
        "avg_planned_risk": trades["planned_risk_usd"].mean(),
        "min_planned_risk": trades["planned_risk_usd"].min(),
        "max_planned_risk": trades["planned_risk_usd"].max(),
        "target_exits": int((trades["exit_reason"] == "target").sum()),
        "stop_exits": int((trades["exit_reason"] == "stop").sum()),
        "time_exits": int((trades["exit_reason"] == "time").sum()),
    }


def export_trade_log(trades: pd.DataFrame, path: Path, initial_capital: float) -> None:
    rows, cumulative = [], 0.0
    for number, trade in enumerate(trades.itertuples(), start=1):
        cumulative += trade.net_pnl
        notional = trade.entry * POINT_VALUE * trade.qty
        run_up_pct = trade.mfe_usd / notional * 100 if notional else 0.0
        drawdown_pct = trade.mae_usd / notional * 100 if notional else 0.0
        base = {column: "" for column in CANONICAL_HEADERS}
        entry = base | {
            "Trade #": number, "Type": "Entry long", "Date/Time": str(trade.entry_time),
            "Signal": "ORB close above 30m high", "Price USD": round(trade.entry, 2),
            "Position size (qty)": trade.qty, "Position size (value)": round(notional, 2),
        }
        exit_row = base | {
            "Trade #": number, "Type": "Exit long", "Date/Time": str(trade.exit_time),
            "Signal": trade.exit_reason, "Price USD": round(trade.exit, 2),
            "Position size (qty)": trade.qty, "Position size (value)": round(notional, 2),
            "Net P&L USD": round(trade.net_pnl, 2),
            "Net P&L %": round(trade.net_pnl / notional * 100, 4),
            "Run-up USD": round(trade.mfe_usd, 2), "Run-up %": round(run_up_pct, 4),
            "Drawdown USD": round(trade.mae_usd, 2), "Drawdown %": round(drawdown_pct, 4),
            "Cumulative P&L USD": round(cumulative, 2),
            "Cumulative P&L %": round(cumulative / initial_capital * 100, 4),
        }
        rows.extend([entry, exit_row])
    pd.DataFrame(rows, columns=CANONICAL_HEADERS).to_csv(path, index=False)


def save_report(results: dict, start: pd.Timestamp, end: pd.Timestamp) -> None:
    report = pd.DataFrame(results).T
    report.index.name = "configuration"
    report.to_csv(OUT_DIR / "summary.csv")
    lines = [
        "# MNQ 30-Minute Opening Range Breakout", "",
        f"- **Sample**: {start.date()} through {end.date()} (end exclusive).",
        "- **Instrument/data**: MNQ volume-rolled continuous futures, 5-minute bars, back-adjusted.",
        "- **Entry**: First completed 5-minute close above the 09:30-10:00 ET opening-range high; fill at next bar open.",
        "- **Exit**: Opening-range low stop, target-R sweep, or 14:00 ET open; a same-bar stop/target collision uses stop-first.",
        f"- **Costs**: ${COMMISSION_PER_SIDE:.2f}/contract/side plus {BASE_SLIPPAGE_TICKS} tick adverse slippage per side; stress test doubles slippage.",
        f"- **Baseline sizing**: {FIXED_MICRO_CONTRACTS} MNQ contracts. The ORB-risk variant floors whole MNQ contracts to a ${RISK_BUDGET:,.0f} per-trade risk budget (1% of a ${VOL_TARGET_CAPITAL:,.0f} account).",
        "- **Equal-risk calculation**: contracts = floor($1,000 / [((entry − stop) + exit slippage) × $2 + round-trip commissions]). Quantities are whole MNQ contracts, so modeled loss never exceeds $1,000; a trade is skipped if even one MNQ exceeds the budget.",
        f"- **Harvey et al. overlay**: 20-session prior-close realized volatility; contracts = floor(5 × {TARGET_ANNUAL_VOL:.0%} / prior annualized vol), clipped to {MIN_MICRO_CONTRACTS}–{MAX_MICRO_CONTRACTS} MNQ.",
        f"- **Moreira–Muir overlay**: same input and clip, but uses the inverse-variance form: floor(5 × ({TARGET_ANNUAL_VOL:.0%} / prior annualized vol)^2).",
        "- **Sensitivity**: the Harvey overlay uses 16- and 24-session lookbacks around the 20-session specification; neither is selected for performance.",
        f"- **Validation**: Historical in-sample target-R comparison; {len(results)} configurations were evaluated, so no result is a live-performance estimate.",
        "", "## Results", "", "```", report.round(2).to_string(), "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start, end = BACKTEST_START, BACKTEST_END
    bars = fetch_bars(start, end)
    session_dates = pd.DatetimeIndex(
        pd.to_datetime(bars.between_time("09:30", "14:00").index.date).unique()
    )
    vol_20 = realized_volatility(bars, VOL_LOOKBACK_SESSIONS)
    configs = {
        f"fixed_5_mnq_target_{target_r:.1f}r": (
            "fixed_5_micros", BASE_SLIPPAGE_TICKS, RISK_BUDGET, INITIAL_CAPITAL, vol_20, target_r
        )
        for target_r in TARGET_R_VALUES
    }
    configs["fixed_5_mnq_target_1.4r_2x_slippage"] = (
        "fixed_5_micros", BASE_SLIPPAGE_TICKS * 2, RISK_BUDGET, INITIAL_CAPITAL, vol_20, 1.4
    )
    results = {}
    for name, (sizing, slippage, risk_budget, initial_capital, annual_vol, target_r) in configs.items():
        trades = simulate(
            bars, sizing=sizing, slippage_ticks=slippage, risk_budget=risk_budget,
            annual_vol=annual_vol, target_r=target_r,
        )
        trades.to_csv(OUT_DIR / f"{name}_trades_detail.csv", index=False)
        export_trade_log(trades, OUT_DIR / f"{name}.tvtrades.csv", initial_capital)
        results[name] = metrics(trades, session_dates)
    save_report(results, start, end)
    print(pd.DataFrame(results).T.round(2).to_string())
    print(f"\nWrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
