"""Backtest: NQ VIX-implied-expected-move band breakout, 2020-01-01 to 2024-12-31.

Usage:
    source .venv/bin/activate
    export QUANTPAD_API_KEY=qp_live_...   # only needed the first time (to build the cache)
    python src/fetch_data.py              # downloads + caches NQ 1m bars and VIX daily closes
    python src/backtest.py                # runs the backtest and prints/saves results
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import fetch_nq_bars, fetch_vix_daily
from strategy import Trade, simulate_session, simulate_session_ib
from divergence import build_ratio_series, add_trailing_signal, decide_mode

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BACKTEST_START = "2020-01-01"
BACKTEST_END = "2026-12-31"

SESSION_TZ = "America/New_York"
SESSION_START = "09:30"
SESSION_END = "16:00"          # exclusive; last bar included opens at 15:59

NQ_POINT_VALUE = 20.0           # $ per index point, E-mini Nasdaq-100
COMMISSION_PER_TRADE_USD = 4.60 # round-turn commission, adjust to your broker
SLIPPAGE_TICKS_PER_TRADE = 2    # 1 tick on entry + 1 tick on exit
TICK_SIZE = 0.25
SLIPPAGE_USD = SLIPPAGE_TICKS_PER_TRADE * TICK_SIZE * NQ_POINT_VALUE

# Vol-normalized sizing uses Micro NQ (MNQ, 1/10th of NQ) contracts so position
# size can flex in finer steps: trade size shrinks when the day's expected
# move (and therefore stop distance) is unusually large, so $ risk per trade
# stays roughly constant instead of scaling up exactly when whipsaw risk is
# highest -- a smooth adjustment rather than the hard on/off cutoffs
# (tighter stop_r, vix_cap) that overfit the 2020 COVID-crash episode.
MICRO_POINT_VALUE = 2.0
MICRO_TICK_SIZE = 0.25
MICRO_TICK_VALUE = MICRO_TICK_SIZE * MICRO_POINT_VALUE       # $0.50
MICRO_COMMISSION_PER_CONTRACT = 0.62                          # round-turn, adjust to your broker
MICRO_SLIPPAGE_TICKS_PER_TRADE = 2
MICRO_SLIPPAGE_USD_PER_CONTRACT = MICRO_SLIPPAGE_TICKS_PER_TRADE * MICRO_TICK_VALUE

RISK_BUDGET_USD = 2000.0        # $ risked per trade under vol-normalized sizing.
                                 # ASSUMPTION -- set this to whatever fraction of your
                                 # actual account you're willing to risk per trade.
MIN_CONTRACTS = 1                # never size below 1 micro, even on extreme-vol days
MAX_CONTRACTS = 50               # sanity cap on very calm/low-VIX days (= 5 NQ-equivalent)

TARGET_R = 1.0                  # target = entry +/- TARGET_R * expected_move beyond entry
STOP_R = 1.0                     # stop distance = STOP_R * (entry - open); 1.0 = stop at session open.
                                 # A tighter stop_r and/or vix_cap look great on 2020-2024 in-sample
                                 # (see tune_drawdown.py) but FAIL on the 2025-2026 holdout -- net P&L
                                 # drops ~80% there. Both were overfit to the single 2020 COVID-crash
                                 # whipsaw cluster, not a stable regularity. Left at the untuned
                                 # baseline until a more robust fix (e.g. vol-normalized position
                                 # sizing) is validated out-of-sample.
VIX_CAP = None


def build_sessions(nq: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    local = nq.copy()
    local["ts"] = local["ts"].dt.tz_convert(SESSION_TZ)
    local["session_date"] = local["ts"].dt.date
    local["time"] = local["ts"].dt.strftime("%H:%M")
    rth = local[(local["time"] >= SESSION_START) & (local["time"] < SESSION_END)].copy()
    rth = rth.drop(columns=["time"])
    rth = rth.sort_values("ts")

    sessions = {}
    for date, grp in rth.groupby("session_date"):
        sessions[pd.Timestamp(date)] = grp.reset_index(drop=True)
    return sessions


def prev_vix_close_lookup(vix: pd.DataFrame) -> pd.Series:
    v = vix.sort_values("date").reset_index(drop=True)
    v["date"] = pd.to_datetime(v["date"]).dt.normalize()
    return v.set_index("date")["vix_close"]


def get_prev_vix(vix_series: pd.Series, session_date: pd.Timestamp) -> float:
    prior = vix_series[vix_series.index < session_date]
    if prior.empty:
        return float("nan")
    return float(prior.iloc[-1])


def load_sessions_and_vix() -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.Series]:
    nq = fetch_nq_bars()
    vix = fetch_vix_daily()
    return build_sessions(nq), prev_vix_close_lookup(vix)


def apply_position_sizing(
    df: pd.DataFrame, stop_r: float, sizing: str = "fixed_1_nq", risk_budget_usd: float = RISK_BUDGET_USD,
) -> pd.DataFrame:
    """sizing='fixed_1_nq': always 1 E-mini contract (original behavior).
    sizing='vol_normalized_micro': size Micro NQ contracts so that
    risk_budget_usd is roughly the $ risked at the stop, regardless of that
    day's expected move. Sizing is decided from entry_price/open_price/stop_r
    only -- known at entry time, no lookahead into the trade's outcome."""
    df = df.copy()
    if sizing == "fixed_1_nq":
        df["contracts"] = 1
        df["pnl_usd_gross"] = df["pnl_points"] * NQ_POINT_VALUE * df["contracts"]
        df["pnl_usd_net"] = df["pnl_usd_gross"] - COMMISSION_PER_TRADE_USD - SLIPPAGE_USD
    elif sizing == "vol_normalized_micro":
        risk_points = (stop_r * (df["entry_price"] - df["open_price"]).abs()).clip(lower=1e-6)
        raw_contracts = risk_budget_usd / (risk_points * MICRO_POINT_VALUE)
        df["contracts"] = raw_contracts.round().clip(lower=MIN_CONTRACTS, upper=MAX_CONTRACTS).astype(int)
        df["pnl_usd_gross"] = df["pnl_points"] * MICRO_POINT_VALUE * df["contracts"]
        df["pnl_usd_net"] = (df["pnl_usd_gross"]
                              - MICRO_COMMISSION_PER_CONTRACT * df["contracts"]
                              - MICRO_SLIPPAGE_USD_PER_CONTRACT * df["contracts"])
    else:
        raise ValueError(f"unknown sizing mode: {sizing}")
    return df


def run_backtest(
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    target_r: float = TARGET_R,
    stop_r: float = STOP_R,
    vix_cap: float | None = VIX_CAP,
    mode: str = "breakout",
    sizing: str = "fixed_1_nq",
    risk_budget_usd: float = RISK_BUDGET_USD,
    sessions: dict[pd.Timestamp, pd.DataFrame] | None = None,
    vix_series: pd.Series | None = None,
    save: bool = True,
) -> pd.DataFrame:
    if sessions is None or vix_series is None:
        sessions, vix_series = load_sessions_and_vix()

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    trades: list[Trade] = []
    for session_date in sorted(sessions):
        if session_date < start_ts or session_date > end_ts:
            continue
        bars = sessions[session_date]
        if len(bars) < 5:
            continue
        pv = get_prev_vix(vix_series, session_date)
        trade = simulate_session(session_date, bars, pv, target_r=target_r, stop_r=stop_r,
                                  vix_cap=vix_cap, mode=mode)
        if trade is not None:
            trades.append(trade)

    rows = [t.__dict__ for t in trades]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No trades generated.")
        return df

    df["r_multiple"] = df["pnl_points"] / df["em_points"]
    df = apply_position_sizing(df, stop_r=stop_r, sizing=sizing, risk_budget_usd=risk_budget_usd)

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "trades.csv", index=False)
    return df


def run_backtest_switching(
    start: str,
    end: str,
    lookback: int,
    low_thresh: float,
    high_thresh: float,
    default_mode: str | None = "breakout",
    target_r: float = TARGET_R,
    stop_r: float = STOP_R,
    sizing: str = "fixed_1_nq",
    risk_budget_usd: float = RISK_BUDGET_USD,
    sessions: dict[pd.Timestamp, pd.DataFrame] | None = None,
    vix_series: pd.Series | None = None,
    save: bool = False,
) -> pd.DataFrame:
    """Like run_backtest, but the mode (breakout/fade/skip) is chosen per
    session from the trailing implied-vs-realized divergence signal (see
    divergence.py) instead of being fixed for the whole run."""
    if sessions is None or vix_series is None:
        sessions, vix_series = load_sessions_and_vix()

    ratio_df = build_ratio_series(sessions, vix_series, get_prev_vix)
    ratio_df = add_trailing_signal(ratio_df, lookback)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    trades: list[Trade] = []
    for session_date in sorted(sessions):
        if session_date < start_ts or session_date > end_ts:
            continue
        if session_date not in ratio_df.index:
            continue
        bars = sessions[session_date]
        if len(bars) < 5:
            continue
        trailing_ratio = ratio_df.loc[session_date, "trailing_ratio"]
        mode = decide_mode(trailing_ratio, low_thresh, high_thresh, default_mode)
        if mode is None:
            continue
        pv = get_prev_vix(vix_series, session_date)
        trade = simulate_session(session_date, bars, pv, target_r=target_r, stop_r=stop_r,
                                  vix_cap=None, mode=mode)
        if trade is not None:
            trades.append(trade)

    rows = [t.__dict__ for t in trades]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No trades generated.")
        return df

    df["r_multiple"] = df["pnl_points"] / df["em_points"]
    df = apply_position_sizing(df, stop_r=stop_r, sizing=sizing, risk_budget_usd=risk_budget_usd)

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "trades.csv", index=False)
    return df


def run_backtest_ib(
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    ib_minutes: int = 30,
    target_r: float = 1.0,
    sizing: str = "fixed_1_nq",
    sessions: dict[pd.Timestamp, pd.DataFrame] | None = None,
    vix_series: pd.Series | None = None,
    save: bool = False,
) -> pd.DataFrame:
    """Idea #5: Initial Balance breakout, VIX-implied expected move used only
    to size the profit target (never the entry trigger). See
    strategy.simulate_session_ib."""
    if sessions is None or vix_series is None:
        sessions, vix_series = load_sessions_and_vix()

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    trades: list[Trade] = []
    for session_date in sorted(sessions):
        if session_date < start_ts or session_date > end_ts:
            continue
        bars = sessions[session_date]
        if len(bars) < ib_minutes + 5:
            continue
        pv = get_prev_vix(vix_series, session_date)
        trade = simulate_session_ib(session_date, bars, pv, ib_minutes=ib_minutes, target_r=target_r)
        if trade is not None:
            trades.append(trade)

    rows = [t.__dict__ for t in trades]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No trades generated.")
        return df

    df["r_multiple"] = df["pnl_points"] / df["em_points"]
    df = apply_position_sizing(df, stop_r=1.0, sizing=sizing)  # IB stop isn't stop_r-based; only matters for vol_normalized sizing

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "trades_ib.csv", index=False)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Summary stats used by both summarize() and the parameter sweep."""
    n = len(df)
    wins = df[df["pnl_usd_net"] > 0]
    losses = df[df["pnl_usd_net"] <= 0]
    win_rate = len(wins) / n
    total_net = df["pnl_usd_net"].sum()
    avg_net = df["pnl_usd_net"].mean()
    gross_profit = wins["pnl_usd_net"].sum()
    gross_loss = -losses["pnl_usd_net"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    daily = df.set_index("date")["pnl_usd_net"].sort_index()
    equity = daily.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()

    daily_ret = daily  # one trade/day max, so daily pnl series doubles as the return series
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else float("nan")

    return dict(n=n, win_rate=win_rate, total_net=total_net, avg_net=avg_net,
                profit_factor=profit_factor, max_dd=max_dd, sharpe=sharpe, equity=equity)


def summarize(df: pd.DataFrame) -> None:
    if df.empty:
        return

    m = compute_metrics(df)
    n, win_rate, total_net, avg_net = m["n"], m["win_rate"], m["total_net"], m["avg_net"]
    profit_factor, max_dd, sharpe, equity = m["profit_factor"], m["max_dd"], m["sharpe"], m["equity"]

    print("=" * 60)
    print(f"NQ VIX-expected-move band breakout  |  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"target_r={TARGET_R}, stop_r={STOP_R}, vix_cap={VIX_CAP}, NQ pt value=${NQ_POINT_VALUE:.0f}")
    print("=" * 60)
    print(f"Trades:            {n}")
    print(f"Win rate:          {win_rate:.1%}")
    print(f"Total net P&L:     ${total_net:,.2f}")
    print(f"Avg P&L / trade:   ${avg_net:,.2f}")
    print(f"Profit factor:     {profit_factor:.2f}")
    print(f"Max drawdown:      ${max_dd:,.2f}")
    print(f"Sharpe (daily,ann):{sharpe:.2f}")
    print()
    print("By direction:")
    print(df.groupby("direction")["pnl_usd_net"].agg(["count", "sum", "mean"]))
    print()
    print("By exit reason:")
    print(df.groupby("exit_reason")["pnl_usd_net"].agg(["count", "sum", "mean"]))
    print()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    print("By year:")
    print(df.groupby("year")["pnl_usd_net"].agg(["count", "sum", "mean"]))

    equity.to_csv(DATA_DIR / "equity_curve.csv", header=["cum_pnl_usd"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        equity.plot(ax=ax)
        ax.set_title("Equity curve (net, cumulative $)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L ($)")
        fig.tight_layout()
        out = DATA_DIR / "equity_curve.png"
        fig.savefig(out, dpi=150)
        print(f"\nSaved equity curve plot -> {out}")
    except ImportError:
        print("\n(matplotlib not installed -- skipped plot; equity_curve.csv still written)")


if __name__ == "__main__":
    trades_df = run_backtest()
    summarize(trades_df)
