"""Runner for the iron condor backtest (src/iron_condor.py). Fetches QQQ
daily spot bars (for the session open) and reuses the VIX cache already
built for the futures strategy. One get_bars/definition call per option leg
per day, so this is much slower than the futures backtest -- keep windows
modest and run long pulls in the background.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import quantpad_data as qpd  # noqa: F401  (import triggers fetch_data's .env loader via package import order)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_data import fetch_vix_daily, _to_ms  # noqa: E402
from backtest import prev_vix_close_lookup, get_prev_vix  # noqa: E402
from iron_condor import run_iron_condor_backtest  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_qqq_daily(start: str, end: str) -> pd.DataFrame:
    s, e = _to_ms(start), _to_ms(end)
    bars = qpd.get_bars("QQQ", "1d", s, e)
    bars = bars.reset_index().rename(columns={"t": "ts"})
    # Daily bar timestamps are already UTC-midnight anchored to the correct
    # trading date (verified: 2023-06-13 00:00:00+00:00 -> that day's open).
    # Do NOT tz_convert -- that shifts the date back a day (UTC midnight is
    # still evening of the prior day in America/New_York). Just strip tz.
    bars["date"] = pd.to_datetime(bars["ts"]).dt.normalize().dt.tz_localize(None)
    bars = bars.set_index("date")[["open", "high", "low", "close", "volume"]]
    return bars


def main(start: str, end: str, wing_width: float, out_csv: str | None, resume: bool = False) -> pd.DataFrame:
    qqq_daily = fetch_qqq_daily(start, end)
    vix = fetch_vix_daily()
    vix_series = prev_vix_close_lookup(vix)

    session_dates = list(qqq_daily.index)
    print(f"{len(session_dates)} sessions to process, {start} -> {end}", flush=True)
    checkpoint = str(DATA_DIR / out_csv) if out_csv else None
    df = run_iron_condor_backtest(session_dates, qqq_daily, vix_series, get_prev_vix,
                                   wing_width=wing_width, progress_every=10, checkpoint_csv=checkpoint,
                                   resume=resume)
    if not df.empty and out_csv:
        df.to_csv(DATA_DIR / out_csv, index=False)
        print(f"saved -> data/{out_csv}")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--wing-width", type=float, default=2.0)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true",
                    help="skip sessions already recorded in --out's checkpoint/progress log from a prior run")
    args = p.parse_args()
    result = main(args.start, args.end, args.wing_width, args.out, resume=args.resume)
    if not result.empty:
        print(result[["date", "qqq_open", "em_points", "entry_credit", "exit_cost", "pnl_usd"]].to_string())
        print(f"\ntotal net P&L: ${result['pnl_usd'].sum():,.2f}  trades: {len(result)}  "
              f"win rate: {(result['pnl_usd'] > 0).mean():.1%}")
