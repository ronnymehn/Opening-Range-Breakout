"""Iron condor strategy (roadmap idea #4): sell the call/put at the
VIX-implied expected-move band, buy further-out wings for defined risk.
Direct expression of "VIX tends to overstate realized vol" -- collect the
premium instead of betting on NQ futures direction.

Entry: at the session's open print, sell 1x short call (nearest listed
strike >= upper band) + 1x short put (nearest listed strike <= lower band),
buy 1x long call/put wing_width further out each for protection, using the
nearest available expiration on/after the session date (QQQ has near-daily
expiries from ~2021 on; 2020 is weekly-only and excluded from this project's
tuning window anyway).

Exit: close the whole structure at the same day's close print (not held to
actual expiration/assignment -- simpler and avoids exercise modeling).

Costs: OPTIONS_COMMISSION_PER_CONTRACT per leg per side (open AND close, so
4 legs x 2 = 8 commission charges per trade). No bid/ask spread modeled --
day-bar open/close prints are used as fill proxies, which is optimistic
(real fills would cross the spread); flagged as a known limitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from options_data import fetch_chain, nearest_expiration, pick_iron_condor_symbols, fetch_day_open_close
from strategy import expected_move_points

CONTRACT_MULTIPLIER = 100.0
OPTIONS_COMMISSION_PER_CONTRACT = 0.65  # per leg, per side (open or close) -- adjust to your broker


@dataclass
class IronCondorTrade:
    date: pd.Timestamp
    qqq_open: float
    prev_vix_close: float
    em_points: float
    upper_band: float
    lower_band: float
    short_call_strike: float
    long_call_strike: float
    short_put_strike: float
    long_put_strike: float
    entry_credit: float      # $ per share collected at entry (before multiplier/commission)
    exit_cost: float         # $ per share paid to close
    pnl_usd: float            # net, after commission, x100 multiplier, 1 contract


def simulate_iron_condor_day(
    session_date: pd.Timestamp, qqq_open: float, prev_vix_close: float, wing_width: float,
) -> "IronCondorTrade | None":
    if pd.isna(prev_vix_close) or qqq_open <= 0:
        return None
    em = expected_move_points(qqq_open, prev_vix_close)
    if em <= 0:
        return None
    upper, lower = qqq_open + em, qqq_open - em

    chain = fetch_chain(session_date)
    if chain.empty:
        return None
    expiration = nearest_expiration(chain, session_date)
    if expiration is None:
        return None
    legs = pick_iron_condor_symbols(chain, expiration, upper, lower, wing_width)
    if legs is None:
        return None

    prices = {}
    for leg in ("short_call", "long_call", "short_put", "long_put"):
        result = fetch_day_open_close(legs[f"{leg}_symbol"], session_date)
        if result is None:
            return None  # a leg had no trades that day -- can't price the structure
        prices[leg] = result  # (open, close)

    entry_credit = ((prices["short_call"][0] - prices["long_call"][0])
                     + (prices["short_put"][0] - prices["long_put"][0]))
    exit_cost = ((prices["short_call"][1] - prices["long_call"][1])
                 + (prices["short_put"][1] - prices["long_put"][1]))

    pnl_per_share = entry_credit - exit_cost
    commission = OPTIONS_COMMISSION_PER_CONTRACT * 4 * 2  # 4 legs, open + close
    pnl_usd = pnl_per_share * CONTRACT_MULTIPLIER - commission

    return IronCondorTrade(
        session_date, qqq_open, prev_vix_close, em, upper, lower,
        legs["short_call_strike"], legs["long_call_strike"],
        legs["short_put_strike"], legs["long_put_strike"],
        entry_credit, exit_cost, pnl_usd,
    )


def _progress_log_path(checkpoint_csv: str) -> Path:
    return Path(checkpoint_csv).with_suffix(".progress.log")


def _load_resume_state(checkpoint_csv: "str | None") -> "tuple[list[IronCondorTrade], set]":
    """Returns (previously-found trades as IronCondorTrade-like rows kept as a
    plain list of dicts, already-processed dates) from a prior run's
    checkpoint + progress log, or ([], set()) if there's nothing to resume."""
    if not checkpoint_csv:
        return [], set()

    prior_trades = []
    cp = Path(checkpoint_csv)
    if cp.exists():
        prior_df = pd.read_csv(cp, parse_dates=["date"])
        prior_trades = prior_df.to_dict("records")

    done_dates = set()
    log_path = _progress_log_path(checkpoint_csv)
    if log_path.exists():
        with open(log_path) as f:
            done_dates = {line.strip() for line in f if line.strip()}
    elif prior_trades:
        # Old-format checkpoint with no progress log (e.g. a run started
        # before resume support existed): the only dates we know for certain
        # were processed are the ones that produced a trade. Skipped/errored
        # dates from that run will be harmlessly redone.
        done_dates = {str(pd.Timestamp(t["date"]).date()) for t in prior_trades}

    return prior_trades, done_dates


def run_iron_condor_backtest(
    session_dates: list[pd.Timestamp],
    qqq_daily: pd.DataFrame,   # indexed by date, column 'open'
    vix_series: pd.Series,
    get_prev_vix,
    wing_width: float = 2.0,
    progress_every: int = 10,
    checkpoint_csv: "str | None" = None,
    resume: bool = False,
) -> pd.DataFrame:
    prior_trade_rows, done_dates = _load_resume_state(checkpoint_csv) if resume else ([], set())
    trades = list(prior_trade_rows)  # dicts from a resumed run, or empty
    log_path = _progress_log_path(checkpoint_csv) if checkpoint_csv else None
    log_file = open(log_path, "a") if log_path else None

    if resume and done_dates:
        remaining = [d for d in session_dates if str(d.date()) not in done_dates]
        print(f"resuming: {len(session_dates) - len(remaining)} sessions already done "
              f"({len(prior_trade_rows)} prior trades kept), {len(remaining)} remaining", flush=True)
        session_dates = remaining

    skipped = 0
    errored = 0
    total = len(session_dates)
    try:
        for i, d in enumerate(session_dates, 1):
            date_str = str(d.date())
            if d not in qqq_daily.index:
                skipped += 1
                if log_file:
                    print(date_str, file=log_file, flush=True)
                continue
            try:
                qqq_open = float(qqq_daily.loc[d, "open"])
                pv = get_prev_vix(vix_series, d)
                t = simulate_iron_condor_day(d, qqq_open, pv, wing_width)
            except Exception as exc:
                # A multi-hour unattended run shouldn't die on one bad/transient
                # API call -- log it, skip the day, keep going.
                errored += 1
                print(f"  [{i}/{total}] {date_str}: ERROR {exc!r}, skipping", flush=True)
                if log_file:
                    print(date_str, file=log_file, flush=True)
                continue
            if t is None:
                skipped += 1
            else:
                trades.append(t.__dict__)
            if log_file:
                print(date_str, file=log_file, flush=True)

            if i % progress_every == 0 or i == total:
                print(f"  [{i}/{total}] {date_str} -- {len(trades)} trades total, "
                      f"{skipped} skipped, {errored} errored this run", flush=True)
                if checkpoint_csv and trades:
                    pd.DataFrame(trades).to_csv(checkpoint_csv, index=False)
    finally:
        if log_file:
            log_file.close()

    print(f"iron condor: {len(trades)} trades total, {skipped} skipped, {errored} errored "
          f"this run (out of {total} sessions processed this run)")
    df = pd.DataFrame(trades)
    if checkpoint_csv and not df.empty:
        df.to_csv(checkpoint_csv, index=False)
    return df
