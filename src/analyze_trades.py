"""Offline re-analysis of the trade logs already committed under data/, for
use without market data (no QuantPad access needed).

For each committed configuration it
  1. removes trades on holiday/early-close sessions (detected from the log:
     a "time" exit after the time-exit bar, i.e. at the 18:00 reopen) and on
     FOMC days (the old runs predate the FOMC filter or the 2020 dates);
  2. computes Sharpe, t-stat, bootstrap CIs, PSR and Deflated Sharpe on daily
     P&L over an approximate NYSE session calendar (flat days included);
  3. applies an analytic slippage stress: k extra ticks on entry and exit;
  4. reports PBO (CSCV) across each target_r sweep.
Caveats: early-close trades the old engine silently dropped cannot be
recovered, the gap-through-stop and trigger-close sizing fixes need a re-run,
and the extra-slippage stress ignores the small shift in target levels.

Usage: python src/analyze_trades.py   (writes data/analysis/)
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orb_stats as st  # noqa: E402
from fomc_dates import FOMC_DATES  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "analysis"
ET = "America/New_York"
TICK = 0.25
TARGETS = [round(0.4 + 0.1 * i, 1) for i in range(17)]
REPO_WIDE_TRIALS = 100  # rough count of every variant tried in this repo so far


class NYSECalendar(AbstractHolidayCalendar):
    """Rule-based approximation of NYSE full-day holidays."""
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def trading_sessions(start, end) -> pd.DatetimeIndex:
    """Approximate full sessions: weekdays minus NYSE holidays and the usual
    13:00/13:15 early closes (day after Thanksgiving, Christmas Eve, July 3)."""
    days = pd.bdate_range(start, end)
    holidays = NYSECalendar().holidays(start, end)
    early = [d for d in days if (d.month == 11 and d.weekday() == 4 and 22 <= d.day - 1 <= 28
                                 and (d - pd.Timedelta(days=1)) in holidays)
             or (d.month == 12 and d.day == 24) or (d.month == 7 and d.day == 3)]
    return days.difference(holidays).difference(pd.DatetimeIndex(early))


def load(path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path)
    trades["session_date"] = pd.to_datetime(trades["session_date"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert(ET)
    return trades


def clean(trades: pd.DataFrame, sessions: pd.DatetimeIndex, time_exit=datetime.time(14, 0)) -> tuple[pd.DataFrame, dict]:
    exit_local = trades["exit_time"]
    invalid = ((exit_local.dt.time > time_exit) | (exit_local.dt.date != trades["session_date"].dt.date)
               | ~trades["session_date"].isin(sessions))
    fomc = trades["session_date"].dt.date.isin(FOMC_DATES)
    kept = trades[~invalid & ~fomc]
    return kept, {"invalid_session_trades": int(invalid.sum()), "invalid_session_pnl": trades.loc[invalid, "net_pnl"].sum(),
                  "fomc_trades": int((fomc & ~invalid).sum()), "fomc_pnl": trades.loc[fomc & ~invalid, "net_pnl"].sum()}


def stressed(trades: pd.DataFrame, extra_ticks: int, point_value: float) -> pd.DataFrame:
    out = trades.copy()
    out["net_pnl"] = out["net_pnl"] - 2 * extra_ticks * TICK * point_value * out["qty"]
    return out


def analyze_set(name: str, files: dict[str, Path], point_value: float, start, end,
                n_trials=(17, REPO_WIDE_TRIALS)) -> tuple[pd.DataFrame, float | None]:
    sessions = trading_sessions(start, end)
    rows, daily_clean = {}, {}
    for config, path in files.items():
        raw = load(path)
        raw = raw[(raw["session_date"] >= pd.Timestamp(start)) & (raw["session_date"] <= pd.Timestamp(end))]
        kept, removed = clean(raw, sessions)
        daily_raw = st.daily_pnl(raw, pd.DatetimeIndex(sorted(set(sessions) | set(raw["session_date"]))))
        daily_clean[config] = st.daily_pnl(kept, sessions)
        row = {"trades_raw": len(raw), "trades_clean": len(kept), **removed,
               "net_raw": raw["net_pnl"].sum(), "net_clean": kept["net_pnl"].sum(),
               "sharpe_raw": st.sharpe_t_stat(daily_raw)[0]}
        for k in (1, 2):
            d = st.daily_pnl(stressed(kept, k, point_value), sessions)
            row[f"net_clean_plus{k}tick"] = d.sum()
            row[f"sharpe_clean_plus{k}tick"] = st.sharpe_t_stat(d)[0]
        rows[config] = row
    table = pd.DataFrame(rows).T
    sig = st.significance_table(daily_clean, trial_counts=n_trials)
    table = table.join(sig)
    table.insert(0, "set", name)
    pbo = st.pbo_cscv(pd.concat(daily_clean.values(), axis=1).to_numpy()) if len(files) >= 2 else None
    return table, pbo


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mnq_dir, nq_dir = DATA / "mnq_orb_reference", DATA / "orb_reference_nq"
    sets = [
        ("MNQ fixed 5, 2021-2024 target sweep",
         {f"{r:.1f}r": mnq_dir / f"fixed_5_mnq_target_{r:.1f}r_trades_detail.csv" for r in TARGETS},
         2.0, "2021-01-01", "2024-12-31"),
        ("NQ fixed 1, 2021-2024 target sweep",
         {f"{r:.1f}r": nq_dir / f"trades_target_{r:.1f}r.csv" for r in TARGETS}, 20.0, "2021-01-01", "2024-12-31"),
        ("NQ fixed 1, 2025-2026 (already-viewed holdout) target sweep",
         {f"{r:.1f}r": nq_dir / f"trades_target_{r:.1f}r.csv" for r in TARGETS}, 20.0, "2025-01-01", "2026-12-31"),
        ("MNQ equal-risk $2,000, 1R, 2021-2024",
         {"1.0r": mnq_dir / "trades_mnq_equalrisk_2021-2024.csv"}, 2.0, "2021-01-01", "2024-12-31"),
        ("MNQ equal-risk $2,000, 1R, 2025-2026 (already-viewed holdout)",
         {"1.0r": mnq_dir / "trades_mnq_equalrisk_2025-2026.csv"}, 2.0, "2025-01-01", "2026-12-31"),
    ]
    tables, lines = [], [
        "# Offline re-analysis of committed trade logs", "",
        "_Generated by src/analyze_trades.py from the CSVs under data/ (no market data needed)._", "",
        "Corrections applied: trades on holiday/early-close sessions (whose time exit filled at the 18:00 "
        "reopen) and FOMC-day trades removed. Daily P&L covers an approximate NYSE session calendar, "
        "flat days included. Not corrected (needs a re-run with market data): early-close trades the old "
        "engine silently dropped, gap-through stop fills, trigger-close sizing.", "",
        f"DSR columns deflate for N=17 (the target sweep) and N={REPO_WIDE_TRIALS} (rough count of every "
        "variant tried in this repo); the truth is in between because the targets are highly correlated. "
        "The '+k tick' columns subtract k extra ticks of slippage on both entry and exit.", "",
    ]
    for name, files, pv, start, end in sets:
        missing = [p for p in files.values() if not p.exists()]
        if missing:
            print(f"skip {name}: missing {missing[0].name}")
            continue
        table, pbo = analyze_set(name, files, pv, start, end)
        tables.append(table)
        cols = ["trades_raw", "trades_clean", "invalid_session_trades", "invalid_session_pnl", "fomc_trades",
                "net_raw", "net_clean", "sharpe_raw", "sharpe_ann", "t_stat", "sharpe_ci_iid", "psr_vs_0",
                "dsr_N17", f"dsr_N{REPO_WIDE_TRIALS}", "sharpe_clean_plus1tick", "sharpe_clean_plus2tick"]
        shown = table[cols].rename(columns={"sharpe_ann": "sharpe_clean"})
        shown = shown.apply(lambda c: c if c.name == "sharpe_ci_iid" else pd.to_numeric(c))
        lines += [f"## {name}", "", "```", shown.round(3).to_string(), "```", ""]
        if pbo is not None:
            lines += [f"PBO (CSCV, 16 blocks) across the target sweep: **{pbo:.2f}**", ""]
    pd.concat(tables).to_csv(OUT / "offline_significance.csv")
    (OUT / "offline_significance.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
