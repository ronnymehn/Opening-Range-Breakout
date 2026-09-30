"""Compare another trade list against the Python backtest, day by day.

Use it to check that the MultiCharts strategy trades exactly like the Python
engine (backtest parity), and later to measure how demo/live fills differ
from what the backtest assumed (slippage).

  python src/compare_trades.py --python data/results/mnq_parity/target_1.0r_slip1_trades_detail.csv \
      --other mc_list_of_trades.csv --point-value 2

--python is a *_trades_detail.csv written by run_orb.py. --other can be:
  - a MultiCharts / TradeStation "List of Trades" CSV export, or a
    TradingView-style list (two rows per trade: entry row, then exit row,
    with Type, Date/Time or Date + Time, Price, optional Signal/Contracts);
  - a one-row-per-trade log with entry/exit time and price columns (for
    example a hand-kept demo fill log: entry_time, entry_price, exit_time,
    exit_price, qty).
Timestamps without a time zone are read as US Eastern.

Platforms stamp trades differently (MultiCharts shows bar-close times, the
Python engine bar-open times), so times are matched within --time-tol-min
and the report shows the typical offset. Prices are compared in ticks;
positive slippage always means worse than the backtest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum() or ch in "#/")


def _find(columns: dict, *must: str, avoid: tuple = ()) -> str | None:
    for key, original in columns.items():
        if all(m in key for m in must) and not any(a in key for a in avoid):
            return original
    return None


def _to_et(values) -> pd.Series:
    stamps = pd.to_datetime(pd.Series(values), format="mixed")
    if stamps.dt.tz is None:
        return stamps.dt.tz_localize(ET)
    return stamps.dt.tz_convert(ET)


def exit_reason_from_signal(signal) -> str:
    s = str(signal).lower()
    if "stop" in s:
        return "stop"
    if "target" in s or "profit" in s:
        return "target"
    if "time" in s or "close" in s or "eod" in s:
        return "time"
    return "unknown"


def load_other(path) -> pd.DataFrame:
    """Normalize a foreign trade list to one row per trade:
    side, entry_time, entry, exit_time, exit, qty, exit_reason."""
    raw = pd.read_csv(path)
    cols = {_norm(c): c for c in raw.columns}

    entry_t = _find(cols, "entry", "time") or _find(cols, "entry", "date")
    exit_t = _find(cols, "exit", "time") or _find(cols, "exit", "date")
    if entry_t and exit_t:  # one row per trade
        entry_p = _find(cols, "entry", "price") or _find(cols, "entry", avoid=("time", "date"))
        exit_p = _find(cols, "exit", "price") or _find(cols, "exit", avoid=("time", "date", "reason"))
        qty = _find(cols, "qty") or _find(cols, "contracts") or _find(cols, "size")
        side = _find(cols, "side") or _find(cols, "direction")
        reason = _find(cols, "reason") or _find(cols, "signal")
        return pd.DataFrame({
            "side": raw[side].str.lower().map(lambda s: "short" if "short" in s or s.startswith("s") else "long")
            if side else "long",
            "entry_time": _to_et(raw[entry_t]).to_numpy(), "entry": raw[entry_p].astype(float).to_numpy(),
            "exit_time": _to_et(raw[exit_t]).to_numpy(), "exit": raw[exit_p].astype(float).to_numpy(),
            "qty": raw[qty].astype(float).to_numpy() if qty else np.nan,
            "exit_reason": raw[reason].map(exit_reason_from_signal).to_numpy() if reason else "unknown",
        })

    # two rows per trade: entry row then exit row
    type_col = _find(cols, "type")
    price_col = _find(cols, "price")
    stamp_col = _find(cols, "date/time") or _find(cols, "datetime")
    date_col, time_col = _find(cols, "date", avoid=("time",)), _find(cols, "time", avoid=("date",))
    if not type_col or not price_col or not (stamp_col or (date_col and time_col)):
        raise ValueError(f"cannot recognise the trade list layout in {path}; columns: {list(raw.columns)}")
    stamps = _to_et(raw[stamp_col] if stamp_col else raw[date_col].astype(str) + " " + raw[time_col].astype(str))
    signal_col = _find(cols, "signal") or _find(cols, "name")
    qty_col = _find(cols, "qty") or _find(cols, "contracts") or _find(cols, "shares")
    number_col = _find(cols, "#") or _find(cols, "tradenumber") or _find(cols, "trade")

    records, current = [], None
    for i, row in raw.iterrows():
        kind = str(row[type_col]).lower()
        is_exit = "exit" in kind or "cover" in kind or (kind.startswith("sell") and "short" not in kind)
        if not is_exit:
            current = {"side": "short" if "short" in kind else "long", "entry_time": stamps.iloc[i],
                       "entry": float(row[price_col]), "qty": float(row[qty_col]) if qty_col else np.nan,
                       "number": row[number_col] if number_col else None}
        elif current is not None:
            records.append({**current, "exit_time": stamps.iloc[i], "exit": float(row[price_col]),
                            "exit_reason": exit_reason_from_signal(row[signal_col]) if signal_col else "unknown"})
            current = None
    return pd.DataFrame(records).drop(columns="number")


def load_python(path) -> pd.DataFrame:
    trades = pd.read_csv(path)
    trades["entry_time"] = _to_et(trades["entry_time"]).to_numpy()
    trades["exit_time"] = _to_et(trades["exit_time"]).to_numpy()
    if "side" not in trades:
        trades["side"] = "long"
    return trades


def compare(python: pd.DataFrame, other: pd.DataFrame, tick: float = 0.25, point_value: float = 2.0,
            time_tol_min: float = 5.0, price_tol_ticks: float = 2.0, clip_to_overlap: bool = True) -> pd.DataFrame:
    """One row per session date present in either list."""
    python = python.assign(date=python["entry_time"].dt.date)
    other = other.assign(date=other["entry_time"].dt.date)
    if clip_to_overlap and not other.empty:
        lo, hi = other["date"].min(), other["date"].max()
        python = python[(python["date"] >= lo) & (python["date"] <= hi)]
    rows = []
    for date in sorted(set(python["date"]) | set(other["date"])):
        p, o = python[python["date"] == date], other[other["date"] == date]
        row = {"date": date, "python_trades": len(p), "other_trades": len(o)}
        if p.empty or o.empty:
            row["status"] = "missing_in_other" if o.empty else "extra_in_other"
            rows.append(row)
            continue
        pt, ot = p.iloc[0], o.iloc[0]
        d = 1 if pt["side"] == "long" else -1
        issues = []
        if len(p) > 1 or len(o) > 1:
            issues.append("multiple trades")
        if ot["side"] != pt["side"]:
            issues.append("side")
        entry_offset = (ot["entry_time"] - pt["entry_time"]).total_seconds() / 60
        exit_offset = (ot["exit_time"] - pt["exit_time"]).total_seconds() / 60
        entry_slip = d * (ot["entry"] - pt["entry"]) / tick       # + = paid more than the backtest
        exit_slip = -d * (ot["exit"] - pt["exit"]) / tick          # + = received less than the backtest
        if abs(entry_offset) > time_tol_min:
            issues.append("entry time")
        if abs(exit_offset) > time_tol_min:
            issues.append("exit time")
        if abs(entry_slip) > price_tol_ticks:
            issues.append("entry price")
        if abs(exit_slip) > price_tol_ticks:
            issues.append("exit price")
        if ot["exit_reason"] != "unknown" and ot["exit_reason"] != pt["exit_reason"]:
            issues.append("exit reason")
        qty = pt["qty"]
        if pd.notna(ot["qty"]) and ot["qty"] != qty:
            issues.append("qty")
        row.update({
            "status": "match" if not issues else "mismatch", "issues": ", ".join(issues),
            "side": pt["side"], "python_entry_time": pt["entry_time"], "other_entry_time": ot["entry_time"],
            "entry_offset_min": entry_offset, "python_entry": pt["entry"], "other_entry": ot["entry"],
            "entry_slip_ticks": entry_slip, "python_exit_time": pt["exit_time"], "other_exit_time": ot["exit_time"],
            "exit_offset_min": exit_offset, "python_exit": pt["exit"], "other_exit": ot["exit"],
            "exit_slip_ticks": exit_slip, "python_reason": pt["exit_reason"], "other_reason": ot["exit_reason"],
            "python_qty": qty, "other_qty": ot["qty"],
            "pnl_diff_usd": -(entry_slip + exit_slip) * tick * point_value * qty,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(result: pd.DataFrame) -> str:
    if result.empty:
        return "No trades to compare."
    counts = result["status"].value_counts()
    both = result[result["status"].isin(["match", "mismatch"])]
    lines = [f"Sessions compared: {len(result)}"]
    lines += [f"  {status}: {counts.get(status, 0)}" for status in
              ("match", "mismatch", "missing_in_other", "extra_in_other")]
    if not both.empty:
        lines += [
            f"Typical time offset (other - python): entry {both['entry_offset_min'].median():+.0f} min, "
            f"exit {both['exit_offset_min'].median():+.0f} min",
            f"Slippage vs backtest (ticks, + = worse): entry mean {both['entry_slip_ticks'].mean():+.2f} "
            f"/ max {both['entry_slip_ticks'].max():+.2f}; exit mean {both['exit_slip_ticks'].mean():+.2f} "
            f"/ max {both['exit_slip_ticks'].max():+.2f}",
            f"Total P&L difference from fills: ${both['pnl_diff_usd'].sum():+,.2f}",
        ]
        issue_counts = both["issues"].str.split(", ").explode().loc[lambda s: s != ""].value_counts()
        if not issue_counts.empty:
            lines.append("Mismatch reasons: " + ", ".join(f"{k} ({v})" for k, v in issue_counts.items()))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--python", required=True, help="*_trades_detail.csv from run_orb.py")
    p.add_argument("--other", required=True, help="MultiCharts/TradeStation export or demo fill log (CSV)")
    p.add_argument("--point-value", type=float, default=2.0, help="MNQ = 2, NQ = 20")
    p.add_argument("--tick", type=float, default=0.25)
    p.add_argument("--time-tol-min", type=float, default=5.0)
    p.add_argument("--price-tol-ticks", type=float, default=2.0)
    p.add_argument("--no-clip", action="store_true", help="compare all dates, not just the other list's range")
    p.add_argument("--out", default=None, help="write the per-session comparison CSV here")
    args = p.parse_args(argv)

    result = compare(load_python(args.python), load_other(args.other), tick=args.tick,
                     point_value=args.point_value, time_tol_min=args.time_tol_min,
                     price_tol_ticks=args.price_tol_ticks, clip_to_overlap=not args.no_clip)
    out = Path(args.out) if args.out else Path(args.other).with_name(Path(args.other).stem + "_comparison.csv")
    result.to_csv(out, index=False)
    print(summarize(result))
    problems = result[result["status"] != "match"]
    if not problems.empty:
        pd.set_option("display.width", 200)
        print("\nFirst differences:")
        print(problems[[c for c in ("date", "status", "issues", "entry_offset_min", "entry_slip_ticks",
                                    "exit_offset_min", "exit_slip_ticks", "python_reason", "other_reason")
                        if c in problems]].head(15).to_string(index=False))
    print(f"\nPer-session detail: {out}")
    return 0 if problems.empty else 1


if __name__ == "__main__":
    sys.exit(main())
