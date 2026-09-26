"""Compare regime characteristics between 2021-2024 (in-sample) and
2025-2026 (holdout) to understand why so many in-sample-optimized results
degraded or inverted on holdout: VIX level, trend/return, realized vol,
and the VIX-vs-ATR relationship (already used once in target_achievability.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orb_vix_runner import fetch_mnq_5m
from target_achievability import fetch_mnq_daily_atr
from backtest import prev_vix_close_lookup
from fetch_data import fetch_vix_daily

PERIODS = {
    "2021-2024 (in-sample)": ("2021-01-01", "2024-12-31"),
    "2025-2026 (holdout)": ("2025-01-01", "2026-12-31"),
}


def analyze_period(start: str, end: str, bars_5m: pd.DataFrame, vix: pd.DataFrame, atr: pd.Series) -> dict:
    vix_sub = vix[(vix["date"] >= start) & (vix["date"] <= end)]
    daily = bars_5m[(bars_5m.index >= pd.Timestamp(start, tz="America/New_York"))
                     & (bars_5m.index <= pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=1))]
    daily_close = daily.between_time("09:30", "16:00")["close"].groupby(lambda ts: ts.date()).last()
    daily_open = daily.between_time("09:30", "09:35")["open"].groupby(lambda ts: ts.date()).first()
    daily_ret = daily_close.pct_change().dropna()

    total_ret = (daily_close.iloc[-1] / daily_close.iloc[0] - 1) * 100
    n_days = len(daily_ret)
    ann_ret = ((1 + total_ret / 100) ** (252 / max(n_days, 1)) - 1) * 100
    ann_vol = daily_ret.std() * np.sqrt(252) * 100

    equity = (1 + daily_ret).cumprod()
    dd = (equity / equity.cummax() - 1) * 100

    atr_sub = atr[(atr.index >= pd.Timestamp(start)) & (atr.index <= pd.Timestamp(end))]

    daily_range = daily.groupby(lambda ts: ts.date()).apply(lambda g: g["high"].max() - g["low"].min())

    return dict(
        vix_mean=vix_sub["vix_close"].mean(), vix_median=vix_sub["vix_close"].median(),
        vix_std=vix_sub["vix_close"].std(), vix_min=vix_sub["vix_close"].min(), vix_max=vix_sub["vix_close"].max(),
        total_return_pct=total_ret, ann_return_pct=ann_ret, ann_realized_vol_pct=ann_vol,
        max_drawdown_pct=dd.min(), pct_up_days=(daily_ret > 0).mean() * 100,
        avg_daily_range_pts=daily_range.mean(),
        atr14_mean=atr_sub.mean() if len(atr_sub) else np.nan,
        n_sessions=n_days,
    )


def main() -> None:
    bars = fetch_mnq_5m("2021-01-01", "2026-12-31")
    vix = fetch_vix_daily()
    vix["date"] = vix["date"].astype(str)
    atr = fetch_mnq_daily_atr("2021-01-01", "2026-12-31")
    atr.index = pd.to_datetime(atr.index)

    rows = {}
    for label, (start, end) in PERIODS.items():
        rows[label] = analyze_period(start, end, bars, vix, atr)

    result = pd.DataFrame(rows).T
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(result.to_string())


if __name__ == "__main__":
    main()
