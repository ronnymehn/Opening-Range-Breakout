"""Sweep the MNQ vol-normalized risk budget on 2021-2024 IN-SAMPLE data (2020's
COVID-crash outlier excluded this time), then validate the chosen budget once,
unmodified, on the 2025-2026 holdout.

stop_r / vix_cap stay at the untuned baseline (1.0 / None) -- an earlier sweep
of those failed the holdout check, so this only tunes position sizing.
"""
from __future__ import annotations

import pandas as pd

from backtest import compute_metrics, load_sessions_and_vix, run_backtest

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-12-31"
HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-12-31"

RISK_BUDGET_GRID = [500, 1000, 1500, 2000, 2500, 3000, 3556, 4000, 5000, 6000, 8000, 10000]


def run_one(risk_budget, start, end, sessions, vix_series):
    df = run_backtest(
        start=start, end=end, target_r=1.0, stop_r=1.0, vix_cap=None,
        sizing="vol_normalized_micro", risk_budget_usd=risk_budget,
        sessions=sessions, vix_series=vix_series, save=False,
    )
    if df.empty:
        return None
    m = compute_metrics(df)
    avg_contracts = df["contracts"].mean()
    return dict(
        risk_budget=risk_budget, trades=m["n"], win_rate=m["win_rate"],
        avg_micros=avg_contracts, avg_nq_equiv=avg_contracts / 10,
        total_net=m["total_net"], profit_factor=m["profit_factor"],
        max_dd=m["max_dd"], sharpe=m["sharpe"],
        dd_over_profit=abs(m["max_dd"]) / m["total_net"] if m["total_net"] > 0 else float("inf"),
    )


def main() -> None:
    sessions, vix_series = load_sessions_and_vix()

    print(f"### IN-SAMPLE sweep: MNQ vol-normalized risk budget, {IN_SAMPLE_START} -> {IN_SAMPLE_END} ###")
    rows = [run_one(rb, IN_SAMPLE_START, IN_SAMPLE_END, sessions, vix_series) for rb in RISK_BUDGET_GRID]
    result = pd.DataFrame([r for r in rows if r is not None]).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(result.to_string(index=False))

    best_budget = result.iloc[0]["risk_budget"]
    print(f"\nBest by Sharpe (in-sample): risk_budget_usd={best_budget}")

    print(f"\n### HOLDOUT check (no re-tuning): risk_budget_usd={best_budget}, {HOLDOUT_START} -> {HOLDOUT_END} ###")
    holdout = run_one(best_budget, HOLDOUT_START, HOLDOUT_END, sessions, vix_series)
    print(pd.Series(holdout).to_string())

    print(f"\n### For reference: fixed 1 NQ contract on the same two windows ###")
    for label, start, end in [("in-sample 2021-2024", IN_SAMPLE_START, IN_SAMPLE_END),
                               ("holdout 2025-2026", HOLDOUT_START, HOLDOUT_END)]:
        df = run_backtest(start=start, end=end, target_r=1.0, stop_r=1.0, vix_cap=None,
                           sizing="fixed_1_nq", sessions=sessions, vix_series=vix_series, save=False)
        m = compute_metrics(df)
        print(f"{label}: net=${m['total_net']:,.0f}  max_dd=${m['max_dd']:,.0f}  sharpe={m['sharpe']:.2f}")


if __name__ == "__main__":
    main()
