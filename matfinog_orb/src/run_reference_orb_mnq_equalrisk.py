"""MNQ (micro) equal-dollar-risk sizing at the same $2,000/trade budget used
for the NQ equal-risk test, split 2021-2024 vs 2025-2026 -- checks whether
MNQ's finer $2/pt granularity captures more of the trades NQ's coarse $20/pt
sizing was forced to skip (528 of 866 trades were skipped on NQ at this
budget), rather than raising the budget itself.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_reference_orb_sweep import fetch_bars, simulate, metrics  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mnq_orb_reference"
RISK_BUDGET = 2_000.0
TARGET_R = 1.0

PERIODS = {
    "2021-2024 (in-sample)": ("2021-01-01", "2024-12-31"),
    "2025-2026 (holdout)": ("2025-01-01", "2026-12-31"),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, (start, end) in PERIODS.items():
        start_ts, end_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        bars = fetch_bars(start_ts, end_ts)
        session_dates = pd.DatetimeIndex(pd.to_datetime(bars.between_time("09:30", "14:00").index.date).unique())

        trades = simulate(bars, sizing="orb_risk_target", slippage_ticks=1,
                           risk_budget=RISK_BUDGET, annual_vol=pd.Series(dtype=float), target_r=TARGET_R)
        trades.to_csv(OUT_DIR / f"trades_mnq_equalrisk_{label.split()[0]}.csv", index=False)
        results[label] = metrics(trades, session_dates)

    report = pd.DataFrame(results).T
    report.index.name = "period"
    report.to_csv(OUT_DIR / "summary_mnq_equalrisk_split.csv")
    pd.set_option("display.width", 180)
    print(f"MNQ equal-dollar-risk (${RISK_BUDGET:,.0f}/trade), target_r={TARGET_R}")
    print(report.round(2).to_string())


if __name__ == "__main__":
    main()
