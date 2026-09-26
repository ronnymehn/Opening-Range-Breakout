"""Same reference ORB script/sweep as run_reference_orb_sweep.py, but using
the volatility-targeting sizing schemes (harvey_target_vol, moreira_muir)
instead of fixed_5_micros, to see whether vol-scaling position size changes
which target_r wins or improves results. 2021-2024, same costs/data.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_reference_orb_sweep import (  # noqa: E402
    fetch_bars, realized_volatility, simulate, metrics,
    BACKTEST_START, BACKTEST_END, VOL_LOOKBACK_SESSIONS, RISK_BUDGET, INITIAL_CAPITAL,
    BASE_SLIPPAGE_TICKS, TARGET_R_VALUES,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mnq_orb_reference"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = fetch_bars(BACKTEST_START, BACKTEST_END)
    session_dates = pd.DatetimeIndex(
        pd.to_datetime(bars.between_time("09:30", "14:00").index.date).unique()
    )
    vol_20 = realized_volatility(bars, VOL_LOOKBACK_SESSIONS)

    results = {}
    for sizing in ["harvey_target_vol", "moreira_muir"]:
        for target_r in TARGET_R_VALUES:
            name = f"{sizing}_target_{target_r:.1f}r"
            trades = simulate(bars, sizing=sizing, slippage_ticks=BASE_SLIPPAGE_TICKS,
                               risk_budget=RISK_BUDGET, annual_vol=vol_20, target_r=target_r)
            if trades.empty:
                print(f"{name}: no trades (likely no valid vol estimate) -- skipped")
                continue
            results[name] = metrics(trades, session_dates)

    report = pd.DataFrame(results).T
    report.index.name = "configuration"
    report.to_csv(OUT_DIR / "summary_voltarget.csv")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 60)
    print(report.round(2).to_string())
    print(f"\nWrote results to {OUT_DIR / 'summary_voltarget.csv'}")


if __name__ == "__main__":
    main()
