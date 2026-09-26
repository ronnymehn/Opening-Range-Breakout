"""Grid-search stop_r (stop tightness) x vix_cap (regime filter) on the
2020-2024 IN-SAMPLE period only, to see whether either knob reduces max
drawdown without giving back the strategy's edge.

2025-2026 is deliberately excluded here -- it stays out-of-sample. Whatever
config looks best on 2020-2024 should be re-checked against 2025-2026 as a
holdout, not re-tuned on it.
"""
from __future__ import annotations

import pandas as pd

from backtest import compute_metrics, load_sessions_and_vix, run_backtest

IN_SAMPLE_START = "2020-01-01"
IN_SAMPLE_END = "2024-12-31"

STOP_R_GRID = [1.0, 0.75, 0.5, 0.35]
VIX_CAP_GRID = [None, 45, 40, 35]


def main() -> None:
    sessions, vix_series = load_sessions_and_vix()

    rows = []
    for stop_r in STOP_R_GRID:
        for vix_cap in VIX_CAP_GRID:
            df = run_backtest(
                start=IN_SAMPLE_START, end=IN_SAMPLE_END,
                target_r=1.0, stop_r=stop_r, vix_cap=vix_cap,
                sessions=sessions, vix_series=vix_series, save=False,
            )
            if df.empty:
                continue
            m = compute_metrics(df)
            rows.append(dict(
                stop_r=stop_r, vix_cap=vix_cap if vix_cap is not None else "none",
                trades=m["n"], win_rate=m["win_rate"], total_net=m["total_net"],
                profit_factor=m["profit_factor"], max_dd=m["max_dd"], sharpe=m["sharpe"],
                dd_over_profit=abs(m["max_dd"]) / m["total_net"] if m["total_net"] > 0 else float("inf"),
            ))

    result = pd.DataFrame(rows).sort_values("max_dd", ascending=False)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print(f"2020-2024 in-sample sweep: stop_r x vix_cap  (baseline = stop_r=1.0, vix_cap=none)")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
