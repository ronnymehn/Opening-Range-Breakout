import pandas as pd

import orb_engine as eng
import run_orb
from synthetic import random_walk_sessions


def test_parse_targets():
    assert run_orb.parse_targets("0.4:0.6:0.1") == [0.4, 0.5, 0.6]
    assert run_orb.parse_targets("1.0,none") == [1.0, None]


def test_full_run_writes_report(tmp_path, monkeypatch):
    bars = random_walk_sessions(n_days=80, seed=11)
    monkeypatch.setattr(eng, "fetch_5m", lambda spec, start, end, **kw: bars)
    result = run_orb.main(["--instrument", "MNQ", "--start", "2023-01-03", "--end", "2023-05-01",
                           "--targets", "0.5:1.5:0.5", "--slippage-ticks", "1,2", "--robustness",
                           "--benchmark", "--out-dir", str(tmp_path), "--tag", "test"])
    summary = result["summary"]
    assert {"target_1.0r_slip1", "target_1.0r_slip2", "target_1.0r_allin_cost", "target_1.0r_stopslip3",
            "notarget_exit1400", "notarget_exit1555", "target_1.0r_exit1555",
            "benchmark_long_1005_to_1400", "control_target_1.0r_slip1_no_stop_no_target"} <= set(summary.index)
    # Same session denominator for every 14:00-exit config; 15:55 configs share theirs.
    assert summary.loc["target_0.5r_slip1", "sessions"] == summary.loc["target_1.5r_slip2", "sessions"]
    assert summary.loc["target_1.0r_slip2", "net_pnl"] < summary.loc["target_1.0r_slip1", "net_pnl"]
    assert summary.loc["notarget_exit1400", "target_exits"] == 0
    assert result["pbo"] is not None and 0.0 <= result["pbo"] <= 1.0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Deflated Sharpe" in report and "$0.85/contract/side" in report and "NOT back-adjusted" in report
    sig = pd.read_csv(tmp_path / "significance.csv", index_col=0)
    assert "dsr_N100" in sig.columns
    assert (tmp_path / "target_1.0r_slip1.tvtrades.csv").exists()
