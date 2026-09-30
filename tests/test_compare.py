import pandas as pd
import pytest

import compare_trades as cmp
import orb_engine as eng
import run_orb
from synthetic import make_day, make_days, random_walk_sessions

OR = {"09:30": (100.0, 101.0, 99.0, 100.0)}
TRIGGER = {"10:00": (100.0, 101.5, 100.0, 101.25), "10:05": (101.25, 101.5, 101.0, 101.25)}
CFG = eng.StrategyConfig(fixed_qty=1, target_r=1.0)


@pytest.fixture
def python_trades(tmp_path):
    bars = make_days(make_day("2024-03-04", OR | TRIGGER | {"10:30": (101.5, 101.5, 98.5, 99.0)}),  # stop
                     make_day("2024-03-05", OR | TRIGGER | {"10:30": (101.5, 104.5, 101.0, 104.0)}),  # target
                     make_day("2024-03-06", OR | TRIGGER))  # time exit
    trades = eng.simulate(bars, eng.MNQ, CFG)
    path = tmp_path / "py_trades_detail.csv"
    trades.to_csv(path, index=False)
    return trades, path


def test_own_tradingview_export_matches_exactly(python_trades, tmp_path):
    trades, path = python_trades
    tv = tmp_path / "list.tvtrades.csv"
    eng.export_trade_log(trades, tv, eng.MNQ)
    result = cmp.compare(cmp.load_python(path), cmp.load_other(tv))
    assert list(result["status"]) == ["match"] * 3
    assert (result["entry_offset_min"] == 0).all() and (result["entry_slip_ticks"] == 0).all()


def mc_export(tmp_path, rows):
    path = tmp_path / "mc_trades.csv"
    pd.DataFrame(rows, columns=["#", "Type", "Date", "Time", "Signal", "Price", "Contracts"]).to_csv(path, index=False)
    return path


def test_multicharts_style_export_with_close_stamps_and_slippage(python_trades, tmp_path):
    _, path = python_trades
    # MultiCharts stamps bar-close times (+5 min) and here fills one tick worse on entry.
    other = mc_export(tmp_path, [
        (1, "Buy", "03/04/2024", "10:10", "ORB LE", 101.75, 1), (1, "Sell", "03/04/2024", "10:35", "Stop Loss", 98.75, 1),
        (2, "Buy", "03/05/2024", "10:10", "ORB LE", 101.75, 1), (2, "Sell", "03/05/2024", "10:35", "Profit Target", 103.75, 1),
        (3, "Buy", "03/06/2024", "10:10", "ORB LE", 101.75, 1), (3, "Sell", "03/06/2024", "14:05", "ORB time", 99.75, 1),
    ])
    result = cmp.compare(cmp.load_python(path), cmp.load_other(other))
    assert list(result["status"]) == ["match"] * 3
    assert (result["entry_offset_min"] == 5).all()
    assert (result["entry_slip_ticks"] == 1).all()
    assert result["pnl_diff_usd"].sum() == pytest.approx(-3 * 0.25 * 2)
    assert "entry mean +1.00" in cmp.summarize(result)


def test_detects_missing_extra_and_wrong_exit(python_trades, tmp_path):
    _, path = python_trades
    other = mc_export(tmp_path, [
        (1, "Buy", "03/04/2024", "10:10", "ORB LE", 101.5, 1), (1, "Sell", "03/04/2024", "14:05", "ORB time", 99.75, 1),
        (2, "Buy", "03/06/2024", "10:10", "ORB LE", 101.5, 1), (2, "Sell", "03/06/2024", "14:05", "ORB time", 99.75, 1),
        (3, "Buy", "03/07/2024", "11:10", "ORB LE", 101.5, 1), (3, "Sell", "03/07/2024", "14:05", "ORB time", 99.0, 1),
    ])
    result = cmp.compare(cmp.load_python(path), cmp.load_other(other)).set_index("date")
    by_day = result["status"].to_dict()
    assert by_day[pd.Timestamp("2024-03-04").date()] == "mismatch"
    assert "exit reason" in result.loc[pd.Timestamp("2024-03-04").date(), "issues"]
    assert by_day[pd.Timestamp("2024-03-05").date()] == "missing_in_other"
    assert by_day[pd.Timestamp("2024-03-06").date()] == "match"
    assert by_day[pd.Timestamp("2024-03-07").date()] == "extra_in_other"


def test_one_row_demo_fill_log(python_trades, tmp_path):
    _, path = python_trades
    log = tmp_path / "demo_fills.csv"
    log.write_text("entry_time,entry_price,exit_time,exit_price,qty,exit_reason\n"
                   "2024-03-06 10:05:02,101.75,2024-03-06 14:00:01,99.50,1,time\n")
    result = cmp.compare(cmp.load_python(path), cmp.load_other(log))
    assert len(result) == 1 and result.iloc[0]["status"] == "match"
    assert result.iloc[0]["entry_slip_ticks"] == 1 and result.iloc[0]["exit_slip_ticks"] == 1


def test_main_exit_code(python_trades, tmp_path):
    trades, path = python_trades
    tv = tmp_path / "list.tvtrades.csv"
    eng.export_trade_log(trades, tv, eng.MNQ)
    assert cmp.main(["--python", str(path), "--other", str(tv)]) == 0
    eng.export_trade_log(trades.iloc[:2], tv, eng.MNQ)
    assert cmp.main(["--python", str(path), "--other", str(tv), "--no-clip"]) == 1


# ----------------------------------------------------------------- bars from CSV

def test_load_bars_csv_close_stamped_1m_matches_5m(tmp_path):
    five = make_day("2024-03-04", OR | TRIGGER)
    rows = []
    for ts, bar in five.iterrows():  # expand each 5-min bar into five 1-min bars, close-stamped
        for k in range(5):
            last = k == 4
            rows.append({"Date": (ts + pd.Timedelta(minutes=k + 1)).strftime("%m/%d/%Y"),
                         "Time": (ts + pd.Timedelta(minutes=k + 1)).strftime("%H:%M"),
                         "Open": bar.open if k == 0 else bar.close if k else bar.open,
                         "High": bar.high, "Low": bar.low, "Close": bar.close if last else bar.open if k == 0 else bar.close,
                         "Vol": 20})
    path = tmp_path / "mnq_1m_export.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    loaded = eng.load_bars_csv(path, label="close")
    pd.testing.assert_frame_equal(loaded[["open", "high", "low", "close"]], five[["open", "high", "low", "close"]],
                                  check_freq=False)


def test_run_orb_from_bars_csv(tmp_path):
    bars = random_walk_sessions(n_days=30, seed=3)
    path = tmp_path / "bars.csv"
    stamped = bars.copy()
    stamped.index = stamped.index + pd.Timedelta(minutes=5)  # close-stamped, as MultiCharts exports
    stamped.index = stamped.index.tz_localize(None)
    stamped.to_csv(path, index_label="Date/Time")
    result = run_orb.main(["--bars-csv", str(path), "--start", "2023-01-01", "--end", "2023-03-01",
                           "--targets", "1.0", "--out-dir", str(tmp_path / "out")])
    expected = eng.simulate(bars, eng.MNQ, eng.StrategyConfig(target_r=1.0))
    assert len(result["trades"]["target_1.0r_slip1"]) == len(expected) > 0
