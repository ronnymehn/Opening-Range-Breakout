import datetime

import numpy as np
import pandas as pd
import pytest

import orb_engine as eng
from fomc_dates import FOMC_DATES, load_event_dates
from synthetic import make_day, make_days, random_walk_sessions

DATE = "2023-01-04"
# Opening range 101 / 99; first close above 101 on the 10:00 bar; entry at the
# 10:05 open (101.25) + 1 tick = 101.50, risk 2.50 points, 1R target 104.00.
OR = {"09:30": (100.0, 101.0, 99.0, 100.0)}
TRIGGER = {"10:00": (100.0, 101.5, 100.0, 101.25), "10:05": (101.25, 101.5, 101.0, 101.25)}
BASE = OR | TRIGGER
CFG = eng.StrategyConfig(fixed_qty=1, commission_per_side=0.0, target_r=1.0)


def run(bars, **changes):
    return eng.simulate(bars, eng.MNQ, CFG.with_(**changes))


def one(bars, **changes):
    trades = run(bars, **changes)
    assert len(trades) == 1, trades
    return trades.iloc[0]


def t(hhmm, date=DATE):
    return pd.Timestamp(f"{date} {hhmm}", tz=eng.ET)


# ---------------------------------------------------------------- entry & timing

def test_entry_levels_and_time_exit():
    trade = one(make_day(DATE, BASE))
    assert trade.trigger_time == t("10:00") and trade.entry_time == t("10:05")
    assert trade.entry == 101.5 and trade.stop == 99.0 and trade.target == 104.0
    assert trade.exit_reason == "time" and trade.exit_time == t("14:00")
    assert trade.exit == 99.75  # 14:00 open 100 - 1 tick
    assert trade.net_pnl == pytest.approx((99.75 - 101.5) * 2)


def test_later_time_exit():
    trade = one(make_day(DATE, BASE), time_exit="16:00")
    assert trade.exit_time == t("16:00") and trade.exit_reason == "time"


def test_opening_range_is_0930_to_0955_bars_only():
    # A spike in the 09:55 bar belongs to the range; the 10:00 bar does not.
    bars = make_day(DATE, OR | {"09:55": (100.0, 101.2, 100.0, 101.1),
                                "10:00": (101.0, 103.0, 100.0, 101.25)})
    trade = one(bars)
    assert trade.range_points == pytest.approx(101.2 - 99.0)
    assert trade.trigger_time == t("10:00") and trade.entry_time == t("10:05")


def test_0955_close_above_range_cannot_trigger():
    bars = make_day(DATE, OR | {"09:55": (100.0, 101.0, 100.0, 101.0)})
    assert run(bars).empty


def test_trigger_on_1355_bar_is_too_late():
    assert run(make_day(DATE, OR | {"13:55": (100.0, 101.5, 100.0, 101.25)})).empty


def test_last_entry_cutoff():
    bars = make_day(DATE, OR | {"11:00": (100.0, 101.5, 100.0, 101.25)})
    assert len(run(bars)) == 1
    assert run(bars, last_entry="11:00").empty


def test_fomc_day_is_skipped():
    date = "2023-02-01"
    assert datetime.date(2023, 2, 1) in FOMC_DATES
    assert run(make_day(date, BASE)).empty
    assert len(run(make_day(date, BASE), skip_dates=frozenset())) == 1


def test_short_opening_range_is_skipped():
    assert run(make_day(DATE, TRIGGER | {"09:35": (100.0, 101.0, 99.0, 100.0)}, start="09:35")).empty


# ------------------------------------------------------------------------ exits

def test_same_bar_stop_and_target_takes_stop():
    trade = one(make_day(DATE, BASE | {"10:30": (101.5, 104.5, 98.5, 100.0)}))
    assert trade.exit_reason == "stop" and trade.exit == 98.75


def test_target_fill_at_limit_minus_slippage():
    trade = one(make_day(DATE, BASE | {"10:30": (101.5, 104.5, 101.0, 104.0)}))
    assert trade.exit_reason == "target" and trade.exit == 103.75


def test_gap_through_stop_fills_at_open():
    trade = one(make_day(DATE, BASE | {"10:30": (98.0, 98.5, 97.0, 98.0)}))
    assert trade.exit_reason == "stop" and trade.exit == 97.75  # open 98 - 1 tick, not 99 - 1 tick


def test_stop_slippage_ticks_separate_from_entry_slippage():
    trade = one(make_day(DATE, BASE | {"10:30": (101.0, 101.0, 98.5, 99.0)}), stop_slippage_ticks=3)
    assert trade.entry == 101.5 and trade.exit == 98.25
    assert trade.planned_risk_per_contract == pytest.approx((2.5 + 0.75) * 2)


def test_commission_override():
    trade = one(make_day(DATE, BASE), commission_per_side=0.85)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - 1.70)


@pytest.mark.parametrize("mode,expected", [("both", "stop"), ("stop_only", "stop"), ("none", "time")])
def test_stop_on_entry_bar(mode, expected):
    bars = make_day(DATE, BASE | {"10:05": (101.25, 101.5, 98.5, 99.5)})
    assert one(bars, exits_active_on_entry_bar=mode).exit_reason == expected


@pytest.mark.parametrize("mode,expected", [("both", "target"), ("stop_only", "time"), ("none", "time")])
def test_target_on_entry_bar(mode, expected):
    bars = make_day(DATE, BASE | {"10:05": (101.25, 104.5, 101.0, 104.0)})
    assert one(bars, exits_active_on_entry_bar=mode).exit_reason == expected


def test_no_target_variant():
    bars = make_day(DATE, BASE | {"10:30": (101.5, 110.0, 101.0, 109.0)})
    trade = one(bars, target_r=None, time_exit="15:55")
    assert trade.exit_reason == "time" and trade.exit_time == t("15:55") and np.isnan(trade.target)


# ------------------------------------------------------------ holidays/sessions

def test_holiday_halt_is_not_traded():
    # Globex holiday: halts at 13:00 ET, reopens 18:00 -- no 14:00 bar.
    bars = make_days(make_day(DATE, BASE, end="12:55"), make_day(DATE, {}, start="18:00", end="18:30"))
    assert run(bars).empty
    trade = one(bars, require_full_session=False)
    assert trade.exit_reason == "session_end" and trade.exit_time == t("12:55")


def test_early_close_is_not_traded_for_any_target():
    bars = make_day(DATE, BASE, end="13:10")
    for target_r in (0.4, 1.0, 2.0):
        assert run(bars, target_r=target_r).empty


def test_session_dates_exclude_holidays_include_skipped_days():
    bars = make_days(make_day("2023-01-31", BASE), make_day("2023-02-01", BASE),
                     make_day("2023-02-02", BASE, end="12:55"))
    dates = eng.session_dates(bars, CFG)
    assert list(dates.date) == [datetime.date(2023, 1, 31), datetime.date(2023, 2, 1)]
    assert list(eng.session_dates(bars, CFG.with_(target_r=2.0)).date) == list(dates.date)


# ---------------------------------------------------------------------- sizing

def test_equal_risk_sized_from_trigger_close():
    # Entry bar gaps up to 102: sizing must use the trigger close (known when
    # the order is sent), P&L the real fill.
    bars = make_day(DATE, OR | {"10:00": TRIGGER["10:00"], "10:05": (102.0, 102.5, 101.5, 102.0)})
    cfg = dict(sizing="orb_risk_target", risk_budget=55.0, fixed_qty=0)
    by_close = one(bars, **cfg)
    assert by_close.qty == 10  # (101.25 + .25 - 99 + .25) * 2 = 5.50 per contract
    assert by_close.entry == 102.25
    assert one(bars, size_from="fill", **cfg).qty == 7  # (102.25 - 99 + .25) * 2 = 7.00


def test_config_validation():
    with pytest.raises(ValueError):
        eng.StrategyConfig(side="sideways")


# ------------------------------------------------------------ short & variants

def _mirror(day: pd.DataFrame) -> pd.DataFrame:
    out = day.copy()
    out["open"], out["close"] = 200 - day["open"], 200 - day["close"]
    out["high"], out["low"] = 200 - day["low"], 200 - day["high"]
    return out


@pytest.mark.parametrize("extra", [{}, {"10:30": (101.5, 104.5, 101.0, 104.0)},
                                   {"10:30": (98.0, 98.5, 97.0, 98.0)}])
def test_short_mirrors_long(extra):
    long_day = make_day(DATE, BASE | extra)
    long_trade = one(long_day)
    short_trade = one(_mirror(long_day), side="short")
    assert short_trade.side == "short" and short_trade.exit_reason == long_trade.exit_reason
    assert short_trade.entry == pytest.approx(200 - long_trade.entry)
    assert short_trade.stop == pytest.approx(200 - long_trade.stop)
    assert short_trade.net_pnl == pytest.approx(long_trade.net_pnl)


def test_both_sides_takes_first_trigger():
    bars = make_day(DATE, OR | {"10:00": (100.0, 100.0, 98.5, 98.75), "11:00": TRIGGER["10:00"]})
    assert run(bars, side="long").iloc[0].side == "long"
    assert one(bars, side="both").side == "short"


def test_or_mid_stop():
    assert one(make_day(DATE, BASE), stop_mode="or_mid").stop == 100.0


def test_atr_frac_stop_and_width_filter():
    bars = make_day(DATE, BASE)
    atr = pd.Series({datetime.date(2023, 1, 4): 4.0})
    cfg = CFG.with_(stop_mode="atr_frac", stop_atr_frac=0.5)
    trade = eng.simulate(bars, eng.MNQ, cfg, atr=atr).iloc[0]
    assert trade.stop == 99.25  # 101.25 - 0.5 * 4, rounded down to the tick
    width_ok = CFG.with_(or_width_atr_range=(0.4, 1.0))    # width 2 / ATR 4 = 0.5
    width_bad = CFG.with_(or_width_atr_range=(0.6, 2.0))
    assert len(eng.simulate(bars, eng.MNQ, width_ok, atr=atr)) == 1
    assert eng.simulate(bars, eng.MNQ, width_bad, atr=atr).empty


def test_atr_uses_only_prior_sessions():
    bars = random_walk_sessions(n_days=30, seed=1)
    atr = eng.daily_atr(bars, 14, roll_dates=())
    last = bars.index[-1].date()
    changed = bars.copy()
    changed.loc[changed.index.date == last, "high"] += 500
    assert eng.daily_atr(changed, 14, roll_dates=()).loc[last] == atr.loc[last]


# ------------------------------------------------------------------ benchmarks

def test_benchmark_buys_1005_sells_time_exit():
    trades = eng.simulate_benchmark(make_day(DATE, {}), eng.MNQ, CFG)
    assert len(trades) == 1
    assert trades.iloc[0].net_pnl == pytest.approx((99.75 - 100.25) * 2)


def test_matched_control_holds_to_time_exit():
    bars = make_day(DATE, BASE | {"10:30": (101.5, 104.5, 101.0, 104.0)})
    trades = run(bars)
    control = eng.matched_control(trades, bars, eng.MNQ, CFG)
    assert trades.iloc[0].exit_reason == "target"
    assert control.iloc[0].exit_time == t("14:00") and control.iloc[0].entry == trades.iloc[0].entry


# ----------------------------------------------------------------- calendars/vol

def test_fomc_dates_complete():
    for year in range(2020, 2027):
        assert sum(d.year == year for d in FOMC_DATES) == 8, year
    assert {datetime.date(2026, 10, 28), datetime.date(2026, 12, 9), datetime.date(2020, 3, 3)} <= FOMC_DATES


def test_load_event_dates(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("date,event\n2024-03-01,ISM\n2024-03-05,ISM Services\n")
    assert load_event_dates(path) == {datetime.date(2024, 3, 1), datetime.date(2024, 3, 5)}


def test_cme_roll_dates():
    assert eng.cme_roll_dates([2024]) == [datetime.date(2024, 3, 7), datetime.date(2024, 6, 13),
                                          datetime.date(2024, 9, 12), datetime.date(2024, 12, 12)]


def test_roll_step_does_not_change_vol():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2023-01-02", periods=70)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
    roll = datetime.date(2023, 3, 9)
    stepped = np.where(dates.date >= roll, closes * 1.01, closes)

    def build(values):
        return make_days(*[make_day(d, {"16:00": (c, c, c, c)}) for d, c in zip(dates, values)])

    plain = eng.realized_volatility(build(closes), 20, roll_dates=[roll])
    step = eng.realized_volatility(build(stepped), 20, roll_dates=[roll])
    pd.testing.assert_series_equal(plain, step, atol=1e-12)
    unmasked = eng.realized_volatility(build(stepped), 20, roll_dates=())
    assert not np.allclose(unmasked.dropna(), eng.realized_volatility(build(closes), 20, roll_dates=()).dropna())


# ------------------------------------------------------------------------- cache

class FakeFetcher:
    def __init__(self):
        self.calls = []

    def __call__(self, symbol, interval, start_ms, end_ms):
        self.calls.append((start_ms, end_ms))
        index = pd.date_range(pd.Timestamp(start_ms, unit="ms", tz="UTC"),
                              pd.Timestamp(end_ms, unit="ms", tz="UTC"), freq="1h", inclusive="left")
        df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=index)
        df.index.name = "t"
        return df


def test_cache_fetches_only_missing_pieces(tmp_path):
    fetch = FakeFetcher()
    now = int(pd.Timestamp("2025-06-01", tz="UTC").timestamp() * 1000)
    load = lambda s, e: eng.load_1m_bars(eng.MNQ, s, e, fetcher=fetch, cache_dir=tmp_path, now_ms=now)

    first = load("2024-01-01", "2024-03-01")
    assert len(fetch.calls) == 1 and not first.empty
    load("2024-01-15", "2024-02-15")
    assert len(fetch.calls) == 1  # subset: served from cache
    load("2023-12-01", "2024-04-01")
    assert len(fetch.calls) == 3  # only the leading and trailing pieces
    assert fetch.calls[1][1] == eng._et_midnight_ms("2024-01-01")
    assert fetch.calls[2][0] == eng._et_midnight_ms("2024-03-01")
    load("2025-01-01", "2026-12-31")  # future end is clamped to now ...
    calls = len(fetch.calls)
    load("2025-01-01", "2026-12-31")  # ... so asking again does not re-download
    assert len(fetch.calls) == calls
    everything = load("2023-12-01", "2025-06-01")
    assert everything["ts"].min() < pd.Timestamp("2024-01-01", tz="UTC")  # union preserved
