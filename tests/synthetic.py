"""Synthetic 5-minute bar builders for engine tests. Bars are left-labelled
(timestamp = bar open time) in America/New_York, matching what
orb_engine.fetch_5m produces."""
import datetime

import numpy as np
import pandas as pd

ET = "America/New_York"
COLUMNS = ["open", "high", "low", "close", "volume"]


def make_day(date, overrides=None, default=(100.0, 100.25, 99.75, 100.0),
             start="09:30", end="16:55"):
    """One session of 5-minute bars from `start` to `end` inclusive. Every bar is
    `default` (o, h, l, c) unless `overrides` maps "HH:MM" -> (o, h, l, c).
    With the defaults the opening range is 100.25 / 99.75."""
    overrides = overrides or {}
    date = pd.Timestamp(date).date()
    index = pd.date_range(pd.Timestamp(f"{date} {start}", tz=ET),
                          pd.Timestamp(f"{date} {end}", tz=ET), freq="5min")
    rows = []
    for ts in index:
        o, h, l, c = overrides.get(ts.strftime("%H:%M"), default)
        rows.append((o, h, l, c, 100.0))
    return pd.DataFrame(rows, index=index, columns=COLUMNS)


def make_days(*days):
    return pd.concat(days).sort_index()


def random_walk_sessions(n_days=60, seed=7, start_date="2023-01-03", base=15000.0,
                         step_sd=8.0, skip_dates=frozenset()):
    """Weekday sessions 09:30-16:55 ET, each open equal to the prior close (no
    gaps), prices rounded to the 0.25 tick. Dates in `skip_dates` are left out."""
    rng = np.random.default_rng(seed)
    price = base
    frames = []
    day = pd.Timestamp(start_date)
    while len(frames) < n_days:
        if day.weekday() < 5 and day.date() not in skip_dates:
            index = pd.date_range(pd.Timestamp(f"{day.date()} 09:30", tz=ET),
                                  pd.Timestamp(f"{day.date()} 16:55", tz=ET), freq="5min")
            drift = rng.normal(0.0, 1.5)
            rows = []
            for _ in index:
                o = price
                c = round((o + drift + rng.normal(0.0, step_sd)) * 4) / 4
                h = max(o, c) + round(abs(rng.normal(0.0, step_sd / 2)) * 4) / 4
                l = min(o, c) - round(abs(rng.normal(0.0, step_sd / 2)) * 4) / 4
                rows.append((o, h, l, c, float(rng.integers(100, 1000))))
                price = c
            frames.append(pd.DataFrame(rows, index=index, columns=COLUMNS))
        day += datetime.timedelta(days=1)
    return pd.concat(frames)
