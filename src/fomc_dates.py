"""FOMC statement dates (2:00 PM ET release, second day of each 2-day
meeting), used to skip taking new ORB trades on days where a Fed rate
decision drops mid-session -- our 14:00 ET time-exit sits right at the
announcement time, so a position could still be open exactly when the
statement/press conference hits, causing an unrepresentative fill.

2020 also includes the 2020-03-03 unscheduled cut, announced at 10:00 ET
(inside the opening range). The 2020-03-15 emergency cut was on a Sunday and
needs no entry.

Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
(2025-2026 checked against it). Add future years from the same page.
"""
import datetime

FOMC_DATES = frozenset({
    # 2020
    datetime.date(2020, 1, 29), datetime.date(2020, 3, 3), datetime.date(2020, 4, 29),
    datetime.date(2020, 6, 10), datetime.date(2020, 7, 29), datetime.date(2020, 9, 16),
    datetime.date(2020, 11, 5), datetime.date(2020, 12, 16),
    # 2021
    datetime.date(2021, 1, 27), datetime.date(2021, 3, 17), datetime.date(2021, 4, 28),
    datetime.date(2021, 6, 16), datetime.date(2021, 7, 28), datetime.date(2021, 9, 22),
    datetime.date(2021, 11, 3), datetime.date(2021, 12, 15),
    # 2022
    datetime.date(2022, 1, 26), datetime.date(2022, 3, 16), datetime.date(2022, 5, 4),
    datetime.date(2022, 6, 15), datetime.date(2022, 7, 27), datetime.date(2022, 9, 21),
    datetime.date(2022, 11, 2), datetime.date(2022, 12, 14),
    # 2023
    datetime.date(2023, 2, 1), datetime.date(2023, 3, 22), datetime.date(2023, 5, 3),
    datetime.date(2023, 6, 14), datetime.date(2023, 7, 26), datetime.date(2023, 9, 20),
    datetime.date(2023, 11, 1), datetime.date(2023, 12, 13),
    # 2024
    datetime.date(2024, 1, 31), datetime.date(2024, 3, 20), datetime.date(2024, 5, 1),
    datetime.date(2024, 6, 12), datetime.date(2024, 7, 31), datetime.date(2024, 9, 18),
    datetime.date(2024, 11, 7), datetime.date(2024, 12, 18),
    # 2025
    datetime.date(2025, 1, 29), datetime.date(2025, 3, 19), datetime.date(2025, 5, 7),
    datetime.date(2025, 6, 18), datetime.date(2025, 7, 30), datetime.date(2025, 9, 17),
    datetime.date(2025, 10, 29), datetime.date(2025, 12, 10),
    # 2026
    datetime.date(2026, 1, 28), datetime.date(2026, 3, 18), datetime.date(2026, 4, 29),
    datetime.date(2026, 6, 17), datetime.date(2026, 7, 29), datetime.date(2026, 9, 16),
    datetime.date(2026, 10, 28), datetime.date(2026, 12, 9),
})


def load_event_dates(path) -> frozenset:
    """Extra skip dates from a CSV with a `date` column (YYYY-MM-DD), e.g. a
    user-verified list of 10:00 ET release days. Other columns are ignored."""
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return frozenset(datetime.date.fromisoformat(row["date"].strip())
                         for row in csv.DictReader(handle) if row.get("date", "").strip())
