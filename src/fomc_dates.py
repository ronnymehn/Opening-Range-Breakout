"""FOMC statement dates (2:00 PM ET release, second day of each 2-day
meeting), used to skip taking new ORB trades on days where a Fed rate
decision drops mid-session -- our 14:00 ET time-exit sits right at the
announcement time, so a position could still be open exactly when the
statement/press conference hits, causing an unrepresentative fill.

Confidence: 2021-2024 are historical (high confidence). 2025 dates were
published well in advance by the Fed and should be reliable. 2026 dates
are lower-confidence / may be incomplete -- verify against
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm before
relying on this for anything beyond a backtest.
"""
import datetime

FOMC_DATES = {
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
    # 2026 -- LOWER CONFIDENCE, verify against the Fed's published calendar
    datetime.date(2026, 1, 28), datetime.date(2026, 3, 18), datetime.date(2026, 4, 29),
    datetime.date(2026, 6, 17), datetime.date(2026, 7, 29), datetime.date(2026, 9, 16),
}
