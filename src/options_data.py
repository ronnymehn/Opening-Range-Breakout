"""QQQ option chain lookups for the iron-condor strategy (roadmap idea #4).

QuantPad's option 'definition' schema (via the SDK, chain-parent symbol
QQQ.OPT) lists every listed contract for a date range: raw OSI symbol,
strike, class (C/P), expiration. Per-contract OHLCV price bars are fetched
with the exact raw_symbol (including its internal padding -- do not strip
it, the API needs it verbatim).
"""
from __future__ import annotations

import pandas as pd
import quantpad_data as qpd

CHAIN_SYMBOL = "QQQ.OPT"


def fetch_chain(date: pd.Timestamp) -> pd.DataFrame:
    """All contracts listed on `date`, with expiration as a plain UTC date
    (the expiration timestamp is a nominal calendar date -- do NOT
    tz_convert it, that shifts it by a day)."""
    start = pd.Timestamp(date).normalize().tz_localize("UTC")
    end = start + pd.Timedelta(days=1)
    s, e = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    chunks = list(qpd.get_ticks(CHAIN_SYMBOL, "definition", s, e,
                                 columns=["raw_symbol", "instrument_class", "strike_price", "expiration"]))
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    df["expiration_date"] = pd.to_datetime(df["expiration"], unit="ns", utc=True).dt.date
    return df[["raw_symbol", "instrument_class", "strike_price", "expiration_date"]]


def nearest_expiration(chain: pd.DataFrame, session_date: pd.Timestamp) -> "pd.Timestamp | None":
    d = pd.Timestamp(session_date).date()
    candidates = sorted(exp for exp in chain["expiration_date"].unique() if exp >= d)
    return candidates[0] if candidates else None


def pick_iron_condor_symbols(
    chain: pd.DataFrame, expiration_date, upper_band: float, lower_band: float, wing_width: float,
) -> "dict | None":
    """Short call/put at the first listed strike beyond the band; long
    call/put (protection) at the first listed strike beyond that by
    wing_width. Returns None if the chain doesn't have enough strikes on
    that expiration to build the full structure."""
    exp_chain = chain[chain["expiration_date"] == expiration_date]
    calls = exp_chain[exp_chain["instrument_class"] == "C"].sort_values("strike_price")
    puts = exp_chain[exp_chain["instrument_class"] == "P"].sort_values("strike_price")
    if calls.empty or puts.empty:
        return None

    call_strikes_above = calls[calls["strike_price"] >= upper_band]
    put_strikes_below = puts[puts["strike_price"] <= lower_band].sort_values("strike_price", ascending=False)
    if call_strikes_above.empty or put_strikes_below.empty:
        return None

    short_call = call_strikes_above.iloc[0]
    short_put = put_strikes_below.iloc[0]

    long_call_candidates = calls[calls["strike_price"] >= short_call["strike_price"] + wing_width]
    long_put_candidates = puts[puts["strike_price"] <= short_put["strike_price"] - wing_width].sort_values(
        "strike_price", ascending=False)
    if long_call_candidates.empty or long_put_candidates.empty:
        return None

    long_call = long_call_candidates.iloc[0]
    long_put = long_put_candidates.iloc[0]

    return dict(
        short_call_symbol=short_call["raw_symbol"], short_call_strike=short_call["strike_price"],
        long_call_symbol=long_call["raw_symbol"], long_call_strike=long_call["strike_price"],
        short_put_symbol=short_put["raw_symbol"], short_put_strike=short_put["strike_price"],
        long_put_symbol=long_put["raw_symbol"], long_put_strike=long_put["strike_price"],
    )


def fetch_day_open_close(raw_symbol: str, session_date: pd.Timestamp) -> "tuple[float, float] | None":
    """Day's official open/close print for one option contract, from the 1d
    OHLCV bar. Returns None if the contract had no trades that day."""
    start = pd.Timestamp(session_date).normalize().tz_localize("UTC")
    end = start + pd.Timedelta(days=1)
    s, e = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    bars = qpd.get_bars(raw_symbol, "1d", s, e)
    if bars.empty:
        return None
    row = bars.iloc[0]
    return float(row["open"]), float(row["close"])
