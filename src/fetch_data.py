"""Download NQ 1-minute bars (QuantPad) and VIX daily closes (FRED) and cache locally.

Requires QUANTPAD_API_KEY to be set in the environment (Profile -> API on quantpad.ai).
Run once before backtest.py; re-run only when you want to refresh/extend the cache.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import quantpad_data as qpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: sets os.environ from KEY=VALUE lines, never overwriting
    a value already set in the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
NQ_SYMBOL = "NQ.C.0"          # continuous front-month E-mini Nasdaq-100, calendar roll
START = "2019-12-01"          # a little before 2020 so the first sessions have a prior VIX close
END = "2026-09-25"            # through "today" (2026-09-24) + 1 day, inclusive

NQ_PARQUET = DATA_DIR / "nq_1m.parquet"
VIX_CSV = DATA_DIR / "vix_daily.csv"


def _to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_nq_bars() -> pd.DataFrame:
    if NQ_PARQUET.exists():
        print(f"[nq] using cached {NQ_PARQUET}")
        return pd.read_parquet(NQ_PARQUET)

    if not os.environ.get("QUANTPAD_API_KEY"):
        sys.exit("QUANTPAD_API_KEY is not set. Get a key at quantpad.ai (Profile -> API) and "
                 "export QUANTPAD_API_KEY=qp_live_... before running this script.")

    print(f"[nq] downloading {NQ_SYMBOL} 1m bars {START}..{END} from QuantPad ...")
    start_ms, end_ms = _to_ms(START), _to_ms(END)

    # Pull year by year to keep each request a manageable size.
    chunks = []
    years = pd.date_range(START, END, freq="YS").tolist()
    if not years or years[0] > pd.Timestamp(START):
        years = [pd.Timestamp(START)] + years
    boundaries = years + [pd.Timestamp(END)]

    for i in range(len(boundaries) - 1):
        s = int(boundaries[i].tz_localize("UTC").timestamp() * 1000)
        e = int(boundaries[i + 1].tz_localize("UTC").timestamp() * 1000)
        if e <= start_ms or s >= end_ms:
            continue
        s = max(s, start_ms)
        e = min(e, end_ms)
        print(f"  fetching {boundaries[i].date()}..{boundaries[i + 1].date()}")
        df = qpd.get_bars(NQ_SYMBOL, "1m", s, e)
        # SDK returns a DatetimeIndex named 't' (tz-aware UTC) plus duplicated
        # short (o/h/l/c/v) and long (open/high/low/close/volume) columns.
        df = df.reset_index().rename(columns={"t": "ts"})
        df = df[["ts", "open", "high", "low", "close", "volume"]]
        chunks.append(df)

    bars = pd.concat(chunks, ignore_index=True)
    if not pd.api.types.is_datetime64_any_dtype(bars["ts"]):
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    elif bars["ts"].dt.tz is None:
        bars["ts"] = bars["ts"].dt.tz_localize("UTC")
    bars = bars.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(NQ_PARQUET, index=False)
    print(f"[nq] cached {len(bars):,} bars -> {NQ_PARQUET}")
    return bars


def fetch_vix_daily() -> pd.DataFrame:
    if VIX_CSV.exists():
        print(f"[vix] using cached {VIX_CSV}")
        return pd.read_csv(VIX_CSV, parse_dates=["date"])

    print("[vix] downloading VIXCLS daily close from FRED ...")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    resp = requests.get(url, params={"id": "VIXCLS"}, timeout=30)
    resp.raise_for_status()

    from io import StringIO
    raw = pd.read_csv(StringIO(resp.text))
    raw.columns = ["date", "vix_close"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["vix_close"] = pd.to_numeric(raw["vix_close"], errors="coerce")
    raw = raw.dropna(subset=["vix_close"])
    raw = raw[(raw["date"] >= "2019-01-01") & (raw["date"] <= END)].reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(VIX_CSV, index=False)
    print(f"[vix] cached {len(raw):,} rows -> {VIX_CSV}")
    return raw


if __name__ == "__main__":
    fetch_nq_bars()
    fetch_vix_daily()
    print("done.")
