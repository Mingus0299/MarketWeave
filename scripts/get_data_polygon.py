from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import os
import time
import random
import requests
import pandas as pd
import pytz
from typing import Optional, Tuple

# Config 
API_KEY = (os.getenv("POLYGON_API_KEY") or "").strip()
if not API_KEY:
    raise RuntimeError("POLYGON_API_KEY not set in .env or environment")

BASE = "https://api.polygon.io"
TICKERS = ["TSLA", "NVDA", "AAPL", "MSFT", "AMZN"]

# Inclusive date range (YYYY-MM-DD)
START = "2025-06-1"
END   = "2025-08-1"

# Rate limit / retry knobs
PACE_SECONDS = 15     # sleep between requests to avoid 429s
MAX_RETRIES  = 6      # on 429/5xx
BACKOFF_BASE = 10     # seconds, exponential + jitter

# Output locations
SAVE_DIR    = Path("data/raw/aggs_1m")
OUT_PARQUET = Path("data/raw/ohlcv_1m.parquet")

# Timezone
MARKET_TZ = pytz.timezone("America/New_York")

# early closes via pandas-market-calendars 
try:
    import pandas_market_calendars as mcal
    NYSE = mcal.get_calendar("XNYS")
except Exception:  # library not installed or unavailable
    NYSE = None

# HTTP session 
session = requests.Session()
# keep apiKey out of printed URLs
session.params = {"apiKey": API_KEY}

# Helpers
def rth_window_utc(date_str: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return (open_utc, close_utc) for regular trading hours on `date_str`.
    If the date is a full holiday/weekend, return None.
    Uses pandas-market-calendars when available; otherwise 09:30–16:00 ET.
    """
    if NYSE is not None:
        sched = NYSE.schedule(start_date=date_str, end_date=date_str)
        if sched.empty:
            return None
        open_local  = sched["market_open"].iloc[0].tz_convert(MARKET_TZ)
        close_local = sched["market_close"].iloc[0].tz_convert(MARKET_TZ)
        return open_local.tz_convert("UTC"), close_local.tz_convert("UTC")

    # Fallback: Mon–Fri, 09:30–16:00 ET (no early close awareness)
    day = pd.Timestamp(date_str).tz_localize(MARKET_TZ)
    if day.weekday() >= 5:  # 5=Sat, 6=Sun
        return None
    open_local  = day.normalize() + pd.Timedelta(hours=9, minutes=30)
    close_local = day.normalize() + pd.Timedelta(hours=16, minutes=0)
    return open_local.tz_convert("UTC"), close_local.tz_convert("UTC")

def business_days(start: str, end: str) -> pd.DatetimeIndex:
    """Business days in market tz; uses market calendar if available."""
    if NYSE is not None:
        days = NYSE.valid_days(start_date=start, end_date=end)
        # valid_days are tz-aware UTC; convert to market tz for consistent date strs
        return days.tz_convert(MARKET_TZ)
    # fallback: simple Mon–Fri
    return pd.date_range(start, end, freq="B", tz=MARKET_TZ)

def fetch_aggs_1m(ticker: str, date: str) -> pd.DataFrame:
    """
    Download 1-min adjusted OHLCV for one ticker and one date.
    Returns a DataFrame filtered to regular session only.
    """
    window = rth_window_utc(date)
    if window is None:
        return pd.DataFrame()  # holiday/weekend

    start_ts_utc, end_ts_utc = window
    # IMPORTANT: end bound is *strictly less than* end_ts to avoid the first post-market bar
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}

    for attempt in range(1, MAX_RETRIES + 1):
        r = session.get(url, params=params, timeout=60)
        if r.status_code == 200:
            js = r.json()
            rows = js.get("results") or []
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            # milliseconds since epoch, start of bar
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)

            # Filter to RTH window (strict end)
            df = df[(df["timestamp"] >= start_ts_utc) & (df["timestamp"] < end_ts_utc)]

            if df.empty:
                return df

            df["ticker"] = ticker
            df = df.rename(
                columns={
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume",
                    "n": "NumTrades",
                    "vw": "VWAP",
                }
            )
            for col in ("NumTrades", "VWAP"):
                if col not in df:
                    df[col] = pd.NA

            return df[["timestamp", "ticker", "Open", "High", "Low", "Close", "Volume", "NumTrades", "VWAP"]]

        if r.status_code in (429, 502, 503, 504):
            # Honor Retry-After if present, else exponential backoff + jitter
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = min(BACKOFF_BASE * (2 ** (attempt - 1)), 90) + random.uniform(0, 2)
            safe_url = r.url.replace(API_KEY, "****")
            print(f"[{ticker} {date}] {r.status_code} -> sleeping {wait:.1f}s  URL: {safe_url}")
            time.sleep(wait)
            continue

        # Other errors: show and raise
        safe_url = r.url.replace(API_KEY, "****")
        print("URL:", safe_url)
        print("Status:", r.status_code)
        print("Body:", r.text[:400])
        r.raise_for_status()

    raise RuntimeError(f"Max retries reached for {ticker} {date}")

# Main
def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    days = business_days(START, END) # business days only
    day_strs = [d.astimezone(MARKET_TZ).strftime("%Y-%m-%d") for d in days]

    for dstr in day_strs:
        print(f"=== {dstr} ===")
        for t in TICKERS:
            out_file = SAVE_DIR / t / f"{t}_{dstr}.parquet"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            if out_file.exists():
                print(f"  {t}: exists, skipping")
            else:
                df = fetch_aggs_1m(t, dstr)
                if df.empty:
                    print(f"  {t}: no regular-session data")
                else:
                    print(f"  {t}: {len(df)} bars -> {out_file}")
                    df.to_parquet(out_file)

                # pace to avoid 429s
                time.sleep(PACE_SECONDS)

    # consolidate everything we have (idempotent)
    parts = []
    for t in TICKERS:
        shard_dir = SAVE_DIR / t
        if shard_dir.exists():
            for p in sorted(shard_dir.glob(f"{t}_*.parquet")):
                parts.append(pd.read_parquet(p))

    if not parts:
        print("No data collected.")
        return

    combined = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["timestamp", "ticker"])
        .sort_values(["timestamp", "ticker"])
        .reset_index(drop=True)
    )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PARQUET)
    print(f"Saved: {OUT_PARQUET}  rows: {len(combined):,}")

if __name__ == "__main__":
    main()
