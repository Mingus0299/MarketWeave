from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import os, time, random, requests, pandas as pd

API_KEY = (os.getenv("POLYGON_API_KEY") or "").strip()
if not API_KEY:
    raise RuntimeError("POLYGON_API_KEY not set in .env or environment")

BASE = "https://api.polygon.io"
TICKERS = ["TSLA","NVDA","AAPL","MSFT","AMZN"]

# Inclusive date range
START = "2025-06-30"
END   = "2025-07-25"

# Rate limit / retry knobs
PACE_SECONDS = 15
MAX_RETRIES  = 6    # on 429 or transient errors
BACKOFF_BASE = 10   # seconds, grows exponentially
SAVE_DIR = Path("data/raw/aggs_1m")
OUT_PARQUET = Path("data/raw/ohlcv_1m.parquet")
# ----------------------------------

session = requests.Session()

def fetch_aggs_1m(ticker: str, date: str) -> pd.DataFrame:
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}

    for attempt in range(1, MAX_RETRIES + 1):
        r = session.get(url, params=params, timeout=60)
        if r.status_code == 200:
            js = r.json()
            rows = js.get("results") or []
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df["ticker"] = ticker
            df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume",
                                    "n":"NumTrades","vw":"VWAP"})
            for col in ["NumTrades","VWAP"]:
                if col not in df:
                    df[col] = pd.NA
            # Regular session filter (13:30–20:00 UTC)
            start_ts = pd.Timestamp(f"{date} 13:30", tz="UTC")
            end_ts   = pd.Timestamp(f"{date} 20:00", tz="UTC")
            df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
            return df[["timestamp","ticker","Open","High","Low","Close","Volume","NumTrades","VWAP"]]

        # Handle rate limit & transient errors
        if r.status_code in (429, 502, 503, 504):
            # prefer Retry-After if provided
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                # exponential backoff + jitter
                wait = min(BACKOFF_BASE * (2 ** (attempt - 1)), 90) + random.uniform(0, 2)
            safe_url = r.url.replace(API_KEY, "****")
            print(f"[{ticker} {date}] {r.status_code} -> waiting {wait:.1f}s; URL: {safe_url}")
            time.sleep(wait)
            continue

        # Other errors: show and raise
        safe_url = r.url.replace(API_KEY, "****")
        print("URL:", safe_url)
        print("Status:", r.status_code)
        print("Body:", r.text[:400])
        r.raise_for_status()

    raise RuntimeError(f"Max retries reached for {ticker} {date}")

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    days = pd.date_range(START, END, freq="D")
    for day in days:
        dstr = day.strftime("%Y-%m-%d")
        print(f"=== {dstr} ===")
        for t in TICKERS:
            out_file = SAVE_DIR / t / f"{t}_{dstr}.parquet"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if out_file.exists():
                print(f"  {t}: exists, skipping")
                continue

            df = fetch_aggs_1m(t, dstr)
            if df.empty:
                print(f"  {t}: no data")
            else:
                print(f"  {t}: {len(df)} bars -> {out_file}")
                df.to_parquet(out_file)

            # pace to avoid 429s
            time.sleep(PACE_SECONDS)

    # consolidate everything we have (idempotent)
    parts = []
    for t in TICKERS:
        for p in sorted((SAVE_DIR / t).glob(f"{t}_*.parquet")):
            parts.append(pd.read_parquet(p))
    if not parts:
        print("No data collected.")
        return

    combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["timestamp","ticker"])
    combined.sort_values(["timestamp","ticker"], inplace=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PARQUET)
    print("Saved:", OUT_PARQUET, "rows:", len(combined))

if __name__ == "__main__":
    main()


