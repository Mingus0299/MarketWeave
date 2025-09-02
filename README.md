## MarketWeave
Two Streamlit apps powered by Polygon’s aggregates:
app/App_draft.py — Price + 30-sec & 60-sec volatility (σ) overlays with weekend/overnight time compression.
app/App.py — Intraday VWAP + VWAP bands (±k·σ) and a simple nowcasting model for the next-minute return using other tickers.
Both apps read the same minute OHLCV parquet and (optionally) an events CSV for vertical markers (FOMC, CPI, etc.).
Data is fetched with scripts/get_data_polygon.py via Polygon’s Aggregates v2 endpoint.

# Features
- Minute OHLCV for 5 large-cap tickers (configurable)
- Trading-session stitching: hide weekends, overnight, and missing/holiday sessions
- Vol overlays: σ(30s) and σ(60s) computed from 1-minute log returns
- VWAP (cumulative, per session) + adaptive bands (±k·σ in $ terms)
- Nowcasting: Ridge regression to predict next-minute return from cross-ticker lags
- Event markers from an optional CSV (FOMC, CPI…)



# basic Repository Layout
.
├─ app/
│  ├─ App_draft.py              # σ(30s) & σ(60s) visualization
│  └─ App.py                    # VWAP + VWAP bands + nowcasting tab
├─ scripts/
│  └─ get_data_polygon.py       # download 1-min OHLCV via Polygon (RTH-filtered)
├─ data/
│  └─ raw/
│     ├─ aggs_1m/               # per-ticker, per-day parquet shards (auto-created)
│     ├─ ohlcv_1m.parquet       # consolidated dataset both apps read
│     └─ events.csv             # optional event markers (see schema below)
├─ .env                         # POLYGON_API_KEY=...
└─ .gitignore


To use this code you need to install packages in your virtual env:
pip install streamlit plotly pandas pyarrow requests python-dotenv pytz
pip install scikit-learn                # App.py (nowcasting) needs this
pip install pandas-market-calendars     # optional: early closes/holidays

You also need to create a .env file at the repo root:
POLYGON_API_KEY=your_real_polygon_key_here


# To add more data you can change it here:
Open scripts/get_data_polygon.py and edit tickers and date range:

TICKERS = ["TSLA", "NVDA", "AAPL", "MSFT", "AMZN"]

#Inclusive date range (YYYY-MM-DD) – zero-padded
START = "2025-06-01"
END   = "2025-08-01"

Then run: python scripts/get_data_polygon.py

# What it does:
- Calls /v2/aggs/ticker/{T}/range/1/minute/{date}/{date} (adjusted, asc).
- Filters to Regular Trading Hours (09:30–16:00 NY). Uses pandas-market-calendars if available; otherwise a 09:30–16:00 fallback.
- Paces requests to avoid 429 rate limits (see PACE_SECONDS) and retries on 429/5xx.
- Writes shards to data/raw/aggs_1m/<T>/<T>_<YYYY-MM-DD>.parquet.
- Consolidates to data/raw/ohlcv_1m.parquet (what both apps read).

# Common issues:
- 401 Unauthorized → key missing/typo in .env.
- 403 Forbidden → wrong plan/endpoint (this uses Aggregates v2, allowed on most plans).
- 429 Too Many Requests → increase PACE_SECONDS, or reduce tickers/dates.


# Finally you can run the app:

# A) σ(30s) & σ(60s) viewer — App_draft.py by running:
streamlit run app/App_draft.py

Use it to:
- Pick Ticker and Date range.
- Toggle “Show σ (30s & 60s)” to plot volatility on the right axis (replaces Volume).
- Choose σ lookback (rolling window over 1-minute log returns).
- Select σ units (percent or bps).
- Time compression via rangebreaks hides weekends, overnight, and missing days.

Interpretation:
- σ1m = rolling std of 1-minute log returns.
- σ30s = σ1m * √(30/60), σ60s = σ1m.
- Spikes tend to coincide with macro/news opens; useful to compare “riskiness” by day.

# B) VWAP + Bands + Nowcasting — App.py bu running:
streamlit run app/App.py

Chart tab:
- VWAP 
- Bands: VWAP ± k·σ$, with σ$ ≈ Close * σ(1m returns) (small-return approx)
- Optional right-axis Volume; event markers if events.csv is present
- Sessions stitched (weekend/overnight breaks hidden)

Nowcasting tab
- Predict next-minute return of the selected ticker using lags of other tickers.
- Controls: feature tickers, lags (0…k), train split, RTH only.
- Model: Ridge regression (linear, L2).
- Reports R², RMSE, MAE, hit-rate and top feature weights.
- Compatible across scikit-learn versions (version-safe RMSE helper baked in).


# N.B.
# Timezones, Duplicates & Gaps
- Downloader stores timestamps in UTC; apps convert to America/Toronto for display.
- Apps drop duplicate timestamps per ticker to avoid “double-line” artifacts.
- If stitching looks odd, verify the downloader’s RTH filter and that event timestamps are UTC.

# Credits
Data: Polygon.io Aggregates v2
Calendars (optional): pandas-market-calendars
UI/Charts: Streamlit + Plotly