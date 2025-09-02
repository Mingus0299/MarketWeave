import pandas as pd, pytz, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# timezone & page
local_tz = pytz.timezone("America/Toronto")
st.set_page_config(page_title="Volatility Explorer", layout="wide")
st.title("Volatility Explorer – Multi-Stock")

# cached loaders
@st.cache_data
def load_bars(path="data/raw/ohlcv_1m.parquet"):
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    # ensure tz-aware UTC then convert to local_tz
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df["timestamp"] = ts.dt.tz_convert(local_tz)
    return df

@st.cache_data
def load_events(path="data/raw/events.csv"):
    try:
        e = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["timestamp_utc","event_type","source","details","timestamp"])
    if "timestamp_utc" not in e.columns:
        return pd.DataFrame(columns=["timestamp_utc","event_type","source","details","timestamp"])
    ts = pd.to_datetime(e["timestamp_utc"], errors="coerce", utc=True)
    e["timestamp"] = ts.dt.tz_convert(local_tz)
    return e

# data
bars = load_bars("data/raw/ohlcv_1m.parquet")
events = load_events("data/raw/events.csv")

tickers = sorted(pd.Series(bars["ticker"]).dropna().unique().tolist())
sel_ticker = st.selectbox("Ticker", tickers, index=0 if tickers else None)

rng = st.date_input("Date range", [])
subset = bars[bars["ticker"] == sel_ticker].copy() if sel_ticker else bars.iloc[0:0].copy()

if not subset.empty:
    # robust tz handling (already local tz from loader)
    subset["timestamp"] = pd.to_datetime(subset["timestamp"], errors="coerce")
    if subset["timestamp"].dt.tz is None:
        subset["timestamp"] = subset["timestamp"].dt.tz_localize(local_tz)
    else:
        subset["timestamp"] = subset["timestamp"].dt.tz_convert(local_tz)

    # date filter
    if rng and len(rng) == 2:
        start = pd.Timestamp(rng[0]).tz_localize(local_tz)
        end   = pd.Timestamp(rng[1]).tz_localize(local_tz) + pd.Timedelta(days=1)
        subset = subset[(subset["timestamp"] >= start) & (subset["timestamp"] < end)]

# plot
if subset.empty:
    st.warning("No data for the selected ticker/date range.")
else:
    # drop any exact duplicate timestamps to avoid accidental overlay
    subset = subset.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    # NEW: volatility controls & computation
    with st.expander("Volatility options (σ)", expanded=False):
        lookback = st.slider("Lookback window (minutes) for σ(1m returns)",
                             min_value=5, max_value=120, value=30, step=5, key="lkb_sigma")
        sigma_units = st.radio("σ units", ["percent", "bps"], horizontal=True, index=0, key="sigma_units")
        show_sigma = st.checkbox("Show σ (30s & 60s) — replaces Volume", value=False, key="show_sigma")

    # 1-minute log returns
    subset["ret1m"] = np.log(subset["Close"]).diff()
    # rolling std of 1-minute returns
    minp = max(5, lookback // 3)  # require a few points before showing σ
    subset["sigma_1m"] = subset["ret1m"].rolling(lookback, min_periods=minp).std()

    # scale to 30s and 60s using sqrt(time)
    subset["sigma_30s"] = subset["sigma_1m"] * np.sqrt(30 / 60)
    subset["sigma_60s"] = subset["sigma_1m"] * 1.0

    # units
    scale = 100.0 if sigma_units == "percent" else 10000.0
    y2_title_sigma = "σ (%)" if sigma_units == "percent" else "σ (bps)"

    # choose a default style: close line for dense views, candles otherwise
    point_count = len(subset)
    default_is_line = point_count > 1500  # tweak if you like
    style = st.radio("Chart style", ["Candles", "Close line"],
                     horizontal=True, index=1 if default_is_line else 0, key="chart_style")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if style == "Close line":
        fig.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=subset["Close"],
                mode="lines",
                name=sel_ticker,
                line=dict(width=1),
            ),
            secondary_y=False,
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=subset["timestamp"],
                open=subset["Open"],
                high=subset["High"],
                low=subset["Low"],
                close=subset["Close"],
                name=sel_ticker,
                increasing_line_width=0.8,   # thinner = less “double line” look
                decreasing_line_width=0.8,
                whiskerwidth=0.2,
            ),
            secondary_y=False,
        )

    # Right axis content: either Volume OR σ lines
    if show_sigma:
        fig.add_trace(
            go.Scatter(
                x=subset["timestamp"], y=subset["sigma_30s"] * scale,
                mode="lines", name="σ 30s", line=dict(width=1, dash="dot"),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["timestamp"], y=subset["sigma_60s"] * scale,
                mode="lines", name="σ 60s", line=dict(width=1),
            ),
            secondary_y=True,
        )
        y2_title = y2_title_sigma
    else:
        fig.add_trace(
            go.Bar(
                x=subset["timestamp"],
                y=subset["Volume"],
                name="Volume",
                opacity=0.30,
            ),
            secondary_y=True,
        )
        y2_title = "Volume"

    # Event markers within visible window
    tmin, tmax = subset["timestamp"].min(), subset["timestamp"].max()
    ev = events[(events["timestamp"] >= tmin) & (events["timestamp"] <= tmax)]
    y_for_annotations = float(subset["High"].max()) if not subset["High"].empty else None
    for _, row in ev.iterrows():
        fig.add_vline(x=row["timestamp"], line_dash="dot")
        if y_for_annotations is not None:
            label = row.get("event_type", "event")
            fig.add_annotation(
                x=row["timestamp"],
                y=y_for_annotations,
                text=str(label),
                showarrow=False,
                yshift=10,
            )

    # collapse off-market time (rangebreaks)
    market_open, market_close = 9.5, 16  # 9:30–16:00 local (ET)
    present_days = pd.DatetimeIndex(subset["timestamp"].dt.normalize().unique())
    full_bd = pd.date_range(
        start=present_days.min().normalize(),
        end=present_days.max().normalize(),
        freq="B",
        tz=local_tz,
    )
    holidays = full_bd.difference(present_days)

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),       # hide weekends
            dict(bounds=[market_close, market_open], pattern="hour"), # hide overnight
            dict(values=list(holidays)),       # hide missing/holiday days
        ]
    )

    # Layout polish (conditional right-axis title)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_yaxes(title_text="Price", secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text=y2_title, secondary_y=True, showgrid=False, side="right")

    st.plotly_chart(fig, use_container_width=True)
