import numpy as np
import pandas as pd, pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import time as dtime

# timezone & page
local_tz = pytz.timezone("America/Toronto")   # display tz (kept from your app)
market_tz = pytz.timezone("America/New_York") # for RTH filtering in model if needed
st.set_page_config(page_title="Volatility Explorer", layout="wide")
st.title("Volatility Explorer – Multi-Stock")

# cached loaders 
@st.cache_data
def load_bars(path="data/raw/ohlcv_1m.parquet"):
    df = pd.read_parquet(path)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df["timestamp"] = ts.dt.tz_convert(local_tz)  # keep everything in display tz
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

bars = load_bars("data/raw/ohlcv_1m.parquet")
events = load_events("data/raw/events.csv")

# controls 
tickers = sorted(pd.Series(bars["ticker"]).dropna().unique().tolist())
sel_ticker = st.selectbox("Ticker", tickers, index=0 if tickers else None)

min_date = bars["timestamp"].min().date() if len(bars) else None
max_date = bars["timestamp"].max().date() if len(bars) else None
rng = st.date_input("Date range", (min_date, max_date) if min_date and max_date else [])

subset = bars[bars["ticker"] == sel_ticker].copy() if sel_ticker else bars.iloc[0:0].copy()
if not subset.empty and rng and len(rng) == 2:
    start = pd.Timestamp(rng[0], tz=local_tz)
    end   = pd.Timestamp(rng[1], tz=local_tz) + pd.Timedelta(days=1)
    subset = subset[(subset["timestamp"] >= start) & (subset["timestamp"] < end)]

# guard
if subset.empty:
    st.warning("No data for the selected ticker/date range.")
    st.stop()

# drop duplicates, sort
subset = subset.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

# CHART TAB 
tab_chart, tab_model = st.tabs(["Chart", "Nowcasting"])

with tab_chart:
    # VWAP + bands controls 
    with st.expander("VWAP & Bands (per session)", expanded=True):
        show_vwap = st.checkbox("Show VWAP", value=True)
        show_bands = st.checkbox("Show VWAP ± k·σ bands", value=True)
        lookback = st.slider("σ lookback (minutes, uses 1m log-returns)", 10, 120, 30, 5)
        k_band = st.slider("Band width k", 0.5, 4.0, 2.0, 0.5)
        # when bands are shown, you can still keep Volume; your call:
        keep_volume = st.checkbox("Keep Volume on right axis", value=False)

    # compute 1m returns and per-session VWAP
    df = subset.copy()
    df["day"] = df["timestamp"].dt.normalize()  # session key in local tz
    df["ret1m"] = np.log(df["Close"]).diff()
    # per-session cumulative VWAP (index-aligned; no MultiIndex issues)
    price = df["Close"].astype("float64")
    vol   = df["Volume"].astype("float64")

    df["pv"]      = price * vol
    df["cum_pv"]  = df.groupby("day")["pv"].cumsum()
    df["cum_vol"] = df.groupby("day")["Volume"].transform(lambda s: s.replace(0, np.nan).cumsum())
    df["vwap"]    = df["cum_pv"] / df["cum_vol"]

    # per-session rolling σ of returns
    minp = max(5, lookback // 3)
    df["sigma"] = df.groupby("day")["ret1m"].transform(lambda s: s.rolling(lookback, min_periods=minp).std())

    # convert σ (log-return) to ~$ scale using current price
    df["sigma_$"] = df["Close"] * df["sigma"]  # small-return approx
    df["vwap_upper"] = df["vwap"] + k_band * df["sigma_$"]
    df["vwap_lower"] = df["vwap"] - k_band * df["sigma_$"]

    # chart style
    style = st.radio("Chart style", ["Candles", "Close line"], horizontal=True, index=0, key="chart_style")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if style == "Close line":
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["Close"], mode="lines", name=sel_ticker, line=dict(width=1)
        ), secondary_y=False)
    else:
        fig.add_trace(go.Candlestick(
            x=df["timestamp"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=sel_ticker,
            increasing_line_width=0.8, decreasing_line_width=0.8, whiskerwidth=0.2
        ), secondary_y=False)

    # VWAP + bands
    if show_vwap:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["vwap"], name="VWAP", mode="lines", line=dict(width=1, dash="solid")
        ), secondary_y=False)
    if show_bands:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["vwap_upper"], name=f"VWAP + {k_band}σ", mode="lines",
            line=dict(width=1, dash="dot")
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["vwap_lower"], name=f"VWAP - {k_band}σ", mode="lines",
            line=dict(width=1, dash="dot")
        ), secondary_y=False)

    # Either Volume or no right-axis content (you can keep both if you want)
    if keep_volume:
        fig.add_trace(go.Bar(
            x=df["timestamp"], y=df["Volume"], name="Volume", opacity=0.30
        ), secondary_y=True)
        y2_title = "Volume"
    else:
        y2_title = ""

    # event markers
    tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    ev = events[(events["timestamp"] >= tmin) & (events["timestamp"] <= tmax)]
    if not df["High"].empty:
        y_annot = float(df["High"].max())
        for _, row in ev.iterrows():
            fig.add_vline(x=row["timestamp"], line_dash="dot")
            fig.add_annotation(x=row["timestamp"], y=y_annot,
                               text=str(row.get("event_type", "event")),
                               showarrow=False, yshift=10)

    # collapse off-market time
    market_open, market_close = 9.5, 16  # 9:30–16:00 local (ET)
    present_days = pd.DatetimeIndex(df["timestamp"].dt.normalize().unique())
    full_bd = pd.date_range(start=present_days.min().normalize(),
                            end=present_days.max().normalize(),
                            freq="B", tz=local_tz)
    holidays = full_bd.difference(present_days)
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "mon"]),
        dict(bounds=[market_close, market_open], pattern="hour"),
        dict(values=list(holidays)),
    ])

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        height=560,
    )
    fig.update_yaxes(title_text="Price", secondary_y=False, showgrid=True)
    if y2_title:
        fig.update_yaxes(title_text=y2_title, secondary_y=True, showgrid=False, side="right")
    st.plotly_chart(fig, use_container_width=True)

# NOWCASTING TAB 
    st.subheader("Next-Minute Return Nowcasting")

    # pick feature tickers (defaults = other tickers)
    default_feats = [t for t in tickers if t != sel_ticker][:4]
    feat_tickers = st.multiselect(
        "Feature tickers (predictors)",
        [t for t in tickers if t != sel_ticker],
        default=default_feats
    )

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        lags = st.slider("Lags per ticker (0…k minutes)", 0, 5, 2)
    with colB:
        train_frac = st.slider("Train split (first % of time)", 50, 90, 70, step=5)
    with colC:
        rth_only = st.checkbox("Use Regular Session only (09:30–16:00 NY)", value=True)

    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_absolute_error
    except Exception:
        st.error("scikit-learn not installed. Run: `pip install scikit-learn`")
        st.stop()

    # version-safe RMSE + metrics (single definition) 
    try:
        # sklearn >= 1.4
        from sklearn.metrics import root_mean_squared_error as _rmse_fn
        def _rmse_compat(y_true, y_hat):
            return float(_rmse_fn(y_true, y_hat))
    except Exception:
        # older sklearn
        from sklearn.metrics import mean_squared_error
        def _rmse_compat(y_true, y_hat):
            try:
                return mean_squared_error(y_true, y_hat, squared=False)  # 1.0–1.3
            except TypeError:
                return float(np.sqrt(mean_squared_error(y_true, y_hat)))  # very old

    def metrics(y_true, y_hat):
        r2   = r2_score(y_true, y_hat)
        rmse = _rmse_compat(y_true, y_hat)
        mae  = mean_absolute_error(y_true, y_hat)
        hit  = (np.sign(y_true) == np.sign(y_hat)).mean()
        return r2, rmse, mae, hit

    # Build a wide Close table for the selected date window
    b = bars.copy()
    # keep same display tz
    b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce")
    if b["timestamp"].dt.tz is None:
        b["timestamp"] = b["timestamp"].dt.tz_localize(local_tz)
    else:
        b["timestamp"] = b["timestamp"].dt.tz_convert(local_tz)

    if rng and len(rng) == 2:
        start = pd.Timestamp(rng[0], tz=local_tz)
        end   = pd.Timestamp(rng[1], tz=local_tz) + pd.Timedelta(days=1)
        b = b[(b["timestamp"] >= start) & (b["timestamp"] < end)]

    if rth_only:
        tt = b["timestamp"].dt.tz_convert(market_tz).dt.time
        b = b[(tt >= dtime(9,30)) & (tt <= dtime(16,0))]

    wide = (b.pivot_table(index="timestamp", columns="ticker", values="Close")
              .sort_index()
              .dropna(how="any", subset=[sel_ticker] + feat_tickers))

    # 1-min log returns
    rets = np.log(wide).diff().dropna()

    # Features: for each feature ticker, lags 0..k
    X = pd.DataFrame(index=rets.index)
    for t in feat_tickers:
        for k in range(lags + 1):
            X[f"{t}_lag{k}"] = rets[t].shift(k)

    # Target: next-minute return of sel_ticker
    y = rets[sel_ticker].shift(-1)

    # align & drop NaNs from shifting
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    if data.empty:
        st.warning("Not enough overlapping data to train. Try shorter lag, more features, or a different date window.")
        st.stop()

    X_clean = data.drop(columns=["y"])
    y_clean = data["y"]

    # chronological split
    split = int(len(X_clean) * (train_frac / 100.0))
    Xtr, Xte = X_clean.iloc[:split], X_clean.iloc[split:]
    ytr, yte = y_clean.iloc[:split], y_clean.iloc[split:]

    # train
    alpha = st.slider("Ridge alpha (L2 strength)", 0.1, 50.0, 5.0, 0.1)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr, ytr)

    # predict (in-sample & OOS)
    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)

    # metrics
    r2_tr, rmse_tr, mae_tr, hit_tr = metrics(ytr, yhat_tr)
    r2_te, rmse_te, mae_te, hit_te = metrics(yte, yhat_te)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Train (chronological)**")
        st.write(f"R²: **{r2_tr:.3f}**  ·  RMSE: **{rmse_tr:.6f}**  ·  MAE: **{mae_tr:.6f}**  ·  Hit-rate: **{hit_tr*100:.1f}%**")
    with col2:
        st.markdown("**Test / OOS**")
        st.write(f"R²: **{r2_te:.3f}**  ·  RMSE: **{rmse_te:.6f}**  ·  MAE: **{mae_te:.6f}**  ·  Hit-rate: **{hit_te*100:.1f}%**")

    # plot cumulative actual vs predicted (test window)
    test_df = pd.DataFrame({
        "timestamp": yte.index,
        "actual": yte.values,
        "pred": yhat_te,
    }).set_index("timestamp")
    test_df["cum_actual"] = test_df["actual"].cumsum()
    test_df["cum_pred"]   = test_df["pred"].cumsum()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_df.index, y=test_df["cum_actual"], name="Cum actual (test)", mode="lines"))
    fig2.add_trace(go.Scatter(x=test_df.index, y=test_df["cum_pred"],   name="Cum predicted (test)", mode="lines"))
    fig2.update_layout(
        title=f"Nowcasting {sel_ticker}: cumulative returns (test window)",
        xaxis_rangeslider_visible=False, height=420, margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # show top coefficients
    coefs = pd.Series(model.coef_, index=X_clean.columns).sort_values(key=lambda s: s.abs(), ascending=False)
    st.caption("Top features by absolute weight")
    st.dataframe(coefs.head(20).to_frame("weight"))
