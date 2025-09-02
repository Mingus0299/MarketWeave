import pandas as pd, numpy as np
from scipy.signal import correlate
TICKERS = ["TSLA","NVDA","AAPL","MSFT","AMZN"]

bars = pd.read_parquet("data/raw/ohlcv_1m.parquet")
# minute returns
bars["ret"] = bars.groupby("ticker")["Close"].pct_change().fillna(0.0)
# pivot to wide matrix time x ticker
mat = bars.pivot_table(index="timestamp", columns="ticker", values="ret").dropna()

def lag_at_max_xcorr(a, b, max_lag=60):  # +/- 60 minutes
    # cross-corr with lags
    lags = np.arange(-max_lag, max_lag+1)
    xcs = [np.corrcoef(a.shift(-k).dropna().align(b, join="inner")[0])[0,1] for k in lags]
    best = lags[int(np.nanargmax(xcs))]
    return best, np.nanmax(xcs)

leaders=[]
for i in TICKERS:
    for j in TICKERS:
        if i==j: continue
        k, corr = lag_at_max_xcorr(mat[i], mat[j])
        if k<0:  # j leads i if j’s move comes earlier (negative lag applied to i)
            leaders.append((j, i, k, corr))
pd.DataFrame(leaders, columns=["leader","follower","lag_minutes","corr"]).sort_values("corr", ascending=False).head(10).to_csv("data/processed/lead_lag_top.csv", index=False)
