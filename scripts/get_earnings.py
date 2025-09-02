import pandas as pd, yfinance as yf
TICKERS = ["TSLA","NVDA","AAPL","MSFT","AMZN"]
rows=[]
for t in TICKERS:
    df = yf.Ticker(t).get_earnings_dates(limit=24)  # 6 yrs-ish of events if available
    if df is not None:
        df = df.reset_index().rename(columns={"Earnings Date":"timestamp"})
        df["ticker"] = t
        rows.append(df[["timestamp","ticker"]])
pd.concat(rows).to_csv("data/raw/earnings_dates.csv", index=False)
