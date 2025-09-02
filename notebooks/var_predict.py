import pandas as pd
from statsmodels.tsa.api import VAR
TICKERS = ["TSLA","NVDA","AAPL","MSFT","AMZN"]
bars = pd.read_parquet("data/raw/ohlcv_1m.parquet")
bars["ret"] = bars.groupby("ticker")["Close"].pct_change()
mat = bars.pivot_table(index="timestamp", columns="ticker", values="ret").dropna()

train = mat.loc[: mat.index[int(len(mat)*0.8)]]
test  = mat.loc[ mat.index[int(len(mat)*0.8)]:]

model = VAR(train)
res = model.fit(maxlags=5, ic="aic")  # try 2–10
pred = res.forecast(train.values[-res.k_ar:], steps=len(test))
pred_df = pd.DataFrame(pred, index=test.index, columns=test.columns)
# Evaluate for one target, e.g. TSLA (directional accuracy)
acc = ((pred_df["TSLA"] * test["TSLA"])>0).mean()
print("Directional accuracy:", round(acc,3))
res.test_causality("TSLA", ["NVDA","AAPL","MSFT","AMZN"], kind="f").summary()
