import pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

bars = pd.read_parquet("data/raw/ohlcv_1m.parquet")
bars["ret"] = bars.groupby("ticker")["Close"].pct_change()
wide = bars.pivot_table(index="timestamp", columns="ticker", values="ret").dropna()

# Features: last k returns of peer stocks
K=5
X_list=[]
for t in ["TSLA","NVDA","AAPL","MSFT","AMZN"]:
    for k in range(1,K+1):
        X_list.append(wide[t].shift(k).rename(f"{t}_lag{k}"))
X = pd.concat(X_list, axis=1).dropna()
y = (wide["TSLA"].reindex_like(X) > 0).astype(int)  # predict TSLA up/down
mask = ~y.isna()
X, y = X[mask], y[mask]

tscv = TimeSeriesSplit(n_splits=5)
clf = LogisticRegression(max_iter=1000, n_jobs=-1)
preds=[]; trues=[]
for tr, te in tscv.split(X):
    clf.fit(X.iloc[tr], y.iloc[tr])
    p = clf.predict(X.iloc[te]); preds.append(p); trues.append(y.iloc[te].values)
preds = np.concatenate(preds); trues = np.concatenate(trues)
print(classification_report(trues, preds, digits=3))

# Sharp-jump classifier (|return| > 0.7% next minute)
y_jump = (wide["TSLA"].shift(-1).abs() > 0.007).astype(int).reindex_like(X).dropna()
Xj, yj = X.loc[y_jump.index], y_jump
clf2 = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xj, yj)
