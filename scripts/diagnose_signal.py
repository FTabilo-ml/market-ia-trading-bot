# scripts/diagnose_signal.py
from __future__ import annotations
import sys, pathlib, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

TKS = ["AAPL","MSFT","NVDA"]

def zeros_like(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index, dtype="float64")

for tk in TKS:
    df = pd.read_parquet(ROOT/f"data/processed/{tk}.parquet", engine="fastparquet")
    close = pd.to_numeric(df["Close"], errors="coerce")
    sma50  = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    base   = (sma50 > sma200)               # Serie booleana

    sent = pd.to_numeric(df.get("sentiment_score", zeros_like(df)), errors="coerce").fillna(0.0)
    flows = pd.to_numeric(df.get("net_buy", zeros_like(df)), errors="coerce").fillna(0.0)

    base_days   = int(base.astype(int).sum())
    sent_days   = int((sent != 0).astype(int).sum())
    flows_days  = int((flows >= 1).astype(int).sum())
    both_days   = int(((base) & (sent.rolling(3, min_periods=1).mean() > 0) & (flows >= 1)).astype(int).sum())

    print(f"\n== {tk} ==")
    print("Días totales:", len(df))
    print("Base SMA>:", base_days)
    print("Días con sentiment!=0:", sent_days)
    print("Días con net_buy>=1:", flows_days)
    print("Días con las 3 condiciones:", both_days)
