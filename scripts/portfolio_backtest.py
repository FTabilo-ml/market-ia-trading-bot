# scripts/portfolio_backtest.py
from __future__ import annotations
import sys, pathlib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.strategy.rules_engine import signal_sma_params

TICKERS = ["AAPL","MSFT","NVDA"]
FAST, SLOW = 50, 200  # cambia según tu grid

def metrics(ret: pd.Series) -> dict:
    ann = 252
    eq = (1+ret).cumprod()
    cagr = (eq.iloc[-1]/eq.iloc[0])**(ann/len(ret)) - 1
    sharpe = (ret.mean()*ann) / (ret.std()*(ann**0.5)+1e-12)
    mdd = (eq/eq.cummax()-1).min()
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(mdd)}

def main():
    # Construye matriz de retornos del activo y matriz de señales 0/1
    rets = []
    sigs = []
    for tk in TICKERS:
        df = pd.read_parquet(ROOT/f"data/processed/{tk}.parquet", engine="fastparquet")
        rets.append(pd.to_numeric(df["daily_return"], errors="coerce").rename(tk))
        sigs.append(signal_sma_params(df, FAST, SLOW).rename(tk).astype(float))
    R = pd.concat(rets, axis=1).dropna()
    S = pd.concat(sigs, axis=1).reindex(R.index).fillna(0.0)

    # Peso 1/N entre los tickers activos cada día
    active = S.sum(axis=1).replace(0, pd.NA)
    W = S.div(active, axis=0).fillna(0.0)

    # Retorno diario de cartera (sin costes; añade bps si quieres)
    port_ret = (W * R).sum(axis=1)

    print(metrics(port_ret))

if __name__ == "__main__":
    main()
