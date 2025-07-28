# scripts/quick_backtest.py
from __future__ import annotations
import sys, pathlib, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.evaluation.backtest import run_backtest, summary, BTConfig
from src.strategy.rules_engine import signal_sma_params  # ← usa SMA parametrizable

TICKERS = ["AAPL", "MSFT", "NVDA"]
FAST, SLOW = 20, 150   # ← de tu grid search

def main():
    rets = []
    eqs  = []
    for tk in TICKERS:
        df = pd.read_parquet(ROOT/f"data/processed/{tk}.parquet", engine="fastparquet")
        sig = signal_sma_params(df, fast=FAST, slow=SLOW)
        bt  = run_backtest(df, sig, BTConfig(fee_bps=5))
        print(tk, summary(bt))
        rets.append(bt["ret_strategy"].rename(tk))
        eqs.append(bt["equity"].rename(tk))

    # Cartera 1/N por retornos (igual que portfolio_backtest)
    R = pd.concat(rets, axis=1).dropna()
    port_ret = R.mean(axis=1)
    ann = 252
    eq = (1 + port_ret).cumprod()
    cagr = (eq.iloc[-1]/eq.iloc[0])**(ann/len(port_ret)) - 1
    mdd  = (eq/eq.cummax()-1).min()
    sharpe = (port_ret.mean()*ann) / (port_ret.std()*(ann**0.5) + 1e-12)
    print("\nPortfolio:", {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(mdd)})

if __name__ == "__main__":
    main()
