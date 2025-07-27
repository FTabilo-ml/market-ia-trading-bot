# scripts/grid_search_sma.py
from __future__ import annotations
import sys, pathlib, itertools, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from src.evaluation.backtest import run_backtest, summary, BTConfig
from src.strategy.rules_engine import signal_sma_params

TICKERS = ["AAPL","MSFT","NVDA"]
FASTS = [20, 30, 50, 100]
SLOWS = [150, 200, 250]

def backtest_ticker(tk: str, fast: int, slow: int) -> dict:
    df = pd.read_parquet(ROOT/f"data/processed/{tk}.parquet", engine="fastparquet")
    sig = signal_sma_params(df, fast=fast, slow=slow)
    bt  = run_backtest(df, sig, BTConfig(fee_bps=5))
    m   = summary(bt)
    m.update({"ticker": tk, "fast": fast, "slow": slow})
    return m

def main():
    rows = []
    for fast, slow in itertools.product(FASTS, SLOWS):
        if fast >= slow:
            continue
        for tk in TICKERS:
            rows.append(backtest_ticker(tk, fast, slow))
    res = pd.DataFrame(rows)
    # promedio por (fast,slow)
    avg = res.groupby(["fast","slow"], as_index=False)[["CAGR","Sharpe","MaxDD"]].mean()
    print(avg.sort_values("Sharpe", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
