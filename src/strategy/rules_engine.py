# src/strategy/rules_engine.py
from __future__ import annotations
import pandas as pd

__all__ = [
    "signal_sma",
    "signal_sma_sent",
    "signal_sma_sent_congress",
    "signal_sma_params",
]

def _series_or_zeros(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        return s.fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")

def _sma(series: pd.Series, w: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(w, min_periods=w).mean()

def signal_sma(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    return (_sma(close, 50) > _sma(close, 200)).astype(int)

def signal_sma_params(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    close = df["Close"]
    return (_sma(close, fast) > _sma(close, slow)).astype(int)

def signal_sma_sent(df: pd.DataFrame) -> pd.Series:
    s = signal_sma(df).astype(float)
    sent = _series_or_zeros(df, "sentiment_score")
    sent3 = sent.rolling(3, min_periods=1).mean()
    size = (sent3 > 0).map({True: 1.0, False: 0.5})
    return (s * size).clip(0, 1)

def signal_sma_sent_congress(df: pd.DataFrame) -> pd.Series:
    s = signal_sma_sent(df)
    flows = _series_or_zeros(df, "net_buy")
    boost = (flows >= 1).map({True: 1.0, False: 0.8})
    return (s * boost).clip(0, 1)
