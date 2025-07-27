# src/evaluation/backtest.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

@dataclass
class BTConfig:
    start_capital: float = 10_000.0
    fee_bps: float = 5.0          # 5 bps por trade = 0.05%
    slippage_bps: float = 0.0
    max_leverage: float = 1.0     # 1 = sin apalancamiento

def _costs(turnover: pd.Series, fee_bps: float, slippage_bps: float) -> pd.Series:
    bps = (fee_bps + slippage_bps) / 1e4
    return turnover.abs() * bps

def run_backtest(df: pd.DataFrame, signal: pd.Series, cfg: BTConfig = BTConfig()) -> pd.DataFrame:
    """
    df: debe tener 'daily_return' (rendimiento del activo).
    signal: Serie alineada al índice del df, en {0,1} (long/flat).
    No hay lookahead: aplicamos shift(1) al signal.
    """
    s = signal.reindex(df.index).fillna(0).clip(0, cfg.max_leverage)
    s_eff = s.shift(1).fillna(0)                      # ejecutas al cierre siguiente
    ret = pd.to_numeric(df["daily_return"], errors="coerce").fillna(0)

    turnover = s_eff.diff().abs().fillna(s_eff.abs()) # cambios de posición
    costs = _costs(turnover, cfg.fee_bps, cfg.slippage_bps)

    strat_ret = s_eff * ret - costs
    equity = (1 + strat_ret).cumprod() * cfg.start_capital

    out = pd.DataFrame({
        "ret_asset": ret,
        "signal": s,
        "signal_eff": s_eff,
        "turnover": turnover,
        "costs": costs,
        "ret_strategy": strat_ret,
        "equity": equity,
    }, index=df.index)
    return out

def summary(bt: pd.DataFrame) -> dict:
    ret = bt["ret_strategy"]
    n = len(ret)
    if n == 0:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "Turnover/yr": 0.0}
    ann = 252
    cagr = (bt["equity"].iloc[-1] / max(bt["equity"].iloc[0], 1e-12)) ** (ann / n) - 1
    vol = ret.std() * (ann ** 0.5)
    sharpe = (ret.mean() * ann) / (vol + 1e-12)
    dd = (bt["equity"] / bt["equity"].cummax() - 1).min()
    turn_yr = bt["turnover"].sum() * (ann / n)
    return {"CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(dd), "Turnover/yr": float(turn_yr)}
