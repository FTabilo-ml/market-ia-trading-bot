# scripts/backtest_ml.py
from __future__ import annotations
import sys, pathlib, argparse
from collections import deque
from typing import List
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MLDIR = ROOT / "data/ml"
PROCESSED = ROOT / "data/processed"

def monthly_returns(ticker: str) -> pd.Series:
    df = pd.read_parquet(PROCESSED / f"{ticker}.parquet", engine="fastparquet")
    px = pd.to_numeric(df["Close"], errors="coerce")
    m = px.resample("M").last()            # índice mensual (fin de mes)
    return m.pct_change().rename(ticker)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6, help="Meses a mantener la cohorte")
    ap.add_argument("--topk", type=int, default=5, help="N picks mensuales")
    args = ap.parse_args()

    # 1) Predicciones (una fila por Ticker/Date)
    preds_path = MLDIR / f"preds_{args.horizon}m.parquet"
    preds = pd.read_parquet(preds_path, engine="fastparquet")
    preds["Date"] = pd.to_datetime(preds["Date"])
    preds["Month"] = preds["Date"].dt.to_period("M")

    # Matriz de predicciones: index=Period[M], columns=Ticker
    P = preds.pivot_table(index="Month", columns="Ticker", values="y_pred", aggfunc="mean").sort_index()
    # Normaliza a Timestamp fin de mes (DatetimeIndex)
    P.index = pd.PeriodIndex(P.index, freq="M").to_timestamp("M")   # → DatetimeIndex

    # 2) Retornos mensuales reales de los tickers presentes en P
    tickers = [str(c) for c in P.columns]
    if len(tickers) == 0:
        print("❌ No hay tickers en las predicciones.")
        return

    R = pd.concat([monthly_returns(tk) for tk in tickers], axis=1)
    R.index = pd.DatetimeIndex(R.index)  # asegurar tipo

    # 3) Meses en común (intersección con NumPy para evitar avisos de tipado)
    p_idx = pd.DatetimeIndex(P.index)
    r_idx = pd.DatetimeIndex(R.index)
    months_arr = np.intersect1d(
        p_idx.values.astype("datetime64[ns]"),
        r_idx.values.astype("datetime64[ns]")
    )
    months: List[pd.Timestamp] = [pd.Timestamp(ts) for ts in months_arr]
    months.sort()

    if len(months) == 0:
        print("❌ No hay meses en común entre predicciones (P) y retornos (R).")
        return

    H = int(args.horizon)
    K = int(args.topk)

    # 4) Cartera superpuesta: mantenemos H cohortes (cada cohorte: lista[str])
    cohorts: deque[list[str]] = deque(maxlen=H)
    eq: list[float] = []
    idx: list[pd.Timestamp] = []

    for m in months:
        # Picks Top‑K del mes (forzamos str para calmar a Pylance)
        row = P.loc[m].dropna()
        if row.empty:
            picks: list[str] = []
        else:
            ordered = sorted(row.to_dict().items(), key=lambda kv: kv[1], reverse=True)
            picks = [str(k) for k, _ in ordered[:K]]   # <- list[str]
        cohorts.append(picks)

        # Activos = unión con repetición por cohorte → 1/N por posición
        active = [tk for cohort in cohorts for tk in cohort]

        if len(active) == 0:
            r_m = 0.0
        else:
            # Serie de retornos del mes m (puede contener NaN)
            if m in R.index:
                r_row = R.loc[m]           # Serie
                # toma solo los presentes en 'active' y pásalo a ndarray de float
                valid = [tk for tk in active if tk in r_row.index]
                if len(valid) == 0:
                    r_m = 0.0
                else:
                    vals = r_row.reindex(valid).to_numpy(dtype=float)  # ndarray[float]
                    r_m = float(np.nanmean(vals)) if vals.size else 0.0
                    if not np.isfinite(r_m):
                        r_m = 0.0
            else:
                r_m = 0.0

        # Actualiza equity (base 1.0)
        eq.append((eq[-1] if eq else 1.0) * (1.0 + r_m))
        idx.append(m)

    equity = pd.Series(eq, index=pd.to_datetime(idx))
    ann = 12
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (ann / len(equity)) - 1
    mdd  = (equity / equity.cummax() - 1).min()
    print({"CAGR": float(cagr), "MaxDD": float(mdd)})

if __name__ == "__main__":
    main()
