# scripts/backtest_longterm.py
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, cast

import numpy as np
import pandas as pd
from pandas import DatetimeIndex, PeriodIndex

# ─────────────────────────── Paths ───────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
MLDIR = ROOT / "data" / "ml"
ARTS = ROOT / "artifacts" / "longterm"
ARTS.mkdir(parents=True, exist_ok=True)


# ───────────────────── Helpers de datos ──────────────────────
def monthly_returns(ticker: str) -> pd.Series:
    """
    Retorno mensual (fin de mes) para un ticker, usando precios 'Close'
    desde data/processed/<TICKER>.parquet.
    Devuelve una Series con índice DatetimeIndex (fin de mes).
    """
    fp = PROCESSED / f"{ticker}.parquet"
    df = pd.read_parquet(fp, engine="fastparquet")
    # Asegura índice de fechas
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    px = pd.to_numeric(df["Close"], errors="coerce").astype("float64")
    # 'ME' = month-end; evitamos FutureWarning de 'M'
    mclose = px.resample("ME").last()
    ret = mclose.pct_change()
    ret.name = ticker
    return ret


def load_predictions(horizon: int, tickers_filter: List[str] | None = None) -> pd.DataFrame:
    """
    Carga preds_{horizon}m.parquet -> columnas esperadas ['Date','Ticker','y_pred'].
    Devuelve una matriz P: index=DatetimeIndex (fin de mes), columns=tickers, values=y_pred.
    """
    fp = MLDIR / f"preds_{horizon}m.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"No existe {fp}. Genera predicciones con scripts/train_ml.py --horizon {horizon}")

    preds = pd.read_parquet(fp, engine="fastparquet").copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    if tickers_filter:
        preds = preds[preds["Ticker"].astype(str).isin([t.upper() for t in tickers_filter])]

    # Index mensual como Periodos y Pivot a matriz
    preds["Month"] = preds["Date"].dt.to_period("M")
    P = preds.pivot_table(index="Month", columns="Ticker", values="y_pred", aggfunc="mean").sort_index()

    # Normaliza a Timestamps de fin de mes
    pidx: PeriodIndex = pd.PeriodIndex(P.index, freq="M")
    didx: DatetimeIndex = pidx.to_timestamp(how="end")  # fin de mes
    P.index = didx
    # Ordena columnas por nombre de ticker para estabilidad
    P = P.sort_index(axis=1)
    return P


def backtest_longterm(
    horizon: int,
    topk: int,
    rebalance_every: int | None = None,
    tickers: List[str] | None = None,
    save_csv: bool = True,
    save_png: bool = False,
) -> dict[str, float]:
    H = int(horizon)
    K = int(topk)
    REBAL = int(rebalance_every if rebalance_every is not None else horizon)

    # 1) Matriz de predicciones P[m, tk]
    P = load_predictions(H, tickers_filter=tickers)

    # 2) Retornos mensuales R para el universo de P
    universe = [str(c) for c in P.columns]
    if not universe:
        raise ValueError("No hay tickers en las predicciones. Verifica preds_*.parquet.")

    R = pd.concat([monthly_returns(tk) for tk in universe], axis=1)
    R.index = DatetimeIndex(R.index)  # asegura DatetimeIndex
    R = R.sort_index().astype("float64")

    # 3) Meses comunes entre P y R
    # Asegura que P y R tengan DatetimeIndex
    P.index = DatetimeIndex(P.index)
    R.index = DatetimeIndex(R.index)
    # Intersección (Pylance entiende que devuelve DatetimeIndex)
    months_idx: DatetimeIndex = cast(DatetimeIndex, P.index.intersection(R.index))
    months = list(months_idx.sort_values())
    if not months:
        raise ValueError("No hay meses en común entre predicciones (P) y retornos (R).")

    # 4) Bucle: rebalancea y calcula equity
    current_cohort: list[str] = []
    last_reb_idx = -10**9
    eq_values: list[float] = []
    eq_dates: list[pd.Timestamp] = []

    for i, m in enumerate(months):
        if (i - last_reb_idx) >= REBAL and m in P.index:
            row = P.loc[m].dropna()
            current_cohort = [str(tk) for tk in row.sort_values(ascending=False).index[:K]] if not row.empty else []
            last_reb_idx = i

        if not current_cohort:
            r_m = 0.0
        else:
            row_all = R.loc[m]
            if not isinstance(row_all, pd.Series):
                row_all = pd.Series(row_all)
            r_vals = pd.to_numeric(row_all.reindex(current_cohort), errors="coerce").dropna().astype("float64")
            r_m = float(np.nanmean(r_vals.to_numpy())) if not r_vals.empty else 0.0

        eq_values.append((eq_values[-1] if eq_values else 1.0) * (1.0 + r_m))
        eq_dates.append(m)

    equity = pd.Series(eq_values, index=DatetimeIndex(eq_dates), name="equity")

    # 5) Métricas
    n_months = len(equity)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (12.0 / n_months) - 1.0)
    mdd = float((equity / equity.cummax() - 1.0).min())

    if save_csv:
        out_csv = ARTS / f"longterm_eq_H{H}_K{K}_R{REBAL}.csv"
        equity.to_csv(out_csv)
        print(f"▶ Equity CSV: {out_csv}")

    if save_png:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8,3))
            equity.plot(ax=ax)
            ax.set_title(f"LongTerm H={H}m, TopK={K}, Rebal={REBAL}m")
            ax.grid(alpha=0.3)
            out_png = ARTS / f"longterm_eq_H{H}_K{K}_R{REBAL}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"▶ Equity PNG: {out_png}")
        except Exception as e:
            print(f"(No se pudo guardar PNG: {e})")

    return {"CAGR": cagr, "MaxDD": mdd}


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest de inversión a largo plazo con predicciones mensuales.")
    ap.add_argument("--horizon", type=int, default=6, help="Meses objetivo de la predicción y de mantenimiento de la cohorte.")
    ap.add_argument("--topk", type=int, default=5, help="Número de tickers a comprar en cada rebalanceo.")
    ap.add_argument("--rebalance_every", type=int, default=None, help="Meses entre rebalanceos (por defecto = horizon).")
    ap.add_argument("--tickers", type=str, default="", help="Lista separada por comas para limitar el universo (opcional).")
    ap.add_argument("--no_csv", action="store_true", help="No guardar CSV de la curva de equity.")
    ap.add_argument("--save_png", action="store_true", help="Guardar PNG de la curva de equity.")
    args = ap.parse_args()

    tickers: List[str] | None = None
    if args.tickers.strip():
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    res = backtest_longterm(
        horizon=args.horizon,
        topk=args.topk,
        rebalance_every=args.rebalance_every,
        tickers=tickers,
        save_csv=not args.no_csv,
        save_png=args.save_png,
    )
    print(res)

if __name__ == "__main__":
    main()