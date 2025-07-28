# scripts/make_ml_table.py
from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
MLDATA = ROOT / "data" / "ml"
MLDATA.mkdir(parents=True, exist_ok=True)

# Tipo que Pylance espera para el parámetro 'engine' en parquet
ParquetEngineStr = Literal["auto", "pyarrow", "fastparquet"]


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def parse_tickers(arg: str) -> List[str]:
    """
    'all' -> todos los *.parquet en data/processed
    'AAPL,MSFT' -> lista explícita
    """
    if arg.lower() == "all":
        return sorted([p.stem.upper() for p in PROCESSED.glob("*.parquet")])
    return [t.strip().upper() for t in arg.split(",") if t.strip()]


def parquet_engine() -> ParquetEngineStr:
    """Devuelve engine disponible para parquet, priorizando fastparquet."""
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except Exception:
        try:
            import pyarrow  # noqa: F401
            return "pyarrow"
        except Exception:
            # podríamos devolver "auto", pero si no hay engines instalados,
            # pandas fallará igual; preferimos avisar claro:
            raise RuntimeError(
                "No hay motor parquet disponible. Instala 'fastparquet' o 'pyarrow'."
            )


def load_price_df(ticker: str) -> Optional[pd.DataFrame]:
    """
    Lee data/processed/<TICKER>.parquet; devuelve DataFrame con índice Date (datetime)
    y columnas al menos: Open, High, Low, Close, Volume, Adj Close.
    Requiere 'Adj Close' para evitar artefactos por splits/dividendos.
    """
    fp = PROCESSED / f"{ticker}.parquet"
    if not fp.exists():
        print(f"[WARN] No existe {fp}")
        return None
    try:
        eng: ParquetEngineStr = parquet_engine()
        df = pd.read_parquet(fp, engine=eng).copy()
    except Exception as e:
        print(f"[WARN] No se pudo leer {fp}: {e}")
        return None

    # Normalizar índice
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(set(df.columns)):
        print(f"[WARN] Faltan columnas en {ticker}: {sorted(need - set(df.columns))}")
        return None

    if "Adj Close" not in df.columns:
        print(f"[WARN] {ticker} sin 'Adj Close' -> descartado (evita splits/dividendos).")
        return None

    return df.sort_index()


# ---------------------------------------------------------------------
# Feature engineering (mensual)
# ---------------------------------------------------------------------
def build_features_panel(
    ticker: str,
    horizon_m: int,
    min_hist_months: int = 60,
    clip_target_q: float = 0.0,  # 0 = sin clipping; p.ej. 0.01 recorta a [1%,99%]
) -> Optional[pd.DataFrame]:
    """
    - Resample a mensual (fin de mes).
    - Features: retornos, medias, momentum, volatilidad.
    - Target y_fwd_{horizon_m}m: retorno futuro decimal (pm[t+h]/pm[t] - 1).
    - min_hist_months: descarta tickers con muy poca historia mensual útil.
    - clip_target_q: si >0, recorta el target a quantiles simétricos (q, 1-q).
    """
    df = load_price_df(ticker)
    if df is None or df.empty:
        return None

    # Precio de referencia: 'Adj Close' obligatorio
    price_col = "Adj Close"
    px = df[price_col].astype(float)

    # A mensual (último valor del mes) usando MonthEnd ("ME")
    pm = px.resample("ME").last().dropna()
    if pm.empty or len(pm) < min_hist_months:
        return None

    # Retornos y medias móviles (sobre mensual)
    ret1 = pm.pct_change(1)
    ret3 = pm.pct_change(3)
    ret6 = pm.pct_change(6)
    ret12 = pm.pct_change(12)

    ma3 = pm.rolling(3).mean()
    ma6 = pm.rolling(6).mean()
    ma12 = pm.rolling(12).mean()

    mom3 = pm / ma3 - 1.0
    mom6 = pm / ma6 - 1.0
    mom12 = pm / ma12 - 1.0

    vol3 = ret1.rolling(3).std()
    vol6 = ret1.rolling(6).std()
    vol12 = ret1.rolling(12).std()

    # Target: retorno futuro a horizon_m meses (decimal, no %)
    y = pm.shift(-horizon_m) / pm - 1.0

    # DataFrame de salida
    ycol = f"y_fwd_{horizon_m}m"
    out = pd.DataFrame(
        {
            "Date": pm.index,
            "Ticker": ticker,
            "px": pm.values,
            "ret1": ret1.values,
            "ret3": ret3.values,
            "ret6": ret6.values,
            "ret12": ret12.values,
            "ma3": ma3.values,
            "ma6": ma6.values,
            "ma12": ma12.values,
            "mom3": mom3.values,
            "mom6": mom6.values,
            "mom12": mom12.values,
            "vol3": vol3.values,
            "vol6": vol6.values,
            "vol12": vol12.values,
            ycol: y.values,
        }
    )

    # Elimina NaNs (por ventanas e inevitable NaN del target al final)
    out = out.dropna(subset=[ycol]).reset_index(drop=True)

    # Clipping opcional del target para robustez frente a outliers
    if clip_target_q and 0.0 < clip_target_q < 0.5 and not out.empty:
        lo, hi = out[ycol].quantile([clip_target_q, 1 - clip_target_q])
        out[ycol] = out[ycol].clip(lo, hi)

    return out if not out.empty else None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Genera dataset ML mensual con target futuro en decimales."
    )
    ap.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Horizonte de predicción en meses (target futuro).",
    )
    ap.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="'all' o lista: AAPL,MSFT,NVDA",
    )
    ap.add_argument(
        "--min_hist_months",
        type=int,
        default=60,
        help="Mínimo de meses de historia mensual por ticker (default=60).",
    )
    ap.add_argument(
        "--clip_target_q",
        type=float,
        default=0.0,
        help="Quantil para clipping simétrico del target (0.0 = sin clipping, p.ej. 0.01).",
    )
    args = ap.parse_args()

    tickers = parse_tickers(args.tickers)
    if not tickers:
        raise SystemExit("No hay tickers a procesar (revisa data/processed o la lista).")

    frames: List[pd.DataFrame] = []
    kept, skipped = 0, 0
    for t in tickers:
        f = build_features_panel(
            t,
            args.horizon,
            min_hist_months=args.min_hist_months,
            clip_target_q=args.clip_target_q,
        )
        if f is not None and not f.empty:
            frames.append(f)
            kept += 1
        else:
            skipped += 1

    if not frames:
        raise SystemExit("No se generó dataset (¿hay datos válidos en data/processed y con 'Adj Close'?).")

    ds = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    # Guardar
    eng: ParquetEngineStr = parquet_engine()
    out_fp = MLDATA / f"dataset_{args.horizon}m.parquet"
    ds.to_parquet(out_fp, index=False, engine=eng)

    print(
        f"✅ Dataset ML guardado: {out_fp} — filas: {len(ds):,} | "
        f"Date en columnas: True | tickers OK={kept}, omitidos={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
