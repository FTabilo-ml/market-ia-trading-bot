# scripts/make_ml_table.py
from __future__ import annotations
import sys, pathlib, argparse
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data/processed"
MLDIR = ROOT / "data/ml"
MLDIR.mkdir(parents=True, exist_ok=True)

def build_rows(ticker: str, H: int) -> pd.DataFrame:
    """
    Devuelve mensual (ME) con target y_fwd_{H}m y Date como columna.
    """
    fp = PROCESSED / f"{ticker}.parquet"
    df = pd.read_parquet(fp, engine="fastparquet").copy()

    # Asegurar columna Date
    if "Date" not in df.columns:
        # si viene en el índice, rescatarlo
        df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Mensual a fin de mes
    m = (df.set_index("Date").resample("ME").last()).copy()
    m["Date"] = m.index  # <- columna explícita

    # Target: retorno a H meses
    px = pd.to_numeric(m["Close"], errors="coerce")
    y = (px.shift(-H) / px - 1).rename(f"y_fwd_{H}m")
    m[f"y_fwd_{H}m"] = y
    m["Ticker"] = ticker

    # Opcional: quedarnos con columnas numéricas + claves
    keep = ["Date", "Ticker", f"y_fwd_{H}m"]
    numcols = m.select_dtypes(include=[np.number]).columns.tolist()
    cols = keep + [c for c in numcols if c not in keep]
    out = m[cols].reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA")
    args = ap.parse_args()

    H = int(args.horizon)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    parts = []
    for tk in tickers:
        parts.append(build_rows(tk, H))
    ds = pd.concat(parts, ignore_index=True)

    # Filtra filas sin target
    ds = ds.dropna(subset=[f"y_fwd_{H}m"])

    outp = MLDIR / f"dataset_{H}m.parquet"
    ds.to_parquet(outp, engine="fastparquet", index=False)
    print(f"✅ Dataset ML guardado: {outp} — filas: {len(ds)} | Date en columnas: {'Date' in ds.columns}")

if __name__ == "__main__":
    main()
