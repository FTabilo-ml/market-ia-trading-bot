# scripts/make_ml_table_longterm.py
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

def build_one(ticker: str, H: int) -> pd.DataFrame:
    fp = PROCESSED / f"{ticker}.parquet"
    df = pd.read_parquet(fp, engine="fastparquet")

    # Index mensual (fin de mes). Usamos 'ME' por compatibilidad pandas reciente.
    df["Date"] = pd.to_datetime(df.get("Date", df.index))
    df = df.set_index("Date").sort_index()

    # Toma último valor de cada mes (snapshot de fin de mes)
    m = df.resample("ME").last()

    # Precio y retornos mensuales
    px = pd.to_numeric(m["Close"], errors="coerce")
    r_m = px.pct_change()

    # Features sencillas adicionales (además de tus columnas existentes):
    m["mom_3m"]  = px.pct_change(3)
    m["mom_6m"]  = px.pct_change(6)
    m["mom_12m"] = px.pct_change(12)
    m["vol_6m"]  = r_m.rolling(6).std()
    m["vol_12m"] = r_m.rolling(12).std()

    # Target: retorno anualizado a H meses (shift negativo mira al futuro)
    m["y_forward"] = (px.shift(-H) / px) ** (12 / H) - 1

    # Limpieza mínima
    m["Ticker"] = ticker
    m = m.dropna(subset=["y_forward"])

    # Evita fuga de info: nos quedamos con snapshot de fin de mes
    # (todas las columnas ya están en "last of month")
    m = m.reset_index().rename(columns={"Date": "Date"})
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=12, help="Horizonte en meses (12/36/60/120)")
    ap.add_argument("--tickers", type=str, default="", help="Lista separada por comas; vacío=todo processed/")
    ap.add_argument("--min_months", type=int, default=60, help="mínimo de meses para incluir ticker")
    args = ap.parse_args()

    H = int(args.horizon)
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = [p.stem for p in PROCESSED.glob("*.parquet")]

    dfs = []
    for tk in tickers:
        try:
            d = build_one(tk, H)
            # filtra por longitud útil
            if d["Date"].nunique() >= args.min_months:
                dfs.append(d)
        except Exception:
            pass

    if not dfs:
        print("❌ No se generó dataset.")
        return

    out = pd.concat(dfs, ignore_index=True).sort_values(["Date", "Ticker"])
    outpath = MLDIR / f"dataset_{H}m_long.parquet"
    out.to_parquet(outpath, compression="zstd", engine="fastparquet")
    print(f"✅ Dataset ML largo plazo: {outpath} — filas: {len(out)} — tickers: {len(set(out['Ticker']))}")

if __name__ == "__main__":
    main()
