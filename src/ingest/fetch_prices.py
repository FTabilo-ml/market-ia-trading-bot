# ---------- src/ingest/fetch_prices.py ----------
from __future__ import annotations
import pathlib, glob, click
import pandas as pd

RAW_DIR     = pathlib.Path("data/raw/kaggle/prices")        # etfs/  stocks/
PARQUET_DIR = RAW_DIR / "parquet"                           # destino
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["Open"] == 0, "Open"] = df["Close"]
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df["Adj Close"] = df["Adj Close"].where(df["Adj Close"] > 0)
    return df

def csv_files(tickers: list[str] | None) -> list[pathlib.Path]:
    if tickers:
        files = []
        for t in tickers:
            files += glob.glob(str(RAW_DIR / "*" / f"{t}.csv"))
        return [pathlib.Path(f) for f in files]
    # sin filtro → todos
    return [pathlib.Path(p) for p in glob.glob(str(RAW_DIR / "*" / "*.csv"))]

@click.command()
@click.option("--tickers", default=None,
              help="AAPL,MSFT,…  (si se omite convierte todo el dataset)")
def main(tickers):
    tickers = [s.strip().upper() for s in tickers.split(",")] if tickers else None
    files = csv_files(tickers)
    if not files:
        print("❌ No se encontraron CSV."); return

    for csv in files:
        sym = csv.stem.upper()
        out = PARQUET_DIR / f"{sym}.parquet"
        if out.exists():
            continue
        df = pd.read_csv(csv, parse_dates=["Date"])
        df = clean(df)
        df.to_parquet(out, compression="zstd", engine="fastparquet")
        print(f"✔ {sym}")

    print("✅ Convertidos:", len(list(PARQUET_DIR.glob('*.parquet'))))

if __name__ == "__main__":
    main()
# ---------- fin ----------
