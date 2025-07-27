# src/ingest/fetch_news.py

from __future__ import annotations
import os
from dotenv import load_dotenv
import pathlib, time, requests, pandas as pd, glob, click
from datetime import datetime, timedelta

# ─── 1) Carga de la API key ────────────────────────────────────────
load_dotenv()  
API_KEY = os.getenv("NEWSAPI_KEY")
if not API_KEY:
    raise RuntimeError("Falta definir NEWSAPI_KEY en el entorno o en .env")

# ─── 2) Rutas ─────────────────────────────────────────────────────
DATA_DIR     = pathlib.Path("data/raw/news")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PARQ         = DATA_DIR / "news.parquet"

# Donde guardas los precios ya procesados a parquet:
RAW_PRICES   = pathlib.Path("data/raw/kaggle/prices/parquet")

# ─── 3) Parámetros por defecto ────────────────────────────────────
FROM_DATE = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

# ─── 4) Función de consulta a NewsAPI ─────────────────────────────
def fetch_ticker_news(ticker: str) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": FROM_DATE,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": API_KEY,
        "pageSize": 100,
    }
    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    return res.json().get("articles", [])

# ─── 5) CLI con click ─────────────────────────────────────────────
@click.command()
@click.option(
    "--tickers",
    default=None,
    help="Lista separada por comas de tickers (ej. AAPL,MSFT). Si se omite, procesa todos los tickers de data/raw/kaggle/prices/parquet."
)
def main(tickers: str | None):
    # 5.1. Parseo del argumento
    if tickers:
        tickers_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        # Si no se pasa, leo todos los archivos .parquet en RAW_PRICES
        parquet_files = glob.glob(str(RAW_PRICES / "*.parquet"))
        tickers_list = [pathlib.Path(p).stem.upper() for p in parquet_files]

    if not tickers_list:
        print("❌ No se encontraron tickers para procesar.")
        return

    # 5.2. Fetch & acumulación
    rows = []
    for tk in tickers_list:
        articles = fetch_ticker_news(tk)
        for art in articles:
            rows.append({
                "Ticker": tk,
                "Date": art["publishedAt"][:10],
                "Title": art.get("title",""),
                "Description": art.get("description",""),
            })
        time.sleep(1)
        print(f"✔ {tk}: {len(articles)} articles")

    # 5.3. Guardar todo junto
    df = pd.DataFrame(rows)
    df.to_parquet(PARQ, compression="zstd", engine="fastparquet")
    print(f"✅ News guardado en {PARQ} (total tickers = {len(tickers_list)})")

if __name__ == "__main__":
    main()
