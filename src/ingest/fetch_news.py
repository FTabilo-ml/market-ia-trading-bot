# src/ingest/fetch_news.py

from __future__ import annotations
import pathlib
import glob
import time
import click
import feedparser
import pandas as pd
from datetime import datetime, timedelta

# ─── 1) Rutas y constantes ─────────────────────────────────────────
DATA_DIR    = pathlib.Path("data/raw/news")
DATA_DIR.mkdir(exist_ok=True, parents=True)
OUT_PARQ    = DATA_DIR / "news.parquet"
RAW_PRICEP  = pathlib.Path("data/raw/kaggle/prices/parquet")

# Sólo los últimos 30 días
FROM_DATE   = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
RSS_URL_TPL = (
    "https://news.google.com/rss/search?"
    "q={query}&hl=en-US&gl=US&ceid=US:en"
)


# ─── 2) Extraer RSS de Google News ──────────────────────────────────
def fetch_google_news(ticker: str) -> list[dict]:
    url  = RSS_URL_TPL.format(query=f"{ticker}+stock")
    feed = feedparser.parse(url)
    articles: list[dict] = []

    for entry in feed.entries:
        # 2.1 Intentamos usar published_parsed
        published_parsed = getattr(entry, "published_parsed", None)
        if isinstance(published_parsed, time.struct_time):
            date = datetime(*published_parsed[:6]).strftime("%Y-%m-%d")
        else:
            # 2.2 Fallback: intentar parsear entry.published como string
            raw = getattr(entry, "published", "")
            try:
                # Ejemplo: 'Wed, 23 Jul 2025 12:34:56 GMT'
                dt = datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %Z")
                date = dt.strftime("%Y-%m-%d")
            except Exception:
                date = ""

        articles.append({
            "Ticker":  ticker,
            "Date":    date,
            "Title":   entry.get("title", ""),
            "Link":    entry.get("link", ""),
            "Summary": entry.get("summary", ""),
        })

    return articles


# ─── 3) CLI: acepta --tickers o lee todos los .parquet de precios ────
@click.command()
@click.option(
    "--tickers",
    default=None,
    help=(
      "Lista de tickers separados por coma (ej. AAPL,MSFT). "
      "Si no se pasa, procesará todos los archivos "
      "data/raw/kaggle/prices/parquet/*.parquet."
    )
)
def main(tickers: str | None):
    # 3.1 Preparo lista de tickers
    if tickers:
        tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        files = glob.glob(str(RAW_PRICEP / "*.parquet"))
        tickers_list = [pathlib.Path(f).stem.upper() for f in files]

    if not tickers_list:
        print("❌ No se encontraron tickers para procesar.")
        return

    # 3.2 Recolectar artículos
    rows: list[dict] = []
    for tk in tickers_list:
        arts = fetch_google_news(tk)
        rows.extend(arts)
        print(f"✔ {tk}: {len(arts)} articles")
        time.sleep(1)  # para no golpear demasiado rápido Google

    # 3.3 Guardar en Parquet
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(OUT_PARQ, index=False, compression="zstd", engine="fastparquet")
        print(f"✅ News guardado en {OUT_PARQ}")
    else:
        print("⚠️  No se encontraron artículos para los tickers indicados.")


if __name__ == "__main__":
    main()
