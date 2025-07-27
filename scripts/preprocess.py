#!/usr/bin/env python
# scripts/preprocess.py

from __future__ import annotations
import sys
import pathlib
import pandas as pd
import ta
from typing import Optional

# ───────── Ajuste de ruta para importar módulos desde src ──────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.sentiment_analysis import daily_sentiment

# ───────── Rutas de datos ─────────────────────────────────────────
RAW_PRICE_PARQ = ROOT / "data/raw/kaggle/prices/parquet"
FUND_PARQ      = ROOT / "data/raw/simfin/fundamentals.parquet"
CONGRESS_PARQ  = ROOT / "data/raw/congress/trades.parquet"
NEWS_PARQ      = ROOT / "data/raw/news/news.parquet"
PROCESSED_DIR  = ROOT / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS = 15  # mínimo para indicadores técnicos


# ───────── 1. Indicadores técnicos ─────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < MIN_ROWS:
        out = df.copy()
        out["daily_return"] = out["Close"].pct_change()
        return out

    out = ta.add_all_ta_features(
        df.copy(),
        open="Open", high="High", low="Low",
        close="Close", volume="Volume",
        fillna=True,
    )
    out["daily_return"] = out["Close"].pct_change()
    return out


# ───────── 2. Fundamentales SimFin ─────────────────────────────────
def maybe_load_fundamentals() -> Optional[pd.DataFrame]:
    if not FUND_PARQ.exists():
        return None
    fund = pd.read_parquet(FUND_PARQ, engine="fastparquet").reset_index()
    cols = [
        "Ticker", "Fiscal Year", "FCF_margin", "Debt_to_Equity",
        "Gross_Margin", "Net_Margin", "Current_Ratio",
        "Interest_Coverage", "ROIC", "Revenue_CAGR",
    ]
    fund = fund[cols].drop_duplicates()
    fund["Date"] = pd.to_datetime(fund["Fiscal Year"].astype(str) + "-12-31")
    fund.set_index("Date", inplace=True)
    return fund

FUND = maybe_load_fundamentals()

def merge_fundamentals(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if FUND is None:
        return df
    f = FUND[FUND["Ticker"] == ticker]
    if f.empty:
        return df
    return (
        df.merge(
            f.drop(columns=["Ticker", "Fiscal Year"]),
            left_index=True, right_index=True, how="left"
        )
        .ffill()
    )


# ───────── 3. Flujos del Congreso ──────────────────────────────────
def maybe_load_congress() -> Optional[pd.DataFrame]:
    if not CONGRESS_PARQ.exists():
        return None
    from src.features.congress_flows import congress_flows
    trades = pd.read_parquet(CONGRESS_PARQ, engine="fastparquet")
    return congress_flows(trades)

CONGRESS = maybe_load_congress()

def merge_congress(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if CONGRESS is None:
        return df
    flows = CONGRESS[CONGRESS["Ticker"] == ticker]
    if flows.empty:
        return df
    return (
        df.merge(
            flows.drop(columns=["Ticker"]),
            left_index=True, right_index=True, how="left"
        )
        .fillna({"net_buy": 0, "buy_count": 0, "sell_count": 0})
    )


# ───────── 4. Sentimiento de noticias ──────────────────────────────
def maybe_load_news(use_gpu: bool = False) -> Optional[pd.DataFrame]:
    if not NEWS_PARQ.exists():
        return None
    raw = pd.read_parquet(NEWS_PARQ, engine="fastparquet")
    sent = daily_sentiment(raw, use_gpu=use_gpu, batch_size=64)
    # Indexamos por Date; Ticker queda como columna para poder filtrar
    sent = sent.set_index("Date")
    return sent

NEWS = maybe_load_news(use_gpu=False)

def merge_news(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if NEWS is None:
        return df
    n = NEWS[NEWS["Ticker"] == ticker]
    if n.empty:
        return df
    return (
        df.merge(
            n[["sentiment_score"]],
            left_index=True, right_index=True, how="left"
        )
        .fillna({"sentiment_score": 0})
    )


# ───────── 5. Procesar un ticker ───────────────────────────────────
def process_one(path: pathlib.Path) -> None:
    ticker = path.stem.upper()
    df = pd.read_parquet(path, engine="fastparquet").set_index("Date")

    # indicadores técnicos (a prueba de fallos)
    try:
        df = add_indicators(df)
    except Exception:
        df = df.copy()
        df["daily_return"] = df["Close"].pct_change()

    # merges externos
    df = merge_fundamentals(df, ticker)
    df = merge_congress(df, ticker)
    df = merge_news(df, ticker)

    # guardar
    out = PROCESSED_DIR / f"{ticker}.parquet"
    df.to_parquet(out, compression="zstd", engine="fastparquet")
    print(f"✅ processed {ticker}")


# ───────── 6. Bucle principal ──────────────────────────────────────
def main() -> None:
    files = sorted(RAW_PRICE_PARQ.glob("*.parquet"))
    if not files:
        print(f"❌ No se encontraron Parquet en {RAW_PRICE_PARQ}")
        return
    for fp in files:
        process_one(fp)

if __name__ == "__main__":
    main()
