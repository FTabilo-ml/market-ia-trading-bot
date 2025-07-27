# src/ingest/fetch_congress_trades.py
from __future__ import annotations
import pathlib, subprocess, json, requests, pandas as pd

DATA_DIR = pathlib.Path("data/raw/congress")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PARQ = DATA_DIR / "trades.parquet"

# Fuentes
SENATE_REPO = "https://github.com/timothycarambat/senate-stock-watcher-data.git"
SENATE_FILE = "aggregate/all_transactions.json"  # dentro del repo clonado

import requests, time

HOUSE_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"

def load_house() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketBot/1.0)"}
    for attempt in range(3):
        r = requests.get(HOUSE_URL, headers=headers, timeout=60)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            df["source"] = "house"
            return df
        time.sleep(2)           # pequeño backoff
    print("⚠️ Cámara: 403 persistente, se omite.")
    return pd.DataFrame()

def ensure_senate_repo() -> pathlib.Path:
    repo_dir = DATA_DIR / "senate-stock-watcher-data"
    if repo_dir.exists():
        subprocess.run(["git", "-C", repo_dir, "pull", "--quiet"], check=False)
    else:
        subprocess.run(["git", "clone", "--depth", "1", SENATE_REPO, repo_dir, "--quiet"], check=True)
    return repo_dir

def load_senate() -> pd.DataFrame:
    repo = ensure_senate_repo()
    fp = repo / SENATE_FILE
    data = json.loads(fp.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    df["source"] = "senate"
    return df



RENAME_MAP = {
    "ticker": "Ticker",
    "asset_description": "Ticker",
    "transaction_date": "TransactionDate",
    "disclosure_date": "PublicationDate",
    "reported_date": "PublicationDate",
    "type": "Type",
    "amount": "Amount",
    "senator": "Politician",
    "representative": "Politician",
}

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for k, v in RENAME_MAP.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Si tras renombrar no tenemos las columnas clave, abortamos limpio
    required = {"Ticker", "TransactionDate"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Filtra tickers válidos
    df = df[df["Ticker"].astype(str).str.fullmatch(r"[A-Z]{1,5}")]

    # Fechas: algunas filas pueden no tener PublicationDate → se permite NaT y luego se descarta
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    if "PublicationDate" in df.columns:
        df["PublicationDate"] = pd.to_datetime(df["PublicationDate"], errors="coerce")
    else:
        # fallback: si no existe, usa TransactionDate como proxy
        df["PublicationDate"] = df["TransactionDate"]

    df = df.dropna(subset=["TransactionDate", "PublicationDate"])

    if "Amount" in df.columns:
        df["AmountLow"] = (
            df["Amount"].astype(str)
              .str.extract(r"\$?([\d,]+)", expand=False)
              .str.replace(",", "", regex=False)
              .astype(float)
        )

    df["dir"] = df.get("Type", "").astype(str).str.lower().map(
        lambda x: 1 if "pur" in x else (-1 if "sale" in x else 0)
    ).fillna(0)

    return df[["Ticker", "TransactionDate", "PublicationDate", "dir", "AmountLow", "source"]]

def main():
    try:
        senate = load_senate()
    except Exception as e:
        print("⚠️ No se pudo obtener datos del Senado:", e)
        senate = pd.DataFrame()
    try:
        house = load_house()
    except Exception as e:
        print("⚠️ No se pudo obtener datos de la Cámara:", e)
        house = pd.DataFrame()

    if senate.empty and house.empty:
        print("❌ Sin datos de Congreso.")
        return

    merged = pd.concat([senate, house], ignore_index=True)
    norm = normalize(merged)
    norm.to_parquet(PARQ, compression="zstd", engine="fastparquet")
    print(f"✅ Congress trades guardados en {PARQ} ({len(norm)} filas)")

if __name__ == "__main__":
    main()
