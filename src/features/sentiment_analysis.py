from __future__ import annotations
import pandas as pd
from typing import Optional, List
from transformers.pipelines import pipeline
from transformers.pipelines.text_classification import TextClassificationPipeline


# ─── Carga perezosa del pipeline FinBERT ─────────────────────────────────────
_classifier: Optional[TextClassificationPipeline] = None

def get_finbert_classifier(device: int = -1) -> TextClassificationPipeline:
    """
    device = -1 → CPU, 0 → primera GPU (si tienes PyTorch CUDA instalado).
    """
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=128,
            device=device,
        )
        print(f"[INFO] Modelo FinBERT cargado en {'GPU' if device >= 0 else 'CPU'}")
    return _classifier


def _score_one(result: dict) -> float:
    """Convierte la salida del pipeline (un dict) a score ∈ [-1, 1] ≈ pos - neg."""
    label = str(result.get("label", "")).lower()
    try:
        score = float(result.get("score", 0.0))
    except Exception:
        score = 0.0
    if label.startswith("positive"):
        return score
    if label.startswith("negative"):
        return -score
    return 0.0  # neutral u otros


def daily_sentiment(df: pd.DataFrame, use_gpu: bool = False, batch_size: int = 32) -> pd.DataFrame:
    """
    Espera columnas: ['Ticker','Date','Title', opcional: 'Description'|'Summary'|'Content'].
    Devuelve: ['Ticker','Date','sentiment_score'] con Date normalizada a medianoche.
    """
    df2 = df.copy()

    # Fecha a datetime y normalizada (00:00) para agrupar diario de forma robusta
    df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce").dt.normalize()

    # Series de texto seguras para el type checker (nunca str sueltos)
    if "Title" in df2.columns:
        title_series = df2["Title"].astype(str)
    else:
        title_series = pd.Series("", index=df2.index, dtype="string")

    if "Description" in df2.columns:
        desc_series = df2["Description"].astype(str)
    elif "Summary" in df2.columns:
        desc_series = df2["Summary"].astype(str)
    elif "Content" in df2.columns:
        desc_series = df2["Content"].astype(str)
    else:
        desc_series = pd.Series("", index=df2.index, dtype="string")

    df2["text_for_sentiment"] = (title_series.str.strip() + ". " + desc_series.str.strip()).str.strip()

    # Deduplicar textos para acelerar FinBERT
    uniq = df2[["text_for_sentiment"]].drop_duplicates().copy()
    clf = get_finbert_classifier(device=(0 if use_gpu else -1))
    print(f"[INFO] Clasificando {len(uniq)} textos únicos (batch={batch_size})…")
    raw = clf(uniq["text_for_sentiment"].tolist(), batch_size=batch_size)  # list[dict]
    uniq["sentiment_score"] = [ _score_one(r) for r in raw ]

    # Mapear de vuelta los scores a cada fila original
    df2 = df2.merge(uniq, on="text_for_sentiment", how="left")

    # Agregación por Ticker y por día
    daily = (
        df2.groupby(["Ticker", pd.Grouper(key="Date", freq="D")], as_index=False)
           .agg(sentiment_score=("sentiment_score", "mean"))
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily[["Ticker", "Date", "sentiment_score"]]
