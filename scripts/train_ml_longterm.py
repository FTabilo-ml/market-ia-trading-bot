# scripts/train_ml_longterm.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Literal, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Intento de LightGBM con alias seguro para el type checker
_use_lgbm: bool = False
try:
    from lightgbm import LGBMRegressor as _LGBMRegressor  # type: ignore
    _use_lgbm = True
except Exception:
    _LGBMRegressor = None  # type: ignore
    _use_lgbm = False

from sklearn.ensemble import RandomForestRegressor
import joblib

# ---------------------------------------------------------------------
# Rutas y tipos
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_ML = ROOT / "data" / "ml"
ARTS = ROOT / "artifacts" / "ml"
ARTS.mkdir(parents=True, exist_ok=True)

ParquetEngineStr = Literal["auto", "pyarrow", "fastparquet"]


def parquet_engine() -> ParquetEngineStr:
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except Exception:
        try:
            import pyarrow  # noqa: F401
            return "pyarrow"
        except Exception:
            raise RuntimeError(
                "No hay motor parquet disponible. Instala 'fastparquet' o 'pyarrow'."
            )


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------
# Carga y split
# ---------------------------------------------------------------------
def load_dataset(horizon: int) -> pd.DataFrame:
    fp = DATA_ML / f"dataset_{horizon}m.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"No existe el dataset {fp}. Genera primero con make_ml_table.py")
    eng: ParquetEngineStr = parquet_engine()
    df = pd.read_parquet(fp, engine=eng)
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("El dataset necesita columnas 'Date' y 'Ticker'.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().all():
        raise ValueError("La columna 'Date' no pudo convertirse a datetime (todo NaN).")
    return df


def split_by_date(df: pd.DataFrame, split: str, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evita fuga temporal: para train solo usa filas cuya madurez
    (Date + horizon meses) es estrictamente anterior al split.
    """
    split_dt = pd.to_datetime(split)
    maturity = df["Date"] + pd.DateOffset(months=horizon)
    tr = df[maturity < split_dt].copy()
    te = df[df["Date"] >= split_dt].copy()
    if tr.empty or te.empty:
        raise ValueError(
            f"Split '{split}' genera particiones vacías: "
            f"train={len(tr)} test={len(te)}. Elige otra fecha o revisa el horizonte."
        )
    return tr, te


def find_target_col(df: pd.DataFrame, horizon: int) -> str:
    candidates = [f"y_fwd_{horizon}m", "y"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No se encontró columna objetivo. Probé {candidates}. "
        "Regenera el dataset con make_ml_table.py."
    )


# ---------------------------------------------------------------------
# Preprocesamiento y modelo
# ---------------------------------------------------------------------
def _make_ohe():
    """
    Crea OneHotEncoder compatible con versiones nuevas/antiguas de scikit-learn.
    - En >=1.2 existe 'sparse_output'.
    - En <1.2 se usa 'sparse'.
    """
    try:
        # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str], model_name: str) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", _make_ohe(), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # permite salida sparse si alguna trafo es sparse
    )

    if model_name == "lgbm" and _use_lgbm and _LGBMRegressor is not None:
        model = _LGBMRegressor(  # type: ignore[misc,call-arg]
            random_state=42,
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="huber",   # robusto a outliers en y
            # huber_delta=1.0,   # opcional: afina si quieres
        )
    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features=1.0,  # usar todas las features (válido en regresión)
        )
    else:
        # Fallback
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features=1.0,
        )

    return Pipeline(steps=[("prep", pre), ("model", model)])


# ---------------------------------------------------------------------
# Utilidades de entrenamiento
# ---------------------------------------------------------------------
def month_weights(dts: pd.Series) -> np.ndarray:
    """
    Pondera para que cada mes tenga peso total ≈ 1.
    Usa claves string de mes para evitar problemas de tipado con Period.
    """
    months = pd.to_datetime(dts).dt.to_period("M").astype(str)
    counts = months.value_counts()          # index: 'YYYY-MM'
    per_row_counts = months.map(counts)     # Serie con el conteo de su mes
    w = (1.0 / per_row_counts).to_numpy(dtype=float)
    return w


# ---------------------------------------------------------------------
# Entrenamiento + evaluación
# ---------------------------------------------------------------------
def train_and_eval(
    df: pd.DataFrame,
    horizon: int,
    split: str,
    model_name: str,
    tickers: Optional[List[str]] = None,
) -> None:
    y_col = find_target_col(df, horizon)

    if tickers:
        df = df[df["Ticker"].isin(tickers)].copy()
        if df.empty:
            raise ValueError("Tras filtrar 'tickers', no quedan filas.")

    df = df.dropna(subset=[y_col]).copy()

    # Split sin fuga temporal
    df_tr, df_te = split_by_date(df, split, horizon)

    drop_cols = {"Date", y_col}
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in ["Ticker"] if c in df.columns]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No hay columnas de entrada (X) válidas tras el filtrado de tipos.")

    Xtr = df_tr[numeric_cols + categorical_cols].copy()
    Xte = df_te[numeric_cols + categorical_cols].copy()

    # Transformación robusta del target
    eps = 1e-6
    ytr_raw = df_tr[y_col].astype(float).clip(lower=-0.99 + eps)  # evita log1p<=0
    yte_raw = df_te[y_col].astype(float).clip(lower=-0.99 + eps)
    ytr = np.log1p(ytr_raw)
    yte = np.log1p(yte_raw)

    # Ponderación por mes
    wtr = month_weights(df_tr["Date"])

    log(f"Dataset: filas={len(df):,} | train={len(Xtr):,} | test={len(Xte):,}")
    log(f"Fechas: train {df_tr['Date'].min().date()}..{df_tr['Date'].max().date()} | "
        f"test {df_te['Date'].min().date()}..{df_te['Date'].max().date()}")
    log(f"Cols numéricas ({len(numeric_cols)}): {numeric_cols[:8]}{'...' if len(numeric_cols)>8 else ''}")
    log(f"Cols categóricas ({len(categorical_cols)}): {categorical_cols}")

    pipe = build_pipeline(numeric_cols, categorical_cols, model_name=model_name)

    # Fit con sample_weight al estimador final
    try:
        pipe.fit(Xtr, ytr, model__sample_weight=wtr)
    except TypeError:
        # Algunos estimadores no aceptan sample_weight -> fit normal
        pipe.fit(Xtr, ytr)

    # Predicción en espacio transformado y des-transformación a retornos
    pred_tr_g = pipe.predict(Xtr)
    pred_te_g = pipe.predict(Xte)
    pred_tr = np.expm1(pred_tr_g)
    pred_te = np.expm1(pred_te_g)

    # Métricas en el espacio original
    ytr_true = np.expm1(ytr)
    yte_true = np.expm1(yte)

    mae_tr = mean_absolute_error(ytr_true, pred_tr)
    r2_tr = r2_score(ytr_true, pred_tr)
    mae_te = mean_absolute_error(yte_true, pred_te)
    r2_te = r2_score(yte_true, pred_te)

    log("== Métricas ==")
    log(f"Train: MAE={mae_tr:.4f} | R2={r2_tr:.4f}")
    log(f"Test : MAE={mae_te:.4f} | R2={r2_te:.4f}")

    model_name_eff = "lgbm" if (model_name == "lgbm" and _use_lgbm and _LGBMRegressor is not None) else "rf"
    model_fp = ARTS / f"longterm_{horizon}m_{model_name_eff}.pkl"
    joblib.dump(pipe, model_fp)
    log(f"✅ Modelo guardado: {model_fp}")

    # Guardar predicciones (y_pred en escala original)
    preds_tr = df_tr[["Date", "Ticker"]].copy()
    preds_tr["y_true"] = np.asarray(ytr_true, dtype=float)
    preds_tr["y_pred"] = pred_tr
    preds_tr["split"] = "train"

    preds_te = df_te[["Date", "Ticker"]].copy()
    preds_te["y_true"] = np.asarray(yte_true, dtype=float)
    preds_te["y_pred"] = pred_te
    preds_te["split"] = "test"

    preds = pd.concat([preds_tr, preds_te], axis=0, ignore_index=True)
    preds = preds.sort_values(["Date", "Ticker"])
    eng: ParquetEngineStr = parquet_engine()
    out_preds = DATA_ML / f"preds_{horizon}m_{model_name_eff}.parquet"
    preds.to_parquet(out_preds, index=False, engine=eng)
    log(f"✅ Predicciones guardadas: {out_preds}  (filas={len(preds):,})")


def parse_tickers_arg(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [t.strip().upper() for t in arg.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, help="Horizonte en meses (ej. 36)")
    ap.add_argument("--split", type=str, required=True, help="Fecha de corte (YYYY-MM-DD)")
    ap.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "rf"], help="Modelo a usar")
    ap.add_argument("--tickers", type=str, default="", help="Lista de tickers separados por coma (opcional)")
    args = ap.parse_args()

    df = load_dataset(args.horizon)
    tickers = parse_tickers_arg(args.tickers)
    train_and_eval(df, args.horizon, args.split, args.model, tickers)


if __name__ == "__main__":
    main()
