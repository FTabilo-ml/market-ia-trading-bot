# scripts/train_ml_longterm.py
# Entrenamiento robusto para horizonte largo (m meses) con Pipeline sklearn.
# - Lee data/ml/dataset_{h}m.parquet (desde make_ml_table.py)
# - Split temporal por fecha (--split)
# - Preprocesa: OneHot para 'Ticker', pasa numéricas
# - Modelo: LGBMRegressor (si disponible) o RandomForestRegressor
# - Guarda modelo y predicciones para backtest

import argparse
import pathlib
from typing import Optional, List

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

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ML = ROOT / "data" / "ml"
ARTS = ROOT / "artifacts" / "ml"
ARTS.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dataset(horizon: int) -> pd.DataFrame:
    fp = DATA_ML / f"dataset_{horizon}m.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"No existe el dataset {fp}. Genera primero con make_ml_table.py")
    df = pd.read_parquet(fp, engine="fastparquet")
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("El dataset necesita columnas 'Date' y 'Ticker'.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().all():
        raise ValueError("La columna 'Date' no pudo convertirse a datetime (todo NaN).")
    return df


def split_by_date(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dt = pd.to_datetime(split)
    tr = df[df["Date"] < split_dt].copy()
    te = df[df["Date"] >= split_dt].copy()
    if tr.empty or te.empty:
        raise ValueError(
            f"Split '{split}' genera particiones vacías: "
            f"train={len(tr)} test={len(te)}. Elige otra fecha."
        )
    return tr, te


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str], model_name: str) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    if model_name == "lgbm" and _use_lgbm and _LGBMRegressor is not None:
        model = _LGBMRegressor(  # type: ignore[misc,call-arg]
            random_state=42,
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
        )
    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features=1.0,  # <- 'auto' no es válido en regresión, usamos 1.0 (todas las features)
        )
    else:
        # Fallback si pidieron lgbm pero no está disponible
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            max_features=1.0,
        )

    pipe = Pipeline(steps=[("prep", pre), ("model", model)])
    return pipe


def train_and_eval(
    df: pd.DataFrame,
    horizon: int,
    split: str,
    model_name: str,
    tickers: Optional[List[str]] = None,
) -> None:
    y_col = f"y_fwd_{horizon}m"
    if y_col not in df.columns:
        raise ValueError(f"Falta columna destino '{y_col}' en el dataset.")

    if tickers:
        df = df[df["Ticker"].isin(tickers)].copy()
        if df.empty:
            raise ValueError("Tras filtrar 'tickers', no quedan filas.")

    df = df.dropna(subset=[y_col]).copy()

    df_tr, df_te = split_by_date(df, split)

    drop_cols = {"Date", y_col}
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in ["Ticker"] if c in df.columns]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No hay columnas de entrada (X) válidas tras el filtrado de tipos.")

    Xtr = df_tr[numeric_cols + categorical_cols].copy()
    ytr = df_tr[y_col].astype(float).copy()
    Xte = df_te[numeric_cols + categorical_cols].copy()
    yte = df_te[y_col].astype(float).copy()

    if Xtr.empty or Xte.empty:
        raise ValueError("X de train o test vacío. Revisa 'split' o el filtrado.")
    if ytr.empty or yte.empty:
        raise ValueError("y de train o test vacío. Revisa 'split' o NaNs.")

    log(f"Dataset: filas={len(df)} | train={len(Xtr)} | test={len(Xte)}")
    log(f"Fechas: train {Xtr.index.min()}..{Xtr.index.max()} | test {Xte.index.min()}..{Xte.index.max()}")
    log(f"Cols numéricas ({len(numeric_cols)}): {numeric_cols[:8]}{'...' if len(numeric_cols)>8 else ''}")
    log(f"Cols categóricas ({len(categorical_cols)}): {categorical_cols}")

    pipe = build_pipeline(numeric_cols, categorical_cols, model_name=model_name)

    pipe.fit(Xtr, ytr)

    pred_tr = pipe.predict(Xtr)
    pred_te = pipe.predict(Xte)

    mae_tr = mean_absolute_error(ytr, pred_tr)
    r2_tr = r2_score(ytr, pred_tr)
    mae_te = mean_absolute_error(yte, pred_te)
    r2_te = r2_score(yte, pred_te)

    log("== Métricas ==")
    log(f"Train: MAE={mae_tr:.4f} | R2={r2_tr:.4f}")
    log(f"Test : MAE={mae_te:.4f} | R2={r2_te:.4f}")

    model_name_eff = "lgbm" if (model_name == "lgbm" and _use_lgbm and _LGBMRegressor is not None) else "rf"
    model_fp = ARTS / f"longterm_{horizon}m_{model_name_eff}.pkl"
    joblib.dump(pipe, model_fp)
    log(f"✅ Modelo guardado: {model_fp}")

    preds_tr = df_tr[["Date", "Ticker"]].copy()
    preds_tr["y_true"] = ytr.values
    preds_tr["y_pred"] = pred_tr
    preds_tr["split"] = "train"

    preds_te = df_te[["Date", "Ticker"]].copy()
    preds_te["y_true"] = yte.values
    preds_te["y_pred"] = pred_te
    preds_te["split"] = "test"

    preds = pd.concat([preds_tr, preds_te], axis=0, ignore_index=True)
    preds = preds.sort_values(["Date", "Ticker"])
    out_preds = DATA_ML / f"preds_{horizon}m_{model_name_eff}.parquet"
    preds.to_parquet(out_preds, index=False)
    log(f"✅ Predicciones guardadas: {out_preds}  (filas={len(preds)})")


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
