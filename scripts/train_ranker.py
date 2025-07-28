# scripts/train_ranker.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ndcg_score
import joblib

# --- LightGBM (requerido) ---
try:
    from lightgbm import LGBMRanker
except Exception as e:
    raise SystemExit(
        "LightGBM es requerido para este script. Instala con `pip install lightgbm`.\n"
        f"Detalle: {e}"
    )

# ---------------------------------------------------------------------
# Paths por defecto
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_ML = ROOT / "data" / "ml"
ARTS = ROOT / "artifacts" / "rank"
ARTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def month_key(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").astype(str)


def make_ohe_compat() -> OneHotEncoder:
    """
    Devuelve OneHotEncoder compatible con sklearn >=1.2 (sparse_output)
    y anteriores (sparse). Evita warnings de tipo.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_rel_labels_per_month(
    y: pd.Series, month: pd.Series, bins: int
) -> np.ndarray:
    """
    Convierte retornos continuos (y) en etiquetas ordinales 0..bins-1 por mes (cuantiles).
    """
    if bins < 2:
        raise ValueError("rel_bins debe ser >= 2")

    df_ = pd.DataFrame({"y": y.values, "m": month.values})
    out = np.full(len(df_), np.nan, dtype=float)

    # Respetar el orden: asumimos que el df ya viene ordenado por 'month'
    for m, g in df_.groupby("m", sort=False):
        vals = g["y"].to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            continue
        # ranking ascendente dentro del mes
        ranks = np.argsort(np.argsort(vals[valid]))
        frac = (ranks + 1) / (valid.sum() + 1e-9)
        lab = np.minimum(bins - 1, (frac * bins).astype(int))
        tmp = np.full_like(vals, np.nan, dtype=float)
        tmp[valid] = lab
        out[g.index.values] = tmp

    out = np.where(np.isnan(out), -1, out)
    return out.astype(int)


def groups_from_sorted_month(series_month: pd.Series) -> np.ndarray:
    """
    Devuelve tamaños por mes en el MISMO orden en que aparecen las filas.
    Requiere que el dataframe esté ordenado por 'month'.
    """
    uniq = series_month.drop_duplicates()
    sizes = series_month.value_counts(sort=False)
    sizes = sizes.reindex(uniq, fill_value=0)
    return sizes.to_numpy(dtype=int)


def monthly_ic(
    df: pd.DataFrame, score_col: str, y_col: str
) -> Tuple[float, float, int, pd.DataFrame]:
    """
    IC (Spearman) por mes. Devuelve (mean, std, n_meses, detalle_por_mes).
    """
    rows: List[Dict[str, Any]] = []
    for m, g in df.groupby("month", sort=True):
        s = g[score_col].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)
        if len(s) < 2 or len(y) < 2:
            continue
        if np.all(np.isnan(s)) or np.all(np.isnan(y)):
            continue
        ic = pd.Series(s).corr(pd.Series(y), method="spearman")
        if np.isfinite(ic):
            rows.append({"month": str(m), "ic": float(ic)})
    det = pd.DataFrame(rows)
    if det.empty:
        return float("nan"), 0.0, 0, det
    mean_ic = float(det["ic"].mean())
    std_ic = float(det["ic"].std(ddof=0)) if len(det) > 1 else 0.0
    return mean_ic, std_ic, int(len(det)), det


def ndcg_by_month(
    df: pd.DataFrame, score_col: str, rel_col: str, ks: List[int]
) -> Dict[int, float]:
    """
    NDCG mensual usando relevancias no-negativas (rel_col = etiquetas ordinales).
    """
    out: Dict[int, List[float]] = {k: [] for k in ks}
    for m, g in df.groupby("month", sort=True):
        y_true = g[rel_col].to_numpy(dtype=float).reshape(1, -1)
        y_score = g[score_col].to_numpy(dtype=float).reshape(1, -1)
        if np.isnan(y_true).any() or np.isnan(y_score).any():
            continue
        for k in ks:
            k_eff = max(1, min(k, y_true.shape[1]))
            out[k].append(float(ndcg_score(y_true, y_score, k=k_eff)))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in out.items()}


def top_bottom_spread(
    df: pd.DataFrame, score_col: str, y_col: str, k: int
) -> Tuple[float, float]:
    """
    Promedio (TopK - BottomK) por mes y % de meses con spread positivo.
    """
    spreads: List[float] = []
    pos = 0
    for m, g in df.groupby("month", sort=True):
        gg = g.sort_values(score_col, ascending=False)
        k_eff = max(1, min(k, len(gg)))
        top_y = float(gg.head(k_eff)[y_col].mean())
        bot_y = float(gg.tail(k_eff)[y_col].mean())
        sp = top_y - bot_y
        spreads.append(sp)
        if sp > 0:
            pos += 1
    if not spreads:
        return float("nan"), float("nan")
    return float(np.mean(spreads)), float(pos / len(spreads))


def parse_eval_k(s: str) -> List[int]:
    if not s:
        return [5, 10]
    return [int(x) for x in s.split(",") if x.strip()]


def parquet_engine() -> Literal["auto", "pyarrow", "fastparquet"]:
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except Exception:
        try:
            import pyarrow  # noqa: F401
            return "pyarrow"
        except Exception:
            return "auto"


# ---------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------
def train_and_eval(
    dataset_fp: Path,
    horizon: int,
    split: str,
    rel_bins: int,
    ks: List[int],
    sign_from: Literal["val", "test"],
    val_months: int,
    sign_threshold: float,
    score_future: bool,
    num_leaves: Optional[int],
    min_data_in_leaf: Optional[int],
    reg_lambda: Optional[float],
    reg_alpha: Optional[float],
    learning_rate: float,
    n_estimators: int,
) -> None:
    y_col = f"y_fwd_{horizon}m"

    # Carga
    eng = parquet_engine()
    df = pd.read_parquet(dataset_fp, engine=eng)
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("El dataset requiere columnas 'Date' y 'Ticker'.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["month"] = month_key(df["Date"])

    # Split temporal
    df["split"] = "train"
    cut = pd.to_datetime(split)
    df.loc[(df["Date"] >= cut) & (df[y_col].notna()), "split"] = "test"
    df.loc[(df["Date"] >= cut) & (df[y_col].isna()), "split"] = "future"

    tr = df[(df["split"] == "train") & df[y_col].notna()].copy()
    te = df[(df["split"] == "test") & df[y_col].notna()].copy()
    fu = df[(df["split"] == "future")].copy()

    # Ordenar por mes (y ticker para determinismo)
    tr = tr.sort_values(["month", "Ticker"]).reset_index(drop=True)
    te = te.sort_values(["month", "Ticker"]).reset_index(drop=True)
    fu = fu.sort_values(["month", "Ticker"]).reset_index(drop=True)

    log(f"Dataset rank: filas={len(df):,} | train={len(tr):,} | test={len(te):,}")
    log(
        "Fechas: "
        f"train {tr['Date'].min().date() if not tr.empty else 'NA'}..{tr['Date'].max().date() if not tr.empty else 'NA'} | "
        f"test {te['Date'].min().date() if not te.empty else 'NA'}..{te['Date'].max().date() if not te.empty else 'NA'}"
    )

    # Columnas (solo numéricas + 'Ticker' categórica)
    drop_cols = {"Date", "Ticker", y_col, "split", "month"}
    numeric_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in ["Ticker"] if c in df.columns]

    log(f"Cols numéricas ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")
    log(f"Cols categóricas ({len(categorical_cols)}): {categorical_cols}")

    # Preprocesador
    pre = ColumnTransformer(
        transformers=[
            ("cat", make_ohe_compat(), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    # Etiquetas ordinales y grupos por mes
    tr["rel_label"] = make_rel_labels_per_month(tr[y_col], tr["month"], rel_bins)
    te["rel_label"] = make_rel_labels_per_month(te[y_col], te["month"], rel_bins)

    g_tr = groups_from_sorted_month(tr["month"])
    g_te = groups_from_sorted_month(te["month"]) if not te.empty else None
    assert g_tr.sum() == len(tr), "group train no cuadra con filas"
    if g_te is not None:
        assert g_te.sum() == len(te), "group test no cuadra con filas"

    # Label gain escalado hacia la cola alta
    gains = [0] + [int(round(2 ** (i - 1))) for i in range(1, rel_bins)]
    log(f"Rel bins={rel_bins} | label_gain={gains}")

    # Ranker
    model = LGBMRanker(
        objective="lambdarank",
        random_state=42,
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        subsample=0.9,
        colsample_bytree=0.8,
        metric="ndcg",
        eval_at=ks,
        label_gain=gains,
        n_jobs=-1,
        # Regularización opcional
        num_leaves=num_leaves if num_leaves is not None else 63,
        min_data_in_leaf=min_data_in_leaf if min_data_in_leaf is not None else 50,
        reg_lambda=reg_lambda if reg_lambda is not None else 0.0,
        reg_alpha=reg_alpha if reg_alpha is not None else 0.0,
    )

    # Fit preprocessor en TRAIN y transforma
    Xtr = tr[numeric_cols + categorical_cols]
    pre.fit(Xtr)
    Xtr_t = pre.transform(Xtr)
    ytr = tr["rel_label"].astype(int).values

    eval_sets = None
    eval_groups = None
    if not te.empty:
        Xte = te[numeric_cols + categorical_cols]
        Xte_t = pre.transform(Xte)
        yte = te["rel_label"].astype(int).values
        eval_sets = [(Xte_t, yte)]
        eval_groups = [g_te]

    # Entrenamiento
    if eval_sets is not None:
        model.fit(Xtr_t, ytr, group=g_tr, eval_set=eval_sets, eval_group=eval_groups)
    else:
        model.fit(Xtr_t, ytr, group=g_tr)

    # Pipeline final para .predict()
    pipe = Pipeline([("prep", pre), ("model", model)])

    # Scoring (raw) - usar np.asarray para evitar que Pylance crea que es tuple
    tr["raw_score"] = np.asarray(pipe.predict(tr[numeric_cols + categorical_cols]), dtype=float)
    if not te.empty:
        te["raw_score"] = np.asarray(pipe.predict(te[numeric_cols + categorical_cols]), dtype=float)
    if score_future and not fu.empty:
        fu["raw_score"] = np.asarray(pipe.predict(fu[numeric_cols + categorical_cols]), dtype=float)

    # Chequeo de signo
    if sign_from == "val":
        months_tr = sorted(tr["month"].unique())
        val_set = set(months_tr[-max(1, val_months):])
        chk = tr[tr["month"].isin(val_set)].copy()
    else:
        chk = te.copy()

    direction = +1
    if not chk.empty:
        chk["score_tmp"] = chk["raw_score"]
        ic_mean, ic_std, ic_n, _ = monthly_ic(chk, "score_tmp", y_col)
        log(f"[SIGN CHECK={sign_from}] IC mean={ic_mean:.4f}, std={ic_std:.4f}, n_months={ic_n}")
        if np.isfinite(ic_mean) and ic_mean < -abs(float(sign_threshold)):
            direction = -1
            log("→ Señal invertida (flip) por IC medio negativo.")

    # Aplica signo final
    for fr in (tr, te, fu):
        if fr is not None and not fr.empty:
            fr["score"] = direction * fr["raw_score"]

    # Métricas en TEST (post-flip)
    if not te.empty:
        ic_mean_te, ic_std_te, ic_n_te, _ = monthly_ic(te, "score", y_col)
        ndcg_te = ndcg_by_month(te, "score", "rel_label", ks)  # usar rel_label
        log(f"IC mensual (test) — mean={ic_mean_te:.4f}, std={ic_std_te:.4f}, n={ic_n_te}")
        log("NDCG@K (test) promedio por mes: " + ", ".join([f"ndcg@{k}={ndcg_te[k]:.3f}" for k in ks]))
        for k in ks:
            sp, hit = top_bottom_spread(te, "score", y_col, k)
            log(f"Spread Top{k}-Bottom{k} — mean={sp:.4f}, hit%={hit:.1%}")

    # Guardado
    model_fp = ARTS / f"ranker_{horizon}m_lgbm.pkl"
    joblib.dump(pipe, model_fp)
    log(f"✅ Modelo rank guardado: {model_fp}")

    preds_parts = []
    for fr in (tr, te, fu):
        if fr is not None and not fr.empty:
            preds_parts.append(fr[["Date", "Ticker", "split", "raw_score", "score"]])
    preds = pd.concat(preds_parts, ignore_index=True).sort_values(["Date", "Ticker"])

    out_fp = DATA_ML / f"preds_rank_{horizon}m.parquet"
    eng_w = parquet_engine()  # siempre un Literal permitido
    preds.to_parquet(out_fp, index=False, engine=eng_w)
    log(f"✅ Scores guardados: {out_fp} (filas={len(preds):,})")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena un ranker (LGBMRanker) mensual con etiquetas ordinales por cuantiles.")
    ap.add_argument("--horizon", type=int, required=True, help="Horizonte en meses (p.ej., 36).")
    ap.add_argument("--split", type=str, required=True, help="Fecha de corte (YYYY-MM-DD).")
    ap.add_argument("--rel_bins", type=int, default=4, help="Número de cuantiles por mes para etiquetas ordinales (>=2).")
    ap.add_argument("--eval_k", type=str, default="5,10", help="Lista de K para NDCG, separado por coma. Ej: 5,10")
    ap.add_argument("--sign_from", type=str, default="val", choices=["val", "test"], help="Conjunto para chequeo de signo.")
    ap.add_argument("--val_months", type=int, default=24, help="Meses recientes de train para chequeo de signo (si sign_from=val).")
    ap.add_argument("--sign_threshold", type=float, default=0.01, help="Si IC medio < -threshold se invierte la señal.")
    ap.add_argument("--score_future", action="store_true", help="Puntuar filas future (y NaN) para backtest.")
    ap.add_argument("--dataset_fp", type=str, default="", help="Ruta alternativa al dataset parquet (si no, usa data/ml/dataset_rank_{H}m.parquet).")

    # Regularización / Hiperparámetros principales
    ap.add_argument("--num_leaves", type=int, default=63, help="LightGBM num_leaves (menor => más suave).")
    ap.add_argument("--min_data_in_leaf", type=int, default=50, help="LightGBM min_data_in_leaf (mayor => más suave).")
    ap.add_argument("--reg_lambda", type=float, default=0.0, help="L2 regularization.")
    ap.add_argument("--reg_alpha", type=float, default=0.0, help="L1 regularization.")
    ap.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate.")
    ap.add_argument("--n_estimators", type=int, default=600, help="Número de árboles.")

    args = ap.parse_args()

    ks = parse_eval_k(args.eval_k)
    dataset_fp = Path(args.dataset_fp) if args.dataset_fp else (DATA_ML / f"dataset_rank_{args.horizon}m.parquet")
    if not dataset_fp.exists():
        raise SystemExit(f"No existe dataset: {dataset_fp}")

    train_and_eval(
        dataset_fp=dataset_fp,
        horizon=args.horizon,
        split=args.split,
        rel_bins=int(args.rel_bins),
        ks=ks,
        sign_from="val" if args.sign_from == "val" else "test",
        val_months=int(args.val_months),
        sign_threshold=float(args.sign_threshold),
        score_future=bool(args.score_future),
        num_leaves=int(args.num_leaves) if args.num_leaves else None,
        min_data_in_leaf=int(args.min_data_in_leaf) if args.min_data_in_leaf else None,
        reg_lambda=float(args.reg_lambda) if args.reg_lambda is not None else None,
        reg_alpha=float(args.reg_alpha) if args.reg_alpha is not None else None,
        learning_rate=float(args.learning_rate),
        n_estimators=int(args.n_estimators),
    )


if __name__ == "__main__":
    main()
