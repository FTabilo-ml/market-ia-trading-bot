# scripts/train_ml.py
from __future__ import annotations
import sys, pathlib, argparse, pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

MLDIR = ROOT / "data/ml"; MLDIR.mkdir(exist_ok=True, parents=True)

def get_model():
    try:
        import lightgbm as lgb
        return "lgbm", lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        return "rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

def time_splits(dates: pd.Series, n_splits: int = 4):
    # Corta por fecha en cuantiles (expanding train, test adelante)
    qs = np.linspace(0.5, 0.9, n_splits)  # 50%..90% como cortes
    cuts = [dates.quantile(q) for q in qs]
    last = dates.max()
    for c in cuts:
        train_end = dates <= c
        test_period = (dates > c) & (dates <= dates.quantile(min(1.0, (dates.rank(pct=True).max()))))
        yield train_end, ~train_end  # simple: train hasta c, test después

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()

    data_path = MLDIR / f"dataset_{args.horizon}m.parquet"
    df = pd.read_parquet(data_path, engine="fastparquet").reset_index().rename(columns={"index":"Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    label = f"y_{args.horizon}m"
    y = df[label].astype(float)

    # Features numéricas (quita columnas obvias)
    drop = {label, "Ticker", "Close"}  # Close puede fugar info; ya hay TA
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[float, int]).fillna(0.0)

    name, model = get_model()
    print(f"[INFO] Modelo: {name}")

    # Split temporal sencillo: train = pasado, test = futuro
    # Aquí usamos una sola frontera: 80/20 por fecha
    cut = df["Date"].quantile(0.8)
    tr = df["Date"] <= cut
    te = df["Date"] > cut

    model.fit(X[tr], y[tr])
    preds = model.predict(X[te])

    out = df.loc[te, ["Date","Ticker"]].copy()
    out["y_true"] = y[te].values
    out["y_pred"] = preds
    out_path = MLDIR / f"preds_{args.horizon}m.parquet"
    out.to_parquet(out_path, compression="zstd", engine="fastparquet")
    print(f"✅ Predicciones guardadas en {out_path} (test {te.sum()} filas)")

if __name__ == "__main__":
    main()
