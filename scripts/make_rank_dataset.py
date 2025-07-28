# scripts/make_rank_dataset.py
from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import List, Optional, Literal, Set, Tuple, Dict

import numpy as np
import pandas as pd

# ───────────────────────── Paths ─────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
MLDATA = ROOT / "data" / "ml"
MLDATA.mkdir(parents=True, exist_ok=True)

ParquetEngineStr = Literal["auto", "pyarrow", "fastparquet"]


# ───────────────────── Utils / I/O ──────────────────────
def parquet_engine() -> ParquetEngineStr:
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except Exception:
        try:
            import pyarrow  # noqa: F401
            return "pyarrow"
        except Exception:
            raise RuntimeError("Instala 'fastparquet' o 'pyarrow' para Parquet.")

def parse_tickers(arg: str) -> List[str]:
    if arg.lower() == "all":
        return sorted([p.stem.upper() for p in PROCESSED.glob("*.parquet")])
    return [t.strip().upper() for t in arg.split(",") if t.strip()]

def read_exclude_list(fp: Optional[str]) -> Set[str]:
    if not fp:
        return set()
    path = Path(fp)
    if not path.exists():
        print(f"[WARN] exclude_tickers no existe: {path}")
        return set()
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            out.add(s)
    return out


# ─────────────── Feature engineering (mensual) ───────────────
def _monthly_panel_with_liquidity_and_sector(
    df_daily: pd.DataFrame,
    ticker: str,
    sector_col: Optional[str],
    horizon_m: int,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Devuelve:
      - DataFrame mensual con features + y_fwd_{horizon_m}m (puede traer NaN al final)
      - Serie 'sector' mensual (si sector_col existe), indexada por Date mensual
    """
    # Precio de referencia
    price_col = "Adj Close" if "Adj Close" in df_daily.columns else "Close"
    px = df_daily[price_col].astype(float)

    # Dollar volume diario
    dv_daily = (df_daily["Close"].astype(float) * df_daily["Volume"].astype(float)).rename("dv_daily")

    # A mensual (fin de mes)
    pm = px.resample("M").last().dropna()
    if pm.empty:
        return pd.DataFrame(), None

    # Liquidez mensual (mediana de dollar volume diario por mes)
    dv_median_m = dv_daily.resample("M").median().reindex(index=pm.index)

    # Retornos y MAs (sobre mensual)
    def r(k: int) -> pd.Series: return pm.pct_change(k).rename(f"ret_{k}")
    ret_1, ret_3, ret_6, ret_12, ret_24, ret_36 = (r(k) for k in [1, 3, 6, 12, 24, 36])

    ma_6 = pm.rolling(6).mean().rename("ma_6")
    ma_12 = pm.rolling(12).mean().rename("ma_12")
    ma_24 = pm.rolling(24).mean().rename("ma_24")

    px_ma6 = (pm / ma_6 - 1.0).rename("px_ma6")
    px_ma12 = (pm / ma_12 - 1.0).rename("px_ma12")
    px_ma24 = (pm / ma_24 - 1.0).rename("px_ma24")

    # Vol mensual de retornos (rolling)
    vol_6 = ret_1.rolling(6).std().rename("vol_6")
    vol_12 = ret_1.rolling(12).std().rename("vol_12")

    # Target: retorno futuro a horizon_m meses (decimal)
    ycol = f"y_fwd_{horizon_m}m"
    y = (pm.shift(-horizon_m) / pm - 1.0).rename(ycol)

    # Sector mensual (si existe)
    sector_m: Optional[pd.Series] = None
    if sector_col and sector_col in df_daily.columns:
        # toma último valor del mes y ffill para gaps
        sector_m = (
            df_daily[sector_col]
            .astype(str)     # o .astype("string")
            .resample("M")
            .last()
            .reindex(index=pm.index)
            .ffill()
        ).rename(sector_col)

    out = pd.DataFrame(
        {
            "Date": pm.index,
            "Ticker": ticker,
            "px": pm.values,
            "dv_median": dv_median_m.values,  # liquidez mensual
            "ret_1": ret_1.values,
            "ret_3": ret_3.values,
            "ret_6": ret_6.values,
            "ret_12": ret_12.values,
            "ret_24": ret_24.values,
            "ret_36": ret_36.values,
            "ma_6": ma_6.values,
            "ma_12": ma_12.values,
            "ma_24": ma_24.values,
            "px_ma6": px_ma6.values,
            "px_ma12": px_ma12.values,
            "px_ma24": px_ma24.values,
            "vol_6": vol_6.values,
            "vol_12": vol_12.values,
            ycol: y.values,
        }
    ).set_index("Date")

    if sector_m is not None:
        out[sector_col] = sector_m

    return out, sector_m


def build_features_panel(
    ticker: str,
    horizon_m: int,
    min_hist_months: int,
    min_price: float,
    liq_usd_median: float,
    exclude_set: Set[str],
    sector_col: Optional[str],
) -> Optional[pd.DataFrame]:
    fp = PROCESSED / f"{ticker}.parquet"
    if ticker in exclude_set:
        return None
    if not fp.exists():
        return None

    eng: ParquetEngineStr = parquet_engine()
    try:
        df = pd.read_parquet(fp, engine=eng)
    except Exception as e:
        print(f"[WARN] {ticker}: no se pudo leer ({e})")
        return None

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(set(df.columns)):
        return None

    df = df.sort_index()
    panel, _ = _monthly_panel_with_liquidity_and_sector(
        df, ticker, sector_col=sector_col, horizon_m=horizon_m
    )
    if panel.empty or len(panel) < max(24, min_hist_months):
        return None

    # Filtros de universo (por fila mensual)
    panel = panel[(panel["px"] >= float(min_price)) & (panel["dv_median"] >= float(liq_usd_median))]

    if panel.empty:
        return None

    panel = panel.reset_index()
    return panel


# ───────────── Z‑scores (global y por sector) ─────────────
def zscore_by_group(df: pd.DataFrame, group_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    def _z(g: pd.DataFrame) -> pd.DataFrame:
        gg = g.copy()
        for c in num_cols:
            # Fuerza dtype flotante (evita Series[Any])
            col = pd.to_numeric(gg[c], errors="coerce").astype("float64")

            mu: float = float(np.nanmean(col.to_numpy(dtype=float)))
            sd: float = float(np.nanstd(col.to_numpy(dtype=float)))

            if np.isfinite(sd) and sd > 0.0:
                z = (col - mu) / sd            # Series[float64] - float
                gg[c] = z.astype("float64")    # asigna float64
            else:
                gg[c] = 0.0                    # constante flotante

        return gg

    return (
        df.groupby(group_cols, dropna=False, as_index=False, sort=False)
          .apply(_z)
          .reset_index(drop=True)
    )

# ───────────────────────── Main ─────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Genera dataset mensual para ranking (incluye filas future).")
    ap.add_argument("--horizon", type=int, required=True, help="Horizonte en meses para y_fwd (ej. 36).")
    ap.add_argument("--tickers", type=str, required=True, help="'all' o lista: AAPL,MSFT,NVDA")
    ap.add_argument("--min_hist_months", type=int, default=60, help="Historia mínima mensual por ticker.")
    # Filtros de universo
    ap.add_argument("--min_price", type=float, default=3.0, help="Precio mínimo mensual (px) para incluir fila.")
    ap.add_argument("--liq_usd_median", type=float, default=1_000_000.0,
                    help="Dollar volume mediano mensual mínimo para incluir fila.")
    ap.add_argument("--exclude_tickers", type=str, default="", help="Archivo con tickers a excluir (uno por línea).")
    # Neutralización
    ap.add_argument("--zscore_global", action="store_true", help="Z‑score por mes (global).")
    ap.add_argument("--zscore_sector", action="store_true", help="Z‑score por mes x sector (sector_col requerido).")
    ap.add_argument("--sector_col", type=str, default="Sector", help="Nombre de columna sector en processed.")
    # Etiqueta
    ap.add_argument("--clip_target_q", type=float, default=0.0,
                    help="Clipping simétrico de y por quantil (0.0 = sin clipping).")
    args = ap.parse_args()

    tickers = parse_tickers(args.tickers)
    if not tickers:
        raise SystemExit("No hay tickers (revisa data/processed o la lista).")

    exclude_set = read_exclude_list(args.exclude_tickers)

    frames: List[pd.DataFrame] = []
    kept, skipped = 0, 0
    for t in tickers:
        f = build_features_panel(
            t, args.horizon, args.min_hist_months,
            min_price=args.min_price,
            liq_usd_median=args.liq_usd_median,
            exclude_set=exclude_set,
            sector_col=(args.sector_col if args.sector_col else None),
        )
        if f is not None and not f.empty:
            frames.append(f)
            kept += 1
        else:
            skipped += 1

    if not frames:
        raise SystemExit("No se generó dataset (¿filtros demasiado estrictos?).")

    ds = pd.concat(frames, ignore_index=True)
    ds.sort_values(["Date", "Ticker"], inplace=True)
    ds.reset_index(drop=True, inplace=True)

    # Clipping opcional del target (sin eliminar NaNs)
    ycol = f"y_fwd_{args.horizon}m"
    if args.clip_target_q and 0.0 < args.clip_target_q < 0.5 and ycol in ds.columns:
        non_na = ds[ycol].dropna()
        if not non_na.empty:
            lo, hi = non_na.quantile([args.clip_target_q, 1 - args.clip_target_q])
            ds.loc[ds[ycol].notna(), ycol] = ds.loc[ds[ycol].notna(), ycol].clip(lo, hi)

    # Z‑scores: selecciona columnas numéricas de entrada (sin tocar y ni dv/px si no quieres)
    drop_cols = {"Date", "Ticker", ycol}
    # Mantén sector sin estandarizar
    if args.sector_col:
        drop_cols.add(args.sector_col)

    num_cols = [c for c in ds.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(ds[c])]
    # OJO: no z-score a 'px' ni 'dv_median' si quieres que se usen como filtros/controles
    # si prefieres, elimínalos de num_cols:
    for c_ex in ["px", "dv_median"]:
        if c_ex in num_cols:
            num_cols.remove(c_ex)

    # z‑score global por mes
    if args.zscore_global and num_cols:
        ds = zscore_by_group(ds, group_cols=["Date"], num_cols=num_cols)

    # z‑score por sector
    if args.zscore_sector and args.sector_col and (args.sector_col in ds.columns) and num_cols:
        # aplica z-score dentro de (mes, sector) — pisa valores previos si también hiciste global
        ds = zscore_by_group(ds, group_cols=["Date", args.sector_col], num_cols=num_cols)

    # ── Guardar ──
    eng: ParquetEngineStr = parquet_engine()
    out_fp = MLDATA / f"dataset_rank_{args.horizon}m.parquet"
    ds.to_parquet(out_fp, index=False, engine=eng)

    # Stats
    months = ds["Date"].dt.to_period("M").nunique() if "Date" in ds.columns else np.nan
    y_non_na = int(ds[ycol].notna().sum()) if ycol in ds.columns else 0
    y_na = int(ds[ycol].isna().sum()) if ycol in ds.columns else 0

    print(
        f"✅ Dataset rank guardado: {out_fp} — filas: {len(ds):,} | meses únicos: {months} | "
        f"tickers OK={kept}, omitidos={skipped} | y no-NaN: {y_non_na:,} | y NaN (forward): {y_na:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
