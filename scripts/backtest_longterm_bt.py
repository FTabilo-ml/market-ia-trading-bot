# scripts/backtest_longterm_bt.py
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Literal, cast

import numpy as np
import pandas as pd
import backtrader as bt
from backtrader.feeds import PandasData
from backtrader.utils.date import num2date


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
MLDATA = ROOT / "data" / "ml"
ARTS = ROOT / "artifacts" / "longterm"
ARTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Parquet engine (tipado estricto para Pylance)
# ---------------------------------------------------------------------
ParquetEngine = Literal["auto", "pyarrow", "fastparquet"]


def _parquet_engine() -> ParquetEngine:
    try:
        import fastparquet  # noqa: F401
        return cast(ParquetEngine, "fastparquet")
    except Exception:
        try:
            import pyarrow  # noqa: F401
            return cast(ParquetEngine, "pyarrow")
        except Exception:
            return cast(ParquetEngine, "auto")


# ---------------------------------------------------------------------
# Utilidades auxiliares
# ---------------------------------------------------------------------
def safe_nanpercentile(arr_like: Any, q: float) -> float:
    """
    Percentil robusto para Pylance: convierte a ndarray[float64] y usa nanpercentile.
    q debe venir en [0, 100].
    """
    arr = pd.to_numeric(pd.Series(arr_like), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    qf = float(q)
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, qf))


def last_business_day_before(df: pd.DataFrame, ref_dt: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Último día <= ref_dt en el índice del DataFrame (ya ordenado)."""
    try:
        idx = df.index[df.index <= ref_dt]
        if len(idx) == 0:
            return None
        return cast(pd.Timestamp, idx[-1])
    except Exception:
        return None


def month_end_from_key(month_key: str) -> pd.Timestamp:
    """'YYYY-MM' -> último instante del mes."""
    return pd.Period(str(month_key), freq="M").end_time


# ---------------------------------------------------------------------
# Utilidades de mercado
# ---------------------------------------------------------------------
def load_price_df(ticker: str) -> Optional[pd.DataFrame]:
    fp = PROCESSED / f"{ticker}.parquet"
    if not fp.exists():
        print(f"[WARN] No existe {fp}")
        return None
    eng = _parquet_engine()
    try:
        df = pd.read_parquet(fp, engine=eng).copy()
    except Exception as e:
        print(f"[WARN] No se pudo leer {fp}: {e}")
        return None

    # normalizar índice y columnas
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c not in df.columns:
            print(f"[WARN] Falta columna {c} en {fp}")
            return None

    out = df[cols].sort_index()
    out = out.dropna()
    return out


def calc_vol_daily(ticker: str, ref_month: str, lookback_days: int = 63) -> Optional[float]:
    """Volatilidad diaria (std de rendimientos diarios) en ventana previa a ref_month."""
    df = load_price_df(ticker)
    if df is None or df.empty:
        return None
    ref_end = month_end_from_key(ref_month)
    last_dt = last_business_day_before(df, ref_end)
    if last_dt is None:
        return None
    win = df.loc[df.index <= last_dt].tail(int(lookback_days))
    if win.empty or len(win) < 2:
        return None
    ret = win["Close"].pct_change().dropna()
    stdv = np.asarray(ret.std(), dtype=np.float64).item()  # numpy scalar -> float
    if ret.empty:
        return None
    return float(ret.std())


def calc_adv_usd(ticker: str, ref_month: str, adv_days: int = 5) -> Optional[float]:
    """ADV (promedio USD) últimos adv_days antes de ref_month."""
    df = load_price_df(ticker)
    if df is None or df.empty:
        return None
    ref_end = month_end_from_key(ref_month)
    last_dt = last_business_day_before(df, ref_end)
    if last_dt is None:
        return None
    win = df.loc[df.index <= last_dt].tail(int(max(1, adv_days)))
    if win.empty:
        return None
    adv = np.asarray((win["Close"] * win["Volume"]).mean(), dtype=np.float64).item()
    return float(adv)


def get_price_at_rebalance(ticker: str, ref_month: str) -> Optional[float]:
    df = load_price_df(ticker)
    if df is None or df.empty:
        return None
    ref_end = month_end_from_key(ref_month)
    last_dt = last_business_day_before(df, ref_end)
    if last_dt is None:
        return None
    val = np.asarray(df.loc[last_dt, "Close"], dtype=np.float64).item()
    return val


# ---------------------------------------------------------------------
# Señales (predicciones -> pesos)
# ---------------------------------------------------------------------
def build_signals(
    horizon: int,
    model: str,
    topk: int,
    only_test: bool,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    score_pctl_min: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Devuelve:
      signals: dict["YYYY-MM" -> dict["TICKER" -> weight]] (pesos preliminares iguales)
      months: lista ordenada de meses "YYYY-MM" presentes
    """
    # Localiza archivo de predicciones
    cand: List[Path] = []
    if model.lower() == "rank":
        cand.append(MLDATA / f"preds_rank_{horizon}m.parquet")
        cand.append(MLDATA / f"preds_{horizon}m_rank.parquet")  # fallback
    else:
        cand.append(MLDATA / f"preds_{horizon}m_{model}.parquet")
        cand.append(MLDATA / f"preds_rank_{horizon}m.parquet")  # si quieres forzar ranker

    preds_fp = next((p for p in cand if p.exists()), None)
    if preds_fp is None:
        print(f"[WARN] No encontré archivo de predicciones en: {[str(p) for p in cand]}")
        return {}, []

    # Carga
    eng = _parquet_engine()
    try:
        preds = pd.read_parquet(preds_fp, engine=eng)
    except Exception:
        preds = pd.read_parquet(preds_fp)

    df = preds.copy()
    if "Date" not in df.columns or "Ticker" not in df.columns:
        print(f"[WARN] Faltan columnas 'Date'/'Ticker' en {preds_fp}")
        return {}, []

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Usa 'score' si existe; si no, 'y_pred'
    use_col = "score" if "score" in df.columns else ("y_pred" if "y_pred" in df.columns else None)
    if use_col is None:
        print(f"[WARN] No encuentro ni 'score' ni 'y_pred' en {preds_fp}")
        return {}, []

    df = df.dropna(subset=["Date", "Ticker", use_col])
    if df.empty:
        return {}, []

    # Solo test
    if only_test:
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        elif "is_test" in df.columns:
            df = df[df["is_test"].astype(bool)].copy()

    # Rango de fechas
    if date_from:
        df = df[df["Date"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["Date"] <= pd.to_datetime(date_to)]

    if df.empty:
        return {}, []

    df["month_key"] = df["Date"].dt.to_period("M").astype(str)

    # Señales por mes
    signals: Dict[str, Dict[str, float]] = {}
    months: List[str] = []
    for k, g in df.groupby("month_key", sort=True, dropna=False):
        gg = g.copy()

        # Filtrado por percentil mensual del score
        if score_pctl_min is not None:
            p = float(score_pctl_min)
            if 0.0 < p < 1.0 and not gg.empty:
                thr = safe_nanpercentile(gg[use_col], 100.0 * p)
                gg = gg[gg[use_col] >= thr]

        gg = gg.sort_values(use_col, ascending=False).head(max(1, int(topk)))
        if gg.empty:
            continue

        w = 1.0 / float(len(gg))
        signals[str(k)] = {str(t): float(w) for t in gg["Ticker"].tolist()}
        months.append(str(k))

    months = sorted(set(months))
    return signals, months


# ---------------------------------------------------------------------
# PNG por ticker (marcas BUY/SELL)
# ---------------------------------------------------------------------
def save_per_ticker_pngs(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    horizon: int,
    model: str,
    limit: Optional[int] = None,
) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = pd.DataFrame(rows)
    if tr.empty or "ticker" not in tr.columns or "date" not in tr.columns:
        return

    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
    tickers = [t for t in sorted(tr["ticker"].dropna().unique()) if t]

    out_dir.mkdir(parents=True, exist_ok=True)

    def _y_at_dates(series: pd.Series, dates: pd.Series) -> pd.Series:
        dts = pd.to_datetime(dates)
        return series.reindex(dts, method="nearest")

    created = 0
    for t in tickers:
        if limit and created >= int(limit):
            break
        px = load_price_df(t)
        if px is None or px.empty:
            continue
        px = px.sort_index()
        g = tr[tr["ticker"] == t].copy()
        if g["date"].notna().any():
            d0 = pd.to_datetime(g["date"].min()) - pd.to_timedelta(90, unit="D")
            d1 = pd.to_datetime(g["date"].max()) + pd.to_timedelta(90, unit="D")
            sub = px.loc[(px.index >= d0) & (px.index <= d1)].copy()
            if sub.empty:
                sub = px
        else:
            sub = px

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(sub.index, sub["Close"], linewidth=1)

        for action, marker, label in [("BUY", "^", "BUY"), ("SELL", "v", "SELL")]:
            gg = g[g["action"] == action]
            if gg.empty:
                continue
            dts = pd.to_datetime(gg["date"])
            yy = _y_at_dates(sub["Close"], dts)
            ax.scatter(dts, yy, marker=marker, s=70, label=label)

        ax.set_title(f"{t} — LT {horizon}m {model} TopK trades")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{t}_LT_{horizon}m_{model}.png", dpi=140)
        plt.close(fig)
        created += 1

    print(f"▶ PNGs por ticker guardados en: {out_dir} (archivos={created})", flush=True)


def _flatten(objs):
    if objs is None:
        return []
    if isinstance(objs, (list, tuple)):
        out = []
        for o in objs:
            out.extend(_flatten(o))
        return out
    return [objs]


# ---------------------------------------------------------------------
# Estrategia de rebalanceo mensual
# ---------------------------------------------------------------------
class MonthlyRebalance(bt.Strategy):
    params = dict(
        signals={},    # dict["YYYY-MM"] -> dict[ticker]->weight (objetivo)
        months=[],     # lista ordenada de meses (strings)
        verbose=True,
    )

    def __init__(self):
        self.month_now: Optional[str] = None
        self.by_name = {d._name: d for d in self.datas}  # datafeed por ticker
        self.trades_rows: List[Dict[str, Any]] = []

    def log(self, msg: str) -> None:
        if self.p.verbose:
            print(f"[DBG] {msg}", flush=True)

    def _safe_dt_from_data(self, data: Any) -> Optional[str]:
        try:
            if data is not None and hasattr(data, "datetime") and len(data.datetime):
                return f"{num2date(data.datetime[0]):%Y-%m-%d}"
        except Exception:
            pass
        return None

    def _safe_close_from_data(self, data: Any) -> Optional[float]:
        try:
            if data is not None and hasattr(data, "close") and len(data.close):
                return float(data.close[0])
        except Exception:
            pass
        return None

    def next(self):
        if not self.datas:
            return
        dt = num2date(self.datas[0].datetime[0])
        mkey = f"{dt:%Y-%m}"
        if mkey == self.month_now:
            return

        self.month_now = mkey
        sig = self.p.signals.get(mkey, {})
        s = sum(sig.values())
        weights = {k: (v / s) if s > 0 else 0.0 for k, v in sig.items()}

        if weights:
            self.log(f"Rebalance {mkey} -> {weights}")
        else:
            self.log(f"Rebalance {mkey} -> sin señales (todo a 0)")

        for name, d in self.by_name.items():
            target = float(weights.get(name, 0.0))
            self.order_target_percent(data=d, target=target)

    def notify_order(self, order: bt.Order) -> None:
        if order.status in (order.Submitted, order.Accepted):
            return
        d: Any = getattr(order, "data", None)
        row = {
            "date": self._safe_dt_from_data(d),
            "ticker": getattr(d, "_name", "UNKNOWN"),
            "action": ("BUY" if order.isbuy() else "SELL"),
            "size": float(getattr(order.executed, "size", 0.0) or 0.0),
            "price": float(getattr(order.executed, "price", 0.0) or 0.0),
            "value": float(getattr(order.executed, "value", 0.0) or 0.0),
            "commission": float(getattr(order.executed, "comm", 0.0) or 0.0),
            "status": str(order.getstatusname()),
        }
        self.trades_rows.append(row)

    def notify_trade(self, trade: bt.Trade) -> None:
        if not trade.isclosed:
            return
        d: Any = getattr(trade, "data", None)
        row = {
            "date": self._safe_dt_from_data(d),
            "ticker": getattr(d, "_name", "UNKNOWN"),
            "action": "CLOSE",
            "size": float(getattr(trade, "size", 0.0) or 0.0),
            "price": self._safe_close_from_data(d),
            "value": float(getattr(trade, "pnl", 0.0) or 0.0),
            "commission": float(getattr(trade, "commission", 0.0) or 0.0),
            "status": "TRADE_CLOSE",
        }
        self.trades_rows.append(row)


# ---------------------------------------------------------------------
# Ajustes de pesos (opciones avanzadas)
# ---------------------------------------------------------------------
def _cap_and_norm(weights: Dict[str, float], max_weight: Optional[float]) -> Dict[str, float]:
    if not weights:
        return weights
    w = dict(weights)
    if max_weight is not None and max_weight > 0:
        mw = float(max_weight)
        for k in list(w.keys()):
            w[k] = float(min(float(w[k]), mw))
        s2 = sum(w.values())
        if s2 > 0:
            w = {k: float(v) / float(s2) for k, v in w.items()}
    return w


def _apply_min_price_filter(weights: Dict[str, float], month: str, min_price: Optional[float]) -> Dict[str, float]:
    if not weights or not min_price or min_price <= 0:
        return weights
    w: Dict[str, float] = {}
    thr = float(min_price)
    for t, v in weights.items():
        p = get_price_at_rebalance(t, month)
        if p is not None and p >= thr:
            w[t] = float(v)
    s = sum(w.values())
    return {k: float(v) / float(s) for k, v in w.items()} if s > 0 else {}


def _apply_adv_filter(
    weights: Dict[str, float],
    month: str,
    cash: float,
    adv_frac: Optional[float],
    adv_days: Optional[int],
) -> Dict[str, float]:
    """
    Filtro simple por liquidez usando ADV:
    Requerimos: ADV * adv_days * adv_frac >= cash * weight  (aprox. llenado en adv_days)
    """
    if not weights or adv_frac is None or adv_frac <= 0:
        return weights
    days = int(adv_days or 5)
    kept: Dict[str, float] = {}
    for t, w in weights.items():
        adv = calc_adv_usd(t, month, days)
        if adv is None:
            continue
        if float(adv) * float(days) * float(adv_frac) >= float(cash) * float(w):
            kept[t] = float(w)
    s = sum(kept.values())
    return {k: float(v) / float(s) for k, v in kept.items()} if s > 0 else {}


def _apply_risk_parity(weights: Dict[str, float], month: str, max_weight: Optional[float]) -> Dict[str, float]:
    """Convierte pesos iguales -> pesos ~ 1/vol (vol diaria 63d). Renormaliza y cap."""
    if not weights:
        return weights
    inv_vol: Dict[str, float] = {}
    for t in list(weights.keys()):
        vol = calc_vol_daily(t, month, lookback_days=63)
        if vol is None or vol <= 0:
            continue
        inv_vol[t] = 1.0 / float(vol)

    if not inv_vol:
        return weights

    s = sum(inv_vol.values())
    w = {k: float(v) / float(s) for k, v in inv_vol.items()}
    w = _cap_and_norm(w, max_weight)
    return w


# ---------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------
def run(
    horizon: int,
    model: str,
    topk: int,
    tickers_csv: Optional[str],
    only_test: bool,
    cash: float,
    commission_bps: float,
    save_png: bool,
    trades_csv: Optional[str],
    per_ticker_png: bool = False,
    charts_limit: Optional[int] = None,
    charts_dir: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    # avanzadas (por defecto OFF)
    score_pctl_min: Optional[float] = None,
    max_weight: Optional[float] = None,
    risk_parity: bool = False,
    adv_frac: Optional[float] = None,
    adv_days: Optional[int] = None,
    min_hold_price: Optional[float] = None,
    slippage_bps: Optional[float] = None,
) -> None:

    # Señales preliminares
    signals, months = build_signals(
        horizon, model, topk, only_test,
        date_from=date_from, date_to=date_to,
        score_pctl_min=score_pctl_min,
    )
    if not months:
        raise ValueError("No hay meses con señales (revisa preds, filtros y fechas).")

    # Ventana temporal para feeds
    if date_from or date_to:
        start_dt = pd.to_datetime(date_from) if date_from else pd.to_datetime(min(months) + "-01")
        end_dt = pd.to_datetime(date_to) if date_to else (pd.Period(max(months), freq="M").end_time)
    else:
        start_dt = pd.to_datetime(min(months) + "-01")
        end_dt = pd.Period(max(months), freq="M").end_time

    # Filtro opcional por tickers
    universe = sorted({t for m in months for t in signals.get(m, {}).keys()})
    if tickers_csv:
        filt = {t.strip().upper() for t in tickers_csv.split(",") if t.strip()}
        universe = [t for t in universe if t in filt]
        for m in list(signals.keys()):
            signals[m] = {t: w for t, w in signals[m].items() if t in filt}
        months = [m for m in months if signals.get(m)]

    if not universe:
        raise ValueError("Universo vacío (no hay tickers después del filtrado).")

    # Pre‑ajuste de pesos por restricciones
    adjusted_signals: Dict[str, Dict[str, float]] = {}
    for m in months:
        w = dict(signals.get(m, {}))
        # min price
        w = _apply_min_price_filter(w, m, min_hold_price)
        # ADV
        w = _apply_adv_filter(w, m, float(cash), adv_frac, adv_days)
        # risk parity (1/vol)
        if risk_parity:
            w = _apply_risk_parity(w, m, max_weight)
        # cap y normaliza
        w = _cap_and_norm(w, max_weight)
        if w:
            adjusted_signals[m] = w

    months = [m for m in months if adjusted_signals.get(m)]
    if not months:
        raise ValueError("Todas las señales quedaron filtradas por restricciones (ADV/min_price/etc.).")

    # Cerebro
    cerebro = bt.Cerebro()
    loaded = 0
    for t in sorted(set(k for m in months for k in adjusted_signals[m].keys())):
        df = load_price_df(t)
        if df is None or df.empty:
            print(f"[WARN] No se cargó feed para {t}")
            continue
        df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()
        if df.empty:
            continue
        data = PandasData(dataname=df)  # type: ignore[call-arg]
        cerebro.adddata(data, name=t)
        loaded += 1

    if loaded == 0:
        raise ValueError("Ningún feed de precios se pudo cargar.")

    # Comisión + slippage (sumados) — cast explícitos para Pylance
    total_bps = float(commission_bps) + float(slippage_bps or 0.0)
    commission = float(total_bps) / 10000.0
    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_coc(True)

    cerebro.addstrategy(MonthlyRebalance, signals=adjusted_signals, months=months, verbose=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}", flush=True)
    results = cerebro.run()
    strat_inst: MonthlyRebalance = results[0]
    print(f"Final Portfolio Value:   {cerebro.broker.getvalue():,.2f}", flush=True)

    # PNG resumen (nativo de backtrader)
    if save_png:
        try:
            figs = cerebro.plot(style="candle", iplot=False)
            for obj in _flatten(figs):
                if hasattr(obj, "savefig"):
                    out_png = ARTS / f"LT_{horizon}m_{model}_top{topk}.png"
                    obj.savefig(out_png, dpi=140, bbox_inches="tight")
                    print(f"▶ Gráfico PNG: {out_png}")
                    break
        except Exception as e:
            print(f"(Plot no disponible: {e})", flush=True)

    # Métricas resumidas
    dd = strat_inst.analyzers.dd.get_analysis()
    sh = strat_inst.analyzers.sharpe.get_analysis()
    print("== Métricas ==", {
        "max_drawdown": dd.get("max", {}).get("drawdown", None),
        "sharpe": sh.get("sharperatio", None),
        "meses_con_senal": len(months),
        "tickers": loaded,
    }, flush=True)

    # CSV de trades
    rows = getattr(strat_inst, "trades_rows", [])
    if rows:
        out_csv = Path(trades_csv) if trades_csv else (ARTS / f"trades_{horizon}m_{model}_top{topk}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_tr = pd.DataFrame(rows)
        df_tr.sort_values(["date", "ticker", "status", "action"], inplace=True)
        df_tr.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"✅ CSV de trades guardado: {out_csv} (filas={len(df_tr)})")

        # PNG por ticker (opcional)
        # (solo se generan si has pasado --per_ticker_png)
        if per_ticker_png:
            out_dir = Path(charts_dir) if charts_dir else (ARTS / f"per_ticker_{horizon}m_{model}_top{topk}")
            save_per_ticker_pngs(rows, out_dir, horizon, model, limit=charts_limit)
    else:
        print("⚠️ No se registraron órdenes ejecutadas; no se generó CSV.")


def main():
    ap = argparse.ArgumentParser()
    # rango de fechas (para limitar el backtest)
    ap.add_argument("--date_from", type=str, default="", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--date_to", type=str, default="", help="YYYY-MM-DD (inclusive)")

    # PNG por ticker
    ap.add_argument("--per_ticker_png", action="store_true",
                    help="Guardar un PNG por ticker con sus trades.")
    ap.add_argument("--charts_limit", type=int, default=0,
                    help="Máximo de tickers a graficar (0 = sin límite).")
    ap.add_argument("--charts_dir", type=str, default="",
                    help="Directorio de salida para los PNG por ticker.")

    # parámetros del backtest
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--model", type=str, default="lgbm")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tickers", type=str, default="", help="ej: AAPL,MSFT,NVDA (opcional)")
    ap.add_argument("--only_test", action="store_true")
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--commission_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0, help="Se suma a commission_bps")
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--trades_csv", type=str, default="", help="ruta opcional para guardar el CSV de trades")

    # opciones avanzadas (todas OFF por defecto)
    ap.add_argument("--score_pctl_min", type=float, default=0.0,
                    help="Filtra por percentil mensual del score (0..1). 0=sin filtro.")
    ap.add_argument("--max_weight", type=float, default=0.0,
                    help="Tope de peso por ticker (0=sin tope).")
    ap.add_argument("--risk_parity", action="store_true",
                    help="Repondera por 1/vol (ventana 63d) antes de cap y normalización.")
    ap.add_argument("--adv_frac", type=float, default=0.0,
                    help="Fracción de ADV por día que se puede negociar (0=sin filtro).")
    ap.add_argument("--adv_days", type=int, default=5, help="Días para ADV.")
    ap.add_argument("--min_hold_price", type=float, default=0.0,
                    help="Precio mínimo para mantener/entrar en el rebalance (0=sin filtro).")

    args = ap.parse_args()

    run(
        horizon=args.horizon,
        model=args.model,
        topk=args.topk,
        tickers_csv=(args.tickers or None),
        only_test=args.only_test,
        cash=args.cash,
        commission_bps=args.commission_bps,
        save_png=args.save_png,
        trades_csv=(args.trades_csv or None),
        per_ticker_png=args.per_ticker_png,
        charts_limit=(args.charts_limit or None),
        charts_dir=(args.charts_dir or None),
        date_from=(args.date_from or None),
        date_to=(args.date_to or None),
        score_pctl_min=(args.score_pctl_min if args.score_pctl_min > 0 else None),
        max_weight=(args.max_weight if args.max_weight > 0 else None),
        risk_parity=bool(args.risk_parity),
        adv_frac=(args.adv_frac if args.adv_frac > 0 else None),
        adv_days=int(args.adv_days),
        min_hold_price=(args.min_hold_price if args.min_hold_price > 0 else None),
        slippage_bps=(args.slippage_bps if args.slippage_bps > 0 else None),
    )


if __name__ == "__main__":
    main()
