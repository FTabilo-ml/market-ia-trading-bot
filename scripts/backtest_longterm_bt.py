# scripts/backtest_longterm_bt.py
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
# Utilidades Parquet compatibles (evita warnings de typing en Pylance)
# ---------------------------------------------------------------------
def read_parquet_compat(fp: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(fp, engine="fastparquet")
    except Exception:
        try:
            return pd.read_parquet(fp, engine="pyarrow")
        except Exception:
            # Deja que pandas resuelva engine por defecto si está disponible
            return pd.read_parquet(fp)


# ---------------------------------------------------------------------
# Señales (predicciones -> pesos)
# ---------------------------------------------------------------------
def build_signals(horizon: int,
                  model: str,
                  topk: int,
                  only_test: bool,
                  date_from: Optional[str] = None,
                  date_to: Optional[str] = None
                  ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Devuelve:
      signals: dict["YYYY-MM" -> dict["TICKER" -> weight]]
      months: lista ordenada de meses "YYYY-MM" presentes
    """
    # 1) Localiza archivo de predicciones (ranker o baseline)
    candidates: List[Path] = []
    if model.lower() == "rank":
        candidates.append(MLDATA / f"preds_rank_{horizon}m.parquet")
        candidates.append(MLDATA / f"preds_{horizon}m_rank.parquet")  # fallback
    else:
        candidates.append(MLDATA / f"preds_{horizon}m_{model}.parquet")
        candidates.append(MLDATA / f"preds_rank_{horizon}m.parquet")  # si ya entrenaste ranker

    preds_fp = next((p for p in candidates if p.exists()), None)
    if preds_fp is None:
        print(f"[WARN] No encontré archivo de predicciones en: {[str(p) for p in candidates]}")
        return {}, []

    # 2) Carga
    df = read_parquet_compat(preds_fp)

    # 3) Chequeo mínimo
    if "Date" not in df.columns or "Ticker" not in df.columns:
        print(f"[WARN] Faltan columnas 'Date'/'Ticker' en {preds_fp}")
        return {}, []

    # 4) Columna de score
    use_col = "score" if "score" in df.columns else ("y_pred" if "y_pred" in df.columns else None)
    if use_col is None:
        # como último recurso usar 'raw_score' si existe
        use_col = "raw_score" if "raw_score" in df.columns else None
    if use_col is None:
        print(f"[WARN] No encuentro 'score'/'y_pred'/'raw_score' en {preds_fp}")
        return {}, []

    # 5) Normaliza fecha y filtra
    df = df.dropna(subset=["Date", "Ticker", use_col]).copy()
    if df.empty:
        return {}, []
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Solo test si se pide
    if only_test and "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    elif only_test and "is_test" in df.columns:
        df = df[df["is_test"].astype(bool)].copy()

    # Rango temporal (inclusive)
    if date_from:
        df = df[df["Date"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["Date"] <= pd.to_datetime(date_to)]

    if df.empty:
        return {}, []

    # 6) Construye señales por mes (Top-K equal weight)
    df["month_key"] = df["Date"].dt.to_period("M").astype(str)
    signals: Dict[str, Dict[str, float]] = {}
    months: List[str] = []

    for m, g in df.groupby("month_key", sort=True, dropna=False):
        g2 = g.sort_values(use_col, ascending=False).head(max(1, int(topk)))
        if g2.empty:
            continue
        w = 1.0 / len(g2)
        signals[str(m)] = {str(t): float(w) for t in g2["Ticker"].astype(str).tolist()}
        months.append(str(m))

    months = sorted(set(months))
    return signals, months


# ---------------------------------------------------------------------
# Cargar precios por ticker
# ---------------------------------------------------------------------
def load_price_df(ticker: str) -> Optional[pd.DataFrame]:
    fp = PROCESSED / f"{ticker}.parquet"
    if not fp.exists():
        print(f"[WARN] No existe {fp}")
        return None
    try:
        df = read_parquet_compat(fp).copy()
    except Exception as e:
        print(f"[WARN] No se pudo leer {fp}: {e}")
        return None

    # normalizar índice y columnas
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not set(cols).issubset(df.columns):
        print(f"[WARN] Faltan columnas OHLCV en {fp}")
        return None

    out = df[cols].sort_index().dropna()
    return out


# ---------------------------------------------------------------------
# PNG por ticker con BUY/SELL
# ---------------------------------------------------------------------
def save_per_ticker_pngs(rows: List[Dict[str, Any]],
                         out_dir: Path,
                         horizon: int,
                         model: str,
                         limit: Optional[int] = None) -> None:
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
        return series.reindex(pd.to_datetime(dates), method="nearest")

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
            d0 = g["date"].min() - pd.Timedelta(days=90)
            d1 = g["date"].max() + pd.Timedelta(days=90)
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
        out: List[Any] = []
        for o in objs:
            out.extend(_flatten(o))
        return out
    return [objs]


# ---------------------------------------------------------------------
# Estrategia de rebalanceo mensual
# ---------------------------------------------------------------------
class MonthlyRebalance(bt.Strategy):
    params = dict(
        signals={},    # dict["YYYY-MM"] -> dict[ticker]->weight
        months=[],     # lista ordenada de meses (strings)
        verbose=True
    )

    def __init__(self):
        self.month_now: Optional[str] = None
        self.by_name = {d._name: d for d in self.datas}
        self.trades_rows: List[Dict[str, Any]] = []

    def log(self, msg: str) -> None:
        if self.p.verbose:
            print(f"[DBG] {msg}", flush=True)

    # helpers seguros
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
        # anclar calendario al primer feed
        dt = num2date(self.datas[0].datetime[0])
        mkey = f"{dt:%Y-%m}"
        if mkey == self.month_now:
            return  # mismo mes, no rebalancea

        self.month_now = mkey
        sig = self.p.signals.get(mkey, {})
        s = sum(sig.values())
        weights = {k: (v / s) if s > 0 else 0.0 for k, v in sig.items()}

        # logging
        if weights:
            self.log(f"Rebalance {mkey} -> {weights}")
        else:
            self.log(f"Rebalance {mkey} -> sin señales (todo a 0)")

        # ordena pesos objetivo por ticker presente
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
# Backtest
# ---------------------------------------------------------------------
def run(horizon: int,
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
        benchmark: Optional[str] = None) -> None:

    signals, months = build_signals(
        horizon, model, topk, only_test,
        date_from=date_from, date_to=date_to
    )
    if not months:
        raise ValueError("No hay meses con señales (revisa preds y filtros).")

    # Ventana temporal dura
    if date_from or date_to:
        start_dt = pd.to_datetime(date_from) if date_from else pd.to_datetime(min(months) + "-01")
        # fin = fin de mes del último month key (o date_to si se pasó)
        end_dt = pd.to_datetime(date_to) if date_to else pd.Period(max(months), freq="M").end_time

    else:
        start_dt = pd.to_datetime(min(months) + "-01")
        end_dt = pd.Period(max(months), freq="M").end_time


    # Universo
    universe = sorted({t for m in months for t in signals.get(m, {}).keys()})
    if tickers_csv:
        filt = {t.strip().upper() for t in tickers_csv.split(",") if t.strip()}
        universe = [t for t in universe if t in filt]
        for m in list(signals.keys()):
            signals[m] = {t: w for t, w in signals[m].items() if t in filt}
        months = [m for m in months if signals.get(m)]

    if not universe:
        raise ValueError("Universo vacío (no hay tickers después del filtrado).")

    cerebro = bt.Cerebro()
    loaded = 0
    for t in universe:
        df = load_price_df(t)
        if df is None or df.empty:
            print(f"[WARN] No se cargó feed para {t}")
            continue
        # recorte temporal duro
        df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)].copy()
        if df.empty:
            continue
        data = PandasData(dataname=df)  # type: ignore[call-arg]
        cerebro.adddata(data, name=t)
        loaded += 1

    if loaded == 0:
        raise ValueError("Ningún feed de precios se pudo cargar.")

    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=float(commission_bps) / 10000.0)
    cerebro.broker.set_coc(True)  # compra/sell mismo bar sin 'Margin'

    cerebro.addstrategy(MonthlyRebalance, signals=signals, months=months, verbose=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}", flush=True)
    results = cerebro.run()
    strat_inst: MonthlyRebalance = results[0]
    final_v = float(cerebro.broker.getvalue())
    print(f"Final Portfolio Value:   {final_v:,.2f}", flush=True)

    # PNG resumen (plot nativo backtrader)
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

    # Métricas
    dd = strat_inst.analyzers.dd.get_analysis()
    sh = strat_inst.analyzers.sharpe.get_analysis()
    meses_con_senal = len([m for m in months if signals.get(m)])
    print("== Métricas ==", {
        "max_drawdown": dd.get("max", {}).get("drawdown", None),
        "sharpe": sh.get("sharperatio", None),
        "meses_con_senal": meses_con_senal,
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

    # Benchmark buy & hold
    if benchmark:
        bench_fp = PROCESSED / f"{benchmark}.parquet"
        if bench_fp.exists():
            pxb = read_parquet_compat(bench_fp).copy()
            if "Date" in pxb.columns:
                pxb["Date"] = pd.to_datetime(pxb["Date"], errors="coerce")
                pxb = pxb.set_index("Date")
            pxb.index = pd.to_datetime(pxb.index)
            pxb = pxb.sort_index()
            pxb = pxb.loc[(pxb.index >= start_dt) & (pxb.index <= end_dt)]
            if not pxb.empty and "Close" in pxb.columns:
                start_p = float(pxb["Close"].iloc[0])
                end_p = float(pxb["Close"].iloc[-1])
                bh = (end_p / start_p) - 1.0 if start_p > 0 else np.nan
                print(f"Benchmark {benchmark} buy&hold: {(bh * 100):.2f}% en ventana.", flush=True)
        else:
            print(f"[WARN] Benchmark {benchmark} no encontrado en {bench_fp}")



# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest mensual Top-K equal-weight a partir de señales de ranking.")
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
    ap.add_argument("--model", type=str, default="rank")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--tickers", type=str, default="", help="ej: AAPL,MSFT,NVDA (opcional)")
    ap.add_argument("--only_test", action="store_true")
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--commission_bps", type=float, default=10.0)
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--trades_csv", type=str, default="", help="Ruta opcional para guardar el CSV de trades")

    # benchmark opcional (ej. SPY/QQQ)
    ap.add_argument("--benchmark", type=str, default="", help="Ticker de benchmark en data/processed (ej.: SPY)")

    args = ap.parse_args()

    run(
        horizon=int(args.horizon),
        model=str(args.model),
        topk=int(args.topk),
        tickers_csv=(args.tickers if args.tickers else None),
        only_test=bool(args.only_test),
        cash=float(args.cash),
        commission_bps=float(args.commission_bps),
        save_png=bool(args.save_png),
        trades_csv=(args.trades_csv if args.trades_csv else None),
        per_ticker_png=bool(args.per_ticker_png),
        charts_limit=(args.charts_limit or None),
        charts_dir=(args.charts_dir or None),
        date_from=(args.date_from or None),
        date_to=(args.date_to or None),
        benchmark=(args.benchmark or None),
    )


if __name__ == "__main__":
    main()
