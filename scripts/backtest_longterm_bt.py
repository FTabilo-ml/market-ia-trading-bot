# scripts/backtest_longterm_bt.py
from __future__ import annotations

import sys
import pathlib
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import backtrader as bt
from backtrader.feeds import PandasData
from backtrader.utils.date import num2date
from pathlib import Path

# Rutas
ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
MLDATA = ROOT / "data" / "ml"

# === NUEVO destino de artefactos ===
ARTS = ROOT / "artifacts" / "longterm"
ARTS.mkdir(parents=True, exist_ok=True)


# ---------------------- Señales (predicciones -> pesos) ----------------------
def build_signals(
    horizon: int,
    model: str,
    topk: int,
    only_test: bool,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Devuelve:
      signals: dict["YYYY-MM" -> dict["TICKER" -> weight]]
      months: lista ordenada de meses "YYYY-MM" presentes
    Requiere en preds: Date, Ticker, y_pred. (sin autodetección de nombre)
    """
    preds_fp = MLDATA / f"preds_{horizon}m_{model}.parquet"
    if not preds_fp.exists():
        print(f"[WARN] No existe archivo de predicciones: {preds_fp}")
        return {}, []

    try:
        preds = pd.read_parquet(preds_fp, engine="fastparquet")
    except Exception as e:
        print(f"[WARN] No se pudo leer {preds_fp}: {e}")
        return {}, []

    required = {"Date", "Ticker", "y_pred"}
    missing = required.difference(preds.columns)
    if missing:
        print(f"[WARN] Faltan columnas {sorted(missing)} en {preds_fp}")
        return {}, []

    df = preds.copy()
    # limpieza mínima
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", "y_pred"])
    if df.empty:
        return {}, []

    # filtro de test si existe marca explícita; si no, un holdout temporal simple
    if only_test:
        if "is_test" in df.columns:
            df = df[df["is_test"].astype(bool)]
        else:
            tmp = df.copy()
            tmp["month_key"] = tmp["Date"].dt.to_period("M").astype(str)
            months_sorted = sorted(tmp["month_key"].unique())
            if not months_sorted:
                return {}, []
            n_test = max(1, int(round(0.2 * len(months_sorted))))
            test_months = set(months_sorted[-n_test:])
            df = tmp[tmp["month_key"].isin(test_months)].drop(columns=["month_key"])

    if df.empty:
        return {}, []

    # llave mensual
    df["month_key"] = df["Date"].dt.to_period("M").astype(str)

    signals: Dict[str, Dict[str, float]] = {}
    months: List[str] = []
    for k, g in df.groupby("month_key", sort=True, dropna=False):
        k_str = str(k)  # asegura str para Pylance
        # top-k por predicción descendente
        g2 = g.sort_values("y_pred", ascending=False).head(max(1, int(topk)))
        if g2.empty:
            continue
        w = 1.0 / len(g2)
        signals[k_str] = {str(t): float(w) for t in g2["Ticker"].tolist()}
        months.append(k_str)

    months = sorted(set(months))
    return signals, months

# ---------------------- util: cargar precios por ticker ----------------------
def load_price_df(ticker: str) -> Optional[pd.DataFrame]:
    """
    Carga parquet local data/processed/{ticker}.parquet con columnas
    Open, High, Low, Close, Volume e índice datetime.
    """
    fp = PROCESSED / f"{ticker}.parquet"
    if not fp.exists():
        print(f"[WARN] No existe {fp}")
        return None
    try:
        df = pd.read_parquet(fp, engine="fastparquet").copy()
    except Exception as e:
        print(f"[WARN] No se pudo leer {fp}: {e}")
        return None

    # normalizar índice y columnas
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")

    # forzar índice datetime
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        print(f"[WARN] Índice no datetime en {fp}")
        return None

    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c not in df.columns:
            print(f"[WARN] Falta columna {c} en {fp}")
            return None
    return df[cols].sort_index()

def _flatten(objs):
    if objs is None:
        return []
    if isinstance(objs, (list, tuple)):
        out = []
        for o in objs:
            out.extend(_flatten(o))
        return out
    return [objs]

# ---------------------- Estrategia de rebalanceo mensual ----------------------
class MonthlyRebalance(bt.Strategy):
    params = dict(
        signals={},    # dict["YYYY-MM"] -> dict[ticker]->weight
        months=[],     # lista ordenada de meses (strings)
        verbose=True
    )

    def __init__(self):
        self.month_now: Optional[str] = None
        self.by_name = {d._name: d for d in self.datas}  # datafeed por ticker

        # buffers para exportar al final
        self.trades_rows: List[Dict[str, Any]] = []
        self.equity_rows: List[Dict[str, Any]] = []

    def log(self, msg):
        if self.p.verbose:
            print(f"[DBG] {msg}", flush=True)

    def notify_order(self, order: bt.Order) -> None:
        # registramos solo cuando hay ejecución/cancelación
        if order.status not in [order.Completed, order.Canceled, order.Rejected]:
            return

        # fecha del evento
        try:
            dt = num2date(order.executed.dt)
        except Exception:
            dt = num2date(self.datas[0].datetime[0])
        date_str = f"{dt:%Y-%m-%d}"

        # valores
        action = "BUY" if order.isbuy() else "SELL"
        status = (
            "Completed" if order.status == order.Completed else
            "Canceled" if order.status == order.Canceled else
            "Rejected"
        )
        exec_price = float(getattr(order.executed, "price", 0.0) or 0.0)
        exec_size = float(getattr(order.executed, "size", 0.0) or 0.0)
        exec_value = float(getattr(order.executed, "value", 0.0) or 0.0)
        exec_comm  = float(getattr(order.executed, "comm", 0.0) or 0.0)

        # valor de portafolio en ese momento
        port_val = float(self.broker.getvalue())

        row = dict(
            date=date_str,
            ticker=str(order.data._name) if order.data is not None else "",
            action=action,
            status=status,
            price=exec_price,
            size=exec_size,
            value=exec_value,
            commission=exec_comm,
            portfolio_value=port_val,   # <-- agregado
        )
        self.trades_rows.append(row)

    def next(self):
        if not self.datas:
            return

        # equity curve (cada barra)
        dt = num2date(self.datas[0].datetime[0])
        self.equity_rows.append({
            "date": f"{dt:%Y-%m-%d}",
            "portfolio_value": float(self.broker.getvalue()),
        })

        # rebalance mensual
        mkey = f"{dt:%Y-%m}"
        if mkey == self.month_now:
            return  # mismo mes, nada que hacer

        self.month_now = mkey
        sig = self.p.signals.get(mkey, {})
        s = sum(sig.values())
        weights = {k: (v / s) if s > 0 else 0.0 for k, v in sig.items()}

        # registrar
        if weights:
            self.log(f"Rebalance {mkey} -> {weights}")
        else:
            self.log(f"Rebalance {mkey} -> sin señales (todo a 0)")

        # ordenar pesos objetivo por cada ticker presente en los feeds
        for name, d in self.by_name.items():
            target = float(weights.get(name, 0.0))
            self.order_target_percent(data=d, target=target)

# ---------------------- Orquestador ----------------------
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
):
    signals, months = build_signals(horizon, model, topk, only_test)
    if not months:
        raise ValueError("No hay meses con señales (revisa preds y filtros).")

    # universo de tickers según señales
    universe = sorted({t for m in months for t in signals.get(m, {}).keys()})

    # filtro opcional de CLI
    if tickers_csv:
        filt = {t.strip().upper() for t in tickers_csv.split(",") if t.strip()}
        universe = [t for t in universe if t in filt]
        # filtra señales también
        for m in list(signals.keys()):
            signals[m] = {t: w for t, w in signals[m].items() if t in filt}
        # limpia meses vacíos
        months = [m for m in months if signals.get(m)]

    if not universe:
        raise ValueError("Universo vacío (no hay tickers después del filtrado).")

    cerebro = bt.Cerebro()

    # cargar feeds
    loaded = 0
    for t in universe:
        df = load_price_df(t)
        if df is None or df.empty:
            print(f"[WARN] No se cargó feed para {t}")
            continue
        data = PandasData(dataname=df)  # type: ignore[call-arg]  # backtrader acepta kw 'dataname'
        cerebro.adddata(data, name=t)
        loaded += 1

    if loaded == 0:
        raise ValueError("Ningún feed de precios se pudo cargar.")

    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=commission_bps / 10000.0)

    strat = cerebro.addstrategy(MonthlyRebalance, signals=signals, months=months, verbose=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}", flush=True)
    results = cerebro.run()
    strat = results[0]
    print(f"Final Portfolio Value:   {cerebro.broker.getvalue():,.2f}", flush=True)

    # --- CSV de trades ---
    rows = getattr(strat, "trades_rows", [])
    if rows:
        out_csv = Path(trades_csv) if trades_csv else (ARTS / f"trades_{horizon}m_{model}_top{topk}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_tr = pd.DataFrame(rows)
        df_tr.sort_values(["date", "ticker", "status", "action"], inplace=True)
        df_tr.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"✅ CSV de trades guardado: {out_csv} (filas={len(df_tr)})")
    else:
        print("⚠️ No se registraron órdenes ejecutadas; no se generó CSV.")
    strat_inst: MonthlyRebalance = results[0]
    print(f"Final Portfolio Value:   {cerebro.broker.getvalue():,.2f}", flush=True)

    # CSV de trades
    rows = getattr(strat_inst, "trades_rows", [])
    if rows:
        # Si trades_csv viene como str, lo convertimos a Path; si no, usamos la ruta calculada
        out_csv = Path(trades_csv) if trades_csv else (ARTS / f"trades_{horizon}m_{model}_top{topk}.csv")
        # Nos aseguramos de que el directorio padre exista
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Creamos el DataFrame y lo guardamos
        df_tr = pd.DataFrame(rows)
        df_tr.sort_values(["date", "ticker", "status", "action"], inplace=True)
        df_tr.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"✅ CSV de trades guardado: {out_csv} (filas={len(df_tr)})")
    else:
        print("⚠️ No se registraron órdenes ejecutadas; no se generó CSV.")

    # Plot (robusto)
    if save_png:
        try:
            figs = cerebro.plot(style="candle", iplot=False)
            saved = False
            for obj in _flatten(figs):
                if hasattr(obj, "savefig"):
                    out_png = ARTS / f"LT_{horizon}m_{model}_top{topk}.png"
                    obj.savefig(out_png, dpi=140, bbox_inches="tight")
                    print(f"▶ Gráfico PNG: {out_png}")
                    saved = True
                    break
            if not saved:
                print("(Plot no retornó figura guardable)")
        except Exception as e:
            print(f"(Plot no disponible: {e})", flush=True)

    # Métricas básicas
    dd = strat_inst.analyzers.dd.get_analysis()
    sh = strat_inst.analyzers.sharpe.get_analysis()
    print("== Métricas ==", {
        "max_drawdown": dd.get("max", {}).get("drawdown", None),
        "sharpe": sh.get("sharperatio", None),
        "meses": len(months),
        "tickers": loaded,
    }, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--model", type=str, default="lgbm")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tickers", type=str, default="", help="ej: AAPL,MSFT,NVDA (opcional)")
    ap.add_argument("--only_test", action="store_true")
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--commission_bps", type=float, default=10.0)
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--trades_csv", type=str, default="", help="ruta de salida CSV con órdenes/PNL")
    args = ap.parse_args()

    run(
        horizon=args.horizon,
        model=args.model,
        topk=args.topk,
        tickers_csv=(args.tickers if args.tickers else None),
        only_test=args.only_test,
        cash=args.cash,
        commission_bps=args.commission_bps,
        save_png=args.save_png,
        trades_csv=(args.trades_csv if args.trades_csv else None),
    )

if __name__ == "__main__":
    main()
