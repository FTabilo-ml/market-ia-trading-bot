# scripts/backtrader_rules_plot.py (versión de diagnóstico)
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false
from __future__ import annotations
import sys, pathlib, argparse, traceback
from typing import Any
import pandas as pd
import backtrader as bt
import backtrader.indicators as btind
from backtrader.feeds import PandasData
from backtrader.utils.date import num2date

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data/processed"
ARTS = ROOT / "artifacts" / "backtrader"
ARTS.mkdir(parents=True, exist_ok=True)

def log_step(msg: str):
    print(f"[DBG] {msg}", flush=True)

class SmaCross(bt.Strategy):
    params = dict(fast=20, slow=150)
    def __init__(self):
        log_step("SmaCross.__init__")
        self.sma_fast = btind.SMA(self.data.close, period=self.p.fast)
        self.sma_slow = btind.SMA(self.data.close, period=self.p.slow)
        self.crossover = btind.CrossOver(self.sma_fast, self.sma_slow)
        self.trades_log = []
    def _dt(self):
        return num2date(self.data.datetime[0])
    def notify_order(self, order):
        if order.status in [order.Completed]:
            side = "BUY" if order.isbuy() else "SELL"
            self.trades_log.append({
                "datetime": self._dt(),
                "side": side,
                "price": float(order.executed.price),
                "size": float(order.executed.size),
                "commission": float(order.executed.comm),
            })
    def next(self):
        if not self.position and self.crossover > 0:
            cash = float(self.broker.getcash())
            price = float(self.data.close[0])
            size = int((cash*0.98)/price) if price>0 else 0
            if size>0: self.buy(size=size)
        elif self.position and self.crossover < 0:
            self.close()

def load_df(ticker: str) -> pd.DataFrame:
    fp = PROCESSED / f"{ticker}.parquet"
    log_step(f"load_df: intentando abrir {fp}")
    if not fp.exists():
        raise FileNotFoundError(f"No existe {fp}")
    df = pd.read_parquet(fp, engine="fastparquet").copy()
    df["Date"] = pd.to_datetime(df.get("Date", df.index))
    df = df.set_index("Date").sort_index()
    cols = ["Open","High","Low","Close","Volume"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en {fp}")
    log_step(f"load_df: df shape={df.shape}, rango={df.index.min()}..{df.index.max()}")
    return df[cols]

def _flatten(objs: Any) -> list[Any]:
    if objs is None: return []
    if isinstance(objs, (list, tuple)):
        out = []
        for o in objs: out.extend(_flatten(o))
        return out
    return [objs]

def run(ticker: str, fast: int, slow: int, cash: float, commission_bps: float, plot: bool):
    try:
        log_step("run: start")
        df = load_df(ticker)
        data = PandasData(dataname=df)  # type: ignore[call-arg]
        cerebro = bt.Cerebro()
        cerebro.adddata(data, name=ticker)
        cerebro.addstrategy(SmaCross, fast=fast, slow=slow)
        cerebro.broker.setcash(float(cash))
        cerebro.broker.setcommission(commission=commission_bps/10000.0)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

        print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}", flush=True)
        results = cerebro.run()
        log_step("cerebro.run() terminó")
        strat = results[0]
        print(f"Final Portfolio Value:   {cerebro.broker.getvalue():.2f}", flush=True)

        trades_df = pd.DataFrame(strat.trades_log)
        if not trades_df.empty:
            out_csv = ARTS / f"{ticker}_trades.csv"
            trades_df.to_csv(out_csv, index=False)
            print(f"▶ Trades CSV: {out_csv}", flush=True)
        else:
            print("▶ No hubo operaciones (CSV no generado).", flush=True)

        if plot:
            try:
                figs = cerebro.plot(style="candle", iplot=False)
                for obj in _flatten(figs):
                    if hasattr(obj, "savefig"):
                        out_png = ARTS / f"{ticker}_plot.png"
                        obj.savefig(out_png, dpi=150, bbox_inches="tight")
                        print(f"▶ Gráfico PNG: {out_png}", flush=True)
                        break
            except Exception as e:
                print(f"(Plot no disponible: {e})", flush=True)

        ta = strat.analyzers.trades.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        sh = strat.analyzers.sharpe.get_analysis()
        total_trades = ta.get("total", {}).get("total", None)
        closed_trades = ta.get("total", {}).get("closed", 0)
        won_trades = ta.get("won", {}).get("total", 0)
        winrate = (won_trades / closed_trades) if closed_trades else None
        print("Metrics:", {
            "trades_total": total_trades,
            "win_rate": winrate,
            "max_drawdown": dd.get("max", {}).get("drawdown", None),
            "sharpe": sh.get("sharperatio", None),
        }, flush=True)
    except Exception:
        print("❌ Excepción durante la ejecución:\n" + traceback.format_exc(), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="AAPL")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=150)
    ap.add_argument("--cash", type=float, default=10000.0)
    ap.add_argument("--commission_bps", type=float, default=10.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    run(args.ticker, args.fast, args.slow, args.cash, args.commission_bps, args.plot)

if __name__ == "__main__":
    main()
