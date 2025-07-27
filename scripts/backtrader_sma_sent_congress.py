#!/usr/bin/env python
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
from __future__ import annotations
import sys, pathlib, argparse, traceback
from typing import Any, Optional, List, cast
import pandas as pd
import backtrader as bt
import backtrader.indicators as btind
from backtrader.utils.date import num2date
from matplotlib.figure import Figure

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data/processed"
ARTS = ROOT / "artifacts" / "backtrader"
ARTS.mkdir(parents=True, exist_ok=True)

def log_step(msg: str):
    print(f"[DBG] {msg}", flush=True)

class SentCongressData(bt.feeds.PandasData):
    lines = ("sent", "netbuy",)
    params = (("sent", -1), ("netbuy", -1))

class SmaSentCongress(bt.Strategy):
    params = dict(fast=20, slow=150, sent_thr=0.05, net_buy_min=1.0, ticker="AAPL")

    def __init__(self):
        self.sma_fast = btind.SMA(self.data.close, period=self.p.fast)
        self.sma_slow = btind.SMA(self.data.close, period=self.p.slow)
        self.crossover = btind.CrossOver(self.sma_fast, self.sma_slow)
        self.trades_log: List[dict[str, Any]] = []

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

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades_log.append({
                "datetime": self._dt(),
                "side": "CLOSE",
                "pnl": float(trade.pnlcomm),
            })

    def next(self):
        sent = float(getattr(self.data, "sent")[0]) if hasattr(self.data, "sent") else 0.0
        netb = float(getattr(self.data, "netbuy")[0]) if hasattr(self.data, "netbuy") else 0.0

        long_cond = (self.crossover[0] > 0) and (sent >= self.p.sent_thr) and (netb >= self.p.net_buy_min)
        exit_cond = (self.crossover[0] < 0) or (sent < -self.p.sent_thr)

        if not self.position:
            if long_cond:
                cash = float(self.broker.getcash()); price = float(self.data.close[0])
                size = int((cash*0.98)/price) if price>0 else 0
                if size>0: self.buy(size=size)
        else:
            if exit_cond:
                self.close()

    def stop(self):
        if self.trades_log:
            df = pd.DataFrame(self.trades_log)
            out = ARTS / f"{self.p.ticker}_trades_sent_congress.csv"
            df.to_csv(out, index=False)
            print(f"▶ Trades CSV: {out}", flush=True)

def load_df(ticker: str) -> pd.DataFrame:
    fp = PROCESSED / f"{ticker}.parquet"
    log_step(f"load_df: {fp}")
    if not fp.exists():
        raise FileNotFoundError(f"No existe {fp}")
    df = pd.read_parquet(fp, engine="fastparquet").copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c} en {fp}")
    if "sentiment_score" not in df.columns: df["sentiment_score"] = 0.0
    if "net_buy" not in df.columns: df["net_buy"] = 0.0
    df["Date"] = pd.to_datetime(df.get("Date", df.index))
    df = df.set_index("Date").sort_index()
    df.rename(columns={"sentiment_score":"sent","net_buy":"netbuy"}, inplace=True)
    return df[["Open","High","Low","Close","Volume","sent","netbuy"]]

def _flatten(objs: Any) -> list[Any]:
    if objs is None: return []
    if isinstance(objs, (list, tuple)):
        out: list[Any] = []
        for o in objs: out.extend(_flatten(o))
        return out
    return [objs]

def run(ticker: str, fast: int, slow: int, cash: float, commission_bps: float,
        sent_thr: float, net_buy_min: float, save_png: bool):
    try:
        df = load_df(ticker)
        data = SentCongressData(
            dataname=df.assign(
                sent=lambda x: x.get("sentiment_score", 0).fillna(0.0).astype(float),
                netbuy=lambda x: x.get("net_buy", 0).fillna(0.0).astype(float),
            ),
            datetime=None,
            open='Open', high='High', low='Low', close='Close',
            volume='Volume', openinterest=None,
            sent='sent', netbuy='netbuy',
        )


        cerebro = bt.Cerebro()
        cerebro.adddata(data, name=ticker)
        cerebro.addstrategy(SmaSentCongress, fast=fast, slow=slow,
                            sent_thr=sent_thr, net_buy_min=net_buy_min, ticker=ticker)
        cerebro.broker.setcash(float(cash))
        cerebro.broker.setcommission(commission=commission_bps/10000.0)

        # analyzers (sin TimeReturn)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addobserver(bt.observers.BuySell)

        print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}", flush=True)
        results = cerebro.run()
        strat = results[0]
        print(f"Final Portfolio Value:   {cerebro.broker.getvalue():,.2f}", flush=True)

        if save_png:
            try:
                figs = cerebro.plot(style="candle", iplot=False)
                fig_obj: Optional[Figure] = None
                for obj in _flatten(figs):
                    if hasattr(obj, "savefig"):
                        fig_obj = cast(Figure, obj); break
                if fig_obj is not None:
                    out_png = ARTS / f"{ticker}_sma_sent_congress.png"
                    fig_obj.savefig(out_png, dpi=150, bbox_inches="tight")
                    print(f"▶ Gráfico PNG: {out_png}", flush=True)
            except Exception as e:
                print(f"(Plot no disponible: {e})", flush=True)

        ta = strat.analyzers.trades.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        sh = strat.analyzers.sharpe.get_analysis()
        print("Metrics:", {
            "trades_total": ta.get("total", {}).get("total", None),
            "win_rate": (ta.get("won", {}).get("total", 0) / max(ta.get("total", {}).get("closed", 0), 1)) if ta else None,
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
    ap.add_argument("--sent_thr", type=float, default=0.05)
    ap.add_argument("--net_buy_min", type=float, default=1.0)
    ap.add_argument("--save_png", action="store_true")
    args = ap.parse_args()
    run(args.ticker, args.fast, args.slow, args.cash, args.commission_bps,
        args.sent_thr, args.net_buy_min, args.save_png)

if __name__ == "__main__":
    main()
