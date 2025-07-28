#!/usr/bin/env python
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
from __future__ import annotations
import sys, pathlib, argparse, traceback
from typing import Any, Optional, List, cast
import pandas as pd
import backtrader as bt
import backtrader.indicators as btind
from matplotlib.figure import Figure

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data/processed"
ARTS = ROOT / "artifacts" / "backtrader"
ARTS.mkdir(parents=True, exist_ok=True)

class SentCongressData(bt.feeds.PandasData):
    lines = ("sent", "netbuy",)
    params = (("sent", -1), ("netbuy", -1))

def load_df(ticker: str) -> pd.DataFrame:
    fp = PROCESSED / f"{ticker}.parquet"
    if not fp.exists(): raise FileNotFoundError(fp)
    df = pd.read_parquet(fp, engine="fastparquet").copy()
    if "sentiment_score" not in df.columns: df["sentiment_score"] = 0.0
    if "net_buy" not in df.columns: df["net_buy"] = 0.0
    df["Date"] = pd.to_datetime(df.get("Date", df.index))
    df = df.set_index("Date").sort_index()
    df.rename(columns={"sentiment_score":"sent","net_buy":"netbuy"}, inplace=True)
    return df[["Open","High","Low","Close","Volume","sent","netbuy"]]

class SmaSentCongressPortfolio(bt.Strategy):
    params = dict(fast=20, slow=150, sent_thr=0.05, net_buy_min=1.0)

    def __init__(self):
        self.signals = {}
        for d in self.datas:
            fast = btind.SMA(d.close, period=self.p.fast)
            slow = btind.SMA(d.close, period=self.p.slow)
            cross = btind.CrossOver(fast, slow)
            self.signals[d._name] = dict(cross=cross)

    def next(self):
        valid: List[Any] = []
        for d in self.datas:
            cross = self.signals[d._name]["cross"][0]
            sent = float(getattr(d, "sent")[0])
            netb = float(getattr(d, "netbuy")[0])
            if (cross > 0) and (sent >= self.p.sent_thr) and (netb >= self.p.net_buy_min):
                valid.append(d)

        for d in self.datas:
            if self.getposition(d).size and d not in valid:
                self.close(data=d)

        if valid:
            target = 0.95 / max(len(valid), 1)
            for d in valid:
                self.order_target_percent(data=d, target=target)
        else:
            for d in self.datas:
                if self.getposition(d).size:
                    self.close(data=d)

def _flatten(objs: Any) -> list[Any]:
    if objs is None: return []
    if isinstance(objs, (list, tuple)):
        out: list[Any] = []
        for o in objs: out.extend(_flatten(o))
        return out
    return [objs]

def run(tickers: list[str], fast: int, slow: int, cash: float, commission_bps: float,
        sent_thr: float, net_buy_min: float, save_png: bool):
    try:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(float(cash))
        cerebro.broker.setcommission(commission=commission_bps/10000.0)

        for tk in tickers:
            df = load_df(tk)
            data = SentCongressData(
                dataname=df,
                datetime=None,
                open='Open', high='High', low='Low', close='Close',
                volume='Volume', openinterest=None,
                sent='sent', netbuy='netbuy'
            )

            cerebro.adddata(data, name=tk)

        cerebro.addstrategy(SmaSentCongressPortfolio, fast=fast, slow=slow,
                            sent_thr=sent_thr, net_buy_min=net_buy_min)

        # analyzers (sin TimeReturn)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addobserver(bt.observers.BuySell)

        print(f"Starting Portfolio Value: {cerebro.broker.getvalue():,.2f}", flush=True)
        res = cerebro.run()
        print(f"Final Portfolio Value:   {cerebro.broker.getvalue():,.2f}", flush=True)

        if save_png:
            try:
                figs = cerebro.plot(style="candle", iplot=False)
                fig_obj: Optional[Figure] = None
                for obj in _flatten(figs):
                    if hasattr(obj, "savefig"):
                        fig_obj = cast(Figure, obj); break
                if fig_obj is not None:
                    out_png = ARTS / f"PORT_{'-'.join(tickers)}_sma_sent_congress.png"
                    fig_obj.savefig(out_png, dpi=150, bbox_inches="tight")
                    print(f"▶ Gráfico PNG: {out_png}", flush=True)
            except Exception as e:
                print(f"(Plot no disponible: {e})", flush=True)

    except Exception:
        print("❌ Excepción durante la ejecución:\n" + traceback.format_exc(), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=150)
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--commission_bps", type=float, default=10.0)
    ap.add_argument("--sent_thr", type=float, default=0.05)
    ap.add_argument("--net_buy_min", type=float, default=1.0)
    ap.add_argument("--save_png", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run(tickers, args.fast, args.slow, args.cash, args.commission_bps,
        args.sent_thr, args.net_buy_min, args.save_png)

if __name__ == "__main__":
    main()
