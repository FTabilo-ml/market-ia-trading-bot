# src/features/congress_flows.py
import pandas as pd

def congress_flows(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame de trades del Congreso con columnas mínimas:
      Ticker, TransactionDate, PublicationDate, dir, AmountLow

    Devuelve un DataFrame indexado por Date con agregados diarios:
      net_buy, buy_count, sell_count, amount_net
    """
    df = trades.copy()
    # Asegura columnas
    required = {"Ticker", "PublicationDate", "dir"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas en trades: {required - set(df.columns)}")

    df["PublicationDate"] = pd.to_datetime(df["PublicationDate"], errors="coerce")
    df = df.dropna(subset=["PublicationDate"])
    df["Date"] = df["PublicationDate"].dt.date

    # direction ya viene como 1 compra, -1 venta, 0 otro
    df["dir"] = df["dir"].fillna(0).astype(int)
    df["AmountLow"] = df.get("AmountLow", 0).fillna(0)

    agg = (
        df.groupby(["Ticker", "Date"])
          .agg(
              net_buy=("dir", "sum"),
              buy_count=("dir", lambda x: (x == 1).sum()),
              sell_count=("dir", lambda x: (x == -1).sum()),
              amount_net=("AmountLow", lambda x: (x * df.loc[x.index, "dir"]).sum())
          )
          .reset_index()
    )

    agg["Date"] = pd.to_datetime(agg["Date"])
    return agg.set_index("Date")
