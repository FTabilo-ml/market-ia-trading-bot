import pathlib
import click
import yfinance as yf

RAW_DIR = pathlib.Path("data/raw/prices")
RAW_DIR.mkdir(parents=True, exist_ok=True)

@click.command()
@click.option("--tickers", default="AAPL,MSFT,NVDA", help="Comma-sep tickers")
@click.option("--start", default="2015-01-01", help="YYYY-MM-DD")
def main(tickers: str, start: str) -> None:
    tickers_list = [t.strip().upper() for t in tickers.split(",")]
    df = yf.download(tickers_list, start=start, auto_adjust=True, progress=False)
    out = RAW_DIR / f"{'-'.join(tickers_list)}.parquet"
    df.to_parquet(out, compression="zstd")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
