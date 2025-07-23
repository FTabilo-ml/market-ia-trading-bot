"""Herramienta de backtesting para estrategias."""

from src.strategy.rules_engine import generate_signals
from src.ingest.fetch_prices import fetch_prices


def run_backtest(symbol: str):
    """Ejecuta un backtest simple con precios simulados."""
    prices = fetch_prices(symbol)
    signals = generate_signals(prices)
    return list(zip(prices[2:], signals))
