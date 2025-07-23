"""Motor de reglas de trading."""

from src.features.technical_indicators import moving_average


def generate_signals(prices):
    """Genera se\u00f1ales de compra/venta basadas en medias m\u00f3viles."""
    ma = moving_average(prices)
    signals = []
    for price, avg in zip(prices[2:], ma):
        signals.append("buy" if price > avg else "sell")
    return signals
