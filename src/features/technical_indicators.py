"""C\u00e1lculo de indicadores t\u00e9cnicos."""

from typing import List

def moving_average(prices: List[float], window: int = 3) -> List[float]:
    """Devuelve la media m\u00f3vil simple para una ventana dada."""
    if window <= 0:
        raise ValueError("window must be positive")
    return [sum(prices[i:i+window]) / window for i in range(len(prices) - window + 1)]
