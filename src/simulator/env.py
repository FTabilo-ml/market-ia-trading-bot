"""Entorno de simulaci\u00f3n de trading."""

class TradingEnv:
    """Entorno simplificado para pruebas."""
    def reset(self):
        return []

    def step(self, action):
        return [], 0.0, True, {}
