# market-ia-trading-bot

Este proyecto implementa un bot de trading basado en IA. A continuaci\u00f3n se describe la estructura recomendada:

```
market-ia-trading-bot/
├ data/
│   └ raw/                   <- precios, fundamentos, noticias sin procesar
├ notebooks/
│   └ 01_eda_exploration.ipynb
├ src/
│   ├ ingest/
│   │   ├ fetch_prices.py
│   │   ├ fetch_fundamentals.py
│   │   └ fetch_news.py
│   ├ features/
│   │   ├ technical_indicators.py
│   │   ├ sentiment_analysis.py
│   │   └ fundamental_scores.py
│   ├ strategy/
│   │   └ rules_engine.py
│   └ evaluation/
│       └ backtest.py
├ .gitignore
├ README.md
└ requirements.txt
```

Cada m\u00f3dulo se divide por responsabilidades: `ingest` obtiene los datos, `features` genera variables de inter\u00e9s, `strategy` define las reglas de operaci\u00f3n y `evaluation` permite probar el desempe\u00f1o del bot.
