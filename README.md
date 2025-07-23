# market-ia-trading-bot

Este repositorio contiene un bot de trading basado en IA. La siguiente tabla
muestra la organizaci\u00f3n recomendada del proyecto:

```
market-ia-trading-bot/
├ data/                 # raw & processed datasets (git-ignored, tracked by DVC)
│   ├ raw/
│   └ processed/
├ notebooks/            # exploratory analysis & demos
├ src/
│   ├ ingest/           # data download & parsing
│   ├ features/         # technical \u2192 sentiment \u2192 fundamentals
│   ├ simulator/        # gym environment & execution logic
│   ├ agents/           # ML / RL models
│   ├ strategy/         # signal fusion & risk rules
│   └ evaluation/       # backtests & metrics
├ tests/                # pytest unit / integration tests
├ Makefile              # common tasks (fetch_data, train_rl, backtest)
├ environment.yml       # conda env spec
├ dvc.yaml              # data/version control pipeline (optional)
└ README.md             # you are here
```

Cada carpeta agrupa componentes relacionados. Los datos se excluyen del
control de versiones y pueden gestionarse con DVC.
