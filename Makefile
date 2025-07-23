fetch_data:
	python -m src.ingest.fetch_prices --tickers "AAPL,MSFT,NVDA"

train_rl:
	@echo "Training RL agent..."

backtest:
	python -m src.evaluation.backtest
