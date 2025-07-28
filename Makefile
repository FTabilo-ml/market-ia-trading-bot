.PHONY: env preprocess rank_dataset train_rank backtest_rank fetch_congress fetch_news

# Create/update Conda environment
env:
	conda env update -f environment.yml --prune

# Data download and preprocessing
fetch_congress:
	python -m src.ingest.fetch_congress_trades

fetch_news:
	python -m src.ingest.fetch_news

preprocess: fetch_congress fetch_news
       python scripts/prep/preprocess.py

# Build monthly ranking dataset
rank_dataset:
       python scripts/datasets/make_rank_dataset.py --horizon 36

# Train LightGBM ranker and score future months
train_rank:
       python scripts/training/train_ranker.py \
	  --horizon 36 --split 2017-01-01 \
	  --rel_bins 3 --eval_k 5,10 \
	  --num_leaves 31 --min_data_in_leaf 200 \
	  --reg_lambda 1.0 --reg_alpha 0.1 \
	  --learning_rate 0.03 --n_estimators 1200 \
	  --sign_from val --val_months 36 --sign_threshold 0.00 \
	  --score_future

backtest_rank:
       python scripts/backtests/backtest_longterm_bt.py \
	  --horizon 36 --model rank --topk 5 \
	  --date_from 2017-01-01 --date_to 2020-04-30 \
	  --cash 100000 --commission_bps 10 --save_png
