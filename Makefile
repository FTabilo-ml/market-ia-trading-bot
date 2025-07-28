.PHONY: env ml_dataset train_ml rank_dataset train_rank backtest_rank backtest_bt

# Create/update Conda environment
env:
	conda env update -f environment.yml --prune

# 1) Regression dataset (monthly features + target)
ml_dataset:
	python scripts/make_ml_table.py --horizon 36 --tickers all

# 2) Train long-term regression model and save predictions
train_ml:
	python scripts/train_ml_longterm.py --horizon 36 --model lgbm --split 2016-01-01

# 3) Ranking dataset (TopK selection)
rank_dataset:
	python scripts/make_rank_dataset.py --horizon 36

# 4) Train ranker + scoring (includes future if configured)
train_rank:
	python scripts/train_ranker.py \
	  --horizon 36 --split 2017-01-01 \
	  --rel_bins 3 --eval_k 5,10 \
	  --num_leaves 31 --min_data_in_leaf 200 \
	  --reg_lambda 1.0 --reg_alpha 0.1 \
	  --learning_rate 0.03 --n_estimators 1200 \
	  --sign_from val --val_months 36 --sign_threshold 0.00 \
	  --score_future

# 5) Backtest using ranker signals
backtest_rank:
	python scripts/backtest_longterm_bt.py \
	  --horizon 36 --model rank --topk 5 \
	  --date_from 2017-01-01 --date_to 2020-04-30 \
	  --cash 100000 --commission_bps 10 --save_png

# (Optional) Backtest using standard regression model
backtest_bt:
	python scripts/backtest_longterm_bt.py \
	  --horizon 36 --model lgbm --topk 3 \
	  --only_test --cash 100000 --commission_bps 10 --save_png
