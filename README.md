# Long‑Horizon Cross‑Sectional Ranker + Monthly Backtest (Backtrader)

**v1** — Build a monthly cross‑sectional ranking model (LightGBM LambdaRank) from OHLCV data, generate monthly signals (Top‑K), and run a transaction‑cost aware Backtrader backtest with monthly rebalancing.

🇪🇸 **Nota**: gran parte del código y ejemplos se explican en inglés para el README público. Los comentarios del código están en español.

---

## What this repo does

### 1. Dataset builder (`make_rank_dataset.py`)

- Turns per‑ticker OHLCV history into a monthly panel with engineered features and a forward label `y_fwd_{H}m` (e.g., 36‑month forward return).
- Cleans the universe (min price / median \$ADV filters, ETF/ETN blacklist).
- Optional sector/industry neutralization (monthly z‑scores inside sector).
- Ensures no look‑ahead: future rows keep NaN labels.

### 2. Ranker training (`train_ranker.py`)

- Trains a LightGBM LambdaRank model on monthly groups, using ordinal labels by monthly quantiles (`--rel_bins`).
- Temporal split: **train** (`Date < split`), **test** (`Date ≥ split`, label not‑NaN), **future** (`Date ≥ split`, label NaN).
- Evaluates IC (Spearman) by month, NDCG\@K, TopK‑BottomK monthly spread.
- Sign check: if average IC on a validation window is negative, flip signal sign (saves both `raw_score` and `score`).
- Optionally scores the future set for the backtest.

### 3. Monthly backtest (`backtest_longterm_bt.py`)

- Builds monthly signals from predictions (Top‑K) and runs a monthly rebalancing strategy in Backtrader with commissions in bps.
- Hard date cropping (e.g., `2017-01-01..2020-04-30`).
- Outputs summary metrics (Final Value, MaxDD, Sharpe) and artifacts (PNG, per‑ticker charts, trades CSV).

---

## Folder layout

```
.
├─ scripts/
│  ├─ prep/
│  │  └─ preprocess.py
│  ├─ datasets/
│  │  ├─ make_ml_table_longterm.py
│  │  └─ make_rank_dataset.py
│  ├─ training/
│  │  ├─ train_ml.py
│  │  └─ train_ranker.py
│  ├─ backtests/
│  │  ├─ backtest_longterm.py
│  │  └─ backtest_longterm_bt.py
│  └─ tools/
│     ├─ backtest_ml.py
│     ├─ portfolio_backtest.py
│     ├─ grid_search_sma.py
│     └─ ...
├─ legacy/
│  └─ backtrader_sma_sent_congress.py
├─ data/
│  ├─ processed/          # per-ticker OHLCV parquet: TICKER.parquet
│  └─ ml/                 # dataset / predictions parquet
├─ artifacts/
│  ├─ rank/               # trained models (.pkl)
│  └─ longterm/           # backtest plots / trades
```

**Per‑ticker input format** (`data/processed/TICKER.parquet`):

- Required cols: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- Optional: `Sector`, `Industry` (for sector‑neutral z‑scores)
- Indexed by `Date` or include `Date` column (daily).

---

## Data downloads

1. **Prices** – download the Kaggle [stock market dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) and convert each CSV to Parquet under `data/raw/kaggle/prices/parquet/` (one file per ticker). Any helper that writes `TICKER.parquet` with OHLCV + Adj Close is valid.
2. **Congress trades** – either run `make fetch_congress` (uses `src/ingest/fetch_congress_trades`) or download the [congressional trading dataset](https://www.kaggle.com/datasets/shabbarank/congressional-trading-inception-to-march-23) and save it as `data/raw/congress/trades.parquet`.
3. **News** – run `make fetch_news` to grab recent headlines via Google News RSS.

Once the raw files exist you can preprocess everything:

```bash
make preprocess       # runs scripts/prep/preprocess.py
```

---

## Data preprocessing (`preprocess.py`)

The `preprocess.py` script combines raw price files with fundamentals, congress trading flows
and news sentiment. It outputs one Parquet file per ticker under `data/processed/` with
technical indicators and auxiliary features.

Run it at least once before building datasets:

```bash
python scripts/prep/preprocess.py
```

---

## Requirements

- Python 3.9+
- Packages: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `backtrader`, `matplotlib`, `joblib`, and one parquet engine: `fastparquet` or `pyarrow`.

```bash
pip install pandas numpy scikit-learn lightgbm backtrader matplotlib joblib fastparquet
# or: pip install pyarrow
```

> If you use Conda on Windows, it’s fine—just ensure LightGBM installs successfully.

---

## Quickstart (reproduce v1)

1. **Preprocess raw data**

   ```bash
   make preprocess
   ```

   This downloads congress/news data if missing and runs `scripts/prep/preprocess.py` to populate `data/processed/` with per‑ticker files.

2. **Build the dataset**

   ```bash
   python scripts/datasets/make_rank_dataset.py --horizon 36
   ```

   This writes: `data/ml/dataset_rank_36m.parquet`.

3. **Train the ranker + score future**

   The v1 settings that performed best in our tests (robust, smooth ordering):

   ```bash
   python scripts/training/train_ranker.py \
     --horizon 36 --split 2017-01-01 \
     --rel_bins 3 --eval_k 5,10 \
     --num_leaves 31 --min_data_in_leaf 200 \
     --reg_lambda 1.0 --reg_alpha 0.1 \
     --learning_rate 0.03 --n_estimators 1200 \
     --sign_from val --val_months 36 --sign_threshold 0.00 \
     --score_future
   ```

   **Outputs:**

   - `artifacts/rank/ranker_36m_lgbm.pkl`
   - `data/ml/preds_rank_36m.parquet` with columns: `Date`, `Ticker`, `split`, `raw_score`, `score`
   - IC/NDCG printed for test months (e.g., 2017‑01..2017‑04 if H=36).

4. **Run the backtest (2017‑01..2020‑04)**

   ```bash
   python scripts/backtests/backtest_longterm_bt.py \
     --horizon 36 --model rank --topk 5 \
     --date_from 2017-01-01 --date_to 2020-04-30 \
     --cash 100000 --commission_bps 10 --save_png
   ```

   **Outputs:**

   - Plot: `artifacts/longterm/LT_36m_rank_top5.png`
   - Trades: `artifacts/longterm/trades_36m_rank_top5.csv`
   - Console metrics: Final Value, Max Drawdown, Sharpe, `months_with_signal`, `tickers_loaded`

> Example (our run): Final Value \~178,408 on 100,000 (Jan‑2017..Apr‑2020). This is a historical simulation—no investment advice.

---

## How each script works

### `make_rank_dataset.py`

1. Builds monthly features (returns 1/3/6/12/24/36m, MAs, price/MA, liquidity).
2. Forward label: `y_fwd_{H}m` (e.g., 36m forward return).
3. Filters universe: requires min price & median \$ADV; excludes ETF/ETN/leverage tickers.
4. Monthly standardization; optional sector‑neutral z‑score if `Sector`/`Industry` present.
5. Saves `data/ml/dataset_rank_{H}m.parquet`.

### `train_ranker.py`

1. Loads dataset, creates train/test/future by `--split`.
2. Converts continuous y to ordinal labels by monthly quantiles (`--rel_bins`).
3. Groups by month (LightGBM ranking requires group sizes).
4. Trains `LGBMRanker(objective="lambdarank")` with a pipeline: `OneHotEncoder(Ticker)` + numerical passthrough.
5. Sign check on validation window (`--sign_from val`, `--val_months`): flips sign if IC ≤ -threshold.
6. Scores train/test/future → writes `preds_rank_{H}m.parquet`.

### `backtest_longterm_bt.py`

1. Reads predictions, filters to date range (inclusive).
2. For each month: sorts by `score`, picks Top‑K, assigns equal weights.
3. Loads cropped price feeds into Backtrader.
4. Monthly rebalance on the first bar of each month using `order_target_percent`.
5. Broker costs: `--commission_bps` (bps per trade).
6. Saves overall PNG, per‑ticker charts (optional), and trades CSV.

---

## Key CLI options

### `train_ranker.py`

| Arg                  | Meaning                                             |
| -------------------- | --------------------------------------------------- |
| `--horizon`          | Label horizon in months (e.g., 36).                 |
| `--split`            | Temporal split date (e.g., 2017-01-01).             |
| `--rel_bins`         | Monthly quantile bins for ordinal labels (≥2; 3–4). |
| `--eval_k`           | K list for NDCG (e.g., 5,10).                       |
| `--sign_from`        | Where to do sign check: `val` or `test`.            |
| `--val_months`       | Months used for sign check if `sign_from=val`.      |
| `--sign_threshold`   | Flip if mean IC ≤ −threshold.                       |
| `--score_future`     | Score rows with label NaN (backtest period).        |
| `--num_leaves`,      | Complexity controls (lower/higher → smoother).      |
| `--min_data_in_leaf` |                                                     |
| `--reg_lambda`,      | L2 / L1 regularization.                             |
| `--reg_alpha`        |                                                     |
| `--learning_rate`,   | Booster hyper‑params.                               |
| `--n_estimators`     |                                                     |
| `--dataset_fp`       | Custom dataset path (defaults to `data/ml/...`).    |

### `backtest_longterm_bt.py`

| Arg                | Meaning                                              |
| ------------------ | ---------------------------------------------------- |
| `--horizon`        | Match horizon used during training.                  |
| `--model`          | Use `rank` to read `preds_rank_{H}m.parquet`.        |
| `--topk`           | Number of names per month.                           |
| `--date_from,to`   | Hard backtest window (inclusive).                    |
| `--tickers`        | Comma‑separated allowlist (filters signals & feeds). |
| `--only_test`      | Use only test rows from predictions.                 |
| `--cash`           | Initial capital.                                     |
| `--commission_bps` | Commission in basis points per transaction.          |
| `--save_png`       | Save the overall backtest plot.                      |
| `--per_ticker_png` | Optional per‑ticker trade plots.                     |
| `--charts_limit`   | Maximum per‑ticker charts.                           |
| `--charts_dir`     | Directory for per‑ticker charts.                     |
| `--trades_csv`     | Custom path for trades CSV.                          |

---

## Interpreting metrics

- **IC (Spearman)** by month (test): ordering quality; positive is good.
- **NDCG\@K** by month (test): ranking quality at top‑K with ordinal labels.
- **TopK–BottomK spread**: intuitive performance sanity check.
- **Backtest**: Final equity, MaxDD, Sharpe depend on costs, rebalance frequency, and universe.

---

## Tips to improve out‑of‑sample

- Prefer cleaner universe (price ≥ 3–5, median \$ADV ≥ 1–5M), exclude ETFs/ETNs/leverage.
- Keep `rel_bins` small (3–4).
- Increase `min_data_in_leaf`, reduce `num_leaves`, add `reg_*`.
- Sector/industry neutralization avoids unintended macro bets.
- Consider ensemble ranks (momentum/quality/value/low‑vol) averaged.
- Add walk‑forward re‑training for longer evaluations.

---

## Troubleshooting

- `` → run `make_rank_dataset.py --horizon H`.
- **Parquet engine errors** → install `fastparquet` or `pyarrow`.
- **Ticker dtype error in LightGBM** → don't pass `Ticker` as numeric; it’s encoded via OneHotEncoder.
- **Pylance typing warnings** → safe to ignore; they don’t prevent running.
- **No signals for some months** → check that predictions include future rows (run with `--score_future`).

---

## License & Disclaimer

This code is for research/educational use. **No investment advice.** Past performance in backtests does not guarantee future results. You are responsible for verifying data quality, costs, and assumptions before any real‑world use.

---

## Acknowledgments

- LightGBM (LambdaRank)
- Backtrader

---

*Done.* If you want, I can also generate a minimal `requirements.txt` and a short example notebook that runs the three steps end‑to‑end.
