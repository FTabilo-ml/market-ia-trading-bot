
.PHONY: fetch_congress fetch_news preprocess

# 1. Descargar / actualizar repos con las operaciones de congresistas
fetch_congress:
	python -m src.ingest.fetch_congress_trades

# 2. Descargar noticias recientes y guardarlas en Parquet
fetch_news:
	python -m src.ingest.fetch_news

# 3. Generar los Parquet procesados con TA + fundamentales + congresistas + news
preprocess: fetch_congress fetch_news
	python scripts/preprocess.py
