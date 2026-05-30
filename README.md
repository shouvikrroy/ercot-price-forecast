# ERCOT Day-Ahead Price Forecasting

Benchmarks seven forecasting models on ERCOT HB_NORTH day-ahead prices (2022–2026). A zero-shot foundation model (Amazon Chronos) outperforms all trained models; fine-tuning on domain data provides further marginal improvement.

## Results

Chronological train/val/test split. Test set: Oct 2024 – May 2026 (5,750 hours).

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **Chronos (fine-tuned)** | **5.36** | **32.33** | **12.90%** |
| Chronos (zero-shot) | 5.37 | 32.36 | 12.99% |
| LightGBM | 6.79 | 36.95 | 17.84% |
| LSTM (avg 5 seeds) | 7.48 | 41.83 | 17.92% |
| PatchTST (avg 3 seeds) | 7.95 | 56.09 | 17.79% |
| Prophet | 14.53 | 67.18 | 45.46% |
| SARIMA | 22.41 | 148.03 | 49.99% |
| Persistence (baseline) | 22.83 | 82.56 | 67.14% |

See `INTERPRETATION.md` for regime breakdown, spike detection analysis, and fine-tuning details.

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| ERCOT via [gridstatus](https://github.com/gridstatus/gridstatus) | DAM settlement point prices (HB_NORTH, HB_HOUSTON, HB_SOUTH, HB_WEST) | Hourly |
| EIA Hourly Electric Grid Monitor | Wind and solar generation (ERCO region) | Hourly |
| gridstatus | ERCOT system load | Hourly |
| EIA Natural Gas API | Henry Hub spot price | Daily → forward-filled |

Requires `EIA_API_KEY` in `.env`. See `.env.example`.

## Project Structure

```
src/
  pipeline.py          # Data ingestion, feature engineering, train/val/test splits
  evaluate.py          # MAE, RMSE, MAPE
  models/
    persistence.py     # Naive baseline
    sarima.py          # Rolling SARIMAX
    prophet_model.py   # Prophet with exogenous regressors
    lgbm.py            # LightGBM (43 features)
    lstm.py            # Two-layer LSTM
    patchtst.py        # PatchTST transformer
    chronos_eval.py    # Chronos zero-shot and fine-tuned inference
run_evaluation.py      # Full benchmark — all models, saves timestamped CSV
run_chronos.py         # Chronos-only eval with comparison table
finetune_chronos.py    # Custom PyTorch fine-tuning loop for Chronos T5
analyze_chronos.py     # Regime, spike, temporal, and bias analysis
```

## Setup

```bash
# Install dependencies (requires uv)
uv sync

# Add EIA API key
cp .env.example .env

# Build dataset (~5 min, downloads ~500 MB)
uv run python -c "from src.pipeline import run_pipeline; run_pipeline()"

# Run full benchmark
uv run python run_evaluation.py

# Run Chronos only
uv run python run_chronos.py

# Fine-tune Chronos on ERCOT data (~16 min on RTX 3070)
uv run python finetune_chronos.py

# Run tests
uv run pytest
```
