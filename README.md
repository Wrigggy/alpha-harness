# Crypto Alpha Factor Mining Pipeline

End-to-end pipeline for mining formulaic alpha factors from crypto markets using AlphaGen and AlphaQCM.

## Pipeline Overview

1. **Data Collection** — Fetch 1H OHLCV from Binance for ~500 trading pairs
2. **Data Adapter** — Convert to AlphaGen/AlphaQCM tensor format
3. **Factor Mining** — Run AlphaGen (PPO) and AlphaQCM (IQN) to discover formulaic alphas
4. **Evaluation** — IC/ICIR/RankIC/RankICIR analysis, correlation, IC decay
5. **Backtest** — Top-30 long / bottom-30 short portfolio with transaction costs

## Quick Start

```bash
# 1. Create environment
conda create -n crypto-alpha python=3.10
conda activate crypto-alpha
pip install -r requirements.txt

# 2. Clone external repos
git clone https://github.com/RL-MLDM/alphagen.git external/alphagen
git clone https://github.com/ZhuZhouFan/AlphaQCM.git external/alphaqcm
pip install -r external/alphagen/requirements.txt
pip install -r external/alphaqcm/requirements.txt

# 3. Select universe & download data
python -m src.data_collection.universe_selector
python -m src.data_collection.binance_fetcher

# 4. Clean & align data
python -m src.data_collection.data_cleaner

# 5. Run factor mining (use --small-scale for Mac testing)
python -m src.factor_mining.run_alphagen --small-scale
python -m src.factor_mining.run_alphaqcm --small-scale

# 6. Analyze results in notebooks
jupyter notebook notebooks/
```

## Project Structure

```
config/          — YAML configs for data, AlphaGen, AlphaQCM
data/raw/        — Raw Binance OHLCV parquet files
data/processed/  — Cleaned panel data (timestamp x symbol matrices)
data/factors/    — Discovered factor pools (JSON)
src/
  data_collection/  — Universe selection, data download, cleaning
  data_adapter/     — Convert to AlphaGen tensor format
  factor_mining/    — AlphaGen/AlphaQCM training wrappers
  evaluation/       — IC analysis, factor correlation, IC decay
  backtest/         — Long-short portfolio backtest
  utils/            — Device detection, logging
notebooks/       — Exploration, analysis, backtest notebooks
external/        — AlphaGen and AlphaQCM repos (git-ignored)
```

## Hardware

- **Development**: MacBook Air M4 (MPS backend, small-scale tests)
- **Full training**: Windows laptop with NVIDIA GPU (CUDA)

Device is auto-detected via `src/utils/device.py`.
