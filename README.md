# Crypto Alpha Factor Mining Pipeline

End-to-end pipeline for mining formulaic alpha factors from crypto markets using AlphaGen (PPO) and AlphaQCM (distributional RL).

## Pipeline Overview

1. **Data Collection** — Fetch 1H OHLCV from Binance for ~500 trading pairs
2. **Data Adapter** — Convert to AlphaGen/AlphaQCM tensor format (drop-in StockData replacement)
3. **Factor Mining** — Run AlphaGen (PPO) and AlphaQCM (IQN/QR-DQN) to discover formulaic alphas
4. **Evaluation** — IC/ICIR/RankIC/RankICIR analysis, factor correlation, IC decay
5. **Backtest** — Top-N long / bottom-N short portfolio with transaction costs

## Setup (Full Instructions — works on both Mac and CUDA machines)

### 1. Clone the project

```bash
git clone https://github.com/Wrigggy/crypto-alpha-mining.git
cd crypto-alpha-mining
```

### 2. Create Python environment

**Option A: uv (recommended, fastest)**
```bash
# Install uv if not already installed
# macOS: brew install uv
# Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.10 .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

uv pip install -r requirements.txt
```

**Option B: conda**
```bash
conda create -n crypto-alpha python=3.10
conda activate crypto-alpha
pip install -r requirements.txt
```

### 3. Clone external repos

```bash
git clone https://github.com/RL-MLDM/alphagen.git external/alphagen
git clone https://github.com/ZhuZhouFan/AlphaQCM.git external/alphaqcm
```

### 4. Install AlphaGen + AlphaQCM dependencies

```bash
# Core RL dependencies
pip install gymnasium "sb3_contrib>=2.0" "stable_baselines3>=2.0" shimmy tensorboard

# For CUDA machines, install PyTorch with CUDA support:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Download and prepare data

```bash
# Select top trading pairs from Binance (saves to config/universe.json)
python -m src.data_collection.universe_selector

# Download 1H OHLCV data (takes ~8 min for 500 symbols, 2 years)
python -m src.data_collection.binance_fetcher

# Clean, align, and build panel data
python -m src.data_collection.data_cleaner
```

### 6. Run factor mining

**AlphaGen (PPO-based):**
```bash
# Small-scale test (Mac, ~10-30 min)
python -m src.factor_mining.run_alphagen --small-scale

# Full training (CUDA recommended)
python -m src.factor_mining.run_alphagen
```

**AlphaQCM (distributional RL):**
```bash
# QR-DQN variant
python -m src.factor_mining.run_alphaqcm --model qrdqn --small-scale

# IQN variant (generally better)
python -m src.factor_mining.run_alphaqcm --model iqn --small-scale

# Full training
python -m src.factor_mining.run_alphaqcm --model iqn --pool 20 --std-lam 1.0
```

### 7. Analyze results

```bash
jupyter notebook notebooks/
```

Or use the evaluation modules directly:
```python
from src.data_collection.data_cleaner import load_panel
from src.evaluation.ic_analysis import evaluate_factor
from src.backtest.long_short_backtest import long_short_backtest

panel = load_panel("data/processed")
# ... see notebooks for examples
```

## Project Structure

```
config/                     — YAML configs for data, AlphaGen, AlphaQCM
  data_config.yaml          — Binance API settings, universe filters, train/val/test split
  alphagen_config.yaml      — PPO hyperparameters, search space, pool settings
  alphaqcm_config.yaml      — QCM hyperparameters
  universe.json             — Selected trading pairs (generated)
data/
  raw/                      — Raw Binance OHLCV parquet files (1 per symbol)
  processed/                — Cleaned panel data (timestamp × symbol matrices)
  factors/                  — Discovered factor pools (JSON)
src/
  data_collection/
    universe_selector.py    — Select top-N pairs by volume from Binance
    binance_fetcher.py      — Async OHLCV downloader with incremental updates
    data_cleaner.py         — Clean, align, build panel, compute forward returns
  data_adapter/
    feature_engineering.py  — Normalize OHLCV + compute VWAP
    to_alphagen_format.py   — CryptoStockData (drop-in StockData replacement)
                              + CryptoAlphaCalculator (works with both AlphaGen & QCM)
  factor_mining/
    run_alphagen.py         — AlphaGen PPO training launcher
    run_alphaqcm.py         — AlphaQCM (IQN/QR-DQN/FQF) training launcher
    factor_pool_manager.py  — Factor pool management utilities
  evaluation/
    ic_analysis.py          — IC, ICIR, RankIC, RankICIR
    factor_correlation.py   — Pairwise factor correlation + heatmap
    factor_decay.py         — IC decay over forward horizons
  backtest/
    long_short_backtest.py  — Top-N/bottom-N long-short portfolio
    performance_metrics.py  — AR, Sharpe, MDD, Calmar, turnover, win rate
  utils/
    device.py               — Auto-detect CUDA / MPS / CPU
    logger.py               — Loguru-based logging
notebooks/
  01_data_exploration.ipynb — Data quality, missing data, return distribution
  02_factor_analysis.ipynb  — Factor IC analysis, correlation, decay
  03_backtest_results.ipynb — Long-short backtest visualization
external/                   — AlphaGen and AlphaQCM repos (git-ignored)
```

## Hardware

| | Mac (development) | CUDA (full training) |
|---|---|---|
| Device | MPS (Apple Silicon) | NVIDIA GPU |
| Universe | 50 symbols | 500 symbols |
| Data | 3 months | 2+ years |
| AlphaGen episodes | 1,000 | 50,000+ |
| AlphaQCM steps | 50,000 | 2,000,000 |
| Est. training time | 10-30 min | hours-days |

Device is auto-detected via `src/utils/device.py`.

## Key Design Decisions

- **No Qlib dependency**: CryptoStockData replaces Qlib's data loader entirely, producing the same tensor format `(bars, features, stocks)` that AlphaGen's Expression tree expects.
- **Compatible with both forks**: CryptoAlphaCalculator implements the union of methods needed by upstream AlphaGen's `TensorAlphaCalculator` and AlphaQCM's simpler `AlphaCalculator`.
- **Target**: 8-bar forward return (`Ref(close, -8) / close - 1`), i.e., 8-hour return for 1H data.
- **Chronological splits**: Train 70% / Val 15% / Test 15%, no random shuffling.

## CUDA Machine Quick Start (Copy-Paste)

```bash
# 1. Clone and setup
git clone https://github.com/Wrigggy/crypto-alpha-mining.git
cd crypto-alpha-mining
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium "sb3_contrib>=2.0" "stable_baselines3>=2.0" shimmy tensorboard

# 2. External repos
git clone https://github.com/RL-MLDM/alphagen.git external/alphagen
git clone https://github.com/ZhuZhouFan/AlphaQCM.git external/alphaqcm

# 3. Data pipeline
python -m src.data_collection.universe_selector
python -m src.data_collection.binance_fetcher
python -m src.data_collection.data_cleaner

# 4. Full-scale training
python -m src.factor_mining.run_alphagen
python -m src.factor_mining.run_alphaqcm --model iqn --pool 20 --std-lam 1.0

# 5. Results in data/factors/alphagen_pool.json and data/factors/alphaqcm_pool.json
```
