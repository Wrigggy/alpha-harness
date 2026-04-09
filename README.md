# Alpha Harness

LLM-guided alpha research framework. Combines automated alpha discovery (AlphaGen/AlphaQCM RL search) with LLM-based economic reasoning to produce interpretable, robust quantitative factors.

**Core research question**: Does LLM-based economic reasoning improve the quality of RL-discovered alphas?

## Architecture

```
Data Sources (Crypto / CSI500)
        │
        ▼
Feature Expansion (OHLCV → 50+ features)
        │
        ▼
Factor Search (AlphaGen PPO / AlphaQCM distributional RL)
        │
        ▼
LLM Judge (post-filter: expression → NL → score)
        │   ├── Expression → natural language translation
        │   ├── Tag-match relevant papers from knowledge base
        │   └── Score interpretability (0-1)
        │
        ▼
Validation Gate (IC > 0.03, ICIR > 0.5, turnover < 0.3, decay > 3 bars)
        │
        ▼
Portfolio Combination (equal weight / IC-weighted / ridge regression)
        │
        ▼
Backtest (long-short with transaction costs)
```

## Setup

```bash
git clone https://github.com/Wrigggy/alpha-harness.git
cd alpha-harness

# Python environment
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# External RL repos
git clone https://github.com/RL-MLDM/alphagen.git external/alphagen
git clone https://github.com/ZhuZhouFan/AlphaQCM.git external/alphaqcm

# Optional: Qlib CSI500 data
pip install pyqlib
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## Usage

### 1. Data Pipeline (Crypto)

```bash
python -m src.data_collection.universe_selector
python -m src.data_collection.binance_fetcher
python -m src.data_collection.data_cleaner
```

### 2. Factor Mining

```bash
# AlphaGen (PPO) — small scale for development
python -m src.factor_mining.run_alphagen --small-scale

# AlphaQCM (distributional RL)
python -m src.factor_mining.run_alphaqcm --model iqn --small-scale
```

### 3. Full Pipeline (with LLM Judge)

```bash
# Evaluate a discovered factor pool with LLM scoring
python -m src.pipeline --source crypto --judge --evaluate-pool data/factors/alphagen_pool.json

# Without LLM judge (validation only)
python -m src.pipeline --source crypto --no-judge --evaluate-pool data/factors/alphagen_pool.json
```

### 4. Analysis

```bash
jupyter notebook notebooks/
```

## Project Structure

```
alpha-harness/
├── config/
│   ├── data_config.yaml          # Data source settings
│   ├── alphagen_config.yaml      # PPO hyperparameters
│   ├── alphaqcm_config.yaml      # Distributional RL settings
│   └── judge_config.yaml         # LLM judge configuration
├── src/
│   ├── data_collection/          # Crypto data pipeline (Binance)
│   ├── data_adapter/             # AlphaGen/QCM tensor format bridge
│   ├── data_sources/             # Unified data interface (crypto + Qlib)
│   ├── feature_expansion/        # OHLCV → 50+ features
│   ├── factor_mining/            # AlphaGen + AlphaQCM runners
│   ├── knowledge_base/           # Paper corpus & retrieval
│   ├── llm_judge/                # LLM-based alpha scoring
│   ├── evaluation/               # IC analysis + validation gates
│   ├── portfolio/                # Factor combination strategies
│   ├── backtest/                 # Long-short portfolio backtest
│   ├── pipeline.py               # End-to-end orchestration
│   └── utils/                    # Device detection, logging
├── prompts/                      # LLM prompt templates
├── papers/                       # Paper corpus (18 seed papers)
├── external/                     # AlphaGen + AlphaQCM repos
├── notebooks/                    # Analysis notebooks
└── docs/                         # Design specs
```

## New Modules (vs. crypto-alpha-mining)

| Module | Purpose |
|--------|---------|
| `data_sources/` | Unified data interface supporting crypto (Binance) and equity (Qlib/CSI500) |
| `feature_expansion/` | Expands 5 OHLCV fields to 51 features (returns, volatility, momentum, VWAP, volume profile, etc.) |
| `knowledge_base/` | 18-paper seed corpus with tag-based retrieval for LLM judge context |
| `llm_judge/` | Expression → NL translation + interpretability scoring via Claude (Max plan or API) |
| `evaluation/validation_gate.py` | Formalized multi-gate validation (IC, ICIR, turnover, decay, correlation, judge score) |
| `portfolio/` | Factor combination: equal weight, IC-weighted, ridge regression |
| `pipeline.py` | End-to-end orchestration connecting all layers |

## LLM Judge

The judge operates as a **post-filter** (not in the search loop):

1. **Translate**: Expression tree → natural language description
2. **Retrieve**: Tag-match relevant papers from the knowledge base
3. **Score**: Rate interpretability 0-1 with economic narrative

Calibration: the judge accepts signals with *any* plausible mechanism and only rejects clearly nonsensical expressions. Novelty is not penalized.

**Backend options** (configured in `config/judge_config.yaml`):
- `agent_sdk` (default): Uses Claude via Max plan subscription — no API costs
- `api`: Uses Anthropic API directly — for heavier batch workloads

## Hardware

| | Mac (dev) | CUDA (full) |
|---|---|---|
| Device | MPS | NVIDIA GPU |
| Universe | 50 symbols | 500 symbols |
| AlphaGen | 1,000 steps | 300,000 steps |
| AlphaQCM | 50,000 steps | 300,000 steps |

## Key Design Decisions

- **No Qlib dependency for crypto**: CryptoStockData produces the same tensor format AlphaGen expects
- **Post-filter judge**: LLM scores after search, not during — keeps search fast, enables clean ablation
- **Tag-based retrieval**: No vector DB — JSON corpus + tag matching is sufficient at research scale
- **Max plan first**: Claude Agent SDK uses subscription credits, no per-token API costs
