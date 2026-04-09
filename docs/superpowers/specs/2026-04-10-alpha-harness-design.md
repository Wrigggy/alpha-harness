# Alpha Harness: LLM-Guided Alpha Research Framework

**Date**: 2026-04-10
**Type**: Research framework extension
**Status**: Design approved

## Goal

Extend `crypto-alpha-mining` into `alpha-harness` — a reusable research framework that:
1. Supports multiple asset classes (crypto + CSI500/Qlib)
2. Expands input features from 6 to 50+ dimensions via momentum composites
3. Adds an LLM-based judge (post-filter) that scores discovered alphas for economic interpretability
4. Includes a paper knowledge base for literature-grounded evaluation
5. Adds a portfolio combination layer beyond simple long-short

The core research question: **Does LLM-based economic reasoning improve the quality of RL-discovered alphas?**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│ Data Sources                                         │
│  ├── Crypto (existing Binance pipeline)              │
│  └── CSI500 (Qlib public datasets)                   │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ Feature Expansion                                    │
│  Multi-window returns, volatility, VWAP deviation,   │
│  volume profile, intra-bar range → 50+ features      │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ Factor Search (existing AlphaGen/QCM)                │
│  GP/RL expression tree search on expanded features   │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ LLM Judge (POST-FILTER, not in-loop)                 │
│  1. Expression → NL translation                      │
│  2. Tag-match relevant papers from knowledge base    │
│  3. LLM scores interpretability (0-1)                │
│  4. Filter: keep score > threshold                   │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ Validation (existing IC/ICIR + enhanced filters)     │
│  RankIC > 0.03, ICIR > 0.5, turnover < 0.3,         │
│  IC decay half-life > 3 bars, pool corr < 0.7        │
└──────────────┬──────────────────────────────────────┘
               ▼
┌─────────────────────────────────────────────────────┐
│ Portfolio Combination                                │
│  Ridge regression / equal-weight combiner            │
│  Optional CVXPY optimization with constraints        │
└─────────────────────────────────────────────────────┘
```

---

## Module Design

### 1. Data Sources (`src/data_sources/`)

**New module.** Provides a unified interface for loading data from different sources.

#### `base.py` — Abstract data source
```python
class DataSource(ABC):
    def load_panel(self) -> dict[str, pd.DataFrame]:
        """Return dict of field → DataFrame (timestamp x symbols)."""
    def get_metadata(self) -> dict:
        """Return dataset metadata (asset_class, frequency, etc.)."""
```

#### `qlib_source.py` — Qlib/CSI500 loader
- Downloads Qlib's public CSI500 dataset (`qlib.contrib.data`)
- Converts to the same panel format as the crypto pipeline
- Supports Alpha158 and Alpha360 feature sets as baselines
- If 1-min data available, aggregates to 1h with boundary control

#### `crypto_source.py` — Thin wrapper around existing `data_collection/`
- Reuses `binance_fetcher.py`, `data_cleaner.py` as-is
- Just implements the `DataSource` interface

#### Config addition (`config/data_config.yaml`):
```yaml
sources:
  crypto:
    enabled: true
    # ... existing binance config
  qlib_csi500:
    enabled: true
    dataset: "Alpha158"  # or "Alpha360"
    instruments: "csi500"
    start_date: "2017-01-01"
    end_date: "2024-12-31"
```

### 2. Feature Expansion (`src/feature_expansion/`)

**New module.** Generates 50+ features from OHLCV base data.

#### `expander.py`
Takes a raw OHLCV panel and produces an expanded feature panel:

| Category | Features | Count |
|----------|----------|-------|
| Multi-window returns | ret_1, ret_2, ret_4, ret_8, ret_20, ret_60 | 6 |
| Multi-window volatility | vol_5, vol_10, vol_20, vol_60 | 4 |
| Momentum composites | mom_5_20, mom_10_60, mom_20_60 (short/long ratios) | 3 |
| VWAP deviation | vwap_dev_5, vwap_dev_10, vwap_dev_20 | 3 |
| Volume profile | vol_ratio_5_20, vol_ratio_10_60, vol_skew_20 | 3 |
| Intra-bar features | hl_range, close_to_hl_mid, close_to_vwap | 3 |
| Price position | close_to_high_20, close_to_low_20 | 2 |
| Rolling statistics | skew_20, kurt_20, autocorr_5 | 3 |
| Cross-sectional | rank_ret_20, rank_vol_20, rank_vwap_dev | 3 |

Total: ~30 base features. With window variants, reaches 50+.

All features are computed with strict point-in-time discipline (no lookahead).

#### Integration with AlphaGen
The expanded features are registered as additional `FeatureType` entries so the expression tree search can operate on them directly. This requires extending AlphaGen's `FeatureType` enum — done via a monkey-patch or a thin wrapper class.

### 3. Knowledge Base (`src/knowledge_base/`)

**New module.** Stores structured paper summaries for the LLM judge to reference.

#### `paper_store.py`
- Stores papers as a JSON file: `papers/corpus.json`
- Each paper follows the schema:
```json
{
  "id": "arxiv_2023_momentum_decay",
  "title": "Short-term Momentum Decay in Chinese A-shares",
  "factor_type": ["momentum", "mean_reversion"],
  "mechanism": "Momentum profits decay within 5 days due to institutional rebalancing",
  "asset_class": ["equity"],
  "frequency": ["intraday", "daily"],
  "decay_horizon": "3-5 days",
  "key_finding": "Intraday momentum reverses after 4 hours in CSI500",
  "source": "arxiv",
  "url": "https://arxiv.org/abs/..."
}
```

#### `retriever.py`
- Tag-based matching: given a factor description, find papers with overlapping `factor_type`, `asset_class`, `frequency`
- Returns top-K matches ranked by tag overlap score
- Simple, no vector DB dependency

#### Paper ingestion
- Manual curation initially (we add papers as we read them)
- Optional: LLM-assisted summarization of new papers into the schema (future enhancement)

### 4. LLM Judge (`src/llm_judge/`)

**New module.** The core research contribution.

#### `base.py` — Abstract judge interface
```python
@dataclass
class JudgeResult:
    expression: str
    nl_description: str
    interpretability_score: float  # 0-1
    economic_narrative: str
    matched_papers: list[str]  # paper IDs
    reasoning: str

class AlphaJudge(ABC):
    def translate(self, expression: str) -> str:
        """Convert expression tree to natural language description."""
    def score(self, expression: str, ic: float, context: dict) -> JudgeResult:
        """Score a candidate alpha for economic interpretability."""
    def batch_score(self, candidates: list[dict]) -> list[JudgeResult]:
        """Score multiple candidates efficiently."""
```

#### `claude_agent_judge.py` — Default implementation (Max plan)
- Uses `claude-agent-sdk` to call Claude via Max subscription
- Translation prompt: given an expression like `zscore(rank(vwap_dev_20) - ts_mean(volume, 10))`, produce a plain-English description of what the signal captures
- Scoring prompt: given the NL description + matched papers + IC stats, rate interpretability 0-1 with reasoning
- Includes prompt templates as configurable strings in YAML

#### `api_judge.py` — Optional API-based implementation
- Uses `anthropic` SDK directly for heavier batch workloads
- Same interface, different backend

#### Judge prompt design (critical)
The judge is instructed to:
1. **Accept** signals with any plausible economic mechanism (momentum, mean reversion, liquidity, microstructure)
2. **Reject** only signals that are clearly nonsensical (e.g., `Abs(Abs(Abs(close)))`) or data artifacts
3. **NOT penalize** novel signals that don't match known archetypes — novelty is not a negative signal

This calibration avoids the familiarity bias risk discussed earlier.

#### Config (`config/judge_config.yaml`):
```yaml
judge:
  backend: "agent_sdk"  # or "api"
  model: "claude-opus-4-6"
  score_threshold: 0.3  # minimum interpretability to pass
  batch_size: 10
  cache_results: true  # cache scores to avoid re-evaluating same expressions
  prompts:
    translation: "prompts/translate.txt"
    scoring: "prompts/score.txt"
```

### 5. Validation Layer (`src/evaluation/`)

**Extend existing module.** Add formalized threshold gates.

#### `validation_gate.py` (new file)
Applies hard quantitative filters after the LLM judge:

| Metric | Threshold | Source |
|--------|-----------|--------|
| Rank IC (mean) | > 0.03 | Existing `ic_analysis.py` |
| Rank ICIR | > 0.5 | Existing `ic_analysis.py` |
| Turnover | < 0.3 (daily) | New computation |
| IC decay half-life | > 3 bars | Existing `factor_decay.py` |
| Correlation to pool | < 0.7 | Existing `factor_correlation.py` |
| LLM judge score | > 0.3 | New from `llm_judge/` |

Factors passing ALL gates enter the final alpha pool.

### 6. Portfolio Combination (`src/portfolio/`)

**New module.** Combines surviving factors into a portfolio signal.

#### `combiner.py`
- **Equal weight**: Simple average of factor signals (baseline)
- **IC-weighted**: Weight by historical IC (existing logic in `FactorPoolManager`)
- **Ridge regression**: Learn optimal weights via regularized regression on training data

#### `optimizer.py` (future enhancement)
- CVXPY-based portfolio optimization with constraints:
  - Sector neutrality (for CSI500)
  - Turnover penalty
  - Position limits
- Not in initial implementation — add when the basic combiner works

### 7. Repo Rename & Structure

Rename `crypto-alpha-mining` → `alpha-harness`.

Final directory structure:
```
alpha-harness/
├── config/
│   ├── data_config.yaml          # Extended with sources section
│   ├── alphagen_config.yaml      # Existing
│   ├── alphaqcm_config.yaml      # Existing
│   ├── judge_config.yaml         # NEW
│   └── universe.json             # Existing
├── data/                          # Existing + new Qlib data
├── external/                      # Existing (AlphaGen, AlphaQCM)
├── src/
│   ├── data_collection/           # Existing (crypto)
│   ├── data_adapter/              # Existing (extended)
│   ├── data_sources/              # NEW: unified data interface
│   │   ├── base.py
│   │   ├── crypto_source.py
│   │   └── qlib_source.py
│   ├── feature_expansion/         # NEW: 50+ feature generation
│   │   └── expander.py
│   ├── factor_mining/             # Existing
│   ├── knowledge_base/            # NEW: paper corpus
│   │   ├── paper_store.py
│   │   └── retriever.py
│   ├── llm_judge/                 # NEW: LLM scoring
│   │   ├── base.py
│   │   ├── claude_agent_judge.py
│   │   └── api_judge.py
│   ├── evaluation/                # Existing + validation_gate.py
│   ├── portfolio/                 # NEW: factor combination
│   │   └── combiner.py
│   ├── backtest/                  # Existing
│   └── utils/                     # Existing
├── prompts/                       # NEW: LLM prompt templates
│   ├── translate.txt
│   └── score.txt
├── papers/                        # NEW: paper corpus JSON
│   └── corpus.json
├── notebooks/                     # Existing (extended)
├── docs/                          # Design docs
├── requirements.txt               # Extended
└── README.md                      # Rewritten
```

---

## Implementation Order

1. **Repo rename** — rename directory + update git remote
2. **Data sources abstraction** — `DataSource` base + crypto wrapper + Qlib loader
3. **Feature expansion** — `expander.py` with 50+ features
4. **Knowledge base** — paper store + retriever + seed corpus
5. **LLM judge** — base interface + Claude Agent SDK implementation + prompts
6. **Validation gate** — formalized threshold filters
7. **Portfolio combiner** — equal weight + IC-weighted + ridge
8. **Integration** — end-to-end pipeline script connecting all layers
9. **Config & README** — update all configs, rewrite README
10. **Notebooks** — add analysis notebooks for judge evaluation

---

## What We're NOT Building (YAGNI)

- No vector DB / semantic search (tag matching is sufficient at research scale)
- No in-loop reward shaping (post-filter only, for now)
- No CVXPY portfolio optimization (ridge regression is sufficient initially)
- No automated paper crawling (manual curation)
- No web UI or dashboard
- No 1-minute data aggregation (use pre-aggregated hourly or daily data)

---

## Dependencies to Add

```
# Qlib data
qlib  # Microsoft's quant library for CSI500 data

# LLM Judge
claude-agent-sdk  # Claude via Max plan
anthropic         # Optional: direct API fallback

# Portfolio
scikit-learn      # Ridge regression (already in scipy dep chain)
```
