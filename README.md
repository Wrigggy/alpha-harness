# Alpha Harness

LLM-guided alpha research framework. Combines automated alpha discovery (AlphaGen PPO / AlphaQCM distributional RL) with LLM-based **warm-start composition** and **post-search interpretability scoring** to produce robust, interpretable quantitative factors.

**Core research question**: Where does an LLM add value in formulaic factor mining — as a *prior* (warm-start), as an *interpretability filter* (post-judge), or both?

## Architecture

```
Data Sources (Crypto / A-share CSI300)
        │
        ▼
Feature Expansion (OHLCV → 50+ features)
        │
        ▼
LLM Idea-Agent  ──►  Warm-Start Seeds (pick mode OR compose mode)
        │                       │
        │                       ▼
        ▼              ┌────────────────────┐
Factor Search ◄────────┤ pool.force_load_exprs
(AlphaGen PPO or AlphaQCM distributional RL)
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

# Optional: Qlib CSI300 data for A-share experiments
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

### 2. Build the 80-Factor Library (one-time)

```bash
python scripts/build_factor_library.py    # writes data/factor_library.json
```

Library composition: GTJA-191 + WorldQuant Alpha101 + HTSY/SWS A-share factors + A-share-specific staples. Every expression is validated through the AlphaGen parser at build time.

### 3. LLM Idea-Agent — Warm-Start Seeds

Two modes, controlled by `--mode`. The LLM backend is independent and controlled by `--llm-backend` (see [LLM Backends](#llm-backends) below).

```bash
# Compose mode (default) — LLM composes NEW expressions from library primitives,
# conditioned on the current market regime; backend defaults to claude_code (Max plan)
python -m src.factor_mining.idea_agent \
    --mode compose --seed 42 --top-k 10 \
    --data-source cn \
    --out data/factors/warm_seeds_cn_compose_seed42.json

# Same call, but route through OpenRouter (requires OPENROUTER_API_KEY)
export OPENROUTER_API_KEY=sk-or-...
python -m src.factor_mining.idea_agent \
    --mode compose --seed 42 --top-k 10 \
    --data-source cn \
    --llm-backend openrouter --model deepseek/deepseek-chat \
    --out data/factors/warm_seeds_cn_compose_deepseek_seed42.json

# Pick mode (baseline) — LLM picks factor IDs from the library verbatim
python -m src.factor_mining.idea_agent \
    --mode pick --seed 42 --top-k 10 \
    --data-source cn \
    --out data/factors/warm_seeds_cn_pick_seed42.json
```

Output is a JSON pool of {id, expression, train_IC, hypotheses, [template, used_library_ids]} that any RL runner can consume via `--warm-seeds`.

### 4. Factor Mining (with or without warm-start)

Same interface across both RL backends:

```bash
# AlphaGen on CSI300, compose warm-start
python -m src.factor_mining.run_alphagen_cn \
    --seed 42 --n-steps 100000 \
    --warm-seeds data/factors/warm_seeds_cn_compose_seed42.json \
    --run-name B_compose_alphagen_cn_seed42

# AlphaQCM on CSI300, compose warm-start (distributional RL)
python -m src.factor_mining.run_alphaqcm \
    --data-source cn --seed 42 --model qrdqn \
    --warm-seeds data/factors/warm_seeds_cn_compose_seed42.json \
    --run-name B_compose_qcm_cn_seed42

# AlphaQCM on crypto, vanilla
python -m src.factor_mining.run_alphaqcm --data-source crypto --small-scale
```

### 5. Full Pipeline (with LLM Judge post-filter)

```bash
python -m src.pipeline --source crypto --judge \
    --evaluate-pool data/factors/B_compose_alphagen_cn_seed42_pool.json
```

### 6. Ablation Experiment + Auto Report

```bash
bash scripts/run_experiment_cn.sh
# Runs 6 conditions (2 seeds × 3 conditions), then nbconvert produces
# notebooks/04_llm_ablation.html for one-file scp back to laptop
```

### 7. Analysis

```bash
jupyter notebook notebooks/
```

## LLM Warm-Start — Pick vs Compose

The idea-agent does the same thing both modes: it asks the LLM, conditioned on a set of market hypotheses, to propose factors that the RL agent should start with. The crucial difference is **what the LLM emits**.

### Pick mode (baseline)

LLM returns a JSON array of factor IDs. We look them up in the 80-factor library and load the canonical expressions directly into the RL pool.

**Empirical finding (negative result)**: pick-mode warm-start *hurts* AlphaGen's final IC. The pool starts at val IC ≈ 0.05 from the LLM picks, then test IC **dips** during PPO training before recovering — never beating the vanilla curve.

**Why it fails**: AlphaGen's reward = incremental IC of a new factor added to the pool. Pre-loading high-IC factors makes the marginal contribution of newly-generated factors tiny → advantage signal saturates → PPO explores poorly. The warm-start does not poison parameters; it **kills reward signal density**. This is the same root cause that motivates AlphaQCM's distributional RL.

### Compose mode (default)

LLM receives:
- the 80-factor library as **primitives**, referenced by ID (e.g., `m_008`, `v_003`)
- 10 market hypotheses (short reversal, medium momentum, vol regime, …)
- a **regime summary** computed from the train-data tail (realized vol, median drift, cross-sectional dispersion)

LLM emits templates like `Mul(Sub(0,m_008),Div(v_003,v_004))`. A resolver in `idea_agent.py` expands each F-ID into its canonical RPN expression, the existing parser validates the result, IC is scored on train, negative-IC seeds are sign-flipped via `Mul(-1.0, ...)`, and the top-K are written to the warm-seeds JSON.

**Why compose mode is the contender**: composed factors are *novel* (not in the library) yet *grounded* in vetted building blocks. They don't pre-fill the high-IC slots → RL still has headroom to improve from. Inspired by AlphaAgent (KDD 2025), LLM-MCTS (arXiv 2505.11122), CogALPHA (arXiv 2511.18850), and QuantaAlpha (ICLR 2026 review). Pick mode is preserved as a baseline-negative control.

A B-style **free-form generation** mode (LLM emits raw expressions without a library scaffold, with novelty regularization) is parked as future work.

## Project Structure

```
alpha-harness/
├── config/
│   ├── data_config.yaml          # Crypto + CSI300 segments
│   ├── alphagen_config.yaml      # PPO hyperparameters
│   ├── alphaqcm_config.yaml      # Distributional RL settings
│   └── judge_config.yaml         # LLM judge configuration
├── data/
│   └── factor_library.json       # 80-factor curated library (gitignored)
├── prompts/
│   ├── idea_agent_pick.txt       # Pick-mode prompt
│   ├── idea_agent_compose.txt    # Compose-mode prompt
│   ├── score.txt                 # Judge scoring
│   └── translate.txt             # Expression → NL
├── src/
│   ├── data_collection/          # Crypto data pipeline (Binance)
│   ├── data_adapter/             # AlphaGen/QCM tensor format bridge
│   ├── data_sources/             # Unified data interface (crypto + Qlib)
│   ├── feature_expansion/        # OHLCV → 50+ features
│   ├── factor_mining/
│   │   ├── idea_agent.py         # LLM warm-start (pick + compose)
│   │   ├── _regime.py            # Realized vol/drift/dispersion summary
│   │   ├── _calc_factory.py      # Centralized crypto/cn calculator factory
│   │   ├── _qcm_parser.py        # Parser for QCM-fork Expression types
│   │   ├── run_alphagen.py       # AlphaGen PPO (crypto)
│   │   ├── run_alphagen_cn.py    # AlphaGen PPO (CSI300)
│   │   └── run_alphaqcm.py       # AlphaQCM (crypto OR CSI300)
│   ├── knowledge_base/           # Paper corpus & retrieval
│   ├── llm_client/               # Unified LLM dispatcher (claude_code / openrouter / anthropic)
│   ├── llm_judge/                # Backend-agnostic LLM judge
│   ├── evaluation/               # IC analysis + validation gates
│   ├── portfolio/                # Factor combination strategies
│   ├── backtest/                 # Long-short portfolio backtest
│   ├── pipeline.py               # End-to-end orchestration
│   └── utils/                    # Device detection, logging
├── scripts/
│   ├── build_factor_library.py   # Programmatic library builder + validator
│   ├── run_experiment_cn.sh      # 6-run CSI300 ablation + auto HTML report
│   └── smoke_test.sh             # Mac CPU smoke test on crypto
├── papers/                       # Paper corpus (18 seed papers)
├── external/                     # alphagen + alphaqcm (gitignored)
└── notebooks/                    # Analysis notebooks
```

## LLM Backends

Both the idea-agent (warm-start) and the judge (post-filter) call out to an LLM through a single dispatcher at `src/llm_client/`. Three backends are wired up:

| Backend | Auth | Default model | Notes |
|---|---|---|---|
| `claude_code` *(default)* | Max-plan subscription, **no API key** | `claude-opus-4-7` | Uses `claude_agent_sdk` |
| `openrouter` | `OPENROUTER_API_KEY` | `deepseek/deepseek-chat` | OpenAI-compatible; lets you swap to GPT/Gemini/DeepSeek/etc. |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-4-7` | Direct anthropic SDK |

Selection precedence: **CLI flag** (`--llm-backend ...`) → **env var** (`LLM_BACKEND=...`) → **default** (`claude_code`).

```bash
# One-shot override for a single run
python -m src.factor_mining.idea_agent ... --llm-backend openrouter

# Or session-wide
export LLM_BACKEND=openrouter
export OPENROUTER_API_KEY=sk-or-...
```

For the judge, `config/judge_config.yaml` carries `backend:` and `model:` so the post-filter respects whichever provider you want without touching CLI flags. Legacy slugs `agent_sdk` and `api` are auto-mapped to `claude_code` and `anthropic` for back-compat with older checkouts.

## LLM Judge

The judge operates as a **post-filter** (not in the search loop):

1. **Translate**: Expression tree → natural language description
2. **Retrieve**: Tag-match relevant papers from the knowledge base
3. **Score**: Rate interpretability 0-1 with economic narrative

Calibration: the judge accepts signals with *any* plausible mechanism and only rejects clearly nonsensical expressions. Novelty is not penalized.

The judge is backend-agnostic — see [LLM Backends](#llm-backends) above.

## Hardware

|  | Mac (dev) | CUDA (full) |
|---|---|---|
| Device | MPS / CPU | NVIDIA GPU |
| Universe | 50 crypto symbols | 300 A-share symbols (CSI300) |
| AlphaGen | 1,000 steps (smoke) | 100,000–300,000 steps |
| AlphaQCM | 50,000 steps | 300,000 steps |

`--device {auto,cuda,mps,cpu}` is exposed on every runner. The MPS backend is OK for training but `pool.optimize()` calls float64 ops that MPS does not support, so the smoke test pins to `cpu`.

## Ablation Conditions

`scripts/run_experiment_cn.sh` runs 2 seeds × 3 conditions on CSI300:

| Condition | What it tests |
|---|---|
| A — vanilla | RL search, no LLM at all |
| B — warm-start | RL search starting from LLM-composed factors |
| C — warm-start + judge | B's final pool, judge filter applied to drop low-interpretability factors |

C is built from B's pool with no extra training, which keeps the GPU budget at 4 training runs total. The notebook `notebooks/04_llm_ablation.ipynb` is auto-executed by `nbconvert` at the end of the script and produces a single self-contained HTML report.

## Key Design Decisions

- **No Qlib dependency for crypto**: CryptoStockData produces the same tensor format AlphaGen expects
- **Post-filter judge**: LLM scores after search, not during — keeps search fast, enables clean ablation
- **Tag-based retrieval**: No vector DB — JSON corpus + tag matching is sufficient at research scale
- **Pluggable LLM backend**: all call sites go through `src/llm_client/`; `claude_code` (Max-plan, free) is the default, but the same code can route to `openrouter` (BYO-key, multi-model) or `anthropic` (direct API) without edits
- **Compose-mode resolver**: LLM references library factors by ID (`m_008`) instead of repeating their full RPN. Cuts prompt tokens, lets the LLM stay focused on combination logic, and makes the audit trail (`used_library_ids`) explicit
- **Sign-flip on negative IC**: warm seeds wrap negative-IC factors in `Mul(-1.0, ...)` so the RL pool's `force_load_exprs` never rejects them on sign alone
- **In-tree parser for QCM**: the AlphaQCM fork ships expression.py but not parser.py, and `external/` is gitignored — so a parser targeting QCM Expression types lives at `src/factor_mining/_qcm_parser.py` instead of being patched into the fork
