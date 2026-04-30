# Alpha Harness

## What Changed In This Branch / 这个分支相对原版改了什么

This branch extends the original crypto-oriented `alpha-harness` into a dual-use local research framework:
- keeps the original crypto workflow intact
- adds equity-style local panel research support for China A-shares
- removes the hard dependency on Qlib for equity experiments by supporting local panel data first
- adds public free-data connectors for A-share daily data
- adds local report generation with plots for factor evaluation
- upgrades the pipeline so AlphaGen, AlphaQCM, validation, combination, and backtest can all run on equity-style data

这个分支把原本偏向 crypto 的 `alpha-harness` 扩展成了一个双用途的本地研究框架：
- 保留原有 crypto 研究流程
- 新增适用于中国 A 股的 equity-style 本地 panel 研究能力
- 将 Qlib 从“必需依赖”改成“可选后端”，优先支持本地 panel 数据
- 新增公开免费 A 股日频数据接入脚本
- 新增自动画图的本地因子评估报告脚本
- 打通 AlphaGen、AlphaQCM、验证、组合、回测在 equity-style 数据上的整条链路

## Branch Changelog / 分支改动日志

### 1. Data Layer / 数据层
- Added `LocalPanelSource` to load repo-native panel data for equity research.
- 新增 `LocalPanelSource`，可直接加载仓库原生 panel 格式做股票研究。

- Refactored Qlib integration so local panel data can be used first and Qlib becomes an optional backend.
- 重构 Qlib 接入逻辑，使本地 panel 可优先使用，Qlib 变为可选后端。

- Added reusable equity panel builder utilities.
- 新增通用股票 panel 构建工具。

- Added equity data import script for long-form OHLCV files.
- 新增股票长表 OHLCV 导入脚本。

- Added free public-data download scripts for China A-share daily data via AkShare and BaoStock.
- 新增基于 AkShare 和 BaoStock 的中国 A 股免费日频数据下载脚本。

### 2. Alpha Search / 因子搜索
- Generalized AlphaGen data adapter from crypto-only to panel-based multi-asset usage.
- 将 AlphaGen 数据适配器从 crypto-only 泛化为基于 panel 的多资产适配层。

- Updated AlphaGen runner to read source-specific horizon and annualization settings from config.
- 更新 AlphaGen 启动逻辑，使其从配置读取不同数据源的目标周期和年化参数。

- Updated AlphaQCM runner to use the same panel-based equity/crypto configuration path.
- 更新 AlphaQCM 启动逻辑，使其共享同一套 crypto/equity panel 配置。

- Added a smoke config for faster AlphaQCM CPU validation.
- 新增 AlphaQCM smoke 配置，便于 CPU 上快速验证。

### 3. Evaluation / 评估层
- Added a local alpha report generator with IC, RankIC, decay, quantiles, scatter plots, correlation heatmap, and long-short backtest outputs.
- 新增本地 alpha 报告脚本，输出 IC、RankIC、衰减、分组收益、散点图、相关性热图和多空回测结果。

- Made backtest annualization configurable instead of hardcoding crypto frequency.
- 将回测年化参数改为可配置，不再写死为 crypto 频率。

- Improved liquidity field handling so equity panels can use `volume` or `turnover`-style proxies.
- 改进流动性字段适配，使股票 panel 可以使用 `volume` 或类似 `turnover` 的代理字段。

### 4. Pipeline / 流程编排
- Reworked `src/pipeline.py` into a real evaluation pipeline for expression pools on local panel data.
- 将 `src/pipeline.py` 重写为针对本地 panel 数据的真实表达式池评估入口。

- The pipeline now evaluates expressions, runs validation gates, combines factors, and performs long-short backtests on equity-style data.
- 现在的 pipeline 可以在 equity-style 数据上完成表达式评估、验证筛选、因子组合和多空回测。

### 5. Docs / 文档
- Rewrote README to reflect the current mixed crypto + equity research workflow.
- 重写 README，使其反映当前 crypto + equity 混合研究流程。

## Overview / 项目概述

`alpha-harness` is an LLM-guided alpha research framework. It combines:
- automated factor discovery with AlphaGen and AlphaQCM
- optional LLM-based interpretation and scoring
- formal validation gates
- factor combination and simple long-short backtesting

`alpha-harness` 是一个 LLM 引导的 alpha 研究框架，包含：
- 基于 AlphaGen 和 AlphaQCM 的自动因子发现
- 可选的 LLM 解释和打分
- 正式化验证门槛
- 因子组合与简化多空回测

This branch supports two main workflows:
- crypto intraday research using the original Binance pipeline
- equity-style daily research using local panel data or optional Qlib fallback

这个分支支持两条主要工作流：
- 使用原始 Binance 流程做 crypto 高频/日内研究
- 使用本地 panel 数据或可选 Qlib 回退做 equity-style 日频研究

## Architecture / 框架结构

```text
Data Sources
  |- Crypto: Binance raw -> cleaned panel
  |- Equity: local panel / BaoStock / AkShare / optional Qlib

Feature Expansion
  |- OHLCV -> 50+ engineered features

Factor Mining
  |- AlphaGen (PPO)
  |- AlphaQCM (distributional RL)

Judging and Validation
  |- optional LLM judge
  |- IC / RankIC / ICIR
  |- turnover / decay / correlation gates

Portfolio Construction
  |- equal weight
  |- IC-weighted
  |- ridge regression

Backtest
  |- top-N long / bottom-N short
```

## Setup / 环境准备

```bash
git clone https://github.com/Wrigggy/alpha-harness.git
cd alpha-harness

# create environment
python -m venv .venv

# install dependencies
python -m pip install -r requirements.txt

# external repos
git clone https://github.com/RL-MLDM/alphagen.git external/alphagen
git clone https://github.com/ZhuZhouFan/AlphaQCM.git external/alphaqcm
```

Notes:
- `pyqlib` is optional in this branch.
- For equity-style research, local panel data is the primary path.
- `AkShare` and `BaoStock` are included as free A-share daily data connectors.

说明：
- 这个分支里 `pyqlib` 是可选项。
- 对于 equity-style 研究，本地 panel 数据是主路径。
- 已加入 `AkShare` 和 `BaoStock` 两条免费 A 股日频数据接入方式。

## Main Configs / 主要配置

- `config/data_config.yaml`
  - original crypto-oriented config

- `config/data_config_equity_cn.yaml`
  - China equity-style local research config

- `config/alphagen_config.yaml`
  - AlphaGen hyperparameters

- `config/alphaqcm_config.yaml`
  - AlphaQCM default config

- `config/alphaqcm_config_smoke.yaml`
  - AlphaQCM quick smoke-test config

## Usage / 使用方式

### 1. Crypto Data Pipeline / Crypto 数据流程

```bash
python -m src.data_collection.universe_selector
python -m src.data_collection.binance_fetcher
python -m src.data_collection.data_cleaner
```

### 2. Import Local Equity Panel / 导入本地股票 Panel

Prepare a long-form file with columns:
`date, symbol, open, high, low, close, volume`

Optional columns:
`amount, turnover, vwap, market_cap, industry`

```bash
python scripts/import_equity_panel.py ^
  --input data/raw_equity/csi500_daily.parquet ^
  --output data/equity_panel/csi500_daily ^
  --market cn --universe csi500
```

### 3. Fetch Free A-share Daily Data / 拉取免费 A 股日频数据

Using BaoStock:

```bash
python scripts/fetch_equity_baostock.py ^
  --start 2024-01-01 ^
  --end 2024-06-30 ^
  --symbols 600519,000858,600036,601318,000333 ^
  --output data/equity_panel/cn_baostock5_daily
```

Using AkShare:

```bash
python scripts/fetch_equity_akshare.py ^
  --start 2024-01-01 ^
  --end 2024-03-31 ^
  --symbols 600519,000858,600036 ^
  --output data/equity_panel/cn_real3_daily
```

Note:
- In this environment, BaoStock is more stable than AkShare for repeated downloads.

说明：
- 在当前环境里，BaoStock 的稳定性高于 AkShare，更适合重复下载。

### 4. Run AlphaGen / 运行 AlphaGen

Crypto:

```bash
python -m src.factor_mining.run_alphagen --small-scale
```

Equity:

```bash
python -m src.factor_mining.run_alphagen --small-scale ^
  --data-config config/data_config_equity_cn.yaml
```

### 5. Run AlphaQCM / 运行 AlphaQCM

Equity smoke:

```bash
python -m src.factor_mining.run_alphaqcm --model iqn --small-scale ^
  --config config/alphaqcm_config_smoke.yaml ^
  --data-config config/data_config_equity_cn.yaml
```

### 6. Run Pipeline / 运行完整评估流程

```bash
python -m src.pipeline ^
  --evaluate-pool data/factors/alphagen_pool.json ^
  --data-config config/data_config_equity_cn.yaml ^
  --split test ^
  --combiner ic_weighted ^
  --output-dir out/pipeline_results/equity_smoke
```

### 7. Generate Local Alpha Report / 生成本地图表报告

```bash
python scripts/generate_alpha_report.py ^
  --pool data/factors/alphagen_pool.json ^
  --data-config config/data_config_equity_cn.yaml ^
  --split test ^
  --output-dir out/reports/equity_smoke
```

## Important New Files / 关键新增文件

### Data source and adapter
- `src/data_sources/local_panel_source.py`
- `src/data_sources/qlib_source.py`
- `src/data_adapter/to_alphagen_format.py`

### Equity data tooling
- `src/data_collection/equity_panel_builder.py`
- `scripts/import_equity_panel.py`
- `scripts/fetch_equity_akshare.py`
- `scripts/fetch_equity_baostock.py`

### Configs
- `config/data_config_equity_cn.yaml`
- `config/alphaqcm_config_smoke.yaml`

### Reporting
- `scripts/generate_alpha_report.py`

## Current Status / 当前状态

What is already verified in this branch:
- crypto AlphaGen path still works
- local equity panel import works
- AlphaGen can run on equity-style local panel data
- AlphaQCM can run on equity-style local panel data
- the pipeline can evaluate an expression pool on equity-style data
- the local report script can generate plots and summary files
- BaoStock can fetch real A-share daily data and convert it into local panel format

当前这个分支已经验证过：
- crypto 的 AlphaGen 路径仍然可用
- 本地股票 panel 导入可用
- AlphaGen 可以在 equity-style 本地 panel 上运行
- AlphaQCM 可以在 equity-style 本地 panel 上运行
- pipeline 可以在 equity-style 数据上评估表达式池
- 本地报告脚本可以生成图表和汇总文件
- BaoStock 可以拉取真实 A 股日频并转成本地 panel

## Limitations / 当前限制

- Qlib on Windows is still optional and not the primary path in this branch.
- Strict historical CSI500 constituent tracking is not yet fully built in.
- Free public sources are good enough for local research bootstrapping, but not a perfect substitute for a fully curated institutional equity dataset.

限制项：
- Windows 上的 Qlib 仍是可选项，不是这个分支的主路径。
- 严格意义上的 CSI500 历史成分跟踪还没有完整建好。
- 免费公开源足够做本地研究启动，但不能完全替代机构级整理好的股票数据集。
