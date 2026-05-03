# Alpha Harness

## Branch Update Log / 分支更新日志

### Compared with the original project
This branch turns the original crypto-first `alpha-harness` into a lighter but more complete local research repo for both crypto and equities.

Key changes:
- keeps the original crypto workflow
- adds local equity panel research support
- adds China A-share free-data pipelines
- adds US equity free-data pipelines
- adds local WorldQuant-style single-alpha proxy evaluation
- adds expression translation and submission-ready export scripts
- adds team governance and alpha registry tooling
- removes bundled datasets and generated experiment outputs from Git

### 相比原版项目
这个分支把原本偏 `crypto` 的 `alpha-harness` 扩展成一个更轻量、但研究链路更完整的本地量化研究仓库，同时保留原有加密资产流程。

主要变化：
- 保留原始 crypto workflow
- 新增本地股票 panel 研究能力
- 新增中国 A 股免费数据抓取与建库脚本
- 新增美股免费数据抓取与建库脚本
- 新增本地 WorldQuant 风格单因子代理评估
- 新增表达式翻译与可提交导出脚本
- 新增团队治理与 alpha 注册表工具
- Git 中移除数据文件和实验产物，只保留可复现 pipeline

## Overview / 项目概览

`alpha-harness` is an LLM-assisted alpha research framework for:
- factor generation with AlphaGen and AlphaQCM
- local factor filtering and validation
- portfolio combination and long-short backtesting
- local WorldQuant-style proxy evaluation
- expression translation and export

`alpha-harness` 是一个由 LLM 辅助的因子研究框架，支持：
- 使用 AlphaGen 和 AlphaQCM 生成因子
- 本地因子筛选与验证
- 因子组合与多空回测
- 本地 WorldQuant 风格代理评估
- 因子表达式翻译与导出

## What Is Kept In This Slim Branch / 这个精简分支保留了什么

- full source code under `src/`
- runnable scripts under `scripts/`
- configs for crypto, China equity, and US equity workflows
- data ingestion and panel-building pipelines
- local proxy evaluation and reporting pipeline

保留内容：
- `src/` 下完整源码
- `scripts/` 下可运行脚本
- crypto、中国股票、美股三类配置
- 数据抓取、导入、panel 构建 pipeline
- 本地代理评估与报告生成 pipeline

## What Is Removed From Git / 从 Git 中移除的内容

- raw data files
- processed panel data
- local caches
- training checkpoints
- generated reports and plots
- temporary outputs under `out/`

移除内容：
- 原始数据文件
- 处理后的 panel 数据
- 本地缓存
- 训练 checkpoint
- 自动生成的报告和图表
- `out/` 下的运行结果

## Repository Layout / 仓库结构

```text
config/      configs for research and evaluation
docs/        lightweight docs and notes
external/    local external dependencies cloned by user
prompts/     LLM prompts
scripts/     data, workflow, export, and evaluation scripts
src/         core library code
data/        empty in Git, rebuilt locally
out/         empty in Git, generated locally
```

## Setup / 环境准备

```bash
git clone https://github.com/Wrigggy/alpha-harness.git
cd alpha-harness

python -m venv .venv
python -m pip install -r requirements.txt
```

Clone external dependencies locally:

```bash
git clone https://github.com/RL-MLDM/alphagen.git external/alphagen
git clone https://github.com/ZhuZhouFan/AlphaQCM.git external/alphaqcm
```

本地再额外克隆外部依赖：

```bash
git clone https://github.com/RL-MLDM/alphagen.git external/alphagen
git clone https://github.com/ZhuZhouFan/AlphaQCM.git external/alphaqcm
```

## Data Pipelines / 数据流水线

### China A-share / 中国 A 股

Use the included scripts to fetch constituents and daily bars, then build a local panel:

```bash
python scripts/fetch_csi500_constituents.py
python scripts/fetch_equity_baostock.py --help
python scripts/fetch_equity_akshare.py --help
python scripts/build_csi500_baostock_panel.py --help
```

### US Equity / 美股

Build a local US panel with free sources:

```bash
python scripts/fetch_us_symbol_universe.py --help
python scripts/fetch_equity_us_yfinance.py --help
python scripts/import_equity_panel.py --help
python scripts/build_us_universe_masks.py --help
```

## Core Workflows / 核心流程

### 1. Run AlphaGen

```bash
python -m src.factor_mining.run_alphagen --small-scale
python -m src.factor_mining.run_alphagen --small-scale --data-config config/data_config_equity_cn.yaml
```

### 2. Run AlphaQCM

```bash
python -m src.factor_mining.run_alphaqcm --model iqn --small-scale --config config/alphaqcm_config_smoke.yaml --data-config config/data_config_equity_cn.yaml
```

### 3. Evaluate A Pool

```bash
python -m src.pipeline --evaluate-pool data/factors/alphagen_pool.json --data-config config/data_config_equity_cn.yaml --split test --combiner ic_weighted --output-dir out/pipeline_results/equity_smoke
```

### 4. Generate Local Reports

```bash
python scripts/generate_alpha_report.py --pool data/factors/alphagen_pool.json --data-config config/data_config_equity_cn.yaml --split test --output-dir out/reports/equity_smoke
```

### 5. WorldQuant-Style Proxy Workflow

```bash
python scripts/evaluate_brain_candidates.py --pool data/factors/alphagen_pool.json --data-config config/data_config_equity_brain_top50.yaml --brain-config config/brain_proxy_equity_cn.yaml --split test --output-dir out/brain_proxy/stage1_test
python scripts/export_accepted_worldquant.py --input-csv out/brain_proxy/stage1_test/brain_candidate_metrics.csv --output-dir out/worldquant_submit_ready
```

## LLM Judge / LLM 评审

This branch supports:
- DeepSeek API
- OpenAI / Codex-compatible API
- Anthropic API

Configure them in:
- `config/judge_config.yaml`
- local `.env`

这个分支支持：
- DeepSeek API
- OpenAI / Codex 兼容接口
- Anthropic API

配置入口：
- `config/judge_config.yaml`
- 本地 `.env`

## Important Notes / 重要说明

- This repository does not include datasets.
- All data must be rebuilt locally through the provided pipelines.
- `external/` dependencies are expected to be cloned locally.
- Local WorldQuant-style metrics are only proxy metrics, not platform-equivalent results.

- 本仓库不包含数据集。
- 所有数据需要通过脚本在本地重建。
- `external/` 依赖需要你本地自行克隆。
- 本地 WorldQuant 风格指标只是代理指标，不等同于平台真实结果。
