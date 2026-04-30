# AlphaGen Top50 因子结果包

本目录保存了 2026-05-01 针对 `CSI500 Top50` 日频样本运行 `AlphaGen` 后，当前最值得保留到分支上的结果包。

## 结论

- 当前最佳展示结果是未筛选的 `5k / seed7` 因子池。
- `hard filter` 和 `soft filter` 都没有在完整报告口径下优于基线。
- 因此本次建议保留基线池作为当前阶段的主结果，对筛选版仅做对照留档。

基线核心指标：

- `RankIC mean = 0.065582`
- `RankICIR = 0.279633`
- `Annual Return = 15.51%`
- `Sharpe = 0.6835`
- `Max Drawdown = 7.93%`

## 文件说明

- `alphagen_top50_5k_seed7.json`: AlphaGen 原始因子池及权重。
- `report/summary.md`: 英文版摘要，由报告脚本自动生成。
- `report/summary.json`: 机器可读摘要。
- `report/combined_backtest.png`: 组合回测权益曲线。
- `report/combined_cum_ic.png`: 组合累计 IC 曲线。
- `report/combined_decay.png`: 组合 IC 衰减图。
- `report/combined_quantiles.png`: 分层收益图。
- `report/factor_correlation.png`: 因子相关性热力图。
- `report/factor_metrics.csv`: 单因子指标明细。

## 因子列表与结构化描述

以下描述是对表达式结构的近似翻译，目的是帮助阅读，不代表已经验证过明确的经济学含义。

1. `Less(Sub(2.0,$high),-10.0)`
   近似含义：对当日最高价做线性平移后再截断，属于非常原始的价格水平裁剪表达式。

2. `Div(Less(2.0,Greater($vwap,Add(-10.0,Var($open,5d)))),Mul(0.01,$vwap))`
   近似含义：将 `VWAP` 与 `开盘价 5 日方差减 10` 做比较，再按 `0.01 * VWAP` 归一化；更像“波动约束下的价格尺度信号”。

3. `Max(Sub(1.0,Add(Ref(Std(Mul($vwap,0.5),5d),20d),-2.0)),20d)`
   近似含义：使用 20 日前的 `VWAP` 相关 5 日波动，再做平移和 20 日窗口最大值，属于“滞后波动强度”类表达式。

4. `Sub(Abs(Greater(-0.5,Sub($vwap,0.01))),Add($high,5.0))`
   近似含义：把 `VWAP` 与常数比较后的幅度特征减去 `最高价 + 5`，是一个强非线性的价格偏离表达式。

5. `Delta(Sub(-1.0,$open),40d)`
   近似含义：观察 `-1 - 开盘价` 在 40 日尺度上的变化，本质上是长窗口价格动量/反转变体。

6. `Div(Greater(0.5,$low),$vwap)`
   近似含义：比较 `0.5` 和最低价后，再除以 `VWAP`，可视作“低价位置相对 VWAP 的缩放信号”。

7. `Div(Min($high,40d),-1.0)`
   近似含义：取 40 日窗口最高价相关统计后取负，属于带负号的长窗口价格极值信号。

## 推荐查看顺序

1. 先看 `report/summary.md`
2. 再看 `report/combined_backtest.png`
3. 然后看 `report/combined_cum_ic.png`
4. 最后看 `report/factor_correlation.png`

## 对照结果

筛选版结果放在相邻目录 `../2026-05-01_top50_seed7_filters/`：

- `alphagen_top50_5k_seed7_filtered.json`
- `alphagen_top50_5k_seed7_soft_filtered.json`
- `filtered_summary.json`
- `soft_filtered_summary.json`

其中：

- `hard filter` 完整报告口径下年化收益约 `-50.85%`
- `soft filter` 完整报告口径下年化收益约 `-56.14%`

因此这两版都不作为当前主展示结果。

## 复现入口

建议统一使用项目内解释器入口：

```powershell
.\py312.cmd -m src.pipeline --evaluate-pool data/factors/alphagen_top50_5k_seed7.json --data-config config/data_config_equity_top50_baostock_2023_2024.yaml --split test --combiner ic_weighted --output-dir out/pipeline_results/equity_top50_seed7_5k
```
