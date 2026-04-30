# Alpha Report

- Split: `test`
- IC Horizon: `8` bars
- Factors: `7`

## Combined Signal

- RankIC mean: `0.065582`
- IC mean: `0.008249`
- RankICIR: `0.279633`
- ICIR: `0.040382`
- Backtest annual return: `0.155052`
- Backtest Sharpe: `0.683475`
- Backtest max drawdown: `0.079341`

## Per-Factor Metrics

| Factor | IC | RankIC | ICIR | RankICIR | Weight |
|---|---:|---:|---:|---:|---:|
| f01_Less_Sub_2_0_high_10_0 | -0.006039 | 0.025716 | -0.021775 | 0.086243 | -0.001423 |
| f02_Div_Less_2_0_Greater_vwap_Add_10_0_V | 0.116601 | 0.033246 | 0.386814 | 0.111000 | -0.037618 |
| f03_Max_Sub_1_0_Add_Ref_Std_Mul_vwap_0_5 | -0.040404 | 0.019281 | -0.160905 | 0.076710 | 0.056671 |
| f04_Sub_Abs_Greater_0_5_Sub_vwap_0_01_Ad | -0.007434 | 0.038155 | -0.027042 | 0.125886 | -0.237981 |
| f05_Delta_Sub_1_0_open_40d | 0.050391 | 0.140451 | 0.213582 | 0.554597 | 0.244387 |
| f06_Div_Greater_0_5_low_vwap | -0.051789 | 0.020445 | -0.173077 | 0.080396 | 0.076638 |
| f07_Div_Min_high_40d_1_0 | 0.006151 | 0.016420 | 0.022250 | 0.057784 | 0.345283 |