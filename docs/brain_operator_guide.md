# BRAIN Operator Guide

This document organizes the currently visible WorldQuant BRAIN operators into a practical reference.

It is written for the workflow:
- inspect fields in BRAIN `Data`
- express an alpha using BRAIN operators
- simulate it under a chosen setting
- iterate until it is strong enough to consider submission

Important:
- Operator availability can depend on your account level.
- Some operators shown in local research code may not exist in BRAIN under the same name.
- This guide focuses on intuition and usage, not formal platform syntax edge cases.

## How To Think About BRAIN Formulas

A BRAIN alpha is usually built from three layers:

1. `Raw field`
Examples:
- `close`
- `volume`
- `operating_income`
- `market_cap`

2. `Transform`
Examples:
- compare a stock with its own history: `ts_rank`, `ts_zscore`
- compare a stock with other stocks today: `rank`, `zscore`
- compare a stock only with peers: `group_rank`, `group_neutralize`

3. `Simulation setting`
Examples:
- delay
- region
- universe
- neutralization
- decay
- truncation

Very often, a useful first alpha looks like:

```text
group_neutralize(rank(ts_zscore(some_field, 252)), subindustry)
```

This means:
- detect whether a field is unusual relative to the stock's own history
- rank stocks cross-sectionally
- remove group bias

## Category Map

- `Arithmetic`: basic math building blocks
- `Logical`: conditions and branch logic
- `Time Series`: compare a stock with its own past
- `Cross Sectional`: compare stocks with each other on the same day
- `Vector`: reduce vector-type fields to scalar values
- `Transformational`: bucketing and trade control
- `Group`: compare or neutralize within sector/industry/custom groups

---

## Arithmetic Operators

These are the base math operators. They are used to build the raw shape of an alpha.

### `abs(x)`
- Meaning: absolute value
- Use when:
  - you only care about magnitude, not sign
  - you want to stabilize ratios or deviations
- Example:

```text
abs(ts_delta(close, 5))
```

### `add(x, y, filter=false)` or `x + y`
- Meaning: element-wise addition
- Use when:
  - combining two signals
  - adding a stabilizer constant
- `filter=true` treats NaN as 0 before summing

### `densify(x)`
- Meaning: compresses a sparse grouping field into only the buckets that actually appear
- Use when:
  - working with group labels
  - you want efficient grouping behavior for custom buckets
- Most useful together with group operators

### `divide(x, y)` or `x / y`
- Meaning: ratio
- Use when:
  - converting level into intensity
  - building valuation, efficiency, or normalized signals
- Be careful around zero denominators

### `inverse(x)`
- Meaning: `1 / x`
- Use when:
  - turning a large value into a small score
  - expressing inverse relations

### `log(x)`
- Meaning: natural log
- Use when:
  - compressing large positive ranges
  - reducing skew
  - making multiplicative growth look more linear
- Only safe for positive inputs

### `max(x, y, ...)`
- Meaning: maximum of inputs
- Use when:
  - clipping floors
  - picking stronger of multiple signals

### `min(x, y, ...)`
- Meaning: minimum of inputs
- Use when:
  - clipping ceilings
  - conservative combination logic

### `multiply(x, y, ..., filter=false)` or `x * y`
- Meaning: element-wise multiplication
- Use when:
  - interaction effects
  - gating one signal by another
- Example:

```text
multiply(rank(close), rank(volume))
```

### `power(x, y)`
- Meaning: `x ^ y`
- Use when:
  - nonlinear amplification
  - convex or concave transforms

### `reverse(x)`
- Meaning: `-x`
- Use when:
  - flipping long/short direction
  - turning “high is bad” into “high is good”

### `sign(x)`
- Meaning: returns `+1`, `0`, `-1`
- Use when:
  - only directional information matters
  - discretizing a signal

### `signed_power(x, y)`
- Meaning: power transform while preserving the sign of `x`
- Use when:
  - you want nonlinear scaling without losing long/short direction

### `sqrt(x)`
- Meaning: square root
- Use when:
  - soft-compressing large values
  - variance-like scaling

### `subtract(x, y, filter=false)` or `x - y`
- Meaning: difference
- Use when:
  - spreads
  - deviations
  - relative comparisons

---

## Logical Operators

These are used for conditions, gating, and piecewise rules.

### `and(input1, input2)`
- Meaning: true only if both conditions are true
- Use when:
  - combining filters
  - strict trade conditions

### `if_else(input1, input2, input3)`
- Meaning: if condition is true, return `input2`; else `input3`
- Use when:
  - asymmetric logic
  - switching formula behavior by regime
- Example:

```text
if_else(volume > ts_mean(volume, 20), rank(close), reverse(rank(close)))
```

### Comparison operators
- `input1 < input2`
- `input1 <= input2`
- `input1 == input2`
- `input1 > input2`
- `input1 >= input2`
- `input1 != input2`

Use when:
- defining thresholds
- selecting states
- building trade conditions

### `is_nan(input)`
- Meaning: returns 1 if input is NaN
- Use when:
  - missing-value diagnostics
  - conditional backfilling

### `not(x)`
- Meaning: logical negation

### `or(input1, input2)`
- Meaning: true if at least one condition is true

---

## Time Series Operators

These compare a stock to its own past. This is one of the most important categories in BRAIN.

### `days_from_last_change(x)`
- Meaning: how many days since the value last changed
- Use when:
  - stale fundamentals
  - event recency

### `hump(x, hump=0.01)`
- Meaning: limits the magnitude of day-to-day changes in the signal
- Use when:
  - lowering turnover
  - smoothing unstable alphas
- This is often useful late in refinement, not necessarily in the first prototype

### `kth_element(x, d, k, ignore="NaN")`
- Meaning: returns the K-th value in a lookback window
- Use when:
  - backfill-like logic
  - robust history extraction

### `last_diff_value(x, d)`
- Meaning: most recent different value from the last `d` days
- Use when:
  - detecting actual change rather than repeated stale value

### `ts_arg_max(x, d)`
- Meaning: number of days since the maximum occurred
- Use when:
  - recency of peak
  - “how recently did this stock make a high?”

### `ts_arg_min(x, d)`
- Meaning: number of days since the minimum occurred
- Use when:
  - recency of trough

### `ts_av_diff(x, d)`
- Meaning: `x - ts_mean(x, d)` with NaNs ignored
- Use when:
  - simple mean-reversion logic
  - deviation from average

### `ts_backfill(x, lookback=d, k=1)`
- Meaning: fill missing values from recent valid history
- Use when:
  - sparse fundamental fields
  - coverage improvement

### `ts_corr(x, y, d)`
- Meaning: rolling correlation
- Use when:
  - price-volume relation
  - factor interaction over time
- Example:

```text
ts_corr(rank(close), rank(volume), 20)
```

### `ts_count_nans(x, d)`
- Meaning: number of NaNs in the last `d` days
- Use when:
  - data quality filters

### `ts_covariance(y, x, d)`
- Meaning: rolling covariance
- Use when:
  - joint movement magnitude, not just normalized relation

### `ts_decay_linear(x, d, dense=false)`
- Meaning: weighted rolling average with larger weight on recent data
- Use when:
  - smoothing
  - emphasizing fresh information
- Often a good replacement for more aggressive raw signals

### `ts_delay(x, d)`
- Meaning: value from `d` days ago
- Use when:
  - historical comparison
  - avoiding look-ahead

### `ts_delta(x, d)`
- Meaning: `x - ts_delay(x, d)`
- Use when:
  - momentum
  - acceleration
  - change detection

### `ts_mean(x, d)`
- Meaning: rolling mean
- Use when:
  - smoothing
  - baseline estimation

### `ts_product(x, d)`
- Meaning: product over lookback window
- Use when:
  - compounded growth style logic

### `ts_quantile(x, d, driver="gaussian")`
- Meaning: rolling rank transformed through a distribution
- Use when:
  - reshaping the signal distribution
  - robust normalization

### `ts_rank(x, d, constant=0)`
- Meaning: current value ranked against the stock's own last `d` days
- Use when:
  - identifying historical highs/lows
  - detecting whether today is unusually large/small
- This is one of the best first operators to learn
- Example:

```text
ts_rank(operating_income, 252)
```

### `ts_regression(y, x, d, lag=0, rettype=0)`
- Meaning: rolling regression outputs
- Use when:
  - extracting slope, fit, residual-style relationships
  - trend and sensitivity analysis
- More advanced and easy to misuse without strong intuition

### `ts_scale(x, d, constant=0)`
- Meaning: scales the series to a 0–1 range over the lookback
- Use when:
  - bounded normalization

### `ts_std_dev(x, d)`
- Meaning: rolling standard deviation
- Use when:
  - volatility
  - stability
  - denominator in standardized transforms

### `ts_step(1)`
- Meaning: time counter increasing by 1 each day
- Use when:
  - date-like sequencing logic
- Less common for standard stock alphas

### `ts_sum(x, d)`
- Meaning: rolling sum
- Use when:
  - aggregating recent activity

### `ts_zscore(x, d)`
- Meaning: `(x - recent_mean) / recent_std`
- Use when:
  - standardized deviation from normal behavior
  - comparing fields with different scales
- This is another top-priority operator to learn

---

## Cross Sectional Operators

These compare all stocks with each other on the same date.

### `normalize(x, useStd=false, limit=0.0)`
- Meaning: subtract cross-sectional mean; optionally divide by cross-sectional std
- Use when:
  - removing market-wide level effects
  - standardizing a daily cross section

### `quantile(x, driver=gaussian, sigma=1.0)`
- Meaning: rank values, then reshape the cross section using a distribution
- Use when:
  - controlling outliers
  - distributing signal mass more smoothly

### `rank(x, rate=2)`
- Meaning: rank stocks cross-sectionally between 0 and 1
- Use when:
  - building robust cross-sectional alphas
  - reducing outlier influence
- This is one of the most common BRAIN operators

### `scale(x, scale=1, longscale=1, shortscale=1)`
- Meaning: rescales the signal so the daily absolute exposure sums to the chosen size
- Use when:
  - explicit portfolio-style normalization
  - controlling long and short book separately

### `winsorize(x, std=4)`
- Meaning: clip extreme values by cross-sectional standard deviation thresholds
- Use when:
  - reducing outlier impact
  - stabilizing noisy fields

### `zscore(x)`
- Meaning: daily cross-sectional z-score
- Use when:
  - standardizing today's stock values relative to all others

---

## Vector Operators

These are used when a field itself is vector-valued rather than scalar.

### `vec_avg(x)`
- Meaning: average across vector elements
- Use when:
  - collapsing vector fields into a scalar

### `vec_sum(x)`
- Meaning: sum across vector elements
- Use when:
  - total effect from a vector field

If you are mostly working with standard price/fundamental matrix fields, you may not need vector operators often.

---

## Transformational Operators

These help shape how a signal is grouped or traded.

### `bucket(...)`
- Meaning: divide values into custom buckets
- Typical usage:

```text
bucket(rank(x), range="0, 1, 0.1")
```

- Use when:
  - creating custom quantile groups
  - feeding group operators with custom buckets

### `trade_when(x, y, z)`
- Meaning:
  - change alpha to `y` only when condition `x` is true
  - otherwise keep previous value
  - optionally exit when `z` is true
- Use when:
  - lowering turnover
  - making a signal event-driven
  - only entering when a regime is active

This is one of the most useful “advanced but practical” operators.

---

## Group Operators

These compare stocks only within peer groups such as sector, industry, subindustry, country, or custom buckets.

Group operators are critical because many raw signals are really just sector bets unless you neutralize them.

### `group_backfill(x, group, d, std=4.0)`
- Meaning: fill NaNs using winsorized group history
- Use when:
  - sparse data within industries/groups

### `group_mean(x, weight, group)`
- Meaning: group-level harmonic mean
- Use when:
  - building a group reference point
- More specialized than `group_rank` or `group_neutralize`

### `group_neutralize(x, group)`
- Meaning: subtract the group mean from each stock
- Use when:
  - removing industry/sector bias
  - isolating idiosyncratic effects
- This is one of the most important operators in equity alpha design

### `group_rank(x, group)`
- Meaning: rank stocks only within their own group
- Use when:
  - peer-relative comparison
  - comparing firms inside the same subindustry
- Often safer than raw `rank(x)` for equity fundamentals

### `group_scale(x, group)`
- Meaning: normalize values within each group to 0–1
- Use when:
  - group-relative scaling

### `group_zscore(x, group)`
- Meaning: z-score within each group
- Use when:
  - standardized peer-relative deviation

---

## The Most Useful Starter Operators

If you are new to BRAIN expression writing, learn these first:

- `ts_rank`
- `ts_zscore`
- `ts_delta`
- `rank`
- `group_rank`
- `group_neutralize`
- `ts_decay_linear`
- `winsorize`
- `trade_when`

With these, you can already build many good first-pass alphas.

---

## Common Alpha Design Patterns

### 1. Time-series anomaly

```text
ts_zscore(x, 252)
```

Meaning:
- is today's value unusual relative to this stock's own 1-year history?

### 2. Cross-sectional ranking

```text
rank(x)
```

Meaning:
- which stocks are high or low versus all others today?

### 3. Industry-relative ranking

```text
group_rank(x, subindustry)
```

Meaning:
- which stocks are high or low versus peers in the same subindustry?

### 4. Industry-neutralized anomaly

```text
group_neutralize(ts_zscore(x, 252), subindustry)
```

Meaning:
- unusual relative to own history, then remove group bias

### 5. Smoothed signal

```text
ts_decay_linear(rank(x), 10)
```

Meaning:
- cross-sectional signal with recent-history smoothing

### 6. Event-gated trading

```text
trade_when(volume > ts_mean(volume, 20), rank(x), abs(ts_delta(volume, 5)) < 0.01)
```

Meaning:
- only actively update the alpha when the volume condition is met

---

## Mapping From Local Research Expressions To BRAIN Style

Your local research framework may use forms like:

```text
Mul(...)
Add(...)
Div(...)
Mean(...)
Cov(...)
EMA(...)
```

Typical BRAIN-style mapping is:

- `Mul(a,b)` -> `multiply(a,b)`
- `Add(a,b)` -> `add(a,b)`
- `Div(a,b)` -> `divide(a,b)`
- `Sub(a,b)` -> `subtract(a,b)`
- `Log(x)` -> `log(x)`
- `Mean(x,20d)` -> `ts_mean(x,20)`
- `Cov(x,y,20d)` -> `ts_covariance(x,y,20)`

Important:
- Some local operators may not exist in BRAIN under the same name
- Some local operators may require a structural rewrite in BRAIN
- Account level can restrict which operators are available

---

## Practical Writing Advice

### Start simple
Use a minimal working formula first:

```text
group_rank(ts_zscore(close, 252), subindustry)
```

Do not start with a giant nested expression unless you already understand each piece.

### Prefer stable transforms
Often safer:
- `rank`
- `ts_rank`
- `ts_zscore`
- `group_rank`

Often riskier if used carelessly:
- deep nested arithmetic
- too many conditionals
- raw ratios with unstable denominators

### Use group logic for equities
Fundamental or price-level alphas often need:
- `group_rank`
- `group_neutralize`
- `group_zscore`

Otherwise you may just be rediscovering sector exposure.

### Use smoothing when turnover is too high
Try:
- `ts_decay_linear`
- `hump`
- `trade_when`

### Control outliers
Try:
- `winsorize`
- `rank`
- `zscore`
- `quantile`

---

## A Good First Workflow

1. Pick one field from BRAIN `Data`
2. Write a simple time-series transform
3. Add a cross-sectional rank
4. Add group logic if needed
5. Simulate
6. If turnover is too high, smooth it
7. If results are too group-driven, neutralize more carefully

Example progression:

```text
operating_income
ts_rank(operating_income, 252)
rank(ts_rank(operating_income, 252))
group_rank(ts_rank(operating_income, 252), subindustry)
ts_decay_linear(group_rank(ts_rank(operating_income, 252), subindustry), 10)
```

---

## Final Notes

- `Time Series` answers: how unusual is this stock versus its own past?
- `Cross Sectional` answers: how extreme is this stock versus all others today?
- `Group` answers: how extreme is this stock versus peers?
- `Transformational` answers: when and how should this alpha trade?

If you already have local formulas from AlphaGen or AlphaQCM, the best next step is usually:
- identify the economic intuition
- rebuild the expression using BRAIN-native operators
- keep the first version short and interpretable

