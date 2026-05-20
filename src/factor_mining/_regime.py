"""Compute a one-paragraph market-regime summary from a calculator.

Used to condition the LLM's compose-mode warm-start prompt. Three numbers:
- realized_vol : annualized stdev of cross-sectional-mean daily log return
- median_drift : median per-stock total return over the lookback window
- dispersion   : average daily cross-sectional std of log returns

Reads the raw `(total_bars, n_features, n_stocks)` data tensor on the calculator
(NOT evaluate_alpha — that runs normalize_by_day and z-scores the values).

Loud, simple signals — enough to bias the LLM toward mean-reversion vs momentum
compositions without overspecifying.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _extract_raw_close(calc) -> Tuple[torch.Tensor, str]:
    """Pull raw close prices from the calculator's StockData wrapper.

    Both CryptoStockData and QlibStockData store `data` as
    (total_bars, n_features, n_stocks) where CLOSE is at index 1
    in the FeatureType enum (OPEN, CLOSE, HIGH, LOW, VOLUME, VWAP).
    We also trim the trailing future-buffer so we look at real-history rows.
    """
    sd = calc.data
    raw = sd.data  # (total_bars, n_features, n_stocks)
    if raw.dim() != 3:
        raise RuntimeError(f"Unexpected raw data shape: {tuple(raw.shape)}")
    future = getattr(sd, "max_future_days", 0) or 0
    if future > 0:
        raw = raw[:-future]  # drop forward-fill buffer
    close = raw[:, 1, :].detach().cpu()  # CLOSE index = 1
    return close, "CLOSE@idx1"


def compute_regime_summary(calc, lookback_days: int = 60) -> str:
    try:
        close, _ = _extract_raw_close(calc)
    except Exception as e:
        return f"Regime: failed to extract raw close prices ({e})."

    if close.dim() != 2 or close.shape[0] < 2:
        return "Regime: insufficient data to summarize."

    n = min(lookback_days, close.shape[0])
    tail = close[-n:]

    valid = torch.isfinite(tail) & (tail > 0)
    safe = torch.where(valid, tail, torch.full_like(tail, float("nan")))

    log_p = torch.log(torch.clamp(safe, min=1e-12))
    log_ret = log_p[1:] - log_p[:-1]
    log_ret = torch.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

    # Cross-sectional mean log return per day, then annualize its std
    daily_mean_ret = log_ret.mean(dim=1)
    realized_vol = float(daily_mean_ret.std().item()) * (252 ** 0.5)

    # Per-stock total return across the window, then take median over stocks
    first = safe[0]
    last = safe[-1]
    ratio = last / torch.clamp(first, min=1e-12)
    per_stock_ret = ratio - 1.0
    finite = per_stock_ret[torch.isfinite(per_stock_ret)]
    median_drift = float(finite.median().item()) if finite.numel() else 0.0

    cs_disp = float(log_ret.std(dim=1).mean().item())

    if realized_vol > 0.30:
        flavor = "elevated vol — mean-reversion regimes typically dominate"
    elif realized_vol < 0.15:
        flavor = "low vol — momentum / trend-continuation regimes typically dominate"
    else:
        flavor = "moderate vol — neither reversion nor momentum is heavily favored"

    drift_dir = (
        "upward drift" if median_drift > 0.02
        else "downward drift" if median_drift < -0.02
        else "flat drift"
    )

    return (
        f"Recent {n}-bar market regime (computed from train tail): "
        f"annualized realized vol of cross-section mean return ≈ {realized_vol:.1%}; "
        f"median per-stock total return ≈ {median_drift:+.1%} ({drift_dir}); "
        f"daily cross-sectional dispersion ≈ {cs_disp:.4f}. "
        f"Interpretation: {flavor}."
    )
