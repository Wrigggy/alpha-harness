"""Shared factory for train/val/test calculators.

Centralizes the crypto vs A-share branch so idea_agent.py, apply_judge_filter.py,
and any future caller stay in sync.

Calculators returned implement the AlphaCalculator interface and are compatible
with AlphaGen's pool / env code paths:
- crypto : src.data_adapter.to_alphagen_format.CryptoAlphaCalculator
- cn     : external/alphagen/alphagen_qlib/calculator.QLibStockDataCalculator

The target horizon and form are chosen per-source:
- crypto : 8-bar forward close-to-close return (matches existing pipeline)
- cn     : 10-day VWAP-to-VWAP forward return (AlphaGen-paper-aligned)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.data.expression import Expression, Feature, Ref  # noqa: E402

DataSource = Literal["crypto", "cn"]
Split = Literal["train", "val", "test"]


def _crypto_target() -> Expression:
    from alphagen.data.expression import FeatureType
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -8) / close - 1


def _cn_target() -> Expression:
    from alphagen_qlib.stock_data import FeatureType
    vwap = Feature(FeatureType.VWAP)
    return Ref(vwap, -11) / Ref(vwap, -1) - 1


def build_calculators(
    data_source: DataSource,
    data_config_path: str = "config/data_config.yaml",
    splits_to_load: tuple[Split, ...] = ("train",),
    device: torch.device | None = None,
):
    """Return a dict {split: calculator} for the requested splits.

    The calculator interface guarantees:
        calc.calc_single_IC_ret(expr) -> float
        calc.evaluate_alpha(expr)     -> torch.Tensor
        calc.target                   -> torch.Tensor
        calc.n_days                   -> int
        pool.test_ensemble(calc)      -> (ic, rank_ic)
    """
    from src.utils.device import get_device
    if device is None:
        device = get_device("auto")

    if data_source == "crypto":
        from src.data_adapter.to_alphagen_format import (
            CryptoAlphaCalculator, create_data_splits,
        )
        with open(data_config_path, encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        processed_dir = data_cfg["data"]["processed_dir"]
        splits = create_data_splits(
            processed_dir, data_config_path, device=device,
            max_backtrack_days=100, max_future_days=10,
        )
        target = _crypto_target()
        return {s: CryptoAlphaCalculator(splits[s], target) for s in splits_to_load}

    elif data_source == "cn":
        from alphagen_qlib.stock_data import StockData, initialize_qlib
        from alphagen_qlib.calculator import QLibStockDataCalculator
        with open(data_config_path, encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        cn_cfg = data_cfg.get("cn", {})
        qlib_dir = cn_cfg.get("qlib_data_path", "~/.qlib/qlib_data/cn_data")
        initialize_qlib(qlib_dir)
        instruments = cn_cfg.get("instruments", "csi300")
        segments = cn_cfg.get("segments", {
            "train": ["2015-01-01", "2021-12-31"],
            "val":   ["2022-01-01", "2022-12-31"],
            "test":  ["2023-01-01", "2023-12-31"],
        })
        target = _cn_target()
        result = {}
        for s in splits_to_load:
            start, end = segments[s]
            sd = StockData(
                instrument=instruments,
                start_time=start,
                end_time=end,
                max_backtrack_days=100,
                max_future_days=30,
                device=device,
            )
            result[s] = QLibStockDataCalculator(sd, target)
        return result

    else:
        raise ValueError(f"Unknown data_source: {data_source}")
