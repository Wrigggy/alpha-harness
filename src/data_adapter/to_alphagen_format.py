"""Convert Binance crypto data into AlphaGen/AlphaQCM tensor format.

Provides CryptoStockData as a drop-in replacement for AlphaGen's
qlib-based StockData. The Expression tree system evaluates directly
against CryptoStockData.data with shape (total_bars, n_features, n_stocks).

Path management:
    The caller (run_alphagen.py or run_alphaqcm.py) is responsible for
    adding the correct external repo to sys.path BEFORE importing this
    module. This avoids conflicts between the upstream AlphaGen and
    AlphaQCM's fork.
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from src.data_adapter.feature_engineering import compute_normalized_features
from src.data_collection.data_cleaner import load_panel


def _ensure_alphagen_path():
    """Add external/alphagen to path if alphagen isn't already importable."""
    try:
        import alphagen  # noqa: F401
    except ImportError:
        p = str(Path(__file__).resolve().parents[2] / "external" / "alphagen")
        sys.path.insert(0, p)


_ensure_alphagen_path()

from alphagen_qlib.stock_data import FeatureType
from alphagen.data.expression import Expression
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr


class CryptoStockData:
    """Drop-in replacement for AlphaGen's StockData.

    Loads Binance crypto panel data and exposes the same interface
    that Expression.evaluate(data) expects.

    Data tensor shape: (total_bars, n_features, n_stocks)
    where total_bars = max_backtrack_days + n_days + max_future_days

    Feature order follows FeatureType enum:
        0=OPEN, 1=CLOSE, 2=HIGH, 3=LOW, 4=VOLUME, 5=VWAP
    """

    def __init__(
        self,
        instrument: Union[str, List[str]] = "all",
        start_time: str = "",
        end_time: str = "",
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cpu"),
        preloaded_data: Optional[Tuple[torch.Tensor, pd.Index, pd.Index]] = None,
    ):
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device

        if preloaded_data is not None:
            self.data, self._dates, self._stock_ids = preloaded_data
        else:
            raise ValueError("Use load_crypto_stock_data() factory function")

    def __getitem__(self, slc: slice) -> "CryptoStockData":
        """Get a subview of the data given a date slice or an index slice."""
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        if isinstance(slc.start, str):
            return self[self.find_date_slice(slc.start, slc.stop)]
        start, stop = slc.start, slc.stop
        start = start if start is not None else 0
        stop = (stop if stop is not None else self.n_days) + self.max_future_days + self.max_backtrack_days
        start = max(0, start)
        stop = min(self.data.shape[0], stop)
        idx_range = slice(start, stop)
        data = self.data[idx_range]
        remaining = data.isnan().reshape(-1, data.shape[-1]).all(dim=0).logical_not().nonzero().flatten()
        data = data[:, :, remaining]
        return CryptoStockData(
            instrument=self._instrument,
            start_time=self._dates[start + self.max_backtrack_days].strftime("%Y-%m-%d %H:%M"),
            end_time=self._dates[stop - 1 - self.max_future_days].strftime("%Y-%m-%d %H:%M"),
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            preloaded_data=(data, self._dates[idx_range], self._stock_ids[remaining.tolist()]),
        )

    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        ts = pd.Timestamp(date)
        idx: int = self._dates.searchsorted(ts)
        if exclusive and idx < len(self._dates) and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(f"Date {date} is out of range: [{self._start_time}, {self._end_time}]")
        return idx

    def find_date_slice(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> slice:
        start = None if start_time is None else self.find_date_index(start_time)
        stop = None if end_time is None else self.find_date_index(end_time, exclusive=False)
        return slice(start, stop)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def stock_ids(self) -> pd.Index:
        return self._stock_ids

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)


def load_crypto_stock_data(
    processed_dir: str,
    start_time: str,
    end_time: str,
    max_backtrack_days: int = 100,
    max_future_days: int = 30,
    device: torch.device = torch.device("cpu"),
) -> CryptoStockData:
    """Factory: load crypto data into CryptoStockData."""
    panel = load_panel(processed_dir)
    features = compute_normalized_features(panel)

    full_index = panel["close"].index
    start_ts = pd.Timestamp(start_time, tz="UTC") if full_index.tz else pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time, tz="UTC") if full_index.tz else pd.Timestamp(end_time)

    core_mask = (full_index >= start_ts) & (full_index <= end_ts)
    core_positions = np.where(core_mask)[0]
    if len(core_positions) == 0:
        raise ValueError(f"No data between {start_time} and {end_time}")

    core_start = core_positions[0]
    core_end = core_positions[-1]

    buf_start = max(0, core_start - max_backtrack_days)
    buf_end = min(len(full_index) - 1, core_end + max_future_days)

    actual_backtrack = core_start - buf_start
    actual_future = buf_end - core_end

    selected_index = full_index[buf_start:buf_end + 1]
    symbols = list(panel["close"].columns)

    # Build tensor: (total_bars, n_features, n_stocks)
    feature_order = ["open", "close", "high", "low", "volume", "vwap"]
    arrays = []
    for fname in feature_order:
        arrays.append(features[fname].loc[selected_index].values.astype(np.float32))

    stacked = np.stack(arrays, axis=0)           # (6, total_bars, n_stocks)
    stacked = np.transpose(stacked, (1, 0, 2))   # (total_bars, 6, n_stocks)
    stacked = np.nan_to_num(stacked, nan=0.0)

    data_tensor = torch.tensor(stacked, dtype=torch.float32, device=device)

    n_days = len(selected_index) - actual_backtrack - actual_future
    logger.info(
        f"CryptoStockData: {n_days} bars x {len(symbols)} symbols x {len(feature_order)} features "
        f"(backtrack={actual_backtrack}, future={actual_future})"
    )

    return CryptoStockData(
        instrument="all",
        start_time=start_time,
        end_time=end_time,
        max_backtrack_days=actual_backtrack,
        max_future_days=actual_future,
        features=list(FeatureType),
        device=device,
        preloaded_data=(data_tensor, selected_index, pd.Index(symbols)),
    )


class CryptoAlphaCalculator:
    """Calculator compatible with BOTH upstream AlphaGen and AlphaQCM's fork.

    Implements the union of methods required by:
    - alphagen (upstream): TensorAlphaCalculator interface
    - alphaqcm: simpler AlphaCalculator interface
    Both call calc_single_IC_ret, calc_mutual_IC, calc_pool_IC_ret, calc_pool_rIC_ret.
    Upstream additionally calls calc_pool_all_ret, calc_single_rIC_ret, etc.
    """

    def __init__(self, data: CryptoStockData, target: Optional[Expression] = None):
        self.data = data
        if target is not None:
            self.target_value = normalize_by_day(target.evaluate(data))
        else:
            self.target_value = None

    def _calc_alpha(self, expr: Expression) -> torch.Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    # --- Required by both upstream and QCM ---

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return batch_pearsonr(value, self.target_value).mean().item()

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return batch_spearmanr(value, self.target_value).mean().item()

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        ic = batch_pearsonr(value, self.target_value).mean().item()
        ric = batch_spearmanr(value, self.target_value).mean().item()
        return ic, ric

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        v1, v2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return batch_pearsonr(v1, v2).mean().item()

    def make_ensemble_alpha(self, exprs, weights) -> torch.Tensor:
        factors = [self._calc_alpha(exprs[i]) * weights[i] for i in range(len(exprs))]
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def calc_pool_IC_ret(self, exprs, weights) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return batch_pearsonr(value, self.target_value).mean().item()

    def calc_pool_rIC_ret(self, exprs, weights) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return batch_spearmanr(value, self.target_value).mean().item()

    def calc_pool_all_ret(self, exprs, weights) -> Tuple[float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            ic = batch_pearsonr(value, self.target_value).mean().item()
            ric = batch_spearmanr(value, self.target_value).mean().item()
            return ic, ric

    def calc_pool_all_ret_with_ir(self, exprs, weights) -> Tuple[float, float, float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            ics = batch_pearsonr(value, self.target_value)
            rics = batch_spearmanr(value, self.target_value)
            ic_mean, ic_std = ics.mean().item(), ics.std().item()
            ric_mean, ric_std = rics.mean().item(), rics.std().item()
            return ic_mean, ic_mean / ic_std, ric_mean, ric_mean / ric_std

    # --- Required by upstream's TensorAlphaCalculator interface ---

    def evaluate_alpha(self, expr: Expression) -> torch.Tensor:
        return self._calc_alpha(expr)

    @property
    def target(self) -> torch.Tensor:
        return self.target_value

    @property
    def n_days(self) -> int:
        return self.data.n_days


def create_data_splits(
    processed_dir: str,
    config_path: str = "config/data_config.yaml",
    device: torch.device = torch.device("cpu"),
    max_backtrack_days: int = 100,
    max_future_days: int = 30,
) -> dict:
    """Create train/val/test CryptoStockData splits."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    split_cfg = cfg["split"]
    panel = load_panel(processed_dir)
    index = panel["close"].index

    n = len(index)
    train_end = int(n * split_cfg["train_ratio"])
    val_end = train_end + int(n * split_cfg["val_ratio"])

    splits_def = {
        "train": (str(index[0]), str(index[train_end - 1])),
        "val": (str(index[train_end]), str(index[val_end - 1])),
        "test": (str(index[val_end]), str(index[-1])),
    }

    splits = {}
    for name, (start, end) in splits_def.items():
        logger.info(f"{name}: {start} -> {end}")
        splits[name] = load_crypto_stock_data(
            processed_dir, start, end,
            max_backtrack_days=max_backtrack_days,
            max_future_days=max_future_days,
            device=device,
        )
    return splits
