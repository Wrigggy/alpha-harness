"""Convert local panel data into AlphaGen/AlphaQCM tensor format.

This module originally targeted crypto-only OHLCV data. It is now generalized
to support equity-style panel data as well, while keeping the old public names
(`CryptoStockData`, `CryptoAlphaCalculator`) as compatibility aliases.
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from src.data_collection.data_cleaner import load_panel
from src.data_sources.local_panel_source import panel_directory_exists
from src.data_sources.qlib_source import QlibSource


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


DEFAULT_FEATURE_ORDER = ["open", "close", "high", "low", "volume", "vwap"]


class PanelStockData:
    """Drop-in replacement for AlphaGen's StockData backed by a local panel."""

    def __init__(
        self,
        instrument: Union[str, List[str]] = "all",
        start_time: str = "",
        end_time: str = "",
        max_backtrack_days: int = 100,
        max_future_days: int = 5,
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
            raise ValueError("Use load_panel_stock_data() factory function")

    def __getitem__(self, slc: slice) -> "PanelStockData":
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
        return PanelStockData(
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
        _, _, n_columns = data.shape
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        reshaped = data.reshape(-1, n_columns)
        return pd.DataFrame(reshaped.detach().cpu().numpy(), index=index, columns=columns)


def _build_vwap(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "vwap" in panel:
        return panel["vwap"]
    if "quote_volume" in panel and "volume" in panel:
        return panel["quote_volume"] / panel["volume"].replace(0, np.nan)
    return (panel["high"] + panel["low"] + panel["close"]) / 3.0


def _load_local_panel(processed_dir: str, data_config: str | None = None) -> dict[str, pd.DataFrame]:
    path = Path(processed_dir)
    if panel_directory_exists(path):
        return load_panel(processed_dir)

    if data_config is not None and Path(data_config).exists():
        with open(data_config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        source_cfg = cfg.get("source", {}) if isinstance(cfg, dict) else {}
        source_name = source_cfg.get("name", "crypto")
        panel_dir = source_cfg.get("panel_dir")
        if panel_dir and panel_directory_exists(panel_dir):
            return load_panel(panel_dir)
        if source_name == "qlib":
            qcfg = source_cfg.get("qlib", {})
            source = QlibSource(
                instruments=qcfg.get("instruments", "csi500"),
                start_date=qcfg.get("start_date", "2020-01-01"),
                end_date=qcfg.get("end_date", "2023-12-31"),
                dataset=qcfg.get("dataset", "Alpha158"),
                cache_dir=qcfg.get("cache_dir", "data/qlib_cache"),
                provider_uri=qcfg.get("provider_uri"),
                fallback_panel_dir=panel_dir,
                prefer_local_panel=bool(qcfg.get("prefer_local_panel", False)),
            )
            return source.load_panel()

    raise FileNotFoundError(f"Panel source not found: {processed_dir}")


def load_panel_stock_data(
    processed_dir: str,
    start_time: str,
    end_time: str,
    max_backtrack_days: int = 100,
    max_future_days: int = 5,
    device: torch.device = torch.device("cpu"),
    data_config: str | None = None,
) -> PanelStockData:
    """Factory: load generic panel data into AlphaGen-compatible StockData."""
    panel = _load_local_panel(processed_dir, data_config=data_config)
    vwap = _build_vwap(panel)

    full_index = panel["close"].index
    start_ts = pd.Timestamp(start_time, tz="UTC") if getattr(full_index, "tz", None) else pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time, tz="UTC") if getattr(full_index, "tz", None) else pd.Timestamp(end_time)

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

    if actual_future < max_future_days:
        logger.warning(
            "Requested future buffer {} but only {} bars available",
            max_future_days,
            actual_future,
        )
    if actual_backtrack < max_backtrack_days:
        logger.warning(
            "Requested backtrack buffer {} but only {} bars available",
            max_backtrack_days,
            actual_backtrack,
        )

    selected_index = full_index[buf_start:buf_end + 1]
    symbols = list(panel["close"].columns)
    raw_fields = {
        "open": panel["open"],
        "close": panel["close"],
        "high": panel["high"],
        "low": panel["low"],
        "volume": panel["volume"],
        "vwap": vwap,
    }

    arrays = [raw_fields[name].loc[selected_index].values.astype(np.float32) for name in DEFAULT_FEATURE_ORDER]
    stacked = np.stack(arrays, axis=0)
    stacked = np.transpose(stacked, (1, 0, 2))
    data_tensor = torch.tensor(stacked, dtype=torch.float32, device=device)

    n_days = len(selected_index) - actual_backtrack - actual_future
    logger.info(
        "PanelStockData: {} bars x {} symbols x {} features (backtrack={}, future={})",
        n_days,
        len(symbols),
        len(DEFAULT_FEATURE_ORDER),
        actual_backtrack,
        actual_future,
    )

    return PanelStockData(
        instrument="all",
        start_time=start_time,
        end_time=end_time,
        max_backtrack_days=actual_backtrack,
        max_future_days=actual_future,
        features=list(FeatureType),
        device=device,
        preloaded_data=(data_tensor, selected_index, pd.Index(symbols)),
    )


class PanelAlphaCalculator:
    """Calculator compatible with AlphaGen and AlphaQCM tensor interfaces."""

    def __init__(self, data: PanelStockData, target: Optional[Expression] = None):
        self.data = data
        self.target_value = normalize_by_day(target.evaluate(data)) if target is not None else None

    def _calc_alpha(self, expr: Expression) -> torch.Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    def _empty_alpha(self) -> torch.Tensor:
        if self.target_value is not None:
            return torch.zeros_like(self.target_value)
        return torch.zeros(
            (self.data.n_days, self.data.n_stocks),
            dtype=torch.float32,
            device=self.data.device,
        )

    def calc_single_IC_ret(self, expr: Expression) -> float:
        return batch_pearsonr(self._calc_alpha(expr), self.target_value).mean().item()

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return batch_spearmanr(self._calc_alpha(expr), self.target_value).mean().item()

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        ic = batch_pearsonr(value, self.target_value).mean().item()
        ric = batch_spearmanr(value, self.target_value).mean().item()
        return ic, ric

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return batch_pearsonr(self._calc_alpha(expr1), self._calc_alpha(expr2)).mean().item()

    def make_ensemble_alpha(self, exprs, weights) -> torch.Tensor:
        if len(exprs) == 0:
            return self._empty_alpha()
        factors = [self._calc_alpha(exprs[i]) * weights[i] for i in range(len(exprs))]
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def calc_pool_IC_ret(self, exprs, weights) -> float:
        with torch.no_grad():
            return batch_pearsonr(self.make_ensemble_alpha(exprs, weights), self.target_value).mean().item()

    def calc_pool_rIC_ret(self, exprs, weights) -> float:
        with torch.no_grad():
            return batch_spearmanr(self.make_ensemble_alpha(exprs, weights), self.target_value).mean().item()

    def calc_pool_all_ret(self, exprs, weights) -> Tuple[float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return (
                batch_pearsonr(value, self.target_value).mean().item(),
                batch_spearmanr(value, self.target_value).mean().item(),
            )

    def calc_pool_all_ret_with_ir(self, exprs, weights) -> Tuple[float, float, float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            ics = batch_pearsonr(value, self.target_value)
            rics = batch_spearmanr(value, self.target_value)
            ic_mean, ic_std = ics.mean().item(), ics.std().item()
            ric_mean, ric_std = rics.mean().item(), rics.std().item()
            return ic_mean, ic_mean / ic_std, ric_mean, ric_mean / ric_std

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
    max_future_days: int = 5,
) -> dict:
    """Create train/val/test PanelStockData splits."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split_cfg = cfg["split"]
    panel = _load_local_panel(processed_dir, data_config=config_path)
    index = panel["close"].index

    n = len(index)
    if n <= max_backtrack_days + max_future_days + 5:
        raise ValueError(
            "Not enough bars to create AlphaGen splits. "
            f"Need more than {max_backtrack_days + max_future_days + 5}, got {n}."
        )
    train_end = int(n * split_cfg["train_ratio"])
    val_end = train_end + int(n * split_cfg["val_ratio"])
    safe_end = lambda idx: min(max(idx, max_backtrack_days + 1), n - 1 - max_future_days)
    safe_start = min(max_backtrack_days, n - 1 - max_future_days - 2)

    if train_end <= safe_start + max_future_days:
        train_end = min(n - 2 * max_future_days - 2, max(safe_start + max_future_days + 1, train_end))
    if val_end <= train_end + 1:
        val_end = min(n - max_future_days - 1, train_end + max(2, int(n * split_cfg["val_ratio"])))

    splits_def = {
        "train": (str(index[safe_start]), str(index[train_end - 1 - max_future_days])),
        "val": (str(index[train_end]), str(index[safe_end(val_end - 1 - max_future_days)])),
        "test": (str(index[val_end]), str(index[safe_end(n - 1)])),
    }

    splits = {}
    for name, (start, end) in splits_def.items():
        logger.info("{}: {} -> {}", name, start, end)
        splits[name] = load_panel_stock_data(
            processed_dir,
            start,
            end,
            max_backtrack_days=max_backtrack_days,
            max_future_days=max_future_days,
            device=device,
            data_config=config_path,
        )
    return splits


# Backward-compatible aliases for existing imports.
CryptoStockData = PanelStockData
CryptoAlphaCalculator = PanelAlphaCalculator
load_crypto_stock_data = load_panel_stock_data
