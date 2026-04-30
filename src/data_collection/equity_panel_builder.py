"""Helpers for building local equity panels from long-form daily data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["amount", "turnover", "vwap", "market_cap", "industry"]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    numeric_cols = [col for col in REQUIRED_COLUMNS[2:] + OPTIONAL_COLUMNS if col in frame.columns]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def build_panel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    panel: dict[str, pd.DataFrame] = {}
    for field in REQUIRED_COLUMNS[2:] + OPTIONAL_COLUMNS:
        if field in df.columns:
            panel[field] = df.pivot(index="date", columns="symbol", values=field).sort_index()
    return panel


def save_panel(panel: dict[str, pd.DataFrame], output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for field, matrix in panel.items():
        matrix.to_parquet(output_dir / f"{field}.parquet")

    symbols = list(panel["close"].columns)
    pd.Series(symbols).to_csv(output_dir / "symbols.csv", index=False, header=False)
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
