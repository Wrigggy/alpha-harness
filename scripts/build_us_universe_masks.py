"""Build daily liquidity-ranked universe masks such as USA_TOP3000/TOP1000/TOP500 plus SP500."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]


def load_panel_field(panel_dir: Path, field: str) -> pd.DataFrame:
    path = panel_dir / f"{field}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing panel field: {path}")
    return pd.read_parquet(path)


def load_symbol_list(path: str | None) -> set[str]:
    if not path:
        return set()
    data_path = Path(path)
    if data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.upper())


def build_topn_mask(liquidity_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=liquidity_df.index, columns=liquidity_df.columns)
    for ts in liquidity_df.index:
        row = pd.to_numeric(liquidity_df.loc[ts], errors="coerce").dropna().sort_values(ascending=False)
        if row.empty:
            continue
        keep = row.head(min(top_n, len(row))).index
        mask.loc[ts, keep] = True
    return mask


def build_static_symbol_mask(base_df: pd.DataFrame, symbols: set[str]) -> pd.DataFrame:
    cols = [c for c in base_df.columns if str(c).upper() in symbols]
    mask = pd.DataFrame(False, index=base_df.index, columns=base_df.columns)
    if cols:
        mask.loc[:, cols] = True
    return mask


def write_mask(panel_dir: Path, field_name: str, mask: pd.DataFrame) -> None:
    mask.astype(int).to_parquet(panel_dir / f"{field_name}.parquet")
    logger.info("Wrote {} to {}", field_name, panel_dir / f"{field_name}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build US liquidity universe masks")
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--liquidity-field", default="amount")
    parser.add_argument("--top-levels", default="3000,2000,1000,500,200")
    parser.add_argument("--sp500-file", default=None, help="Optional CSV/parquet with a symbol column")
    args = parser.parse_args()

    panel_dir = Path(args.panel_dir)
    liquidity_df = load_panel_field(panel_dir, args.liquidity_field)
    levels = [int(x.strip()) for x in args.top_levels.split(",") if x.strip()]

    for top_n in levels:
        mask = build_topn_mask(liquidity_df, top_n)
        write_mask(panel_dir, f"universe_usa_top{top_n}", mask)

    sp500_symbols = load_symbol_list(args.sp500_file)
    if sp500_symbols:
        sp500_mask = build_static_symbol_mask(liquidity_df, sp500_symbols)
        write_mask(panel_dir, "universe_sp500", sp500_mask)


if __name__ == "__main__":
    main()
