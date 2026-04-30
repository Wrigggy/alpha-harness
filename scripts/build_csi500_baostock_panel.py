"""Build a local CSI 500-style panel using latest constituents + BaoStock history.

This is a practical approximation:
- constituent list comes from the latest CSI 500 snapshot via AkShare
- price history comes from BaoStock

It is not a full historical-constituent reconstruction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_constituents(path: Path, top_n: int | None = None) -> list[str]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    if "symbol" not in df.columns:
        raise ValueError(f"'symbol' column not found in {path}")
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    if top_n is not None:
        if "weight" in df.columns:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
            df = df.sort_values("weight", ascending=False).head(top_n)
        else:
            df = df.head(top_n)
    return df["symbol"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSI 500-style local panel from latest constituents")
    parser.add_argument("--constituents", default="data/raw_equity/csi500_constituents_latest.parquet")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--top-n", type=int, default=None, help="Optional cap by constituent weight")
    parser.add_argument("--output", default="data/equity_panel/csi500_latest_baostock_daily")
    parser.add_argument("--raw-output", default="data/raw_equity/csi500_latest_baostock_daily.parquet")
    args = parser.parse_args()

    symbols = read_constituents(Path(args.constituents), top_n=args.top_n)
    logger.info("Loaded {} constituent symbols", len(symbols))

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "fetch_equity_baostock.py"),
        "--start", args.start,
        "--end", args.end,
        "--symbols", ",".join(symbols),
        "--output", args.output,
        "--raw-output", args.raw_output,
    ]
    logger.info("Running BaoStock fetch for {} symbols", len(symbols))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    logger.info("CSI 500-style panel build complete: {}", args.output)


if __name__ == "__main__":
    main()
