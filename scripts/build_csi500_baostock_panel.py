"""Build a local CSI 500-style panel using latest constituents + BaoStock history.

This is a practical approximation:
- constituent list comes from the latest CSI 500 snapshot via AkShare
- price history comes from BaoStock

It is not a full historical-constituent reconstruction.
"""

from __future__ import annotations

import argparse
import math
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
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--output", default="data/equity_panel/csi500_latest_baostock_daily")
    parser.add_argument("--raw-output", default="data/raw_equity/csi500_latest_baostock_daily.parquet")
    args = parser.parse_args()

    symbols = read_constituents(Path(args.constituents), top_n=args.top_n)
    logger.info("Loaded {} constituent symbols", len(symbols))

    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, args.batch_size)
    n_batches = math.ceil(len(symbols) / batch_size)
    temp_files: list[Path] = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(symbols))
        batch_symbols = symbols[start_idx:end_idx]
        batch_file = raw_output.with_name(f"{raw_output.stem}_part{batch_idx + 1:03d}{raw_output.suffix}")
        temp_files.append(batch_file)

        if args.resume and batch_file.exists():
            logger.info("Skipping existing batch file {}", batch_file)
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "fetch_equity_baostock.py"),
            "--start", args.start,
            "--end", args.end,
            "--symbols", ",".join(batch_symbols),
            "--raw-output", str(batch_file),
            "--output", str(Path(args.output).with_name(f"{Path(args.output).name}_part{batch_idx + 1:03d}")),
        ]
        logger.info(
            "Running BaoStock fetch batch {}/{} for {} symbols",
            batch_idx + 1,
            n_batches,
            len(batch_symbols),
        )
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    frames: list[pd.DataFrame] = []
    for batch_file in temp_files:
        if batch_file.exists():
            frames.append(pd.read_parquet(batch_file))
    if not frames:
        raise RuntimeError("No batch files were produced by BaoStock fetch")

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "symbol"], keep="last")
    combined.to_parquet(raw_output, index=False)
    logger.info("Merged {} batch files into {}", len(frames), raw_output)

    import_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "import_equity_panel.py"),
        "--input", str(raw_output),
        "--output", args.output,
        "--market", "cn",
        "--universe", f"csi500_latest_top{len(symbols)}",
        "--frequency", "1d",
    ]
    subprocess.run(import_cmd, check=True, cwd=str(ROOT))
    logger.info("CSI 500-style panel build complete: {}", args.output)


if __name__ == "__main__":
    main()
