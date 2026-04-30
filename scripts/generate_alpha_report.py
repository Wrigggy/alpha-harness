r"""Generate a local alpha report with plots for an AlphaGen factor pool.

This script is designed for the current alpha-harness repo layout:
- loads a pool json from data/factors/
- parses AlphaGen expression strings
- evaluates factor values on crypto panel data
- computes IC / RankIC / decay / simple long-short backtests
- writes plots plus markdown/json summaries under out/reports/

Example:
    .\.venv\Scripts\python.exe scripts\generate_alpha_report.py \
        --pool data\factors\alphagen_pool.json --split test
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALPHAGEN = ROOT / "external" / "alphagen"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXTERNAL_ALPHAGEN) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ALPHAGEN))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.data.parser import parse_expression
from src.backtest.long_short_backtest import long_short_backtest, plot_backtest
from src.data_adapter.to_alphagen_format import (
    PanelAlphaCalculator,
    _load_local_panel,
    create_data_splits,
    load_panel_stock_data,
)
from src.evaluation.factor_decay import compute_ic_decay, plot_ic_decay
from src.evaluation.ic_analysis import evaluate_factor


def slugify(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[:max_len] or "factor"


def load_pool(pool_path: Path) -> tuple[list[str], list[float]]:
    with open(pool_path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "exprs" in raw:
        exprs = [str(x) for x in raw["exprs"]]
        weights = [float(x) for x in raw.get("weights", [1.0] * len(exprs))]
        return exprs, weights

    if isinstance(raw, list):
        exprs = [str(item["expression"]) for item in raw]
        weights = [float(item.get("weight", item.get("ic", 1.0))) for item in raw]
        return exprs, weights

    raise ValueError(f"Unsupported pool format: {pool_path}")


def normalize_weights(weights: list[float]) -> list[float]:
    if not weights:
        return []
    total = sum(abs(w) for w in weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def load_data_for_split(
    processed_dir: str,
    data_config: str,
    split: str,
    max_backtrack_days: int = 100,
    max_future_days: int = 10,
):
    if split in {"train", "val", "test"}:
        return create_data_splits(
            processed_dir,
            data_config,
            max_backtrack_days=max_backtrack_days,
            max_future_days=max_future_days,
        )[split]

    panel = _load_local_panel(processed_dir, data_config=data_config)
    index = panel["close"].index
    if len(index) <= max_backtrack_days + max_future_days + 1:
        raise ValueError("Not enough data to build a full split report")
    start = str(index[max_backtrack_days])
    end = str(index[-1 - max_future_days])
    return load_panel_stock_data(
        processed_dir,
        start,
        end,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        data_config=data_config,
    )


def tensor_to_factor_frame(stock_data, tensor: object, name: str = "factor") -> pd.DataFrame:
    frame = stock_data.make_dataframe(tensor, columns=[name])
    return frame[name].unstack(level=1).astype(float)


def build_forward_returns(close_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close_df.shift(-horizon) / close_df - 1.0


def auto_min_observations(requested: int, n_symbols: int) -> int:
    return max(2, min(requested, n_symbols))


def auto_leg_size(requested: int | None, n_symbols: int) -> int:
    if requested is not None:
        return requested
    return max(1, n_symbols // 3)


def plot_cumulative_ic(metrics, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not metrics.ic_series.empty:
        metrics.ic_series.cumsum().plot(ax=ax, label="Cum IC", linewidth=1.2)
    if not metrics.rank_ic_series.empty:
        metrics.rank_ic_series.cumsum().plot(ax=ax, label="Cum RankIC", linewidth=1.2)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Cumulative IC")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_factor_scatter(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    title: str,
    output_path: Path,
    max_points: int = 5000,
) -> None:
    a = factor_df.values.flatten().astype(float)
    r = forward_returns.values.flatten().astype(float)
    mask = np.isfinite(a) & np.isfinite(r)
    a = a[mask]
    r = r[mask]

    if len(a) == 0:
        return
    if len(a) > max_points:
        idx = np.random.default_rng(42).choice(len(a), size=max_points, replace=False)
        a = a[idx]
        r = r[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(a, r, s=6, alpha=0.25)
    if len(a) >= 2 and np.nanstd(a) > 1e-12 and np.nanstd(r) > 1e-12:
        try:
            slope, intercept = np.polyfit(a, r, deg=1)
            xline = np.linspace(float(a.min()), float(a.max()), 100)
            yline = slope * xline + intercept
            ax.plot(xline, yline, color="crimson", linewidth=1.5)
        except np.linalg.LinAlgError:
            pass
    ax.set_title(title)
    ax.set_xlabel("Factor Value")
    ax.set_ylabel("Forward Return")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_quantile_profile(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    quantiles: int,
) -> pd.Series:
    q = max(2, min(quantiles, factor_df.shape[1]))
    bucket_returns: dict[int, list[float]] = {i: [] for i in range(1, q + 1)}

    common_idx = factor_df.index.intersection(forward_returns.index)
    common_cols = factor_df.columns.intersection(forward_returns.columns)

    for ts in common_idx:
        f = factor_df.loc[ts, common_cols].astype(float)
        r = forward_returns.loc[ts, common_cols].astype(float)
        mask = np.isfinite(f.values) & np.isfinite(r.values)
        if mask.sum() < q:
            continue

        f = f[mask]
        r = r[mask]
        ranks = f.rank(method="first", pct=True)
        bucket = np.ceil(ranks * q).astype(int).clip(1, q)
        for b in range(1, q + 1):
            selected = r[bucket == b]
            if not selected.empty:
                bucket_returns[b].append(float(selected.mean()))

    return pd.Series(
        {b: (float(np.mean(vals)) if vals else 0.0) for b, vals in bucket_returns.items()}
    )


def plot_quantile_profile(profile: pd.Series, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(profile.index.astype(str), profile.values, color="steelblue")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Avg Forward Return")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 1.1), max(5, len(corr) * 0.8)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_factor_correlation_local(
    factor_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    names = list(factor_dict.keys())
    n = len(names)
    corr_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            fi = factor_dict[names[i]]
            fj = factor_dict[names[j]]
            common_idx = fi.index.intersection(fj.index)
            common_cols = fi.columns.intersection(fj.columns)

            a = fi.loc[common_idx, common_cols].values.flatten().astype(float)
            b = fj.loc[common_idx, common_cols].values.flatten().astype(float)
            mask = np.isfinite(a) & np.isfinite(b)
            corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 10 else 0.0
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return pd.DataFrame(corr_matrix, index=names, columns=names)


def write_summary_markdown(
    output_path: Path,
    split: str,
    horizon: int,
    metrics_df: pd.DataFrame,
    combined_metrics: dict,
    combined_backtest_metrics: dict,
) -> None:
    lines = [
        "# Alpha Report",
        "",
        f"- Split: `{split}`",
        f"- IC Horizon: `{horizon}` bars",
        f"- Factors: `{len(metrics_df)}`",
        "",
        "## Combined Signal",
        "",
        f"- RankIC mean: `{combined_metrics['rank_ic_mean']:.6f}`",
        f"- IC mean: `{combined_metrics['ic_mean']:.6f}`",
        f"- RankICIR: `{combined_metrics['rank_icir']:.6f}`",
        f"- ICIR: `{combined_metrics['icir']:.6f}`",
        f"- Backtest annual return: `{combined_backtest_metrics['annual_return']:.6f}`",
        f"- Backtest Sharpe: `{combined_backtest_metrics['sharpe_ratio']:.6f}`",
        f"- Backtest max drawdown: `{combined_backtest_metrics['max_drawdown']:.6f}`",
        "",
        "## Per-Factor Metrics",
        "",
        "| Factor | IC | RankIC | ICIR | RankICIR | Weight |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['name']} | {row['ic_mean']:.6f} | {row['rank_ic_mean']:.6f} | "
            f"{row['icir']:.6f} | {row['rank_icir']:.6f} | {row['weight']:.6f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local alpha report with plots")
    parser.add_argument("--pool", default="data/factors/alphagen_pool.json", help="Pool JSON path")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed panel directory")
    parser.add_argument("--data-config", default="config/data_config.yaml", help="Data config path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "full"])
    parser.add_argument("--horizon", type=int, default=8, help="Forward return horizon for IC analysis")
    parser.add_argument("--min-observations", type=int, default=10)
    parser.add_argument("--n-long", type=int, default=None)
    parser.add_argument("--n-short", type=int, default=None)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--output-dir", default=None, help="Optional report output directory")
    args = parser.parse_args()

    pool_path = Path(args.pool)
    with open(args.data_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    source_cfg = cfg.get("source", {})
    source_name = source_cfg.get("name", "crypto")
    default_horizon = int(source_cfg.get("target_horizon", args.horizon))
    bars_per_year = int(source_cfg.get("bars_per_year", 8760 if source_name == "crypto" else 252))

    expr_strings, raw_weights = load_pool(pool_path)
    weights = normalize_weights(raw_weights)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "out" / "reports" / f"{pool_path.stem}_{args.split}"
    )
    factors_dir = output_dir / "factors"
    output_dir.mkdir(parents=True, exist_ok=True)
    factors_dir.mkdir(parents=True, exist_ok=True)

    effective_processed_dir = (
        args.processed_dir if source_name == "crypto" else source_cfg.get("panel_dir", args.processed_dir)
    )
    stock_data = load_data_for_split(effective_processed_dir, args.data_config, args.split)
    panel = _load_local_panel(effective_processed_dir, data_config=args.data_config)

    target_expr = Ref(Feature(FeatureType.CLOSE), -default_horizon) / Feature(FeatureType.CLOSE) - 1
    calculator = PanelAlphaCalculator(stock_data, target_expr)

    parsed_exprs = [parse_expression(expr) for expr in expr_strings]
    factor_frames: dict[str, pd.DataFrame] = {}
    factor_metrics_rows: list[dict] = []

    for idx, (expr_str, expr, weight) in enumerate(zip(expr_strings, parsed_exprs, weights), start=1):
        factor_name = f"f{idx:02d}_{slugify(expr_str, 36)}"
        factor_df = tensor_to_factor_frame(stock_data, calculator.evaluate_alpha(expr))
        close_df = factor_df.copy() * np.nan
        if panel is not None:
            close_df = panel["close"].loc[factor_df.index, factor_df.columns].astype(float)
        else:
            raise FileNotFoundError(f"Local panel not found for report generation: {effective_processed_dir}")
        fwd_ret = build_forward_returns(close_df, default_horizon)
        min_obs = auto_min_observations(args.min_observations, factor_df.shape[1])
        metrics = evaluate_factor(factor_df, fwd_ret, min_observations=min_obs)

        factor_frames[factor_name] = factor_df
        factor_metrics_rows.append(
            {
                "name": factor_name,
                "expression": expr_str,
                "weight": weight,
                "ic_mean": metrics.ic_mean,
                "rank_ic_mean": metrics.rank_ic_mean,
                "icir": metrics.icir,
                "rank_icir": metrics.rank_icir,
            }
        )

        plot_cumulative_ic(metrics, f"{factor_name} - Cumulative IC", factors_dir / f"{factor_name}_cum_ic.png")
        plot_factor_scatter(
            factor_df,
            fwd_ret,
            f"{factor_name} - Factor vs Forward Return",
            factors_dir / f"{factor_name}_scatter.png",
        )
        quantile_profile = compute_quantile_profile(factor_df, fwd_ret, args.quantiles)
        plot_quantile_profile(
            quantile_profile,
            f"{factor_name} - Quantile Forward Returns",
            factors_dir / f"{factor_name}_quantiles.png",
        )

        decay_df = compute_ic_decay(
            factor_df,
            close_df,
            horizons=[1, 2, 4, 8, 24] if bars_per_year > 1000 else [1, 2, 5, 10, 20],
            min_observations=min_obs,
        )
        decay_df.to_csv(factors_dir / f"{factor_name}_decay.csv", index=False)
        fig = plot_ic_decay(decay_df, factor_name=factor_name)
        fig.savefig(factors_dir / f"{factor_name}_decay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    metrics_df = pd.DataFrame(factor_metrics_rows)
    metrics_df.to_csv(output_dir / "factor_metrics.csv", index=False)

    combined_df = tensor_to_factor_frame(
        stock_data,
        calculator.make_ensemble_alpha(parsed_exprs, weights),
        name="combined",
    )
    close_df = panel["close"].loc[combined_df.index, combined_df.columns].astype(float)
    liquidity_field = "quote_volume" if "quote_volume" in panel else "volume"
    liquidity_df = panel[liquidity_field].loc[combined_df.index, combined_df.columns].astype(float)
    fwd_ret = build_forward_returns(close_df, default_horizon)
    next_bar_ret = build_forward_returns(close_df, 1)

    min_obs = auto_min_observations(args.min_observations, combined_df.shape[1])
    combined_ic = evaluate_factor(combined_df, fwd_ret, min_observations=min_obs)

    combined_metrics = {
        "ic_mean": combined_ic.ic_mean,
        "rank_ic_mean": combined_ic.rank_ic_mean,
        "icir": combined_ic.icir,
        "rank_icir": combined_ic.rank_icir,
    }
    with open(output_dir / "combined_metrics.json", "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)

    plot_cumulative_ic(combined_ic, "Combined Signal - Cumulative IC", output_dir / "combined_cum_ic.png")
    plot_factor_scatter(
        combined_df,
        fwd_ret,
        "Combined Signal - Factor vs Forward Return",
        output_dir / "combined_scatter.png",
    )

    combined_quantiles = compute_quantile_profile(combined_df, fwd_ret, args.quantiles)
    plot_quantile_profile(
        combined_quantiles,
        "Combined Signal - Quantile Forward Returns",
        output_dir / "combined_quantiles.png",
    )

    combined_decay = compute_ic_decay(
        combined_df,
        close_df,
        horizons=[1, 2, 4, 8, 24] if bars_per_year > 1000 else [1, 2, 5, 10, 20],
        min_observations=min_obs,
    )
    combined_decay.to_csv(output_dir / "combined_decay.csv", index=False)
    fig = plot_ic_decay(combined_decay, factor_name="Combined Signal")
    fig.savefig(output_dir / "combined_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_long = auto_leg_size(args.n_long, combined_df.shape[1])
    n_short = auto_leg_size(args.n_short, combined_df.shape[1])
    backtest = long_short_backtest(
        combined_df,
        next_bar_ret,
        n_long=n_long,
        n_short=n_short,
        liquidity_filter=liquidity_df,
        transaction_cost_bps=5.0,
        bars_per_year=bars_per_year,
    )
    backtest["equity_curve"].to_csv(output_dir / "combined_equity_curve.csv", header=True)
    with open(output_dir / "combined_backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(backtest["metrics"], f, indent=2)

    fig = plot_backtest(
        backtest["equity_curve"],
        backtest["metrics"],
        title=f"Combined Long-Short Backtest ({args.split})",
    )
    fig.savefig(output_dir / "combined_backtest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if factor_frames:
        corr = compute_factor_correlation_local(factor_frames)
        corr.to_csv(output_dir / "factor_correlation.csv")
        plot_correlation_heatmap(corr, "Factor Correlation Heatmap", output_dir / "factor_correlation.png")

    write_summary_markdown(
        output_dir / "summary.md",
        split=args.split,
        horizon=args.horizon,
        metrics_df=metrics_df,
        combined_metrics=combined_metrics,
        combined_backtest_metrics=backtest["metrics"],
    )

    summary = {
        "pool": str(pool_path),
        "split": args.split,
        "horizon": default_horizon,
        "source": source_name,
        "n_factors": len(expr_strings),
        "symbols": int(combined_df.shape[1]),
        "observations": int(combined_df.shape[0]),
        "combined_metrics": combined_metrics,
        "combined_backtest": backtest["metrics"],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Alpha report written to {}", output_dir)


if __name__ == "__main__":
    main()
