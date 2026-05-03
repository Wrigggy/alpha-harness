"""Build a team-level governance dashboard for alpha ownership and BRAIN tracking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALPHAGEN = ROOT / "external" / "alphagen"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXTERNAL_ALPHAGEN) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ALPHAGEN))

from alphagen.data.parser import parse_expression

from src.brain_proxy.evaluator import (
    build_proxy_context,
    compute_signal_correlation_matrix,
    evaluate_brain_candidate,
    tensor_to_factor_frame,
)
from src.team_governance.registry import (
    load_registry,
    registry_to_frame,
    save_registry,
    update_proxy_metrics,
    utc_now_iso,
)


def _plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(matrix.columns) * 1.2), max(5, len(matrix.index) * 0.9)))
    image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _member_level_correlation(alpha_corr: pd.DataFrame, owner_map: dict[str, str]) -> pd.DataFrame:
    owners = sorted(set(owner_map.values()))
    matrix = pd.DataFrame(np.eye(len(owners)), index=owners, columns=owners)
    for left in owners:
        left_ids = [alpha_id for alpha_id, owner in owner_map.items() if owner == left]
        for right in owners:
            right_ids = [alpha_id for alpha_id, owner in owner_map.items() if owner == right]
            if not left_ids or not right_ids:
                matrix.loc[left, right] = 0.0
                continue
            sub = alpha_corr.loc[left_ids, right_ids].abs()
            matrix.loc[left, right] = float(sub.max().max()) if not sub.empty else 0.0
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the team alpha registry and build governance outputs")
    parser.add_argument("--registry", default="data/governance/team_registry.json")
    parser.add_argument("--data-config", default="config/data_config_equity_cn.yaml")
    parser.add_argument("--brain-config", default="config/brain_proxy_equity_cn.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-alphas", type=int, default=60, help="Cap for correlation analysis runtime")
    parser.add_argument("--include-retired", action="store_true")
    parser.add_argument("--skip-proxy-refresh", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    records = registry.records if args.include_retired else [r for r in registry.records if r.status != "retired"]
    if not records:
        raise ValueError("Registry is empty after filtering")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "out" / "governance" / f"team_registry_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = registry_to_frame(registry)
    if not args.include_retired and not frame.empty:
        frame = frame[frame["status"] != "retired"].copy()
    for column in [
        "proxy_brain_readiness_score",
        "proxy_max_team_correlation",
        "proxy_max_cross_member_correlation",
    ]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame.to_csv(out_dir / "registry_flat.csv", index=False)

    owner_summary = frame.groupby("owner").agg(
        n_alpha=("alpha_id", "count"),
        n_submitted=("brain_status", lambda s: int((s != "not_submitted").sum())),
        n_ready=("status", lambda s: int((s == "ready").sum())),
        avg_proxy_score=("proxy_brain_readiness_score", "mean"),
    ).reset_index()
    owner_summary.to_csv(out_dir / "owner_summary.csv", index=False)

    family_summary = frame.groupby("family").agg(
        n_alpha=("alpha_id", "count"),
        n_submitted=("brain_status", lambda s: int((s != "not_submitted").sum())),
        avg_proxy_score=("proxy_brain_readiness_score", "mean"),
    ).reset_index()
    family_summary.to_csv(out_dir / "family_summary.csv", index=False)

    frame.sort_values(["brain_status", "submitted_at", "owner"], ascending=[True, False, True]).to_csv(
        out_dir / "brain_submission_status.csv",
        index=False,
    )

    selected = sorted(
        records,
        key=lambda r: float(r.proxy_metrics.get("brain_readiness_score", -1.0)),
        reverse=True,
    )[: args.max_alphas]

    if not args.skip_proxy_refresh:
        context = build_proxy_context(
            data_config_path=args.data_config,
            split=args.split,
            brain_config_path=args.brain_config,
        )
    else:
        context = None

    signal_map: dict[str, pd.DataFrame] = {}
    refreshed_at = utc_now_iso()

    for record in selected:
        if context is None:
            continue
        parsed = parse_expression(record.expression)
        factor_df = tensor_to_factor_frame(
            context["stock_data"],
            context["calculator"].evaluate_alpha(parsed),
            name=record.alpha_id,
        )
        signal, summary = evaluate_brain_candidate(
            expression=record.expression,
            factor_df=factor_df,
            panel=context["panel"],
            forward_returns=context["forward_returns"],
            next_bar_returns=context["next_bar_returns"],
            liquidity_df=context["liquidity_df"],
            bars_per_year=context["bars_per_year"],
            cfg=context["brain_cfg"],
        )
        signal_map[record.alpha_id] = signal
        update_proxy_metrics(registry, record.alpha_id, summary, evaluated_at=refreshed_at)

    if signal_map:
        alpha_corr = compute_signal_correlation_matrix(signal_map)
        alpha_corr.to_csv(out_dir / "alpha_correlation.csv")
        alpha_labels = {record.alpha_id: f"{record.owner}:{record.family}:{record.alpha_id}" for record in selected}
        alpha_corr_labeled = alpha_corr.rename(index=alpha_labels, columns=alpha_labels)
        _plot_heatmap(alpha_corr_labeled, "Alpha Correlation", out_dir / "alpha_correlation.png")

        owner_map = {record.alpha_id: record.owner for record in selected}
        member_corr = _member_level_correlation(alpha_corr, owner_map)
        member_corr.to_csv(out_dir / "member_correlation.csv")
        _plot_heatmap(member_corr, "Cross-Member Max |Correlation|", out_dir / "member_correlation.png")

        for record in registry.records:
            if record.alpha_id not in alpha_corr.index:
                continue
            abs_corr = alpha_corr.loc[record.alpha_id].abs().drop(index=record.alpha_id, errors="ignore")
            record.proxy_metrics["max_team_correlation"] = float(abs_corr.max()) if not abs_corr.empty else 0.0
            cross_member = [
                abs_corr.loc[other.alpha_id]
                for other in selected
                if other.alpha_id in abs_corr.index and other.owner != record.owner
            ]
            record.proxy_metrics["max_cross_member_correlation"] = float(max(cross_member)) if cross_member else 0.0

    save_registry(registry, args.registry)

    submitted = int((frame["brain_status"] != "not_submitted").sum()) if not frame.empty else 0
    summary = {
        "n_registry_records": int(len(frame)),
        "n_submitted_to_brain": submitted,
        "n_not_submitted": int(len(frame) - submitted),
        "owners": sorted(frame["owner"].dropna().unique().tolist()) if not frame.empty else [],
        "families": sorted(frame["family"].dropna().unique().tolist()) if not frame.empty else [],
        "analysis_split": args.split,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Team governance dashboard written to {}", out_dir)


if __name__ == "__main__":
    main()
