"""End-to-end alpha evaluation pipeline for crypto and equity-style panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml
from loguru import logger
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALPHAGEN = ROOT / "external" / "alphagen"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXTERNAL_ALPHAGEN) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ALPHAGEN))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.data.parser import parse_expression

from src.backtest.long_short_backtest import (
    long_short_backtest,
    plot_backtest,
    plot_worldquant_style_panels,
)
from src.brain_proxy.expression_translation import BrainExpressionTranslator
from src.brain_proxy.expression_translation import build_brain_variants, render_brain_submission_template
from src.data_adapter.to_alphagen_format import (
    PanelAlphaCalculator,
    _load_local_panel,
    create_data_splits,
)
from src.data_sources.crypto_source import CryptoSource
from src.data_sources.local_panel_source import LocalPanelSource, panel_directory_exists
from src.data_sources.qlib_source import QlibSource
from src.evaluation.factor_decay import compute_ic_decay
from src.evaluation.ic_analysis import evaluate_factor
from src.evaluation.validation_gate import ValidationConfig, ValidationGate
from src.feature_expansion.expander import expand_features, get_feature_names
from src.knowledge_base.paper_store import PaperStore
from src.knowledge_base.retriever import PaperRetriever
from src.llm_judge.base import JudgeResult
from src.portfolio.combiner import EqualWeightCombiner, ICWeightedCombiner, RidgeCombiner
from src.utils.pool_io import load_pool, normalize_weights


def load_judge(config_path: str = "config/judge_config.yaml"):
    """Load the LLM judge based on config."""
    cfg_path = Path(config_path)
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        judge_cfg = cfg.get("judge", {})
    else:
        judge_cfg = {"backend": "api", "provider": "openai", "model": "gpt-5-mini"}

    backend = str(judge_cfg.get("backend", "agent_sdk")).lower()
    model = judge_cfg.get("model", "gpt-5-mini")

    if backend in {"agent_sdk", "claude"}:
        from src.llm_judge.claude_agent_judge import ClaudeAgentJudge
        return ClaudeAgentJudge(model=model)
    if backend in {"api", "openai", "deepseek", "codex"}:
        from src.llm_judge.api_judge import ApiJudge
        provider = str(judge_cfg.get("provider", "openai")).lower()
        if backend != "api":
            provider = backend
        return ApiJudge(
            provider=provider,
            model=model,
            max_tokens=int(judge_cfg.get("max_tokens", 1024)),
            base_url=judge_cfg.get("base_url"),
            api_key_env=judge_cfg.get("api_key_env"),
            translate_prompt_path=judge_cfg.get("prompts", {}).get("translation", "prompts/translate.txt"),
            score_prompt_path=judge_cfg.get("prompts", {}).get("scoring", "prompts/score.txt"),
        )
    raise ValueError(f"Unknown judge backend: {backend}")


def load_data_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_source_name(explicit_source: str | None, data_cfg: dict) -> str:
    if explicit_source is not None:
        return explicit_source
    return data_cfg.get("source", {}).get("name", "crypto")


def get_data_source(source: str, data_cfg: dict):
    source_cfg = data_cfg.get("source", {})
    if source == "crypto":
        return CryptoSource(processed_dir=data_cfg["data"]["processed_dir"])
    if source == "panel":
        panel_dir = source_cfg.get("panel_dir", data_cfg["data"]["processed_dir"])
        return LocalPanelSource(
            panel_dir=panel_dir,
            asset_class="equity",
            frequency=data_cfg["data"].get("interval", "1d"),
            source_name="local_panel",
        )
    if source == "qlib":
        qcfg = source_cfg.get("qlib", {})
        return QlibSource(
            instruments=qcfg.get("instruments", "csi500"),
            start_date=qcfg.get("start_date", "2020-01-01"),
            end_date=qcfg.get("end_date", "2023-12-31"),
            dataset=qcfg.get("dataset", "Alpha158"),
            cache_dir=qcfg.get("cache_dir", "data/qlib_cache"),
            provider_uri=qcfg.get("provider_uri"),
            fallback_panel_dir=source_cfg.get("panel_dir"),
            prefer_local_panel=bool(qcfg.get("prefer_local_panel", False)),
        )
    raise ValueError(f"Unknown source: {source}")


def tensor_to_factor_frame(stock_data, tensor: object, name: str) -> pd.DataFrame:
    frame = stock_data.make_dataframe(tensor, columns=[name])
    return frame[name].unstack(level=1).astype(float)


def build_forward_returns(close_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close_df.shift(-horizon) / close_df - 1.0


def default_validation_config(source_name: str) -> ValidationConfig:
    if source_name == "crypto":
        return ValidationConfig(
            min_rank_ic=0.03,
            min_rank_icir=0.5,
            max_turnover=0.35,
            min_decay_halflife=3,
            max_pool_correlation=0.7,
            min_judge_score=0.3,
        )
    return ValidationConfig(
        min_rank_ic=0.02,
        min_rank_icir=0.3,
        max_turnover=0.5,
        min_decay_halflife=2,
        max_pool_correlation=0.7,
        min_judge_score=0.3,
    )


def get_combiner(combiner_type: str):
    if combiner_type == "equal":
        return EqualWeightCombiner()
    if combiner_type == "ic_weighted":
        return ICWeightedCombiner()
    if combiner_type == "ridge":
        return RidgeCombiner()
    raise ValueError(f"Unknown combiner: {combiner_type}")


def maybe_score_with_judge(
    expressions: list[str],
    factor_metrics: list[dict],
    use_judge: bool,
) -> tuple[list[JudgeResult], dict[str, float]]:
    if not use_judge:
        return [], {}

    judge = load_judge()
    store = PaperStore()
    store.load()
    retriever = PaperRetriever(store)
    judge_results: list[JudgeResult] = []
    judge_scores: dict[str, float] = {}

    for expr, row in zip(expressions, factor_metrics):
        matched = retriever.retrieve(
            factor_type=["momentum", "mean_reversion", "volatility", "liquidity"],
            top_k=3,
        )
        matched_dicts = [{"title": p.title, "abstract": p.mechanism} for p in matched]
        result = judge.score(expr, row["rank_ic"], matched_dicts)
        judge_results.append(result)
        judge_scores[expr] = result.interpretability_score
        logger.info("Judge: {} -> {:.2f}", expr[:80], result.interpretability_score)

    return judge_results, judge_scores


def run_pipeline(
    source: str | None = None,
    use_judge: bool = False,
    pool_path: str | None = None,
    combiner_type: str = "ic_weighted",
    data_config_path: str = "config/data_config.yaml",
    split: str = "test",
    output_dir: str = "out/pipeline_results",
    backtest_years: int = 5,
):
    """Run the alpha pipeline on a pre-computed expression pool."""
    if pool_path is None:
        raise ValueError("pool_path is required. Run factor mining first.")

    data_cfg = load_data_config(data_config_path)
    source_name = resolve_source_name(source, data_cfg)
    source_cfg = data_cfg.get("source", {})
    horizon = int(source_cfg.get("target_horizon", 8))
    bars_per_year = int(source_cfg.get("bars_per_year", 8760 if source_name == "crypto" else 252))

    logger.info("Step 1: Loading source={} with config={}", source_name, data_config_path)
    data_source = get_data_source(source_name, data_cfg)
    panel = data_source.load_panel()
    logger.info("Loaded panel: {} fields, {} symbols", len(panel), len(panel["close"].columns))

    logger.info("Step 2: Expanding engineered features for diagnostics")
    expanded = expand_features(panel)
    logger.info("Expanded to {} engineered features; sample={}", len(expanded), get_feature_names()[:5])

    processed_dir = (
        data_cfg["data"]["processed_dir"]
        if source_name == "crypto"
        else source_cfg.get("panel_dir", data_cfg["data"]["processed_dir"])
    )
    splits = create_data_splits(
        processed_dir,
        data_config_path,
        max_backtrack_days=100,
        max_future_days=max(5, horizon),
    )
    if split not in splits:
        raise ValueError(f"Unknown split: {split}")
    stock_data = splits[split]

    expressions, raw_weights = load_pool(pool_path)
    weights = normalize_weights(raw_weights)
    translator = BrainExpressionTranslator()
    logger.info("Step 3: Loaded {} expressions from {}", len(expressions), pool_path)

    target_expr = Ref(Feature(FeatureType.CLOSE), -horizon) / Feature(FeatureType.CLOSE) - 1
    calculator = PanelAlphaCalculator(stock_data, target_expr)
    parsed_exprs = [parse_expression(expr) for expr in expressions]

    close_df = panel["close"].loc[stock_data.make_dataframe(calculator.target, ["target"]).index.levels[0], stock_data.stock_ids].astype(float)
    forward_returns = build_forward_returns(close_df, horizon)
    next_bar_returns = build_forward_returns(close_df, 1)
    liquidity_field = "quote_volume" if "quote_volume" in panel else "volume"
    liquidity_df = panel[liquidity_field].loc[close_df.index, close_df.columns].astype(float)

    factor_frames: dict[str, pd.DataFrame] = {}
    factor_metrics: list[dict] = []
    translation_rows: list[dict] = []

    logger.info("Step 4: Evaluating factor expressions on {} split", split)
    for idx, (expr_str, expr, weight) in enumerate(zip(expressions, parsed_exprs, weights), start=1):
        factor_name = f"factor_{idx:02d}"
        factor_df = tensor_to_factor_frame(stock_data, calculator.evaluate_alpha(expr), factor_name)
        factor_frames[expr_str] = factor_df
        metrics = evaluate_factor(factor_df, forward_returns, min_observations=max(2, min(10, factor_df.shape[1])))
        translation = translator.translate(expr)
        factor_metrics.append(
            {
                "name": factor_name,
                "expression": expr_str,
                "worldquant_expression": translation.worldquant_expression,
                "weight": weight,
                "ic": metrics.ic_mean,
                "rank_ic": metrics.rank_ic_mean,
                "icir": metrics.icir,
                "rank_icir": metrics.rank_icir,
            }
        )
        translation_rows.append(
            {
                "name": factor_name,
                "expression": expr_str,
                "worldquant_expression": translation.worldquant_expression,
                "translation_supported": translation.supported,
                "translation_notes": " | ".join(translation.notes),
            }
        )

    judge_results, judge_scores = maybe_score_with_judge(expressions, factor_metrics, use_judge)

    logger.info("Step 5: Validation gate")
    gate = ValidationGate(default_validation_config(source_name))
    accepted_exprs: list[str] = []
    accepted_frames: dict[str, pd.DataFrame] = {}
    validation_rows: list[dict] = []

    for row in factor_metrics:
        expr_str = row["expression"]
        factor_df = factor_frames[expr_str]
        existing_pool = list(accepted_frames.values())
        result = gate.validate(
            factor_df,
            next_bar_returns,
            existing_pool=existing_pool,
            judge_score=judge_scores.get(expr_str) if use_judge else None,
        )
        validation_rows.append(
            {
                **row,
                "passed": result.passed,
                "failures": "; ".join(result.failures),
                **result.metrics,
            }
        )
        if result.passed:
            accepted_exprs.append(expr_str)
            accepted_frames[expr_str] = factor_df

    if not accepted_frames:
        logger.warning("No factors passed validation; falling back to top 1 by |RankIC|")
        best = max(factor_metrics, key=lambda x: abs(x["rank_ic"]))
        accepted_exprs = [best["expression"]]
        accepted_frames = {best["expression"]: factor_frames[best["expression"]]}

    logger.info("Accepted {}/{} factors", len(accepted_frames), len(expressions))

    logger.info("Step 6: Fit combiner={}", combiner_type)
    combiner = get_combiner(combiner_type)
    combiner.fit(accepted_frames, next_bar_returns)
    combined_signal = combiner.combine(accepted_frames)
    combined_metrics = evaluate_factor(
        combined_signal,
        forward_returns,
        min_observations=max(2, min(10, combined_signal.shape[1])),
    )
    combined_decay = compute_ic_decay(
        combined_signal,
        close_df,
        horizons=[1, 2, 4, 8, 24] if bars_per_year > 1000 else [1, 2, 5, 10, 20],
        min_observations=max(2, min(10, combined_signal.shape[1])),
    )

    if backtest_years and backtest_years > 0:
        cutoff = pd.to_datetime(combined_signal.index.max()) - pd.DateOffset(years=backtest_years)
        combined_signal = combined_signal.loc[pd.to_datetime(combined_signal.index) >= cutoff]
        next_bar_returns = next_bar_returns.loc[combined_signal.index, combined_signal.columns]
        liquidity_df = liquidity_df.loc[combined_signal.index, combined_signal.columns]

    n_leg = max(1, combined_signal.shape[1] // 3)
    backtest = long_short_backtest(
        combined_signal,
        next_bar_returns,
        n_long=n_leg,
        n_short=n_leg,
        liquidity_filter=liquidity_df,
        transaction_cost_bps=5.0,
        bars_per_year=bars_per_year,
    )

    logger.info("Step 7: Saving outputs")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submit_dir = out_dir / "submit_ready_worldquant"
    submit_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(factor_metrics).to_csv(out_dir / "factor_metrics.csv", index=False)
    pd.DataFrame(translation_rows).to_csv(out_dir / "worldquant_expression_map.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(out_dir / "validation_results.csv", index=False)
    combined_decay.to_csv(out_dir / "combined_decay.csv", index=False)
    combined_signal.to_csv(out_dir / "combined_signal.csv")
    backtest["equity_curve"].to_csv(out_dir / "combined_equity_curve.csv", header=True)
    backtest["daily_metrics"].to_csv(out_dir / "combined_backtest_daily.csv", index=True)
    backtest["yearly_metrics"].to_csv(out_dir / "combined_backtest_yearly.csv", index=False)
    pd.DataFrame(
        [{"metric": k, "value": v} for k, v in backtest["aggregate_metrics"].items() if k != "Years"]
    ).to_csv(out_dir / "aggregate_data.csv", index=False)
    backtest["yearly_metrics"].to_csv(out_dir / "yearly_data.csv", index=False)

    variant_rows: list[dict] = []
    for row in translation_rows:
        template = render_brain_submission_template(
            alpha_name=row["name"],
            base_expression=row["worldquant_expression"],
            local_expression=row["expression"],
        )
        (submit_dir / f"{row['name']}.txt").write_text(template, encoding="utf-8")
        for variant in build_brain_variants(row["worldquant_expression"]):
            variant_rows.append(
                {
                    "name": row["name"],
                    "variant": variant.label,
                    "expression": variant.expression,
                    "notes": " | ".join(variant.notes),
                }
            )
    pd.DataFrame(variant_rows).to_csv(out_dir / "brain_expression_variants.csv", index=False)

    fig = plot_backtest(
        backtest["equity_curve"],
        backtest["metrics"],
        title=f"Pipeline Backtest ({source_name}, {split})",
    )
    fig.savefig(out_dir / "combined_backtest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_worldquant_style_panels(
        backtest["daily_metrics"],
        title=f"Pipeline WorldQuant-Style Panels ({source_name}, {split})",
    )
    fig.savefig(out_dir / "combined_worldquant_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "source": source_name,
        "data_config": data_config_path,
        "split": split,
        "pool_path": pool_path,
        "horizon": horizon,
        "bars_per_year": bars_per_year,
        "n_candidates": len(expressions),
        "n_accepted": len(accepted_frames),
        "accepted_expressions": accepted_exprs,
        "combiner": combiner_type,
        "backtest_years": backtest_years,
        "combined_metrics": {
            "ic": combined_metrics.ic_mean,
            "rank_ic": combined_metrics.rank_ic_mean,
            "icir": combined_metrics.icir,
            "rank_icir": combined_metrics.rank_icir,
        },
        "backtest_metrics": backtest["metrics"],
        "aggregate_data": backtest["aggregate_metrics"],
        "yearly_data": backtest["yearly_metrics"].to_dict(orient="records"),
        "worldquant_expression_map": translation_rows,
    }

    if judge_results:
        summary["judge_thresholded"] = len(accepted_frames)
        with open(out_dir / "judge_results.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "expression": r.expression,
                        "nl_description": r.nl_description,
                        "interpretability_score": r.interpretability_score,
                        "economic_narrative": r.economic_narrative,
                        "matched_papers": r.matched_papers,
                        "reasoning": r.reasoning,
                    }
                    for r in judge_results
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Pipeline summary saved to {}", out_dir / "summary.json")


def main():
    parser = argparse.ArgumentParser(description="Alpha Harness Pipeline")
    parser.add_argument("--source", default=None, choices=["crypto", "qlib", "panel"], help="Data source override")
    parser.add_argument("--judge", action="store_true", default=False, help="Enable LLM judge scoring")
    parser.add_argument("--no-judge", dest="judge", action="store_false", help="Disable LLM judge scoring")
    parser.add_argument("--evaluate-pool", type=str, required=True, help="Path to pre-computed factor pool JSON")
    parser.add_argument("--combiner", default="ic_weighted", choices=["equal", "ic_weighted", "ridge"])
    parser.add_argument("--data-config", default="config/data_config.yaml", help="Data config path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Evaluation split")
    parser.add_argument("--output-dir", default="out/pipeline_results", help="Output directory")
    parser.add_argument("--backtest-years", type=int, default=5, help="Limit backtest output to recent N years; <=0 uses all history")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        use_judge=args.judge,
        pool_path=args.evaluate_pool,
        combiner_type=args.combiner,
        data_config_path=args.data_config,
        split=args.split,
        output_dir=args.output_dir,
        backtest_years=args.backtest_years,
    )


if __name__ == "__main__":
    main()
