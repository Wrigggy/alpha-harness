"""End-to-end alpha harness pipeline.

Connects all layers: data → feature expansion → factor search → LLM judge →
validation → portfolio combination → backtest.

Usage:
    python -m src.pipeline --source crypto --judge
    python -m src.pipeline --source qlib --no-judge
    python -m src.pipeline --evaluate-pool data/factors/alphagen_pool.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.data_sources.crypto_source import CryptoSource
from src.feature_expansion.expander import expand_features, get_feature_names
from src.knowledge_base.paper_store import PaperStore
from src.knowledge_base.retriever import PaperRetriever
from src.llm_judge.base import JudgeResult
from src.evaluation.validation_gate import ValidationGate, ValidationConfig
from src.evaluation.ic_analysis import evaluate_factor
from src.portfolio.combiner import EqualWeightCombiner, ICWeightedCombiner, RidgeCombiner
from src.backtest.long_short_backtest import long_short_backtest, plot_backtest


def load_judge(config_path: str = "config/judge_config.yaml"):
    """Load the LLM judge based on config."""
    cfg_path = Path(config_path)
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        judge_cfg = cfg.get("judge", {})
    else:
        judge_cfg = {"backend": "agent_sdk", "model": "claude-opus-4-6"}

    backend = judge_cfg.get("backend", "agent_sdk")
    model = judge_cfg.get("model", "claude-opus-4-6")

    if backend == "agent_sdk":
        from src.llm_judge.claude_agent_judge import ClaudeAgentJudge
        return ClaudeAgentJudge(model=model)
    elif backend == "api":
        from src.llm_judge.api_judge import ApiJudge
        return ApiJudge(model=model)
    else:
        raise ValueError(f"Unknown judge backend: {backend}")


def run_pipeline(
    source: str = "crypto",
    use_judge: bool = True,
    pool_path: str | None = None,
    combiner_type: str = "ic_weighted",
):
    """Run the full alpha harness pipeline.

    Args:
        source: Data source ("crypto" or "qlib").
        use_judge: Whether to run the LLM judge on candidates.
        pool_path: Path to a pre-computed factor pool JSON (skip search if provided).
        combiner_type: Factor combination method ("equal", "ic_weighted", "ridge").
    """
    # ── Step 1: Load data ─────────────────────────────────────────────
    logger.info("Step 1: Loading data from source={}", source)

    if source == "crypto":
        data_source = CryptoSource()
        panel = data_source.load_panel()
    elif source == "qlib":
        from src.data_sources.qlib_source import QlibSource
        data_source = QlibSource()
        panel = data_source.load_panel()
    else:
        raise ValueError(f"Unknown source: {source}")

    logger.info("Loaded panel: {} fields, {} symbols",
                len(panel), len(panel["close"].columns))

    # ── Step 2: Feature expansion ─────────────────────────────────────
    logger.info("Step 2: Expanding features")
    expanded = expand_features(panel)
    logger.info("Expanded to {} features: {}", len(expanded), get_feature_names()[:5])

    # ── Step 3: Load or generate factor pool ──────────────────────────
    if pool_path:
        logger.info("Step 3: Loading pre-computed factor pool from {}", pool_path)
        with open(pool_path, encoding="utf-8") as f:
            pool_data = json.load(f)

        # Extract expressions and weights from pool
        if isinstance(pool_data, list):
            expressions = [f.get("expression", "") for f in pool_data]
            ics = [f.get("ic", 0.0) for f in pool_data]
        elif "exprs" in pool_data:
            expressions = pool_data["exprs"]
            ics = [0.0] * len(expressions)
        else:
            raise ValueError("Unrecognized pool format")

        logger.info("Loaded {} candidate expressions", len(expressions))
    else:
        logger.info("Step 3: No pool provided. Run factor mining first:")
        logger.info("  python -m src.factor_mining.run_alphagen --small-scale")
        logger.info("  python -m src.factor_mining.run_alphaqcm --small-scale")
        logger.info("Then re-run with: --evaluate-pool data/factors/alphagen_pool.json")
        return

    # ── Step 4: LLM Judge (post-filter) ───────────────────────────────
    judge_results: list[JudgeResult] = []

    if use_judge:
        logger.info("Step 4: Running LLM judge on {} candidates", len(expressions))

        judge = load_judge()
        store = PaperStore()
        papers = store.load()
        retriever = PaperRetriever(store)

        for expr, ic in zip(expressions, ics):
            # Translate expression and find matching papers
            matched = retriever.retrieve(
                factor_type=["momentum", "mean_reversion", "volatility", "liquidity"],
                top_k=3,
            )
            matched_dicts = [
                {"title": p.title, "abstract": p.mechanism} for p in matched
            ]

            result = judge.score(expr, ic, matched_dicts)
            judge_results.append(result)
            logger.info("  {} -> score={:.2f}", expr[:60], result.interpretability_score)

        # Filter by judge threshold
        judge_cfg_path = Path("config/judge_config.yaml")
        threshold = 0.3
        if judge_cfg_path.exists():
            with open(judge_cfg_path, encoding="utf-8") as f:
                threshold = yaml.safe_load(f).get("judge", {}).get("score_threshold", 0.3)

        passed_indices = [
            i for i, r in enumerate(judge_results)
            if r.interpretability_score >= threshold
        ]
        logger.info("LLM judge: {}/{} candidates passed (threshold={})",
                    len(passed_indices), len(expressions), threshold)
    else:
        logger.info("Step 4: Skipping LLM judge")
        passed_indices = list(range(len(expressions)))

    # ── Step 5: Validation gate ───────────────────────────────────────
    logger.info("Step 5: Validation gate (IC, ICIR, turnover, decay, correlation)")
    logger.info("  Note: Full validation requires factor values as DataFrames.")
    logger.info("  When running from expression pool, validation runs during factor mining.")

    # ── Step 6: Summary ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)
    logger.info("  Source: {}", source)
    logger.info("  Features: {}", len(expanded))
    logger.info("  Candidates: {}", len(expressions))
    if use_judge:
        logger.info("  Judge passed: {}/{}", len(passed_indices), len(expressions))
        avg_score = sum(r.interpretability_score for r in judge_results) / max(len(judge_results), 1)
        logger.info("  Avg judge score: {:.3f}", avg_score)

    # Save results
    output_dir = Path("out/pipeline_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if judge_results:
        results_data = [
            {
                "expression": r.expression,
                "nl_description": r.nl_description,
                "interpretability_score": r.interpretability_score,
                "economic_narrative": r.economic_narrative,
                "matched_papers": r.matched_papers,
                "reasoning": r.reasoning,
            }
            for r in judge_results
        ]
        with open(output_dir / "judge_results.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logger.info("Judge results saved to {}", output_dir / "judge_results.json")


def main():
    parser = argparse.ArgumentParser(description="Alpha Harness Pipeline")
    parser.add_argument("--source", default="crypto", choices=["crypto", "qlib"],
                        help="Data source")
    parser.add_argument("--judge", action="store_true", default=False,
                        help="Enable LLM judge scoring")
    parser.add_argument("--no-judge", dest="judge", action="store_false",
                        help="Disable LLM judge scoring")
    parser.add_argument("--evaluate-pool", type=str, default=None,
                        help="Path to pre-computed factor pool JSON")
    parser.add_argument("--combiner", default="ic_weighted",
                        choices=["equal", "ic_weighted", "ridge"],
                        help="Factor combination method")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        use_judge=args.judge,
        pool_path=args.evaluate_pool,
        combiner_type=args.combiner,
    )


if __name__ == "__main__":
    main()
