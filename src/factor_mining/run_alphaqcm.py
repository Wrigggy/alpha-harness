"""Launch AlphaQCM (distributional RL) with crypto or CSI300 data.

Mirrors run_alphagen.py / run_alphagen_cn.py interface so the LLM warm-start
framework (idea_agent → warm_seeds.json → pool.force_load_exprs) plugs in
identically across both RL backends.

AlphaQCM uses its own fork of AlphaGen in external/alphaqcm/alphagen/.
We use that fork's AlphaPool and the QCM agents from fqf_iqn_qrdqn/.

IMPORTANT: This script must force-load AlphaQCM's alphagen fork BEFORE
any other module imports the upstream alphagen.

Usage:
    # Crypto, vanilla
    python -m src.factor_mining.run_alphaqcm --data-source crypto --small-scale

    # CSI300, compose-mode warm-start
    python -m src.factor_mining.run_alphaqcm \\
        --data-source cn --seed 42 \\
        --warm-seeds data/factors/warm_seeds_cn_compose_seed42.json \\
        --run-name B_warm_qcm_cn_seed42
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# CRITICAL: AlphaQCM's fork must be loaded first.
# Remove any cached upstream alphagen modules, then put QCM's path first.
_ROOT = Path(__file__).resolve().parents[2]
_QCM_PATH = str(_ROOT / "external" / "alphaqcm")
_ALPHAGEN_PATH = str(_ROOT / "external" / "alphagen")


def _setup_qcm_paths():
    """Remove cached upstream alphagen and ensure QCM fork has priority."""
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("alphagen"):
            del sys.modules[mod_name]
    if _QCM_PATH in sys.path:
        sys.path.remove(_QCM_PATH)
    sys.path.insert(0, _QCM_PATH)
    if _ALPHAGEN_PATH in sys.path:
        sys.path.remove(_ALPHAGEN_PATH)
    sys.path.insert(1, _ALPHAGEN_PATH)


_setup_qcm_paths()

import torch
import yaml
from loguru import logger

# Now import from QCM's fork
from alphagen.data.expression import Feature, Ref  # noqa: E402
from alphagen.config import OPERATORS  # noqa: E402
from alphagen.models.alpha_pool import AlphaPool  # noqa: E402
from alphagen.rl.env.wrapper import AlphaEnv  # noqa: E402
from alphagen_qlib.stock_data import FeatureType, StockData  # noqa: E402
from alphagen_qlib.calculator import QLibStockDataCalculator  # noqa: E402

from fqf_iqn_qrdqn.agent import QRQCMAgent, IQCMAgent, FQCMAgent  # noqa: E402

# QCM fork lacks parser.py — ship our own that targets QCM Expression objects
from src.factor_mining._qcm_parser import (  # noqa: E402
    ExpressionParser, ExpressionParsingError,
)
from src.utils.device import get_device  # noqa: E402


def _build_parser() -> ExpressionParser:
    return ExpressionParser(
        operators=OPERATORS,
        ignore_case=False,
        time_deltas_need_suffix=True,
        non_positive_time_deltas_allowed=False,
        feature_need_dollar_sign=True,
    )


def _initialize_qcm_qlib(qlib_dir: str) -> None:
    """Init qlib with OUR config path, bypassing QCM's hardcoded one."""
    import qlib
    from qlib.config import REG_CN
    expanded = os.path.expanduser(qlib_dir)
    qlib.init(provider_uri=expanded, region=REG_CN)
    # Mark QCM's StockData as already initialized so it skips its own qlib.init
    StockData._qlib_initialized = True
    logger.info(f"Qlib initialized at {expanded} (QCM fork)")


def load_config(path: str = "config/alphaqcm_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_warm_seed_exprs(path: str):
    """Re-parse warm seeds using the QCM fork's parser.

    We don't import src.factor_mining.idea_agent because that would re-import
    upstream alphagen modules and pollute the QCM-prioritized sys.modules.
    """
    parser = _build_parser()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    exprs = []
    for s in payload["seeds"]:
        try:
            exprs.append(parser.parse(s["expr"]))
        except ExpressionParsingError as e:
            logger.warning(f"Re-parse failed for {s.get('id', s.get('expr'))!r}: {e}")
    return exprs


def _build_data_layer(
    data_source: str,
    data_config_path: str,
    device: torch.device,
):
    """Returns (train_calc, valid_calc, test_calc, meta_dict)."""
    with open(data_config_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if data_source == "crypto":
        from src.data_adapter.to_alphagen_format import (
            CryptoAlphaCalculator, create_data_splits,
        )
        processed_dir = data_cfg["data"]["processed_dir"]
        splits = create_data_splits(
            processed_dir, data_config_path, device=device,
            max_backtrack_days=100, max_future_days=10,
        )
        close = Feature(FeatureType.CLOSE)
        target = Ref(close, -8) / close - 1
        return (
            CryptoAlphaCalculator(splits["train"], target),
            CryptoAlphaCalculator(splits["val"], target),
            CryptoAlphaCalculator(splits["test"], target),
            {
                "instruments": "crypto-universe",
                "segments": {k: [None, None] for k in ("train", "val", "test")},
                "n_train_bars": splits["train"].n_days,
                "n_train_stocks": splits["train"].n_stocks,
            },
        )

    if data_source == "cn":
        cn = data_cfg.get("cn", {})
        qlib_dir = cn.get("qlib_data_path", "~/.qlib/qlib_data/cn_data")
        instruments = cn.get("instruments", "csi300")
        segments = cn["segments"]
        _initialize_qcm_qlib(qlib_dir)

        def get_data(start: str, end: str) -> StockData:
            return StockData(
                instrument=instruments,
                start_time=start,
                end_time=end,
                max_backtrack_days=100,
                max_future_days=30,
                device=device,
            )

        data_train = get_data(*segments["train"])
        data_valid = get_data(*segments["val"])
        data_test = get_data(*segments["test"])

        # Match AlphaGen-CN target exactly: 10-day VWAP-to-VWAP forward return
        vwap = Feature(FeatureType.VWAP)
        target = Ref(vwap, -11) / Ref(vwap, -1) - 1

        return (
            QLibStockDataCalculator(data_train, target),
            QLibStockDataCalculator(data_valid, target),
            QLibStockDataCalculator(data_test, target),
            {
                "instruments": instruments,
                "segments": segments,
                "n_train_bars": data_train.n_days,
                "n_train_stocks": len(data_train._stock_ids),
            },
        )

    raise ValueError(f"Unknown data_source: {data_source}")


def run_alphaqcm(
    config_path: str = "config/alphaqcm_config.yaml",
    data_config_path: str = "config/data_config.yaml",
    model_type: str = "qrdqn",
    seed: int = 0,
    std_lam: float = 1.0,
    pool_capacity: int = 20,
    small_scale: bool = False,
    data_source: str = "crypto",
    warm_seeds_path: Optional[str] = None,
    run_name: Optional[str] = None,
):
    cfg = load_config(config_path)
    device = get_device(cfg.get("device", "auto"))
    use_cuda = device.type == "cuda"

    qcm_config_path = _ROOT / "config" / f"qcm_{model_type}.yaml"
    if not qcm_config_path.exists():
        qcm_config_path = _ROOT / "external" / "alphaqcm" / "qcm_config" / f"{model_type}.yaml"
    logger.info(f"QCM agent config: {qcm_config_path}")
    with open(qcm_config_path, encoding="utf-8") as f:
        agent_config = yaml.safe_load(f)

    agent_config["num_steps"] = cfg.get("n_episodes", 300_000)
    if small_scale:
        agent_config["num_steps"] = cfg.get("small_scale", {}).get("n_episodes", 50_000)
        logger.info(f"Small-scale mode: {agent_config['num_steps']} steps")

    logger.info(f"Data source: {data_source}")
    train_calc, valid_calc, test_calc, meta = _build_data_layer(
        data_source=data_source, data_config_path=data_config_path, device=device,
    )
    logger.info(
        f"Train: {meta['n_train_bars']} bars x {meta['n_train_stocks']} stocks"
    )

    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=train_calc,
        ic_lower_bound=None,
        l1_alpha=5e-3,
    )

    if warm_seeds_path:
        seed_exprs = _load_warm_seed_exprs(warm_seeds_path)
        logger.info(f"Loading {len(seed_exprs)} warm-start expressions into pool")
        pool.force_load_exprs(seed_exprs)
        val_ic_ws, _ = pool.test_ensemble(valid_calc)
        test_ic_ws, _ = pool.test_ensemble(test_calc)
        logger.info(
            f"Pool after warm-start: size={pool.size}, "
            f"val_IC={val_ic_ws:.4f}, test_IC={test_ic_ws:.4f}"
        )

    env = AlphaEnv(pool=pool, device=device, print_expr=True)
    if not hasattr(env, "pool"):
        env.pool = env.unwrapped.pool

    # Log directory
    time_str = datetime.now().strftime("%Y%m%d-%H%M")
    tag = run_name or f"alphaqcm_{data_source}_{model_type}_seed{seed}_{time_str}"
    log_dir = str(
        _ROOT / "data" / "qcm_logs"
        / f"pool_{pool_capacity}_QCM_{std_lam}"
        / tag
    )
    os.makedirs(log_dir, exist_ok=True)

    agent_cls = {"qrdqn": QRQCMAgent, "iqn": IQCMAgent, "fqf": FQCMAgent}[model_type]
    agent = agent_cls(
        env=env,
        valid_calculator=valid_calc,
        test_calculator=test_calc,
        log_dir=log_dir,
        seed=seed,
        std_lam=std_lam,
        cuda=use_cuda,
        **agent_config,
    )

    logger.info(f"Starting AlphaQCM ({model_type}): {agent_config['num_steps']} steps on {device}")
    logger.info(f"Logs: {log_dir}")

    agent.run()

    output_dir = Path("data/factors")
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_state = pool.to_dict()
    val_ic, val_ric = pool.test_ensemble(valid_calc)
    test_ic, test_ric = pool.test_ensemble(test_calc)
    pool_state.update({
        "val_ic": val_ic, "val_ric": val_ric,
        "test_ic": test_ic, "test_ric": test_ric,
        "model": model_type,
        "data_source": data_source,
        "instruments": meta["instruments"],
        "segments": meta["segments"],
        "seed": seed,
        "n_steps": agent_config["num_steps"],
        "warm_seeds_path": warm_seeds_path,
        "run_name": tag,
        "timestamp": datetime.now().isoformat(),
    })

    pool_filename = f"{tag}_pool.json" if run_name else f"alphaqcm_{data_source}_pool.json"
    with open(output_dir / pool_filename, "w") as f:
        json.dump(pool_state, f, indent=2)

    logger.info(f"Pool saved: {pool.size} factors, best IC={pool.best_ic_ret:.4f}")
    logger.info(f"Val IC={val_ic:.4f} RankIC={val_ric:.4f}")
    logger.info(f"Test IC={test_ic:.4f} RankIC={test_ric:.4f}")
    logger.info(f"Final pool: {output_dir / pool_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qrdqn", choices=["qrdqn", "iqn", "fqf"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--std-lam", type=float, default=1.0)
    parser.add_argument("--small-scale", action="store_true")
    parser.add_argument("--config", default="config/alphaqcm_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--data-source", choices=["crypto", "cn"], default="crypto")
    parser.add_argument("--warm-seeds", type=str, default=None,
                        help="Path to warm_seeds JSON from idea_agent (pick or compose mode)")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    run_alphaqcm(
        config_path=args.config,
        data_config_path=args.data_config,
        model_type=args.model,
        seed=args.seed,
        std_lam=args.std_lam,
        pool_capacity=args.pool,
        small_scale=args.small_scale,
        data_source=args.data_source,
        warm_seeds_path=args.warm_seeds,
        run_name=args.run_name,
    )
