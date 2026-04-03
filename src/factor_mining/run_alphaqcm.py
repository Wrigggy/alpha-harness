"""Launch AlphaQCM (distributional RL) training with crypto data.

AlphaQCM uses its own fork of AlphaGen in external/alphaqcm/alphagen/.
We use that fork's AlphaPool and the QCM agents from fqf_iqn_qrdqn/.

IMPORTANT: This script must force-load AlphaQCM's alphagen fork BEFORE
any other module imports the upstream alphagen.

Usage:
    python -m src.factor_mining.run_alphaqcm --small-scale
    python -m src.factor_mining.run_alphaqcm --model iqn
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# CRITICAL: AlphaQCM's fork must be loaded first.
# Remove any cached upstream alphagen modules, then put QCM's path first.
_ROOT = Path(__file__).resolve().parents[2]
_QCM_PATH = str(_ROOT / "external" / "alphaqcm")
_ALPHAGEN_PATH = str(_ROOT / "external" / "alphagen")

# Remove cached alphagen modules so QCM's fork takes priority
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("alphagen"):
        del sys.modules[mod_name]

# QCM path MUST be before alphagen path
if _QCM_PATH in sys.path:
    sys.path.remove(_QCM_PATH)
sys.path.insert(0, _QCM_PATH)
if _ALPHAGEN_PATH in sys.path:
    sys.path.remove(_ALPHAGEN_PATH)
sys.path.insert(1, _ALPHAGEN_PATH)

import torch
import yaml
from loguru import logger

# Now import from QCM's fork — AlphaPool exists here
from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv

from fqf_iqn_qrdqn.agent import QRQCMAgent, IQCMAgent, FQCMAgent

from src.data_adapter.to_alphagen_format import (
    CryptoAlphaCalculator,
    create_data_splits,
)
from src.utils.device import get_device


def load_config(path: str = "config/alphaqcm_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_alphaqcm(
    config_path: str = "config/alphaqcm_config.yaml",
    data_config_path: str = "config/data_config.yaml",
    model_type: str = "qrdqn",
    seed: int = 0,
    std_lam: float = 1.0,
    pool_capacity: int = 20,
    small_scale: bool = False,
):
    cfg = load_config(config_path)
    device = get_device(cfg.get("device", "auto"))
    use_cuda = device.type == "cuda"

    # Load QCM agent config
    qcm_config_path = _ROOT / "external" / "alphaqcm" / "qcm_config" / f"{model_type}.yaml"
    with open(qcm_config_path) as f:
        agent_config = yaml.safe_load(f)

    if small_scale:
        agent_config["num_steps"] = cfg.get("small_scale", {}).get("n_episodes", 50_000)
        logger.info(f"Small-scale mode: {agent_config['num_steps']} steps")

    # Load data
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)
    processed_dir = data_cfg["data"]["processed_dir"]

    splits = create_data_splits(
        processed_dir, data_config_path, device=device,
        max_backtrack_days=100, max_future_days=30,
    )

    # Target: 20-bar forward return
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    train_calc = CryptoAlphaCalculator(splits["train"], target)
    valid_calc = CryptoAlphaCalculator(splits["val"], target)
    test_calc = CryptoAlphaCalculator(splits["test"], target)

    # AlphaQCM's own AlphaPool
    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=train_calc,
        ic_lower_bound=None,
        l1_alpha=5e-3,
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    # Log directory
    time_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = str(
        _ROOT / "data" / "qcm_logs"
        / f"pool_{pool_capacity}_QCM_{std_lam}"
        / f"{model_type}-seed{seed}-{time_str}"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create agent
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
    logger.info(f"Train: {splits['train'].n_days} bars x {splits['train'].n_stocks} symbols")
    logger.info(f"Tensorboard: {log_dir}")

    # Train
    agent.run()

    # Save pool results
    output_dir = Path("data/factors")
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_state = pool.to_dict()
    val_ic, val_ric = pool.test_ensemble(valid_calc)
    test_ic, test_ric = pool.test_ensemble(test_calc)

    pool_state["val_ic"] = val_ic
    pool_state["val_ric"] = val_ric
    pool_state["test_ic"] = test_ic
    pool_state["test_ric"] = test_ric
    pool_state["model"] = model_type
    pool_state["timestamp"] = datetime.now().isoformat()

    with open(output_dir / "alphaqcm_pool.json", "w") as f:
        json.dump(pool_state, f, indent=2)

    logger.info(f"Pool saved: {pool.size} factors, best IC={pool.best_ic_ret:.4f}")
    logger.info(f"Val IC={val_ic:.4f} RankIC={val_ric:.4f}")
    logger.info(f"Test IC={test_ic:.4f} RankIC={test_ric:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qrdqn", choices=["qrdqn", "iqn", "fqf"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--std-lam", type=float, default=1.0)
    parser.add_argument("--small-scale", action="store_true")
    parser.add_argument("--config", default="config/alphaqcm_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    args = parser.parse_args()
    run_alphaqcm(
        args.config, args.data_config,
        model_type=args.model, seed=args.seed,
        std_lam=args.std_lam, pool_capacity=args.pool,
        small_scale=args.small_scale,
    )
