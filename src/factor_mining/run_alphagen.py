"""Launch AlphaGen (Maskable PPO) training with crypto data.

Usage:
    python -m src.factor_mining.run_alphagen --small-scale
    python -m src.factor_mining.run_alphagen  # full training
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import yaml
from loguru import logger

# External repos
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet

from sb3_contrib import MaskablePPO

from src.data_adapter.to_alphagen_format import (
    load_crypto_stock_data,
    CryptoAlphaCalculator,
    create_data_splits,
)
from src.utils.device import get_device


def load_config(path: str = "config/alphagen_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_alphagen(
    config_path: str = "config/alphagen_config.yaml",
    data_config_path: str = "config/data_config.yaml",
    small_scale: bool = False,
):
    cfg = load_config(config_path)
    device = get_device(cfg.get("device", "auto"))

    if small_scale:
        small = cfg.get("small_scale", {})
        n_steps = small.get("n_episodes", 1000)
        logger.info(f"Small-scale mode: {n_steps} timesteps")
    else:
        n_steps = cfg["n_episodes"]

    # Load data splits
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)
    processed_dir = data_cfg["data"]["processed_dir"]

    splits = create_data_splits(
        processed_dir, data_config_path, device=device,
        max_backtrack_days=100, max_future_days=30,
    )
    data_train = splits["train"]
    data_valid = splits["val"]
    data_test = splits["test"]

    # Define target: 20-bar forward return (20 hours for 1H data)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    # Create calculators
    train_calc = CryptoAlphaCalculator(data_train, target)
    valid_calc = CryptoAlphaCalculator(data_valid, target)
    test_calc = CryptoAlphaCalculator(data_test, target)

    # Create factor pool
    pool = MseAlphaPool(
        capacity=cfg["pool_size"],
        calculator=train_calc,
        ic_lower_bound=cfg.get("ic_threshold"),
        l1_alpha=5e-3,
        device=device,
    )

    # Create RL environment
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    # Create PPO agent with LSTM feature extractor
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        gamma=cfg["gamma"],
        clip_range=cfg["ppo_clip"],
        ent_coef=cfg["entropy_coef"],
        verbose=1,
        device=device,
        tensorboard_log="./out/tensorboard",
    )

    logger.info(f"Starting AlphaGen PPO: {n_steps} timesteps on {device}")
    logger.info(f"Train: {data_train.n_days} bars x {data_train.n_stocks} symbols")

    # Train
    model.learn(total_timesteps=n_steps)

    # Evaluate on validation and test sets
    val_ic, val_ric = pool.test_ensemble(valid_calc)
    test_ic, test_ric = pool.test_ensemble(test_calc)
    logger.info(f"Validation IC={val_ic:.4f}, RankIC={val_ric:.4f}")
    logger.info(f"Test IC={test_ic:.4f}, RankIC={test_ric:.4f}")

    # Save results
    output_dir = Path("data/factors")
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_state = pool.to_json_dict()
    pool_state["val_ic"] = val_ic
    pool_state["val_ric"] = val_ric
    pool_state["test_ic"] = test_ic
    pool_state["test_ric"] = test_ric
    pool_state["timestamp"] = datetime.now().isoformat()

    with open(output_dir / "alphagen_pool.json", "w") as f:
        json.dump(pool_state, f, indent=2)

    logger.info(f"Pool saved: {pool.size} factors, best IC={pool.best_ic_ret:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-scale", action="store_true")
    parser.add_argument("--config", default="config/alphagen_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    args = parser.parse_args()
    run_alphagen(args.config, args.data_config, args.small_scale)
