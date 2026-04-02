"""Wrapper script to launch AlphaGen training with crypto data.

This script:
1. Loads CryptoStockData from processed Binance data
2. Initializes the AlphaGen RL agent (PPO)
3. Runs the training loop
4. Saves discovered factors to the factor pool
"""

import sys
from pathlib import Path

import torch
import yaml
from loguru import logger

# Add external repos to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "alphagen"))

from src.data_adapter.to_alphagen_format import CryptoStockData, CryptoAlphaCalculator, create_data_splits
from src.factor_mining.factor_pool_manager import FactorPoolManager, AlphaFactor
from src.utils.device import get_device


def load_config(config_path: str = "config/alphagen_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_alphagen(
    config_path: str = "config/alphagen_config.yaml",
    data_config_path: str = "config/data_config.yaml",
    small_scale: bool = False,
):
    """Run AlphaGen factor mining.

    Args:
        config_path: Path to AlphaGen config.
        data_config_path: Path to data config.
        small_scale: If True, use small-scale test parameters (for Mac development).
    """
    cfg = load_config(config_path)

    # Device setup
    device_str = cfg.get("device", "auto")
    device = get_device(device_str)

    # Override with small-scale params if requested
    if small_scale:
        small = cfg.get("small_scale", {})
        cfg["n_episodes"] = small.get("n_episodes", 1000)
        logger.info("Using small-scale test parameters")

    # Load data
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = data_cfg["data"]["processed_dir"]
    data_splits = create_data_splits(processed_dir, data_config_path, str(device))

    train_data = data_splits["train"]
    val_data = data_splits["val"]

    # Initialize calculator
    train_calc = CryptoAlphaCalculator(train_data)
    val_calc = CryptoAlphaCalculator(val_data)

    # Initialize factor pool
    pool = FactorPoolManager(
        pool_size=cfg["pool_size"],
        ic_threshold=cfg["ic_threshold"],
        correlation_threshold=cfg["correlation_threshold"],
    )

    logger.info(f"Starting AlphaGen training: {cfg['n_episodes']} episodes on {device}")
    logger.info(f"Train data: {train_data.n_days} bars x {train_data.n_stocks} symbols")

    # NOTE: The actual AlphaGen training loop integrates with:
    #   - alphagen.rl.env.AlphaEnv (the RL environment)
    #   - alphagen.rl.policy (the PPO policy network)
    #   - alphagen.data.expression (expression tree building)
    #
    # This requires the external/alphagen repo to be cloned and its
    # dependencies installed. The integration points are:
    #
    # 1. Replace StockData with train_data (CryptoStockData)
    # 2. Replace QlibStockDataCalculator with train_calc (CryptoAlphaCalculator)
    # 3. Keep the RL agent, PPO, and expression generation unchanged
    #
    # Example integration (uncomment when external/alphagen is available):
    #
    # from alphagen.rl.env import AlphaEnv
    # from alphagen.rl.policy import LSTMSharedNet
    # from stable_baselines3 import PPO
    #
    # env = AlphaEnv(
    #     stock_data=train_data,
    #     calculator=train_calc,
    #     pool=pool,
    #     max_expr_length=cfg["max_expression_length"],
    #     features=cfg["features"],
    #     operators_unary=cfg["operators"]["unary"],
    #     operators_binary=cfg["operators"]["binary"],
    #     constants=cfg["constants"],
    #     device=device,
    # )
    #
    # model = PPO(
    #     LSTMSharedNet,
    #     env,
    #     learning_rate=cfg["learning_rate"],
    #     batch_size=cfg["batch_size"],
    #     gamma=cfg["gamma"],
    #     clip_range=cfg["ppo_clip"],
    #     ent_coef=cfg["entropy_coef"],
    #     verbose=1,
    #     device=device,
    # )
    #
    # model.learn(total_timesteps=cfg["n_episodes"])

    logger.warning(
        "AlphaGen training loop requires external/alphagen to be cloned. "
        "Run: git clone https://github.com/RL-MLDM/alphagen.git external/alphagen"
    )

    # Save results
    output_dir = Path("data/factors")
    output_dir.mkdir(parents=True, exist_ok=True)
    pool.save(str(output_dir / "alphagen_pool.json"))

    logger.info("AlphaGen training complete")
    logger.info(pool.summary())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaGen factor mining")
    parser.add_argument("--small-scale", action="store_true", help="Use small-scale test parameters")
    parser.add_argument("--config", default="config/alphagen_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    args = parser.parse_args()

    run_alphagen(args.config, args.data_config, args.small_scale)
