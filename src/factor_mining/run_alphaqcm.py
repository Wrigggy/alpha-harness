"""Wrapper script to launch AlphaQCM training with crypto data.

AlphaQCM uses distributional RL (IQN/QR-DQN) instead of PPO, providing
better exploration via quantile-based uncertainty estimation.
"""

import sys
from pathlib import Path

import torch
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "alphaqcm"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "alphagen"))

from src.data_adapter.to_alphagen_format import CryptoStockData, CryptoAlphaCalculator, create_data_splits
from src.factor_mining.factor_pool_manager import FactorPoolManager, AlphaFactor
from src.utils.device import get_device


def load_config(config_path: str = "config/alphaqcm_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_alphaqcm(
    config_path: str = "config/alphaqcm_config.yaml",
    data_config_path: str = "config/data_config.yaml",
    small_scale: bool = False,
):
    """Run AlphaQCM factor mining.

    Args:
        config_path: Path to AlphaQCM config.
        data_config_path: Path to data config.
        small_scale: If True, use small-scale test parameters.
    """
    cfg = load_config(config_path)

    device_str = cfg.get("device", "auto")
    device = get_device(device_str)

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

    train_calc = CryptoAlphaCalculator(train_data)
    val_calc = CryptoAlphaCalculator(val_data)

    pool = FactorPoolManager(
        pool_size=cfg["pool_size"],
        ic_threshold=cfg["ic_threshold"],
        correlation_threshold=cfg["correlation_threshold"],
    )

    logger.info(f"Starting AlphaQCM training: {cfg['n_episodes']} episodes on {device}")
    logger.info(f"Model: {cfg['model']} | Quantiles: {cfg['n_quantiles']} | Lambda: {cfg['std_lambda']}")
    logger.info(f"Train data: {train_data.n_days} bars x {train_data.n_stocks} symbols")

    # NOTE: The actual AlphaQCM training loop integrates with:
    #   - AlphaQCM's QCM agent (distributional Q-learning)
    #   - Same AlphaGen environment, but different RL algorithm
    #
    # Key differences from AlphaGen:
    #   - Uses IQN (Implicit Quantile Network) or QR-DQN instead of PPO
    #   - Action selection uses Q-value + variance-based exploration bonus
    #   - Controlled by std_lambda: higher = more exploration
    #
    # Example integration (uncomment when external repos are available):
    #
    # from alphagen.rl.env import AlphaEnv
    # from AlphaQCM.agent import QCMAgent  # or similar import
    #
    # env = AlphaEnv(
    #     stock_data=train_data,
    #     calculator=train_calc,
    #     pool=pool,
    #     device=device,
    # )
    #
    # agent = QCMAgent(
    #     env=env,
    #     model_type=cfg["model"],
    #     n_quantiles=cfg["n_quantiles"],
    #     std_lambda=cfg["std_lambda"],
    #     learning_rate=cfg["learning_rate"],
    #     batch_size=cfg["batch_size"],
    #     device=device,
    # )
    #
    # agent.train(n_episodes=cfg["n_episodes"], eval_frequency=cfg["eval_frequency"])

    logger.warning(
        "AlphaQCM training loop requires external repos. Run:\n"
        "  git clone https://github.com/RL-MLDM/alphagen.git external/alphagen\n"
        "  git clone https://github.com/ZhuZhouFan/AlphaQCM.git external/alphaqcm"
    )

    # Save results
    output_dir = Path("data/factors")
    output_dir.mkdir(parents=True, exist_ok=True)
    pool.save(str(output_dir / "alphaqcm_pool.json"))

    logger.info("AlphaQCM training complete")
    logger.info(pool.summary())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaQCM factor mining")
    parser.add_argument("--small-scale", action="store_true", help="Use small-scale test parameters")
    parser.add_argument("--config", default="config/alphaqcm_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    args = parser.parse_args()

    run_alphaqcm(args.config, args.data_config, args.small_scale)
