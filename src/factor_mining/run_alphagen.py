"""Launch AlphaGen (Maskable PPO) training with crypto data.

Includes AlphaGen's CustomCallback for tensorboard logging of:
- pool/size, pool/best_ic_ret, pool/eval_cnt
- test/ic, test/rank_ic on validation and test sets
- Model checkpoints saved every rollout

Usage:
    python -m src.factor_mining.run_alphagen --small-scale
    python -m src.factor_mining.run_alphagen
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import torch
import yaml
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

# External repos
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.models.linear_alpha_pool import MseAlphaPool, LinearAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.rl.policy import LSTMSharedNet

from sb3_contrib import MaskablePPO

from src.data_adapter.to_alphagen_format import (
    CryptoAlphaCalculator,
    create_data_splits,
)
from src.utils.device import get_device


class AlphaGenCallback(BaseCallback):
    """Callback that logs pool metrics to tensorboard and saves checkpoints.

    Adapted from AlphaGen's original CustomCallback in scripts/rl.py.
    Logs at every rollout end:
    - pool/size: number of factors in the pool
    - pool/significant: factors with |weight| > 1e-4
    - pool/best_ic_ret: best ensemble IC on training data
    - pool/eval_cnt: total expression evaluations
    - test/ic_N, test/rank_ic_N: IC on each test calculator
    - test/ic_mean, test/rank_ic_mean: weighted average test IC
    """

    def __init__(
        self,
        save_path: str,
        test_calculators: List[CryptoAlphaCalculator],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

    @property
    def pool(self) -> LinearAlphaPool:
        return self.training_env.envs[0].unwrapped.pool

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pool = self.pool

        # Pool state metrics
        self.logger.record("pool/size", pool.size)
        self.logger.record("pool/significant", int((np.abs(pool.weights[:pool.size]) > 1e-4).sum()))
        self.logger.record("pool/best_ic_ret", pool.best_ic_ret)
        self.logger.record("pool/eval_cnt", pool.eval_cnt)

        # Test metrics on validation/test calculators
        n_days_total = sum(calc.n_days for calc in self.test_calculators)
        ic_mean, ric_mean = 0.0, 0.0
        for i, calc in enumerate(self.test_calculators, start=1):
            ic, ric = pool.test_ensemble(calc)
            weight = calc.n_days / n_days_total
            ic_mean += ic * weight
            ric_mean += ric * weight
            self.logger.record(f"test/ic_{i}", ic)
            self.logger.record(f"test/rank_ic_{i}", ric)
        self.logger.record("test/ic_mean", ic_mean)
        self.logger.record("test/rank_ic_mean", ric_mean)

        # Save checkpoint
        self._save_checkpoint()

    def _save_checkpoint(self):
        path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
        self.model.save(path)
        with open(f"{path}_pool.json", "w") as f:
            json.dump(self.pool.to_json_dict(), f)


def load_config(path: str = "config/alphagen_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
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
    with open(data_config_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    processed_dir = data_cfg["data"]["processed_dir"]

    splits = create_data_splits(
        processed_dir, data_config_path, device=device,
        max_backtrack_days=100, max_future_days=10,
    )
    data_train = splits["train"]
    data_valid = splits["val"]
    data_test = splits["test"]

    # Define target: 8-bar forward return (8 hours for 1H data)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -8) / close - 1

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

    # Output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = str(_ROOT / "out" / "results" / f"alphagen_{timestamp}")
    tb_log_dir = str(_ROOT / "out" / "tensorboard")

    # Callback for tensorboard logging and checkpoints
    callback = AlphaGenCallback(
        save_path=save_path,
        test_calculators=[valid_calc, test_calc],
        verbose=1,
    )

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
        n_steps=cfg.get("n_steps", 128),
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 1.0),
        clip_range=cfg["ppo_clip"],
        ent_coef=cfg["entropy_coef"],
        verbose=1,
        device=device,
        tensorboard_log=tb_log_dir,
    )

    logger.info(f"Starting AlphaGen PPO: {n_steps} timesteps on {device}")
    logger.info(f"Train: {data_train.n_days} bars x {data_train.n_stocks} symbols")
    logger.info(f"Tensorboard: {tb_log_dir}")
    logger.info(f"Checkpoints: {save_path}")

    # Train with callback
    model.learn(
        total_timesteps=n_steps,
        callback=callback,
        tb_log_name=f"alphagen_{timestamp}",
    )

    # Final evaluation
    val_ic, val_ric = pool.test_ensemble(valid_calc)
    test_ic, test_ric = pool.test_ensemble(test_calc)
    logger.info(f"Validation IC={val_ic:.4f}, RankIC={val_ric:.4f}")
    logger.info(f"Test IC={test_ic:.4f}, RankIC={test_ric:.4f}")

    # Save final results
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
