"""AlphaGen training on A-share CSI300 (Qlib bin data).

Mirrors src/factor_mining/run_alphagen.py exactly in interface but swaps the
data layer to alphagen_qlib.StockData + QLibStockDataCalculator. Universe and
date segments come from config/data_config.yaml -> `cn` section.

Target: 10-day VWAP-to-VWAP forward return  (Ref($vwap,-11) / Ref($vwap,-1) - 1)
Universe: csi300 (configurable via data_config.yaml)

Usage:
    python -m src.factor_mining.run_alphagen_cn \\
        --seed 42 --n-steps 100000 \\
        --warm-seeds data/factors/warm_seeds_cn_seed42.json \\
        --run-name B_warm_cn_seed42
"""

from __future__ import annotations

import os
import random
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import yaml
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.data.expression import Feature, Ref  # noqa: E402
from alphagen.models.linear_alpha_pool import MseAlphaPool, LinearAlphaPool  # noqa: E402
from alphagen.rl.env.wrapper import AlphaEnv  # noqa: E402
from alphagen.rl.policy import LSTMSharedNet  # noqa: E402
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib  # noqa: E402
from alphagen_qlib.calculator import QLibStockDataCalculator  # noqa: E402

from sb3_contrib import MaskablePPO  # noqa: E402

from src.utils.device import get_device  # noqa: E402


class AlphaGenCNCallback(BaseCallback):
    """Same metrics as the crypto callback, but works on QLibStockDataCalculator."""

    def __init__(self, save_path: str, test_calculators: List[QLibStockDataCalculator], verbose: int = 0):
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
        self.logger.record("pool/size", pool.size)
        self.logger.record("pool/significant", int((np.abs(pool.weights[:pool.size]) > 1e-4).sum()))
        self.logger.record("pool/best_ic_ret", pool.best_ic_ret)
        self.logger.record("pool/eval_cnt", pool.eval_cnt)

        n_days_total = sum(calc.data.n_days for calc in self.test_calculators)
        ic_mean, ric_mean = 0.0, 0.0
        for i, calc in enumerate(self.test_calculators, start=1):
            ic, ric = pool.test_ensemble(calc)
            weight = calc.data.n_days / n_days_total
            ic_mean += ic * weight
            ric_mean += ric * weight
            self.logger.record(f"test/ic_{i}", ic)
            self.logger.record(f"test/rank_ic_{i}", ric)
        self.logger.record("test/ic_mean", ic_mean)
        self.logger.record("test/rank_ic_mean", ric_mean)

        path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
        self.model.save(path)
        with open(f"{path}_pool.json", "w") as f:
            json.dump(self.pool.to_json_dict(), f)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str = "config/alphagen_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_alphagen_cn(
    config_path: str = "config/alphagen_config.yaml",
    data_config_path: str = "config/data_config.yaml",
    seed: Optional[int] = None,
    n_steps_override: Optional[int] = None,
    warm_seeds_path: Optional[str] = None,
    run_name: Optional[str] = None,
):
    cfg = load_config(config_path)
    device = get_device(cfg.get("device", "auto"))

    seed = seed if seed is not None else int(cfg.get("seed", 42))
    _set_global_seed(seed)
    logger.info(f"Global seed set to {seed}")

    n_steps = n_steps_override if n_steps_override is not None else cfg["n_episodes"]
    logger.info(f"Total PPO timesteps: {n_steps}")

    # ----- Data via Qlib -----
    with open(data_config_path, encoding="utf-8") as f:
        cn = yaml.safe_load(f).get("cn", {})
    qlib_dir = cn.get("qlib_data_path", "~/.qlib/qlib_data/cn_data")
    instruments = cn.get("instruments", "csi300")
    segments = cn["segments"]
    logger.info(f"Initializing Qlib from {qlib_dir} for {instruments}")
    initialize_qlib(qlib_dir)

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
    logger.info(
        f"Train: {data_train.n_days} bars x {data_train.n_stocks} stocks  "
        f"({segments['train'][0]} → {segments['train'][1]})"
    )

    # Target: 10-day VWAP-to-VWAP forward return
    vwap = Feature(FeatureType.VWAP)
    target = Ref(vwap, -11) / Ref(vwap, -1) - 1

    train_calc = QLibStockDataCalculator(data_train, target)
    valid_calc = QLibStockDataCalculator(data_valid, target)
    test_calc = QLibStockDataCalculator(data_test, target)

    # ----- Pool -----
    pool = MseAlphaPool(
        capacity=cfg["pool_size"],
        calculator=train_calc,
        ic_lower_bound=cfg.get("ic_threshold"),
        l1_alpha=5e-3,
        device=device,
    )

    if warm_seeds_path:
        from src.factor_mining.idea_agent import load_seed_expressions
        seed_exprs = load_seed_expressions(Path(warm_seeds_path))
        logger.info(f"Loading {len(seed_exprs)} warm-start expressions into pool")
        pool.force_load_exprs(seed_exprs)
        val_ic_ws, _ = pool.test_ensemble(valid_calc)
        test_ic_ws, _ = pool.test_ensemble(test_calc)
        logger.info(
            f"Pool after warm-start: size={pool.size}, "
            f"val_IC={val_ic_ws:.4f}, test_IC={test_ic_ws:.4f}"
        )

    # ----- RL env -----
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = run_name if run_name else f"alphagen_cn_{timestamp}"
    save_path = str(_ROOT / "out" / "results" / tag)
    tb_log_dir = str(_ROOT / "out" / "tensorboard")

    callback = AlphaGenCNCallback(
        save_path=save_path,
        test_calculators=[valid_calc, test_calc],
        verbose=1,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2, d_model=128, dropout=0.1, device=device,
            ),
        ),
        n_steps=cfg.get("n_steps", 128),
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 1.0),
        clip_range=cfg["ppo_clip"],
        ent_coef=cfg["entropy_coef"],
        seed=seed,
        verbose=1,
        device=device,
        tensorboard_log=tb_log_dir,
    )

    logger.info(f"Starting AlphaGen-CN PPO: {n_steps} timesteps on {device}")
    logger.info(f"Tensorboard: {tb_log_dir}")
    logger.info(f"Checkpoints : {save_path}")

    model.learn(total_timesteps=n_steps, callback=callback, tb_log_name=tag)

    val_ic, val_ric = pool.test_ensemble(valid_calc)
    test_ic, test_ric = pool.test_ensemble(test_calc)
    logger.info(f"Validation IC={val_ic:.4f}, RankIC={val_ric:.4f}")
    logger.info(f"Test       IC={test_ic:.4f}, RankIC={test_ric:.4f}")

    output_dir = Path("data/factors")
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_state = pool.to_json_dict()
    pool_state.update({
        "val_ic": val_ic,
        "val_ric": val_ric,
        "test_ic": test_ic,
        "test_ric": test_ric,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "n_steps": n_steps,
        "warm_seeds_path": warm_seeds_path,
        "run_name": tag,
        "data_source": "cn",
        "instruments": instruments,
        "segments": segments,
    })
    pool_filename = f"{tag}_pool.json" if run_name else "alphagen_cn_pool.json"
    with open(output_dir / pool_filename, "w") as f:
        json.dump(pool_state, f, indent=2)
    logger.info(f"Pool saved: {pool.size} factors, best IC={pool.best_ic_ret:.4f}")
    logger.info(f"Final pool: {output_dir / pool_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/alphagen_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--warm-seeds", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    run_alphagen_cn(
        config_path=args.config,
        data_config_path=args.data_config,
        seed=args.seed,
        n_steps_override=args.n_steps,
        warm_seeds_path=args.warm_seeds,
        run_name=args.run_name,
    )
