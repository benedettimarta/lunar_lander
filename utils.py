"""
Utility functions used by all scripts.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from config import ENV_ID


def set_global_seed(seed: int) -> None:
    """Set Python and NumPy seeds for more repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)


def make_lunar_lander_env(seed: int | None = None, render_mode: str | None = None):
    """
    Create the LunarLander environment.

    It first tries LunarLander-v3, then falls back to LunarLander-v2 in case
    your Gymnasium installation is older.
    """
    env_ids_to_try = [ENV_ID]
    if ENV_ID != "LunarLander-v2":
        env_ids_to_try.append("LunarLander-v2")

    last_error = None
    for env_id in env_ids_to_try:
        try:
            env = gym.make(env_id, render_mode=render_mode)
            if seed is not None:
                env.reset(seed=seed)
                env.action_space.seed(seed)
            return env
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not create LunarLander environment. Install dependencies with:\n"
        "pip install gymnasium[box2d] stable-baselines3 pandas matplotlib numpy"
    ) from last_error


def make_monitored_env(seed: int | None = None, log_dir: Path | None = None):
    """Create LunarLander wrapped with Monitor to record episode rewards."""
    env = make_lunar_lander_env(seed=seed)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(log_dir / "monitor.csv"))
    else:
        env = Monitor(env)
    return env


def evaluate_model(model, n_episodes: int = 50, seed: int = 42) -> Dict[str, Any]:
    """
    Evaluate a trained model.

    Returns average reward, standard deviation, success rate, and per-episode data.
    A successful landing is counted when episode reward >= 200.
    """
    rewards: List[float] = []
    lengths: List[int] = []
    successes: List[int] = []

    env = make_lunar_lander_env(seed=seed)

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(1 if total_reward >= 200.0 else 0)

    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "success_rate": float(np.mean(successes)),
        "n_episodes": int(n_episodes),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "episode_successes": successes,
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_monitor_rewards(monitor_csv: Path) -> pd.DataFrame:
    """
    Load Monitor CSV produced by Stable-Baselines3.
    The first line starts with # and is skipped by pandas.
    """
    if not monitor_csv.exists():
        raise FileNotFoundError(f"Monitor file not found: {monitor_csv}")
    df = pd.read_csv(monitor_csv, skiprows=1)
    df = df.rename(columns={"r": "reward", "l": "episode_length", "t": "time"})
    df["episode"] = np.arange(1, len(df) + 1)
    df["moving_average_reward"] = df["reward"].rolling(window=20, min_periods=1).mean()
    return df
