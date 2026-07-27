"""
Shared environment, wrappers, training, and evaluation utilities.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import Wrapper, ObservationWrapper, ActionWrapper
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from config import ENV_ID, CRASH_REWARD_THRESHOLD, SUCCESS_REWARD_THRESHOLD


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class ObservationNoiseWrapper(ObservationWrapper):
    """
    Adds Gaussian noise to the observed state.

    This represents navigation/sensor uncertainty. The true environment dynamics
    are unchanged, but the policy receives a noisy state estimate.
    """
    def __init__(self, env, obs_noise_std=0.0):
        super().__init__(env)
        self.obs_noise_std = float(obs_noise_std)

    def observation(self, observation):
        if self.obs_noise_std <= 0.0:
            return observation
        noise = np.random.normal(0.0, self.obs_noise_std, size=observation.shape)
        return (observation + noise).astype(observation.dtype)


class ActionDropoutWrapper(ActionWrapper):
    """
    Randomly ignores non-zero thruster commands.

    LunarLander uses discrete actions: 0 = no engine, 1/3 = side engines,
    2 = main engine. With probability action_dropout_prob, a requested thruster
    command is replaced by action 0. This approximates actuator/command uncertainty.
    """
    def __init__(self, env, action_dropout_prob=0.0):
        super().__init__(env)
        self.action_dropout_prob = float(action_dropout_prob)

    def action(self, action):
        action_int = int(action)
        if action_int != 0 and np.random.random() < self.action_dropout_prob:
            return 0
        return action_int


class RewardShapingWrapper(Wrapper):
    """
    Optional reward-shaping wrapper for LunarLander.

    The original environment reward is retained, then additional penalties are added:
    - fuel_penalty penalizes firing engines,
    - angle_penalty penalizes large absolute lander angle,
    - velocity_penalty penalizes high horizontal/vertical velocity.

    This allows a reward-design sensitivity analysis without modifying Gymnasium internals.
    """
    def __init__(self, env, fuel_penalty=0.0, angle_penalty=0.0, velocity_penalty=0.0):
        super().__init__(env)
        self.fuel_penalty = float(fuel_penalty)
        self.angle_penalty = float(angle_penalty)
        self.velocity_penalty = float(velocity_penalty)
        self.last_action = 0
        self.episode_fuel_proxy = 0.0

    def reset(self, **kwargs):
        self.last_action = 0
        self.episode_fuel_proxy = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action_int = int(action)
        self.last_action = action_int

        # Discrete LunarLander actions: 0 none, 1 left, 2 main, 3 right.
        fuel_use = 1.0 if action_int != 0 else 0.0
        self.episode_fuel_proxy += fuel_use

        angle = abs(float(obs[4]))
        speed = float(np.sqrt(obs[2] ** 2 + obs[3] ** 2))

        shaped_reward = float(reward)
        shaped_reward -= self.fuel_penalty * fuel_use
        shaped_reward -= self.angle_penalty * angle
        shaped_reward -= self.velocity_penalty * speed

        info = dict(info)
        info["fuel_proxy_step"] = fuel_use
        info["episode_fuel_proxy"] = self.episode_fuel_proxy
        info["base_reward"] = float(reward)
        info["shaped_reward"] = shaped_reward
        return obs, shaped_reward, terminated, truncated, info


def make_env(
    seed: int | None = None,
    render_mode: str | None = None,
    enable_wind: bool = False,
    wind_power: float = 0.0,
    turbulence_power: float = 0.0,
    reward_kwargs: Dict[str, float] | None = None,
    obs_noise_std: float = 0.0,
    action_dropout_prob: float = 0.0,
):
    env_ids = [ENV_ID]
    if ENV_ID != "LunarLander-v2":
        env_ids.append("LunarLander-v2")

    last_error = None
    for env_id in env_ids:
        try:
            kwargs = {"render_mode": render_mode}
            if enable_wind:
                kwargs.update({
                    "enable_wind": enable_wind,
                    "wind_power": wind_power,
                    "turbulence_power": turbulence_power,
                })
            env = gym.make(env_id, **kwargs)
            if reward_kwargs is not None:
                env = RewardShapingWrapper(env, **reward_kwargs)
            if action_dropout_prob > 0.0:
                env = ActionDropoutWrapper(env, action_dropout_prob=action_dropout_prob)
            if obs_noise_std > 0.0:
                env = ObservationNoiseWrapper(env, obs_noise_std=obs_noise_std)
            if seed is not None:
                env.reset(seed=seed)
                env.action_space.seed(seed)
            return env
        except TypeError:
            try:
                env = gym.make(env_id, render_mode=render_mode)
                if reward_kwargs is not None:
                    env = RewardShapingWrapper(env, **reward_kwargs)
                if action_dropout_prob > 0.0:
                    env = ActionDropoutWrapper(env, action_dropout_prob=action_dropout_prob)
                if obs_noise_std > 0.0:
                    env = ObservationNoiseWrapper(env, obs_noise_std=obs_noise_std)
                if seed is not None:
                    env.reset(seed=seed)
                    env.action_space.seed(seed)
                return env
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not create LunarLander. Try: pip install gymnasium[box2d] stable-baselines3"
    ) from last_error


def make_monitored_env(seed=None, log_dir: Path | None = None, **env_kwargs):
    env = make_env(seed=seed, **env_kwargs)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(log_dir / "monitor.csv"))
    return Monitor(env)


def clean_value(value: Any) -> str:
    if isinstance(value, list):
        return "x".join(str(v) for v in value)
    return str(value).replace(".", "p").replace("-", "m")


def dqn_kwargs(hp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "learning_rate": hp["learning_rate"],
        "gamma": hp["gamma"],
        "exploration_fraction": hp["exploration_fraction"],
        "exploration_final_eps": hp["exploration_final_eps"],
        "buffer_size": hp["buffer_size"],
        "learning_starts": hp["learning_starts"],
        "batch_size": hp["batch_size"],
        "target_update_interval": hp["target_update_interval"],
        "train_freq": hp["train_freq"],
        "gradient_steps": hp["gradient_steps"],
        "policy_kwargs": {"net_arch": hp["net_arch"]},
    }


def ppo_kwargs(hp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "learning_rate": hp["learning_rate"],
        "gamma": hp["gamma"],
        "n_steps": hp["n_steps"],
        "batch_size": hp["batch_size"],
        "n_epochs": hp["n_epochs"],
        "clip_range": hp["clip_range"],
        "ent_coef": hp["ent_coef"],
        "gae_lambda": hp["gae_lambda"],
        "policy_kwargs": {"net_arch": hp["net_arch"]},
    }


def make_model(algorithm: str, env, hp: Dict[str, Any], seed: int, verbose: int = 0):
    algorithm = algorithm.upper()
    if algorithm == "DQN":
        return DQN("MlpPolicy", env, seed=seed, verbose=verbose, **dqn_kwargs(hp))
    if algorithm == "PPO":
        return PPO("MlpPolicy", env, seed=seed, verbose=verbose, **ppo_kwargs(hp))
    raise ValueError(f"Unknown algorithm: {algorithm}")


def load_model(algorithm: str, model_path: Path):
    algorithm = algorithm.upper()
    if algorithm == "DQN":
        return DQN.load(model_path)
    if algorithm == "PPO":
        return PPO.load(model_path)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def save_json(data: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def evaluate_model(model, n_episodes: int, seed: int, **env_kwargs) -> Dict[str, Any]:
    env = make_env(seed=seed, **env_kwargs)
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_successes: List[int] = []
    episode_crashes: List[int] = []
    episode_fuel: List[float] = []
    final_distance_to_pad: List[float] = []
    mean_abs_angle: List[float] = []
    mean_speed: List[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        fuel_proxy = 0.0
        angles = []
        speeds = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += float(reward)
            steps += 1
            fuel_proxy += 1.0 if action_int != 0 else 0.0
            angles.append(abs(float(obs[4])))
            speeds.append(float(np.sqrt(obs[2] ** 2 + obs[3] ** 2)))

        # Distance to the landing pad center is approximated by final x/y state.
        final_dist = float(np.sqrt(obs[0] ** 2 + obs[1] ** 2))

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(1 if total_reward >= SUCCESS_REWARD_THRESHOLD else 0)
        episode_crashes.append(1 if total_reward <= CRASH_REWARD_THRESHOLD else 0)
        episode_fuel.append(fuel_proxy)
        final_distance_to_pad.append(final_dist)
        mean_abs_angle.append(float(np.mean(angles)) if angles else 0.0)
        mean_speed.append(float(np.mean(speeds)) if speeds else 0.0)

    env.close()
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "success_rate": float(np.mean(episode_successes)),
        "crash_rate": float(np.mean(episode_crashes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_fuel_proxy": float(np.mean(episode_fuel)),
        "mean_final_distance_to_pad": float(np.mean(final_distance_to_pad)),
        "mean_abs_angle": float(np.mean(mean_abs_angle)),
        "mean_speed": float(np.mean(mean_speed)),
        "n_episodes": int(n_episodes),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
        "episode_crashes": episode_crashes,
        "episode_fuel_proxy": episode_fuel,
    }


def load_monitor_rewards(monitor_csv: Path) -> pd.DataFrame:
    if not monitor_csv.exists():
        raise FileNotFoundError(f"Monitor file not found: {monitor_csv}")
    df = pd.read_csv(monitor_csv, skiprows=1)
    df = df.rename(columns={"r": "reward", "l": "episode_length", "t": "time"})
    df["episode"] = np.arange(1, len(df) + 1)
    df["moving_average_reward"] = df["reward"].rolling(window=20, min_periods=1).mean()
    return df
