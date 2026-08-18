"""
Configuration for the advanced Lunar Lander RL project.

Project features:
- algorithm comparison: DQN vs PPO
- expanded hyperparameter sensitivity
- reward-shaping sensitivity
- robustness testing under sensor noise and thruster-command uncertainty
- multiple evaluation metrics
- optional short curriculum/fine-tuning experiment
"""

from pathlib import Path

ENV_ID = "LunarLander-v3"

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
LOGS_DIR = ROOT_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# General settings
DEFAULT_SEED = 42
SEEDS = [42, 123, 999]
N_EVAL_EPISODES = 200
SUCCESS_REWARD_THRESHOLD = 200.0
CRASH_REWARD_THRESHOLD = -100.0

# Training durations
BASELINE_TIMESTEPS = 300_000
SENSITIVITY_TIMESTEPS = 150_000
REWARD_SENSITIVITY_TIMESTEPS = 150_000
CURRICULUM_STAGE_TIMESTEPS = 150_000

# Baseline DQN hyperparameters
DQN_BASE = {
    "learning_rate": 1e-4, #5e-5
    "gamma": 0.995, # 0.995 GOOD
    "exploration_fraction": 0.20,
    "exploration_final_eps": 0.01,
    "buffer_size": 100_000,
    "learning_starts": 1_000,
    "batch_size": 128, # 64
    "target_update_interval": 1_000, #2000
    "train_freq": 4,
    "gradient_steps": 1,
    "net_arch": [128, 128],
}

# Baseline PPO hyperparameters
PPO_BASE = {
    "learning_rate": 5e-4, # 1e-4
    "gamma": 0.99,
    "n_steps": 4096, # 2048
    "batch_size": 64,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.01, # 0.001
    "gae_lambda": 0.95,
    "net_arch": [512, 512], # [256, 256]
}

# One-at-a-time hyperparameter sensitivity experiments.
# Algorithms are kept separate because DQN and PPO do not have identical parameters.
DQN_SENSITIVITY = [
    {"parameter": "learning_rate", "values": [5e-5, 1e-4, 2e-4]},
    {"parameter": "gamma", "values": [0.98, 0.99, 0.995]},
    {"parameter": "exploration_final_eps", "values": [0.005, 0.01, 0.02]},
    {"parameter": "batch_size", "values": [64, 128, 256]},
    {"parameter": "buffer_size", "values": [50_000, 75_000, 100_000]},
    {"parameter": "target_update_interval", "values": [500, 1_000, 2_000]},
    {"parameter": "net_arch", "values": [[64, 64], [128, 128], [256, 256]]},
]

PPO_SENSITIVITY = [
    {"parameter": "learning_rate", "values": [5e-4, 1e-3, 1.5e-3]},
    {"parameter": "gamma", "values": [0.99, 0.999, 0.9995]},
    {"parameter": "clip_range", "values": [0.2, 0.3, 0.4]},
    {"parameter": "ent_coef", "values": [0.001, 0.01, 0.02]},
    {"parameter": "n_steps", "values": [1024, 2048, 4096]},
    {"parameter": "batch_size", "values": [32, 64, 128]},
    {"parameter": "net_arch", "values": [[128, 128], [256, 256], [512, 512]]},
]

# Reward-shaping sensitivity.
# These add extra shaping terms to the original LunarLander reward.
REWARD_CASES = [
    {
        "case": "original_reward",
        "fuel_penalty": 0.0,
        "angle_penalty": 0.0,
        "velocity_penalty": 0.0,
    },
    {
        "case": "fuel_saving",
        "fuel_penalty": 0.03,
        "angle_penalty": 0.0,
        "velocity_penalty": 0.0,
    },
    {
        "case": "attitude_stability",
        "fuel_penalty": 0.0,
        "angle_penalty": 0.10,
        "velocity_penalty": 0.0,
    },
    {
        "case": "soft_landing",
        "fuel_penalty": 0.0,
        "angle_penalty": 0.0,
        "velocity_penalty": 0.10,
    },
    {
        "case": "combined_shaping",
        "fuel_penalty": 0.02,
        "angle_penalty": 0.05,
        "velocity_penalty": 0.05,
    },
]

# Robustness evaluation cases. These are physically meaningful for lunar landing.
# The controller is trained in the nominal environment and evaluated under:
# - measurement uncertainty, representing imperfect navigation sensors;
# - thruster command dropout, representing actuator/command uncertainty;
# - a combined sensor and actuator uncertainty case.
ROBUSTNESS_CASES = [
    {"case": "nominal", "obs_noise_std": 0.0, "action_dropout_prob": 0.0},
    {"case": "low_sensor_noise", "obs_noise_std": 0.02, "action_dropout_prob": 0.0},
    {"case": "high_sensor_noise", "obs_noise_std": 0.05, "action_dropout_prob": 0.0},
    {"case": "low_thruster_dropout", "obs_noise_std": 0.0, "action_dropout_prob": 0.05},
    {"case": "high_thruster_dropout", "obs_noise_std": 0.0, "action_dropout_prob": 0.15},
    {"case": "combined_noise_dropout", "obs_noise_std": 0.02, "action_dropout_prob": 0.05},
]
