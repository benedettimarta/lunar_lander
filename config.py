"""
Shared configuration for the Lunar Lander RL assignment.

This upgraded version increases the assignment complexity by:
1. varying six DQN hyperparameters,
2. repeating each case over multiple random seeds,
3. evaluating robustness under wind disturbances.
"""
from pathlib import Path

ENV_ID = "LunarLander-v3"

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
LOGS_DIR = ROOT_DIR / "logs"

for folder in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    folder.mkdir(exist_ok=True)

# Default training settings
DEFAULT_TOTAL_TIMESTEPS = 300_000
SENSITIVITY_TIMESTEPS = 150_000
DEFAULT_SEED = 42
SEEDS = [42, 123]  # Increase to [42, 123, 999] if your computer can handle it.

# Default DQN hyperparameters
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_GAMMA = 0.99
DEFAULT_EXPLORATION_FRACTION = 0.20
DEFAULT_EXPLORATION_FINAL_EPS = 0.05
DEFAULT_BATCH_SIZE = 64
DEFAULT_BUFFER_SIZE = 100_000
DEFAULT_TARGET_UPDATE_INTERVAL = 1_000
DEFAULT_TRAIN_FREQ = 4
DEFAULT_NET_ARCH = [128, 128]

# Evaluation settings
N_EVAL_EPISODES = 50
SUCCESS_REWARD_THRESHOLD = 200.0

# Hyperparameter sensitivity analysis.
# This is one-at-a-time sensitivity: all parameters stay at the default except the one being varied.
SENSITIVITY_EXPERIMENTS = [
    {"parameter": "learning_rate", "values": [1e-4, 5e-4, 1e-3]},
    {"parameter": "gamma", "values": [0.90, 0.95, 0.99]},
    {"parameter": "exploration_final_eps", "values": [0.01, 0.05, 0.10]},
    {"parameter": "batch_size", "values": [32, 64, 128]},
    {"parameter": "buffer_size", "values": [50_000, 100_000, 200_000]},
    {"parameter": "target_update_interval", "values": [500, 1_000, 2_000]},
    {"parameter": "net_arch", "values": [[64, 64], [128, 128], [256, 256]]},
]

# Robustness environments used after training.
# These evaluate whether the trained controller generalizes to disturbances.
ROBUSTNESS_CASES = [
    {"case": "no_wind", "enable_wind": False, "wind_power": 0.0, "turbulence_power": 0.0},
    {"case": "light_wind", "enable_wind": True, "wind_power": 5.0, "turbulence_power": 0.5},
    {"case": "strong_wind", "enable_wind": True, "wind_power": 15.0, "turbulence_power": 1.5},
]
