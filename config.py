"""
Shared configuration for the Lunar Lander RL assignment.
Edit values here if you want to change the experiment globally.
"""
from pathlib import Path

# Gymnasium environment. LunarLander-v3 is current in recent Gymnasium versions.
# If your installed version only has v2, utils.make_lunar_lander_env() will fall back automatically.
ENV_ID = "LunarLander-v3"

# Project folders
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
LOGS_DIR = ROOT_DIR / "logs"

for folder in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    folder.mkdir(exist_ok=True)

# Default training settings
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_SEED = 42

# Default DQN hyperparameters
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GAMMA = 0.99
DEFAULT_EXPLORATION_FRACTION = 0.20
DEFAULT_EXPLORATION_FINAL_EPS = 0.05

# Evaluation settings
N_EVAL_EPISODES = 50
SUCCESS_REWARD_THRESHOLD = 200.0  # Common LunarLander solved threshold
