"""
Watch the trained baseline agent perform one landing.
"""

from stable_baselines3 import DQN

from config import DEFAULT_SEED, MODELS_DIR
from utils import make_lunar_lander_env


def main():
    model_path = MODELS_DIR / "baseline_dqn.zip"
    if not model_path.exists():
        raise FileNotFoundError("No baseline model found. Run 01_train_baseline.py first.")

    model = DQN.load(model_path)
    env = make_lunar_lander_env(seed=DEFAULT_SEED, render_mode="human")

    obs, info = env.reset(seed=DEFAULT_SEED)
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

    env.close()
    print(f"Episode reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
