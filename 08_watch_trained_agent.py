"""
Watch a trained model.

Examples:
    python 08_watch_agent.py --algorithm DQN
    python 08_watch_agent.py --algorithm PPO --disturbance low_noise
    python 08_watch_agent.py --algorithm DQN --disturbance combined
"""
import argparse
from config import DEFAULT_SEED, MODELS_DIR
from core import load_model, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["DQN", "PPO"], default="PPO")
    parser.add_argument("--disturbance", choices=["none", "low_noise", "high_noise", "dropout", "combined"], default="none")
    args = parser.parse_args()

    model_path = MODELS_DIR / f"baseline_{args.algorithm.lower()}.zip"
    model = load_model(args.algorithm, model_path)

    disturbance_kwargs = {}
    if args.disturbance == "low_noise":
        disturbance_kwargs = {"obs_noise_std": 0.02}
    elif args.disturbance == "high_noise":
        disturbance_kwargs = {"obs_noise_std": 0.05}
    elif args.disturbance == "dropout":
        disturbance_kwargs = {"action_dropout_prob": 0.10}
    elif args.disturbance == "combined":
        disturbance_kwargs = {"obs_noise_std": 0.02, "action_dropout_prob": 0.05}

    env = make_env(seed=DEFAULT_SEED, render_mode="human", **disturbance_kwargs)
    obs, info = env.reset(seed=DEFAULT_SEED)
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)

    env.close()
    print(f"Episode reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
