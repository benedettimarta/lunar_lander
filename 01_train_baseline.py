"""
Train one baseline DQN agent on Lunar Lander.
"""

from stable_baselines3 import DQN

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_EXPLORATION_FINAL_EPS,
    DEFAULT_EXPLORATION_FRACTION,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NET_ARCH,
    DEFAULT_SEED,
    DEFAULT_TARGET_UPDATE_INTERVAL,
    DEFAULT_TOTAL_TIMESTEPS,
    DEFAULT_TRAIN_FREQ,
    LOGS_DIR,
    MODELS_DIR,
    N_EVAL_EPISODES,
    RESULTS_DIR,
)
from utils import dqn_kwargs_from_config, evaluate_model, make_monitored_env, save_json, set_global_seed


def main():
    set_global_seed(DEFAULT_SEED)

    run_name = "baseline_dqn"
    log_dir = LOGS_DIR / run_name
    env = make_monitored_env(seed=DEFAULT_SEED, log_dir=log_dir)

    hp = {
        "learning_rate": DEFAULT_LEARNING_RATE,
        "gamma": DEFAULT_GAMMA,
        "exploration_fraction": DEFAULT_EXPLORATION_FRACTION,
        "exploration_final_eps": DEFAULT_EXPLORATION_FINAL_EPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "buffer_size": DEFAULT_BUFFER_SIZE,
        "target_update_interval": DEFAULT_TARGET_UPDATE_INTERVAL,
        "train_freq": DEFAULT_TRAIN_FREQ,
        "net_arch": DEFAULT_NET_ARCH,
    }

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=DEFAULT_SEED,
        **dqn_kwargs_from_config(hp),
    )

    model.learn(total_timesteps=DEFAULT_TOTAL_TIMESTEPS)

    model_path = MODELS_DIR / f"{run_name}.zip"
    model.save(model_path)
    env.close()

    metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=DEFAULT_SEED)
    metrics.update({
        "run_name": run_name,
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS,
        **hp,
        "model_path": str(model_path),
    })
    save_json(metrics, RESULTS_DIR / f"{run_name}_evaluation.json")

    print("\nBaseline training finished.")
    print(f"Model saved to: {model_path}")
    print(f"Mean reward: {metrics['mean_reward']:.2f}")
    print(f"Success rate: {100 * metrics['success_rate']:.1f}%")


if __name__ == "__main__":
    main()
