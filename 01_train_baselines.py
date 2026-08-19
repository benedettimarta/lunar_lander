"""
Train baseline DQN and PPO models.
"""

from config import BASELINE_TIMESTEPS, DEFAULT_SEED, DQN_BASE, LOGS_DIR, MODELS_DIR, N_EVAL_EPISODES, PPO_BASE, RESULTS_DIR
from core import evaluate_model, make_model, make_monitored_env, save_json, set_global_seed


def train_baseline(algorithm, hp):
    seed = DEFAULT_SEED
    set_global_seed(seed)
    run_name = f"baseline_{algorithm.lower()}"
    env = make_monitored_env(seed=seed, log_dir=LOGS_DIR / run_name)
    model = make_model(algorithm, env, hp, seed=seed, verbose=1)
    model.learn(total_timesteps=BASELINE_TIMESTEPS)
    model_path = MODELS_DIR / f"{run_name}.zip"
    model.save(model_path)
    env.close()

    metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=seed)
    result = {
        "run_name": run_name,
        "algorithm": algorithm,
        "total_timesteps": BASELINE_TIMESTEPS,
        **hp,
        **metrics,
        "model_path": str(model_path),
    }
    save_json(result, RESULTS_DIR / f"{run_name}_evaluation.json")
    print(f"\n{algorithm} baseline saved to {model_path}")
    print(f"Mean reward: {metrics['mean_reward']:.2f}")
    print(f"Success rate: {100 * metrics['success_rate']:.1f}%")
    print(f"Crash rate: {100 * metrics['crash_rate']:.1f}%")


def main():
    train_baseline("DQN", DQN_BASE.copy())
    #train_baseline("PPO", PPO_BASE.copy())


if __name__ == "__main__":
    main()
