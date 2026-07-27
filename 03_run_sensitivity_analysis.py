"""
Run expanded sensitivity analysis for selected DQN hyperparameters.

This version:
1. varies seven hyperparameters,
2. repeats each case with multiple random seeds,
3. reports mean and standard deviation across seeds.

If it is too slow, reduce SEEDS or SENSITIVITY_TIMESTEPS in config.py.
"""

import pandas as pd
from stable_baselines3 import DQN

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_EXPLORATION_FINAL_EPS,
    DEFAULT_EXPLORATION_FRACTION,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NET_ARCH,
    DEFAULT_TARGET_UPDATE_INTERVAL,
    DEFAULT_TRAIN_FREQ,
    LOGS_DIR,
    MODELS_DIR,
    N_EVAL_EPISODES,
    RESULTS_DIR,
    SEEDS,
    SENSITIVITY_EXPERIMENTS,
    SENSITIVITY_TIMESTEPS,
)
from utils import (
    clean_value_for_filename,
    dqn_kwargs_from_config,
    evaluate_model,
    make_monitored_env,
    save_json,
    set_global_seed,
)


def default_hp():
    return {
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


def train_one_run(parameter_name, parameter_value, seed):
    hp = default_hp()
    hp[parameter_name] = parameter_value

    safe_value = clean_value_for_filename(parameter_value)
    run_name = f"{parameter_name}_{safe_value}_seed_{seed}"
    log_dir = LOGS_DIR / run_name

    set_global_seed(seed)
    env = make_monitored_env(seed=seed, log_dir=log_dir)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=seed,
        **dqn_kwargs_from_config(hp),
    )

    print(f"Training {run_name}...")
    model.learn(total_timesteps=SENSITIVITY_TIMESTEPS)

    model_path = MODELS_DIR / f"{run_name}.zip"
    model.save(model_path)
    env.close()

    metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=seed)
    result = {
        "run_name": run_name,
        "parameter": parameter_name,
        "value": str(parameter_value),
        "raw_value": parameter_value,
        "seed": seed,
        "total_timesteps": SENSITIVITY_TIMESTEPS,
        **hp,
        "mean_reward": metrics["mean_reward"],
        "std_reward": metrics["std_reward"],
        "min_reward": metrics["min_reward"],
        "max_reward": metrics["max_reward"],
        "success_rate": metrics["success_rate"],
        "model_path": str(model_path),
    }
    save_json({**metrics, **result}, RESULTS_DIR / f"{run_name}_evaluation.json")
    return result


def main():
    individual_results = []

    for experiment in SENSITIVITY_EXPERIMENTS:
        parameter = experiment["parameter"]
        for value in experiment["values"]:
            for seed in SEEDS:
                individual_results.append(train_one_run(parameter, value, seed))

    individual_df = pd.DataFrame(individual_results)
    individual_path = RESULTS_DIR / "sensitivity_individual_runs.csv"
    individual_df.to_csv(individual_path, index=False)

    summary_df = (
        individual_df
        .groupby(["parameter", "value"], as_index=False)
        .agg(
            mean_reward=("mean_reward", "mean"),
            reward_std_across_seeds=("mean_reward", "std"),
            mean_success_rate=("success_rate", "mean"),
            success_rate_std_across_seeds=("success_rate", "std"),
            n_runs=("seed", "count"),
        )
    )
    summary_path = RESULTS_DIR / "sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSensitivity analysis finished.")
    print(f"Individual runs saved to: {individual_path}")
    print(f"Summary saved to: {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()
