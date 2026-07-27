"""
Run sensitivity analysis for selected DQN hyperparameters.

This trains several agents while changing one parameter at a time:
- learning rate
- discount factor gamma
- final exploration epsilon

Run:
    python 03_run_sensitivity_analysis.py

Note: This can take time. If it is too slow, reduce TOTAL_TIMESTEPS_PER_RUN.
"""

import pandas as pd
from stable_baselines3 import DQN

from config import (
    DEFAULT_EXPLORATION_FRACTION,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    N_EVAL_EPISODES,
)
from utils import evaluate_model, make_monitored_env, save_json, set_global_seed

# Use fewer timesteps for sensitivity analysis if you need quick results.
TOTAL_TIMESTEPS_PER_RUN = 100_000

EXPERIMENTS = [
    {
        "parameter": "learning_rate",
        "values": [1e-4, 5e-4, 1e-3],
    },
    {
        "parameter": "gamma",
        "values": [0.90, 0.95, 0.99],
    },
    {
        "parameter": "exploration_final_eps",
        "values": [0.01, 0.05, 0.10],
    },
]


def train_one_run(parameter_name, parameter_value, run_index):
    seed = DEFAULT_SEED + run_index
    set_global_seed(seed)

    learning_rate = DEFAULT_LEARNING_RATE
    gamma = DEFAULT_GAMMA
    exploration_final_eps = 0.05

    if parameter_name == "learning_rate":
        learning_rate = parameter_value
    elif parameter_name == "gamma":
        gamma = parameter_value
    elif parameter_name == "exploration_final_eps":
        exploration_final_eps = parameter_value
    else:
        raise ValueError(f"Unknown parameter: {parameter_name}")

    run_name = f"{parameter_name}_{parameter_value:g}"
    log_dir = LOGS_DIR / run_name
    env = make_monitored_env(seed=seed, log_dir=log_dir)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_fraction=DEFAULT_EXPLORATION_FRACTION,
        exploration_final_eps=exploration_final_eps,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        target_update_interval=1_000,
        train_freq=4,
        gradient_steps=1,
        verbose=0,
        seed=seed,
    )

    print(f"Training {run_name}...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_RUN)

    model_path = MODELS_DIR / f"{run_name}.zip"
    model.save(model_path)
    env.close()

    metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=seed)
    metrics_summary = {
        "run_name": run_name,
        "parameter": parameter_name,
        "value": parameter_value,
        "seed": seed,
        "total_timesteps": TOTAL_TIMESTEPS_PER_RUN,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "exploration_fraction": DEFAULT_EXPLORATION_FRACTION,
        "exploration_final_eps": exploration_final_eps,
        "mean_reward": metrics["mean_reward"],
        "std_reward": metrics["std_reward"],
        "min_reward": metrics["min_reward"],
        "max_reward": metrics["max_reward"],
        "success_rate": metrics["success_rate"],
        "model_path": str(model_path),
    }

    save_json({**metrics, **metrics_summary}, RESULTS_DIR / f"{run_name}_evaluation.json")
    return metrics_summary


def main():
    all_results = []
    run_index = 0

    for experiment in EXPERIMENTS:
        parameter = experiment["parameter"]
        for value in experiment["values"]:
            result = train_one_run(parameter, value, run_index)
            all_results.append(result)
            run_index += 1

    df = pd.DataFrame(all_results)
    output_path = RESULTS_DIR / "sensitivity_summary.csv"
    df.to_csv(output_path, index=False)

    print("\nSensitivity analysis finished.")
    print(f"Summary saved to: {output_path}")
    print(df[["parameter", "value", "mean_reward", "std_reward", "success_rate"]])


if __name__ == "__main__":
    main()
