"""
Run DQN and PPO hyperparameter sensitivity analysis.
If runtime is too high, reduce SEEDS or SENSITIVITY_TIMESTEPS in config.py.
"""

import pandas as pd
from config import (
    DQN_BASE, DQN_SENSITIVITY, LOGS_DIR, MODELS_DIR, N_EVAL_EPISODES,
    PPO_BASE, PPO_SENSITIVITY, RESULTS_DIR, SEEDS, SENSITIVITY_TIMESTEPS
)
from core import clean_value, evaluate_model, make_model, make_monitored_env, save_json, set_global_seed


def train_case(algorithm, base_hp, parameter, value, seed):
    hp = base_hp.copy()
    hp[parameter] = value
    safe_value = clean_value(value)
    run_name = f"sens_{algorithm.lower()}_{parameter}_{safe_value}_seed_{seed}"
    set_global_seed(seed)
    env = make_monitored_env(seed=seed, log_dir=LOGS_DIR / run_name)
    model = make_model(algorithm, env, hp, seed=seed, verbose=0)
    print(f"Training {run_name}")
    model.learn(total_timesteps=SENSITIVITY_TIMESTEPS)
    model_path = MODELS_DIR / f"{run_name}.zip"
    model.save(model_path)
    env.close()

    metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=seed)
    row = {
        "run_name": run_name,
        "algorithm": algorithm,
        "parameter": parameter,
        "value": str(value),
        "seed": seed,
        "total_timesteps": SENSITIVITY_TIMESTEPS,
        **hp,
        "mean_reward": metrics["mean_reward"],
        "std_reward": metrics["std_reward"],
        "success_rate": metrics["success_rate"],
        "crash_rate": metrics["crash_rate"],
        "mean_episode_length": metrics["mean_episode_length"],
        "mean_fuel_proxy": metrics["mean_fuel_proxy"],
        "mean_final_distance_to_pad": metrics["mean_final_distance_to_pad"],
        "mean_abs_angle": metrics["mean_abs_angle"],
        "mean_speed": metrics["mean_speed"],
        "model_path": str(model_path),
    }
    save_json({**metrics, **row}, RESULTS_DIR / f"{run_name}_evaluation.json")
    return row


def run_for_algorithm(algorithm, base_hp, experiments):
    rows = []
    for exp in experiments:
        for value in exp["values"]:
            for seed in SEEDS:
                rows.append(train_case(algorithm, base_hp, exp["parameter"], value, seed))
    return rows


def main():
    rows = []
    rows.extend(run_for_algorithm("DQN", DQN_BASE.copy(), DQN_SENSITIVITY))
    rows.extend(run_for_algorithm("PPO", PPO_BASE.copy(), PPO_SENSITIVITY))

    individual = pd.DataFrame(rows)
    individual_path = RESULTS_DIR / "sensitivity_individual_runs.csv"
    individual.to_csv(individual_path, index=False)

    summary = individual.groupby(["algorithm", "parameter", "value"], as_index=False).agg(
        mean_reward=("mean_reward", "mean"),
        reward_std_across_seeds=("mean_reward", "std"),
        mean_success_rate=("success_rate", "mean"),
        mean_crash_rate=("crash_rate", "mean"),
        mean_fuel_proxy=("mean_fuel_proxy", "mean"),
        mean_final_distance_to_pad=("mean_final_distance_to_pad", "mean"),
        n_runs=("seed", "count"),
    )
    summary_path = RESULTS_DIR / "sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary)
    print(f"Saved: {individual_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
