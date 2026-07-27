"""
Run reward-design sensitivity analysis.

This trains DQN and PPO with different reward shaping terms.

Run:
    python 04_reward_sensitivity.py
"""
import pandas as pd
from config import (
    DQN_BASE, LOGS_DIR, MODELS_DIR, N_EVAL_EPISODES, PPO_BASE,
    RESULTS_DIR, REWARD_CASES, REWARD_SENSITIVITY_TIMESTEPS, SEEDS
)
from core import evaluate_model, make_model, make_monitored_env, save_json, set_global_seed


def train_reward_case(algorithm, base_hp, reward_case, seed):
    case_name = reward_case["case"]
    reward_kwargs = {k: v for k, v in reward_case.items() if k != "case"}
    run_name = f"reward_{algorithm.lower()}_{case_name}_seed_{seed}"
    set_global_seed(seed)
    env = make_monitored_env(seed=seed, log_dir=LOGS_DIR / run_name, reward_kwargs=reward_kwargs)
    model = make_model(algorithm, env, base_hp.copy(), seed=seed, verbose=0)
    print(f"Training {run_name}")
    model.learn(total_timesteps=REWARD_SENSITIVITY_TIMESTEPS)
    model_path = MODELS_DIR / f"{run_name}.zip"
    model.save(model_path)
    env.close()

    # Evaluate with the same reward definition used during training.
    metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=seed, reward_kwargs=reward_kwargs)
    row = {
        "run_name": run_name,
        "algorithm": algorithm,
        "reward_case": case_name,
        "seed": seed,
        **reward_kwargs,
        "mean_reward": metrics["mean_reward"],
        "std_reward": metrics["std_reward"],
        "success_rate": metrics["success_rate"],
        "crash_rate": metrics["crash_rate"],
        "mean_episode_length": metrics["mean_episode_length"],
        "mean_fuel_proxy": metrics["mean_fuel_proxy"],
        "mean_abs_angle": metrics["mean_abs_angle"],
        "mean_speed": metrics["mean_speed"],
        "model_path": str(model_path),
    }
    save_json({**metrics, **row}, RESULTS_DIR / f"{run_name}_evaluation.json")
    return row


def main():
    rows = []
    for algorithm, hp in [("DQN", DQN_BASE), ("PPO", PPO_BASE)]:
        for reward_case in REWARD_CASES:
            for seed in SEEDS:
                rows.append(train_reward_case(algorithm, hp, reward_case, seed))

    individual = pd.DataFrame(rows)
    individual_path = RESULTS_DIR / "reward_sensitivity_individual_runs.csv"
    individual.to_csv(individual_path, index=False)

    summary = individual.groupby(["algorithm", "reward_case"], as_index=False).agg(
        mean_reward=("mean_reward", "mean"),
        reward_std_across_seeds=("mean_reward", "std"),
        mean_success_rate=("success_rate", "mean"),
        mean_crash_rate=("crash_rate", "mean"),
        mean_fuel_proxy=("mean_fuel_proxy", "mean"),
        mean_abs_angle=("mean_abs_angle", "mean"),
        mean_speed=("mean_speed", "mean"),
        n_runs=("seed", "count"),
    )
    summary_path = RESULTS_DIR / "reward_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary)
    print(f"Saved: {individual_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
