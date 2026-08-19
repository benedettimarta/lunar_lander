"""
Evaluate trained baseline DQN and PPO models under realistic lunar-lander disturbances.

The disturbance cases are sensor/measurement noise, thruster command dropout,
and combined sensor plus actuator uncertainty.
"""

import pandas as pd
from config import DEFAULT_SEED, MODELS_DIR, N_EVAL_EPISODES, RESULTS_DIR, ROBUSTNESS_CASES
from core import evaluate_model, load_model, save_json


def main():
    rows = []
    for algorithm in ["DQN", "PPO"]:
        model_path = MODELS_DIR / f"baseline_{algorithm.lower()}.zip"
        if not model_path.exists():
            print(f"Skipping {algorithm}: missing {model_path}")
            continue
        model = load_model(algorithm, model_path)
        for case in ROBUSTNESS_CASES:
            case_name = case["case"]
            kwargs = {k: v for k, v in case.items() if k != "case"}
            print(f"Evaluating {algorithm} robustness case: {case_name}")
            metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=DEFAULT_SEED, **kwargs)
            row = {
                "algorithm": algorithm,
                "case": case_name,
                **kwargs,
                "mean_reward": metrics["mean_reward"],
                "std_reward": metrics["std_reward"],
                "success_rate": metrics["success_rate"],
                "crash_rate": metrics["crash_rate"],
                "mean_episode_length": metrics["mean_episode_length"],
                "mean_fuel_proxy": metrics["mean_fuel_proxy"],
                "mean_final_distance_to_pad": metrics["mean_final_distance_to_pad"],
                "mean_abs_angle": metrics["mean_abs_angle"],
                "mean_speed": metrics["mean_speed"],
            }
            rows.append(row)
            #save_json({**metrics, **row}, RESULTS_DIR / f"robustness_{algorithm.lower()}_{case_name}.json")

    df = pd.DataFrame(rows)
    output = RESULTS_DIR / "robustness_summary.csv"
    df.to_csv(output, index=False)
    print(df)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
