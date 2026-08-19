"""
Re-evaluate saved DQN and PPO baselines.
"""

import pandas as pd
from config import DEFAULT_SEED, MODELS_DIR, N_EVAL_EPISODES, RESULTS_DIR
from core import evaluate_model, load_model, save_json


def main():
    rows = []
    for algorithm in ["DQN", "PPO"]:
        model_path = MODELS_DIR / f"baseline_{algorithm.lower()}.zip"
        if not model_path.exists():
            print(f"Skipping {algorithm}: {model_path} not found")
            continue
        model = load_model(algorithm, model_path)
        metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=DEFAULT_SEED)
        save_json(metrics, RESULTS_DIR / f"baseline_{algorithm.lower()}_reevaluation.json")
        row = {"algorithm": algorithm, **{k: v for k, v in metrics.items() if not k.startswith("episode_")}}
        rows.append(row)

    df = pd.DataFrame(rows)
    output = RESULTS_DIR / "baseline_algorithm_comparison.csv"
    df.to_csv(output, index=False)
    print(df)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
