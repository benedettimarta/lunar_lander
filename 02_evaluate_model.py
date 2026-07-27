"""
Evaluate a saved model and print results.

Run:
    python 02_evaluate_model.py
"""
from stable_baselines3 import DQN

from config import DEFAULT_SEED, MODELS_DIR, RESULTS_DIR, N_EVAL_EPISODES
from utils import evaluate_model, save_json


def main():
    model_path = MODELS_DIR / "baseline_dqn.zip"
    if not model_path.exists():
        raise FileNotFoundError("No baseline model found. Run 01_train_baseline.py first.")

    model = DQN.load(model_path)
    metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=DEFAULT_SEED)
    save_json(metrics, RESULTS_DIR / "baseline_dqn_re_evaluation.json")

    print("Evaluation results")
    print("------------------")
    print(f"Episodes:      {metrics['n_episodes']}")
    print(f"Mean reward:   {metrics['mean_reward']:.2f}")
    print(f"Std reward:    {metrics['std_reward']:.2f}")
    print(f"Min reward:    {metrics['min_reward']:.2f}")
    print(f"Max reward:    {metrics['max_reward']:.2f}")
    print(f"Success rate:  {100 * metrics['success_rate']:.1f}%")


if __name__ == "__main__":
    main()
