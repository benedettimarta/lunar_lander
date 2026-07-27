"""
Optional curriculum/fine-tuning experiment.

Idea:
1. Start from the nominal baseline policy.
2. Fine-tune it with mild sensor noise and thruster dropout.
3. Evaluate before/after fine-tuning under stronger uncertainty.

Run:
    python 06_curriculum_experiment.py

This is optional but useful for the report if you have time.
"""
import pandas as pd
from config import CURRICULUM_STAGE_TIMESTEPS, DEFAULT_SEED, DQN_BASE, LOGS_DIR, MODELS_DIR, N_EVAL_EPISODES, RESULTS_DIR
from core import evaluate_model, load_model, make_monitored_env, save_json


def fine_tune_algorithm(algorithm):
    base_model_path = MODELS_DIR / f"baseline_{algorithm.lower()}.zip"
    if not base_model_path.exists():
        print(f"Skipping {algorithm}: missing baseline {base_model_path}")
        return []

    rows = []
    model = load_model(algorithm, base_model_path)

    for case_name, kwargs in [
        ("before_finetune_low_uncertainty", {"obs_noise_std": 0.02, "action_dropout_prob": 0.05}),
        ("before_finetune_high_uncertainty", {"obs_noise_std": 0.05, "action_dropout_prob": 0.15}),
    ]:
        metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=DEFAULT_SEED, **kwargs)
        rows.append({"algorithm": algorithm, "stage": case_name, **kwargs, **{k: v for k, v in metrics.items() if not k.startswith("episode_")}})

    env = make_monitored_env(
        seed=DEFAULT_SEED,
        log_dir=LOGS_DIR / f"curriculum_{algorithm.lower()}_low_uncertainty",
        obs_noise_std=0.02,
        action_dropout_prob=0.05,
    )
    model.set_env(env)
    print(f"Fine-tuning {algorithm} under mild sensor/actuator uncertainty")
    model.learn(total_timesteps=CURRICULUM_STAGE_TIMESTEPS, reset_num_timesteps=False)
    out_path = MODELS_DIR / f"curriculum_{algorithm.lower()}_low_uncertainty.zip"
    model.save(out_path)
    env.close()

    for case_name, kwargs in [
        ("after_finetune_low_uncertainty", {"obs_noise_std": 0.02, "action_dropout_prob": 0.05}),
        ("after_finetune_high_uncertainty", {"obs_noise_std": 0.05, "action_dropout_prob": 0.15}),
    ]:
        metrics = evaluate_model(model, n_episodes=N_EVAL_EPISODES, seed=DEFAULT_SEED, **kwargs)
        rows.append({"algorithm": algorithm, "stage": case_name, **kwargs, **{k: v for k, v in metrics.items() if not k.startswith("episode_")}})

    save_json({"rows": rows}, RESULTS_DIR / f"curriculum_{algorithm.lower()}_results.json")
    return rows


def main():
    rows = []
    for algorithm in ["DQN", "PPO"]:
        rows.extend(fine_tune_algorithm(algorithm))
    df = pd.DataFrame(rows)
    output = RESULTS_DIR / "curriculum_summary.csv"
    df.to_csv(output, index=False)
    print(df)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
