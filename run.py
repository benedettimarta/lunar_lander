"""
Simple launcher for the project.

Run:
    python run.py
"""
import subprocess
import sys

OPTIONS = {
    "1": ("Train DQN and PPO baselines", "01_train_baselines.py"),
    "2": ("Evaluate baselines", "02_evaluate_baselines.py"),
    "3": ("Run DQN/PPO hyperparameter sensitivity", "03_sensitivity_analysis.py"),
    "4": ("Run reward-design sensitivity", "04_reward_sensitivity.py"),
    "5": ("Evaluate robustness under sensor/actuator uncertainty", "05_robustness_evaluation.py"),
    "6": ("Optional curriculum/fine-tuning experiment", "06_curriculum_experiment.py"),
    "7": ("Generate plots", "07_plot_results.py"),
    "8": ("Watch DQN baseline", "08_watch_agent.py --algorithm DQN"),
    "9": ("Watch PPO baseline", "08_watch_agent.py --algorithm PPO"),
}


def main():
    print("\nAdvanced Lunar Lander RL Project")
    print("--------------------------------")
    for key, (label, _) in OPTIONS.items():
        print(f"{key}. {label}")
    choice = input("\nChoose an option: ").strip()
    if choice not in OPTIONS:
        print("Invalid option")
        return
    command = OPTIONS[choice][1].split()
    subprocess.run([sys.executable] + command, check=True)


if __name__ == "__main__":
    main()
