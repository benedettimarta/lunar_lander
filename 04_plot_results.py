"""
Create report figures from the baseline and sensitivity-analysis results.

Run:
    python 04_plot_results.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, LOGS_DIR, RESULTS_DIR
from utils import load_monitor_rewards


def plot_baseline_learning_curve():
    monitor_file = LOGS_DIR / "baseline_dqn" / "monitor.csv"
    df = load_monitor_rewards(monitor_file)

    plt.figure()
    plt.plot(df["episode"], df["reward"], label="Episode reward", alpha=0.4)
    plt.plot(df["episode"], df["moving_average_reward"], label="20-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Baseline DQN learning curve")
    plt.legend()
    plt.tight_layout()
    output = FIGURES_DIR / "baseline_learning_curve.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved {output}")


def plot_sensitivity_summary():
    summary_file = RESULTS_DIR / "sensitivity_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError("No sensitivity_summary.csv found. Run 03_run_sensitivity_analysis.py first.")

    df = pd.read_csv(summary_file)

    for parameter, group in df.groupby("parameter"):
        group = group.sort_values("value")

        plt.figure()
        plt.errorbar(
            group["value"].astype(str),
            group["mean_reward"],
            yerr=group["std_reward"],
            marker="o",
            capsize=4,
        )
        plt.xlabel(parameter)
        plt.ylabel("Mean evaluation reward")
        plt.title(f"Sensitivity of performance to {parameter}")
        plt.tight_layout()
        output = FIGURES_DIR / f"sensitivity_{parameter}_reward.png"
        plt.savefig(output, dpi=200)
        plt.close()
        print(f"Saved {output}")

        plt.figure()
        plt.bar(group["value"].astype(str), 100 * group["success_rate"])
        plt.xlabel(parameter)
        plt.ylabel("Success rate [%]")
        plt.title(f"Landing success rate vs {parameter}")
        plt.tight_layout()
        output = FIGURES_DIR / f"sensitivity_{parameter}_success_rate.png"
        plt.savefig(output, dpi=200)
        plt.close()
        print(f"Saved {output}")


def main():
    plot_baseline_learning_curve()
    plot_sensitivity_summary()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
