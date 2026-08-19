"""
Generate plots for the report.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from config import FIGURES_DIR, LOGS_DIR, RESULTS_DIR
from core import load_monitor_rewards


def save_bar(df, x, y, title, ylabel, filename, hue=None):
    plt.figure(figsize=(8, 4.5))
    if hue is None:
        plt.bar(df[x].astype(str), df[y])
    else:
        # simple grouped bar plot without relying on seaborn
        groups = list(df[x].astype(str).unique())
        hues = list(df[hue].astype(str).unique())
        width = 0.8 / max(len(hues), 1)
        positions = range(len(groups))
        for i, h in enumerate(hues):
            vals = []
            for g in groups:
                row = df[(df[x].astype(str) == g) & (df[hue].astype(str) == h)]
                vals.append(float(row[y].iloc[0]) if len(row) else 0.0)
            offset = (i - (len(hues)-1)/2) * width
            plt.bar([p + offset for p in positions], vals, width=width, label=h)
        plt.xticks(list(positions), groups)
        plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25)
    plt.tight_layout()
    out = FIGURES_DIR / filename
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def baseline_learning_curves():
    for alg in ["dqn", "ppo"]:
        f = LOGS_DIR / f"baseline_{alg}" / "monitor.csv"
        if not f.exists():
            continue
        df = load_monitor_rewards(f)
        plt.figure()
        plt.plot(df["episode"], df["reward"], alpha=0.35, label="Episode reward")
        plt.plot(df["episode"], df["moving_average_reward"], label="20-episode moving average")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Baseline {alg.upper()} learning curve")
        plt.legend()
        plt.tight_layout()
        out = FIGURES_DIR / f"baseline_{alg}_learning_curve.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved {out}")


def baseline_comparison():
    f = RESULTS_DIR / "baseline_algorithm_comparison.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    save_bar(df, "algorithm", "mean_reward", "Baseline algorithm comparison", "Mean reward", "baseline_algorithm_reward.png")
    save_bar(df, "algorithm", "success_rate", "Baseline success rate", "Success rate", "baseline_algorithm_success.png")
    save_bar(df, "algorithm", "crash_rate", "Baseline crash rate", "Crash rate", "baseline_algorithm_crash.png")


def sensitivity_plots():
    f = RESULTS_DIR / "sensitivity_summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    for (alg, parameter), group in df.groupby(["algorithm", "parameter"]):
        group = group.sort_values("value")
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(
            group["value"].astype(str),
            group["mean_reward"],
            yerr=group["reward_std_across_seeds"].fillna(0.0),
            marker="o",
            capsize=4,
        )
        plt.xlabel(parameter)
        plt.ylabel("Mean evaluation reward")
        plt.title(f"{alg}: sensitivity of reward to {parameter}")
        plt.xticks(rotation=25)
        plt.tight_layout()
        out = FIGURES_DIR / f"sensitivity_{alg.lower()}_{parameter}_reward.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved {out}")


def reward_sensitivity_plots():
    f = RESULTS_DIR / "reward_sensitivity_summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    save_bar(df, "reward_case", "mean_common_reward", "Reward-design sensitivity", "Mean reward", "reward_sensitivity_reward.png", hue="algorithm")
    save_bar(df, "reward_case", "mean_control_use_proxy", "Fuel use under reward shaping", "Mean fuel proxy", "reward_sensitivity_fuel.png", hue="algorithm")
    save_bar(df, "reward_case", "mean_crash_rate", "Crash rate under reward shaping", "Crash rate", "reward_sensitivity_crash.png", hue="algorithm")


def robustness_plots():
    f = RESULTS_DIR / "robustness_summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    save_bar(df, "case", "mean_reward", "Robustness to sensor/actuator uncertainty", "Mean reward", "robustness_reward.png", hue="algorithm")
    save_bar(df, "case", "success_rate", "Success rate under uncertainty", "Success rate", "robustness_success.png", hue="algorithm")
    save_bar(df, "case", "crash_rate", "Crash rate under uncertainty", "Crash rate", "robustness_crash.png", hue="algorithm")


def curriculum_plots():
    f = RESULTS_DIR / "curriculum_summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    save_bar(df, "stage", "mean_reward", "Effect of uncertainty fine-tuning", "Mean reward", "curriculum_reward.png", hue="algorithm")


def main():
    baseline_learning_curves()
    baseline_comparison()
    sensitivity_plots()
    reward_sensitivity_plots()
    robustness_plots()
    curriculum_plots()
    print("Done generating available plots.")


if __name__ == "__main__":
    main()
