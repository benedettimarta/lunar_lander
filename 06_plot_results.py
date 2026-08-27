"""
Generate plots for the report.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from config import FIGURES_DIR, LOGS_DIR, RESULTS_DIR
from core import load_monitor_rewards

# Plot formatting for report readability
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 16
LEGEND_SIZE = 14

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})


def save_bar(df, x, y, title, ylabel, filename, hue=None):
    plt.figure(figsize=(10,6))
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
        plt.xlabel("Training episodes")
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


def sensitivity_report_plots():
    f = RESULTS_DIR / "sensitivity_summary.csv"
    if not f.exists():
        return

    df = pd.read_csv(f)

    report_params = {
        "DQN": [
            "learning_rate",
            "gamma",
            "exploration_final_eps",
            "batch_size",
        ],
        "PPO": [
            "learning_rate",
            "n_steps",
            "batch_size",
            "net_arch",
        ],
    }

    display_names = {
        "learning_rate": "Learning Rate",
        "gamma": "Discount Factor",
        "exploration_final_eps": "Final Exploration Rate",
        "batch_size": "Batch Size",
        "n_steps": "Rollout Length",
        "net_arch": "Network Architecture",
    }

    for alg, params in report_params.items():

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for ax, parameter in zip(axes, params):

            group = df[
                (df["algorithm"] == alg)
                & (df["parameter"] == parameter)
            ].copy()

            group = group.sort_values("value")

            ax.errorbar(
                group["value"].astype(str),
                group["mean_reward"],
                yerr=group[
                    "reward_std_across_seeds"
                ].fillna(0.0),
                marker="o",
                markersize=10,
                linewidth=2.5,
                capsize=6,
                capthick=2,
            )

            ax.set_title("")
            ax.set_xlabel(display_names[parameter], fontsize=26, labelpad=10)
            ax.set_ylabel("Mean Reward", fontsize=26, labelpad=10)
            ax.tick_params(axis="both", which="major", labelsize=24)
            ax.grid(True, alpha=0.3)

        fig.tight_layout(
            pad=3.0,
            h_pad=3.0,
            w_pad=3.0,
        )

        out = (FIGURES_DIR / f"{alg.lower()}_sensitivity_summary.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


def reward_sensitivity_plots():
    f = RESULTS_DIR / "reward_sensitivity_summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    save_bar(df, "reward_case", "mean_common_reward", "Reward-design sensitivity", "Mean reward", "reward_sensitivity_reward.png", hue="algorithm")
    save_bar(df, "reward_case", "mean_control_use_proxy", "Fuel use under reward shaping", "Mean fuel proxy", "reward_sensitivity_fuel.png", hue="algorithm")
    save_bar(df, "reward_case", "mean_success_rate", "Reward design vs success rate", "Success rate", "reward_sensitivity_success.png", hue="algorithm")


def robustness_plots():
    f = RESULTS_DIR / "robustness_summary.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    save_bar(df, "case", "mean_reward", "Robustness to sensor/actuator uncertainty", "Mean reward", "robustness_reward.png", hue="algorithm")
    save_bar(df, "case", "success_rate", "Success rate under uncertainty", "Success rate", "robustness_success.png", hue="algorithm")
    save_bar(df, "case", "crash_rate", "Crash rate under uncertainty", "Crash rate", "robustness_crash.png", hue="algorithm")

def robustness_noise_plot():

    f = RESULTS_DIR / "robustness_summary.csv"
    if not f.exists():
        return

    df = pd.read_csv(f)

    noise_df = df[
        df["action_dropout_prob"] == 0.0
    ]

    plt.figure(figsize=(10,6))

    for alg in ["DQN", "PPO"]:

        group = noise_df[
            noise_df["algorithm"] == alg
        ].sort_values("obs_noise_std")

        plt.plot(
            group["obs_noise_std"],
            group["success_rate"],
            marker="o",
            label=alg,
        )

    plt.xlabel("Observation noise standard deviation")
    plt.ylabel("Success rate")
    plt.title("Robustness to sensor noise")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "robustness_sensor_noise.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

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
    sensitivity_report_plots()
    reward_sensitivity_plots()
    robustness_plots()
    robustness_noise_plot()
    curriculum_plots()
    print("Done generating available plots.")


if __name__ == "__main__":
    main()
