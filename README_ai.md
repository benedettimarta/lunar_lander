# Lunar Lander RL Assignment Code

This folder contains the Python scripts for a DQN-based Lunar Lander assignment.

## Installation

```bash
pip install -r requirements.txt
```

If Box2D fails on Windows, try installing `swig` first, then rerun the command.

## Run order

1. Train one baseline agent:

```bash
python 01_train_baseline.py
```

2. Evaluate the baseline:

```bash
python 02_evaluate_model.py
```

3. Run sensitivity analysis:

```bash
python 03_run_sensitivity_analysis.py
```

4. Generate figures:

```bash
python 04_plot_results.py
```

5. Optional: watch the trained agent:

```bash
python 05_watch_trained_agent.py
```

## Output folders

- `models/`: trained models
- `logs/`: training reward logs
- `results/`: evaluation metrics and sensitivity summary
- `figures/`: PNG plots for the report
