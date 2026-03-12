import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("results/extreme_value_modelling")
OUT = Path("results/extreme_value_analysis/comparison")

LOCATIONS = ["fauskane", "fedjeosen"]
MODELS = ["GEV", "GPD"]
PERIODS = [10, 20, 50]
DATASET_GROUPS = {
    "local": "local_",
    "transfer": "transfer_",
}


def plot(location, model, periods, group_name, prefix):

    df = pd.read_csv(ROOT / location / "summary_metrics.csv")
    df = df[df.model == model]
    df = df[df["dataset"].astype(str).str.startswith(prefix)]

    obs_col = "rl_obs" if "rl_obs" in df.columns else "rl_raw"
    model_col = "rl_model"

    if obs_col not in df.columns or model_col not in df.columns:
        raise KeyError(
            f"Expected columns '{obs_col}' and '{model_col}' in {location}/summary_metrics.csv"
        )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for i, return_period in enumerate(periods):
        ax = axes[i]

        subset = df[df.return_period == return_period].copy()
        subset = subset.dropna(subset=[obs_col, model_col])

        if subset.empty:
            ax.set_title(f"RL{return_period}")
            ax.set_xlabel("Raw Hindcast (m)")
            ax.set_ylabel("Model (m)")
            ax.grid(alpha=0.3)
            continue

        for _, row in subset.iterrows():
            ax.scatter(
                row[obs_col],
                row[model_col],
                label=row.dataset,
            )

        obs = subset[obs_col].to_numpy(dtype=float)
        model_vals = subset[model_col].to_numpy(dtype=float)
        center = float(np.mean(obs))

        max_dev = max(
            float(np.max(np.abs(obs - center))),
            float(np.max(np.abs(model_vals - center))),
        )
        half_span = max(0.75, max_dev * 1.15)
        x_min = center - half_span
        x_max = center + half_span

        ax.plot([x_min, x_max], [x_min, x_max], "k--")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)

        ax.set_title(f"RL{return_period}")
        ax.set_xlabel("Raw Hindcast (m)")
        ax.set_ylabel("Model (m)")
        ax.grid(alpha=0.3)

    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=5,
            frameon=False,
        )

    fig.suptitle(f"{location} - {model} ({group_name})")

    OUT.mkdir(parents=True, exist_ok=True)

    period_tag = "_".join(str(p) for p in periods)
    path = OUT / f"{location}_{group_name}_{period_tag}_{model.lower()}.png"

    plt.tight_layout(rect=(0, 0.17, 1, 0.95))
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved", path)


def main():

    for loc in LOCATIONS:
        for model in MODELS:
            for group_name, prefix in DATASET_GROUPS.items():
                plot(loc, model, PERIODS, group_name, prefix)


if __name__ == "__main__":
    main()