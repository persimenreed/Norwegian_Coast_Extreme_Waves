from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("results/extreme_value_modelling")
OUT = Path("results/extreme_value_analysis/ciwidth_heatmap")

RETURN_PERIOD = 50.0
STUDY_LOCATIONS = ["stavanger", "bergen", "kristiansund"]
BUOY_LOCATIONS = ["fauskane", "fedjeosen"]
MODELS = ["GEV", "GPD"]
STUDY_DATASETS_DEFAULT = [
    "raw",
    "ensemble_fedjeosen",
    "ensemble_fauskane",
    "ensemble_combined",
]


def load_summary(location: str) -> pd.DataFrame:
    path = ROOT / location / "summary_return_levels.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def dataset_sort_key(name: str):
    if name == "raw":
        return (0, name)
    if name.startswith("local_"):
        return (1, name)
    if name.startswith("transfer_"):
        return (2, name)
    if name.startswith("pooled_"):
        return (3, name)
    return (4, name)


def dataset_label(name: str) -> str:
    return name


def opposite_buoy_location(location: str) -> str | None:
    mapping = {
        "fauskane": "fedjeosen",
        "fedjeosen": "fauskane",
    }
    return mapping.get(location)


def load_rows(location: str, model: str | None = None, models: list[str] | None = None) -> pd.DataFrame:
    df = load_summary(location).copy()
    df["return_period"] = pd.to_numeric(df["return_period"], errors="coerce")
    df = df[np.isclose(df["return_period"], RETURN_PERIOD)].copy()
    if model is not None:
        df = df[df["model"] == model].copy()
    if models is not None:
        df = df[df["model"].isin(models)].copy()
    return df


def discover_datasets(locations, models, family_prefix: str) -> list[str]:
    found = set()

    for location in locations:
        df = load_rows(location, models=models)

        for dataset in df["dataset"].dropna().astype(str).unique():
            if dataset == "raw" or dataset.startswith(f"{family_prefix}_"):
                found.add(dataset)

    ordered = sorted(found, key=dataset_sort_key)
    if "raw" not in ordered:
        ordered = ["raw"] + ordered
        ordered = sorted(set(ordered), key=dataset_sort_key)
    return ordered


def plot_heatmap(locations, model, datasets=None):
    rows = []
    for location in locations:
        df_loc = load_rows(location, model=model)
        if datasets is not None:
            df_loc = df_loc[df_loc["dataset"].isin(datasets)].copy()

        for _, row in df_loc.iterrows():
            rows.append({
                "location": location,
                "dataset": row["dataset"],
                "ci_width": float(row["ci_width"]),
            })

    if not rows:
        raise ValueError(f"No rows found for model={model}, RP={RETURN_PERIOD} after dataset filtering")

    df = pd.DataFrame(rows)

    pivot = df.pivot(index="dataset", columns="location", values="ci_width")

    # preserve location order from CLI
    pivot = pivot.reindex(columns=locations)

    # Keep a deterministic base order for ties and fallback behavior.
    sorted_index = sorted(pivot.index, key=dataset_sort_key)
    pivot = pivot.reindex(sorted_index)

    # Primary ordering: lowest CI width at Stavanger first.
    if "stavanger" in pivot.columns:
        pivot = pivot.sort_values(by="stavanger", ascending=True, na_position="last", kind="mergesort")
    else:
        print("Warning: 'stavanger' not found in selected locations; using default dataset ordering.")

    vals = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 2.5, 0.45 * len(pivot.index) + 2.5))
    im = ax.imshow(vals, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([dataset_label(x) for x in pivot.index])

    norm = im.norm
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
            color = "black" if np.isfinite(v) and norm(v) > 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"{model} 50-year CI width")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CI width (m)")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"study_area_CI_50yr_{model.lower()}.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


def plot_buoy_model_heatmap(location: str, family_prefix: str):
    df = load_rows(location, models=MODELS)
    keep = discover_datasets([location], MODELS, family_prefix)

    if family_prefix == "transfer":
        opposite = opposite_buoy_location(location)
        ensemble_opposite = f"ensemble_{opposite}" if opposite else None
        available = set(df["dataset"].dropna().astype(str))
        if ensemble_opposite and ensemble_opposite in available:
            keep = keep + [ensemble_opposite]

    df = df[df["dataset"].isin(keep)].copy()

    if df.empty:
        print(f"Skipped {location} {family_prefix}: no matching rows")
        return

    pivot = df.pivot_table(index="dataset", columns="model", values="ci_width", aggfunc="first")
    pivot = pivot.reindex(index=sorted(pivot.index, key=dataset_sort_key))
    pivot = pivot.reindex(columns=MODELS)

    if "GEV" in pivot.columns:
        pivot = pivot.sort_values(by="GEV", ascending=True, na_position="last", kind="mergesort")

    vals = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 2.5, 0.45 * len(pivot.index) + 2.5))
    im = ax.imshow(vals, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([dataset_label(x) for x in pivot.index])

    norm = im.norm
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
            color = "black" if np.isfinite(v) and norm(v) > 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"{location} {family_prefix} 50-year CI width")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CI width (m)")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{location}_CI_50yr_{family_prefix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_vestfjorden_gev_gpd_heatmap():
    """Plot combined GEV/GPD heatmap for vestfjorden with ensemble datasets."""
    try:
        df = load_rows("vestfjorden", models=MODELS)
    except FileNotFoundError:
        print("Skipped vestfjorden: summary_return_levels.csv not found")
        return

    keep_datasets = [
        "raw",
        "ensemble_fauskane",
        "ensemble_fedjeosen",
        "ensemble_combined",
    ]

    df = df[df["dataset"].isin(keep_datasets)].copy()

    if df.empty:
        print("Skipped vestfjorden: no matching ensemble datasets")
        return

    pivot = df.pivot_table(index="dataset", columns="model", values="ci_width", aggfunc="first")
    pivot = pivot.reindex(index=sorted(pivot.index, key=dataset_sort_key))
    pivot = pivot.reindex(columns=MODELS)

    if "GEV" in pivot.columns:
        pivot = pivot.sort_values(by="GEV", ascending=True, na_position="last", kind="mergesort")

    vals = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 2.5, 0.45 * len(pivot.index) + 2.5))
    im = ax.imshow(vals, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([dataset_label(x) for x in pivot.index])

    norm = im.norm
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt = "nan" if not np.isfinite(v) else f"{v:.2f}"
            color = "black" if np.isfinite(v) and norm(v) > 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title("vestfjorden ensemble 50-year CI width")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CI width (m)")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "vestfjorden_CI_50yr_ensemble.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main():
    # 1) Study-area heatmaps for fixed study locations, one figure per model.
    plot_heatmap(STUDY_LOCATIONS, "GEV", datasets=STUDY_DATASETS_DEFAULT)
    plot_heatmap(STUDY_LOCATIONS, "GPD", datasets=STUDY_DATASETS_DEFAULT)

    # 2) Buoy heatmaps split by location and dataset family, with GEV/GPD on x-axis.
    for location in BUOY_LOCATIONS:
        plot_buoy_model_heatmap(location, "local")
        plot_buoy_model_heatmap(location, "transfer")

    # 3) Vestfjorden combined GEV/GPD heatmap if available.
    plot_vestfjorden_gev_gpd_heatmap()


if __name__ == "__main__":
    main()