import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from src.extreme_value_modelling.common import DECLUSTER_HOURS, THRESHOLD_QUANTILE
from src.extreme_value_modelling.common import dataset_name
from src.extreme_value_modelling.extreme_preprocessing import compute_pot, decluster_clustermax, load_data
from src.extreme_value_modelling.paths import resolve_diagnostics_dir, resolve_input_path


def _threshold_grid(values: np.ndarray, chosen_threshold: float) -> np.ndarray:
    thresholds = np.linspace(np.percentile(values, 90), np.percentile(values, 99), 30)
    return np.unique(np.sort(np.append(thresholds, chosen_threshold)))


def run(location, mode, corr_method="pqm", transfer_source=None):
    dataset = dataset_name(mode, corr_method=corr_method, transfer_source=transfer_source)
    input_path = resolve_input_path(location, mode, corr_method=corr_method, transfer_source=transfer_source)
    out_dir = resolve_diagnostics_dir(location)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    hs = df["hs"].values
    _, chosen_threshold, _, _ = compute_pot(df, quantile=THRESHOLD_QUANTILE, decluster_hours=DECLUSTER_HOURS)

    thresholds = _threshold_grid(hs, chosen_threshold)

    stats = {"mean_excess": [], "xi": [], "sigma": [], "n_peaks": []}

    for u in thresholds:
        peaks = decluster_clustermax(df[df["hs"] > u], DECLUSTER_HOURS)
        excess = peaks.to_numpy(dtype=float) - u
        excess = excess[excess > 0]

        stats["n_peaks"].append(len(excess))

        if len(excess) > 30:
            stats["mean_excess"].append(np.mean(excess))
            shape, _, scale = genpareto.fit(excess, floc=0)
            stats["xi"].append(shape)
            stats["sigma"].append(scale)
            continue

        stats["mean_excess"].append(np.nan)
        stats["xi"].append(np.nan)
        stats["sigma"].append(np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    panels = (
        (axs[0], "mean_excess", "Mean Excess", "Mean Residual Life"),
        (axs[1], "xi", "Shape ξ", "Shape Stability"),
    )

    for ax, key, ylabel, title in panels:
        ax.plot(thresholds, stats[key])
        ax.axvline(chosen_threshold, color="red", linestyle="--")
        ax.set_xlabel("Threshold (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()

    plt.tight_layout()
    plt.savefig(out_dir / f"evt_diagnostics_{dataset}.png", dpi=300)
    plt.close()
