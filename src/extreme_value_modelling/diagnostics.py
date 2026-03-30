import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from src.extreme_value_modelling.common import dataset_name
from src.extreme_value_modelling.extreme_preprocessing import load_data
from src.extreme_value_modelling.paths import resolve_diagnostics_dir, resolve_input_path


def run(location, mode, corr_method="pqm", transfer_source=None):
    dataset = dataset_name(mode, corr_method=corr_method, transfer_source=transfer_source)
    input_path = resolve_input_path(location, mode, corr_method=corr_method, transfer_source=transfer_source)
    out_dir = resolve_diagnostics_dir(location)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    hs = df["hs"].values

    thresholds = np.linspace(np.percentile(hs, 90), np.percentile(hs, 99), 30)

    stats = {"mean_excess": [], "xi": [], "sigma": [], "n_exceed": []}

    for u in thresholds:
        exceed = hs[hs > u]
        excess = exceed - u

        stats["n_exceed"].append(len(exceed))

        if len(excess) > 30:
            stats["mean_excess"].append(np.mean(excess))
            shape, _, scale = genpareto.fit(excess, floc=0)
            stats["xi"].append(shape)
            stats["sigma"].append(scale)
            continue

        stats["mean_excess"].append(np.nan)
        stats["xi"].append(np.nan)
        stats["sigma"].append(np.nan)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    panels = (
        (axs[0, 0], "mean_excess", "Mean Excess", "Mean Residual Life"),
        (axs[0, 1], "xi", "Shape ξ", "Shape Stability"),
        (axs[1, 0], "sigma", "Scale σ", "Scale Stability"),
        (axs[1, 1], "n_exceed", "Number of Exceedances", "Threshold Stability"),
    )

    for ax, key, ylabel, title in panels:
        ax.plot(thresholds, stats[key])
        ax.set_xlabel("Threshold (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()

    plt.tight_layout()
    plt.savefig(out_dir / f"evt_diagnostics_{dataset}.png", dpi=300)
    plt.close()
