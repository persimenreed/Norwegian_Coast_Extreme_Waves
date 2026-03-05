import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from src.extreme_value_modelling.paths import resolve_input_path, resolve_diagnostics_dir
from src.extreme_value_modelling.extreme_preprocessing import load_data
from src.extreme_value_modelling.common import dataset_name


def run(location, mode, corr_method="qm", pooling=False):

    dataset = dataset_name(mode, corr_method, pooling)

    input_path = resolve_input_path(location, mode, corr_method, pooling)
    out_dir = resolve_diagnostics_dir(location)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(str(input_path))
    hs = df["hs"].values

    thresholds = np.linspace(np.percentile(hs, 90), np.percentile(hs, 99), 30)

    mean_excess = []
    xi = []
    sigma = []
    n_exceed = []

    for u in thresholds:

        exceed = hs[hs > u]
        excess = exceed - u

        n_exceed.append(len(exceed))

        if len(excess) > 30:

            mean_excess.append(np.mean(excess))

            shape, _, scale = genpareto.fit(excess, floc=0)

            xi.append(shape)
            sigma.append(scale)

        else:

            mean_excess.append(np.nan)
            xi.append(np.nan)
            sigma.append(np.nan)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Mean residual life
    axs[0,0].plot(thresholds, mean_excess)
    axs[0,0].set_xlabel("Threshold (m)")
    axs[0,0].set_ylabel("Mean Excess")
    axs[0,0].set_title("Mean Residual Life")

    # Shape stability
    axs[0,1].plot(thresholds, xi)
    axs[0,1].set_xlabel("Threshold (m)")
    axs[0,1].set_ylabel("Shape ξ")
    axs[0,1].set_title("Shape Stability")

    # Scale stability
    axs[1,0].plot(thresholds, sigma)
    axs[1,0].set_xlabel("Threshold (m)")
    axs[1,0].set_ylabel("Scale σ")
    axs[1,0].set_title("Scale Stability")

    # Threshold stability (number of exceedances)
    axs[1,1].plot(thresholds, n_exceed)
    axs[1,1].set_xlabel("Threshold (m)")
    axs[1,1].set_ylabel("Number of Exceedances")
    axs[1,1].set_title("Threshold Stability")

    for ax in axs.flatten():
        ax.grid()

    plt.tight_layout()
    plt.savefig(out_dir / f"evt_diagnostics_{dataset}.png", dpi=300)
    plt.close()