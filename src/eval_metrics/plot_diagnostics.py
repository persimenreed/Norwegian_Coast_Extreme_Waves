import numpy as np
import matplotlib.pyplot as plt


def _finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _style_for_name(name):
    if name == "raw":
        return {"linestyle": "--", "linewidth": 1.8}

    if name.startswith("localcv_"):
        return {"linestyle": "-", "linewidth": 1.4}

    if name.startswith("transfer_"):
        return {"linestyle": ":", "linewidth": 1.4}

    if name.startswith("pooled_"):
        return {"linestyle": "-.", "linewidth": 1.6}

    if name.startswith("ensemble_"):
        return {"linestyle": "-", "linewidth": 2.2}

    return {"linestyle": "-", "linewidth": 1.2}


def _pdf_cdf_style(name):
    # All solid lines for readability in PDF/CDF comparisons.
    style = {"linestyle": "-", "linewidth": 1.2, "alpha": 0.40}

    if name == "raw":
        style.update({"linewidth": 2.0, "alpha": 0.9})
    elif name.startswith("ensemble_") or name in {"ensemble_transfer", "ensemble_pooling"}:
        style.update({"linewidth": 2.0, "alpha": 0.9})

    return style


def _split_series_for_eval_plots(series_dict):
    local = {}
    transfer = {}
    ensemble = {}

    for name, values in series_dict.items():
        if name == "raw":
            local[name] = values
            transfer[name] = values
            ensemble[name] = values
            continue

        if name.startswith("localcv_"):
            local[name] = values

        if (
            name.startswith("transfer_")
            or name.startswith("pooled_")
            or name.startswith("ensemble_")
            or name in {"ensemble_transfer", "ensemble_pooling"}
        ):
            transfer[name] = values

        if name.startswith("ensemble_") or name in {"ensemble_transfer", "ensemble_pooling"}:
            ensemble[name] = values

    return {"local": local, "transfer": transfer, "ensemble": ensemble}


def _plot_pdf_single(obs, series_dict, out_path, title_suffix):
    plt.figure(figsize=(7, 4))

    obs = _finite(obs)
    if len(obs) > 10:
        h, edges = np.histogram(obs, bins="fd", density=True)
        c = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(c, h, label="buoy", linewidth=2.6, alpha=1.0, linestyle="-")

    for name, values in series_dict.items():
        x = _finite(values)
        if len(x) < 10:
            continue

        h, edges = np.histogram(x, bins="fd", density=True)
        c = 0.5 * (edges[:-1] + edges[1:])

        style = _pdf_cdf_style(name)
        plt.plot(c, h, label=name, **style)

    plt.xlabel("Hs (m)")
    plt.ylabel("Density")
    plt.title(f"PDF comparison ({title_suffix})")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_cdf_single(obs, series_dict, out_path, title_suffix):
    plt.figure(figsize=(7, 4))

    obs = np.sort(_finite(obs))
    if len(obs) == 0:
        plt.close()
        return

    p = np.arange(1, len(obs) + 1) / len(obs)
    plt.plot(obs, p, label="buoy", linewidth=2.6, alpha=1.0, linestyle="-")

    for name, values in series_dict.items():
        x = np.sort(_finite(values))
        if len(x) == 0:
            continue

        p = np.arange(1, len(x) + 1) / len(x)
        style = _pdf_cdf_style(name)
        plt.plot(x, p, label=name, **style)

    plt.xlabel("Hs (m)")
    plt.ylabel("Empirical CDF")
    plt.title(f"CDF comparison ({title_suffix})")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _residual_style(name):
    style = {"alpha": 0.40, "s": 8}

    if name == "raw":
        style.update({"alpha": 0.9, "s": 12})
    elif name.startswith("ensemble_") or name in {"ensemble_transfer", "ensemble_pooling"}:
        style.update({"alpha": 0.9, "s": 12})

    return style


def _plot_qq_single(obs, series_dict, out_path, title_suffix):
    q = np.linspace(0.01, 0.995, 200)

    obs = _finite(obs)
    if len(obs) == 0:
        return

    obs_q = np.quantile(obs, q)

    plt.figure(figsize=(6, 6))

    for name, values in series_dict.items():
        x = _finite(values)
        if len(x) == 0:
            continue

        xq = np.quantile(x, q)
        style = _pdf_cdf_style(name)
        plt.plot(obs_q, xq, label=name, **style)

    lo = float(np.nanmin(obs_q))
    hi = float(np.nanmax(obs_q))
    plt.plot([lo, hi], [lo, hi], color="k", linestyle="-", linewidth=1.1, label="1:1")

    plt.xlabel("Buoy quantiles")
    plt.ylabel("Model quantiles")
    plt.title(f"QQ plot ({title_suffix})")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_residuals_single(obs, series_dict, out_path, title_suffix):
    plt.figure(figsize=(7, 4))

    y = np.asarray(obs, float)

    for name, values in series_dict.items():
        x = np.asarray(values, float)

        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) == 0:
            continue

        r = x[m] - y[m]
        style = _residual_style(name)
        plt.scatter(y[m], r, label=name, **style)

    plt.axhline(0, color="k", linestyle="-", linewidth=1)

    plt.xlabel("Buoy Hs (m)")
    plt.ylabel("Residual (model − buoy)")
    plt.title(f"Residuals ({title_suffix})")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pdf(obs, series_dict, out_dir):
    groups = _split_series_for_eval_plots(series_dict)

    if groups["local"]:
        _plot_pdf_single(obs, groups["local"], f"{out_dir}/pdf_local.png", "local")

    if groups["transfer"]:
        _plot_pdf_single(obs, groups["transfer"], f"{out_dir}/pdf_transfer.png", "transfer")

    if groups["ensemble"]:
        _plot_pdf_single(obs, groups["ensemble"], f"{out_dir}/pdf_ensemble.png", "ensemble")


def plot_cdf(obs, series_dict, out_dir):
    groups = _split_series_for_eval_plots(series_dict)

    if groups["local"]:
        _plot_cdf_single(obs, groups["local"], f"{out_dir}/cdf_local.png", "local")

    if groups["transfer"]:
        _plot_cdf_single(obs, groups["transfer"], f"{out_dir}/cdf_transfer.png", "transfer")

    if groups["ensemble"]:
        _plot_cdf_single(obs, groups["ensemble"], f"{out_dir}/cdf_ensemble.png", "ensemble")


def plot_qq(obs, series_dict, out_dir):
    groups = _split_series_for_eval_plots(series_dict)

    if groups["local"]:
        _plot_qq_single(obs, groups["local"], f"{out_dir}/qq_local.png", "local")

    if groups["transfer"]:
        _plot_qq_single(obs, groups["transfer"], f"{out_dir}/qq_transfer.png", "transfer")

    if groups["ensemble"]:
        _plot_qq_single(obs, groups["ensemble"], f"{out_dir}/qq_ensemble.png", "ensemble")


def plot_residuals(obs, series_dict, out_dir):
    groups = _split_series_for_eval_plots(series_dict)

    if groups["local"]:
        _plot_residuals_single(obs, groups["local"], f"{out_dir}/residuals_local.png", "local")

    if groups["transfer"]:
        _plot_residuals_single(obs, groups["transfer"], f"{out_dir}/residuals_transfer.png", "transfer")

    if groups["ensemble"]:
        _plot_residuals_single(obs, groups["ensemble"], f"{out_dir}/residuals_ensemble.png", "ensemble")
