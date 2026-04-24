import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CLASSICAL_METHODS = {"dagqm", "linear", "pqm"}
ML_METHODS = {"gpr", "transformer", "xgboost"}
CDF_YMIN = 0.80
CDF_XMIN_QUANTILE = 0.80
CDF_XMAX_QUANTILE = 0.99
QQ_MIN_QUANTILE = 0.90
CDF_QQ_FIGSIZE = (6.5, 4.6)
METHOD_COLORS = {
    "Observed": "#000000",
    "buoy": "#000000",
    "raw": "#ff7f0e",
    "NORA3": "#ff7f0e",
    "MoE": "#2ca02c",
    "MoE Local": "#2ca02c",
    "MoE Transfer": "#4c78a8",
    "MoE Fauskane": "#2ca02c",
    "MoE Fedjeosen": "#4c78a8",
    "MoE Combined": "#bcbd22",
    "dagqm": "#d62728",
    "DAGQM": "#d62728",
    "linear": "#8c564b",
    "Linear": "#8c564b",
    "pqm": "#e377c2",
    "PQM": "#e377c2",
    "gpr": "#7f7f7f",
    "GPR": "#7f7f7f",
    "transformer": "#9467bd",
    "Transformer": "#9467bd",
    "xgboost": "#6b6ecf",
    "XGBoost": "#6b6ecf",
    "1:1": "#111111",
}
METHOD_LABELS = {
    "raw": "NORA3",
    "buoy": "Observed",
    "linear": "Linear",
    "pqm": "PQM",
    "dagqm": "DAGQM",
    "gpr": "GPR",
    "xgboost": "XGBoost",
    "transformer": "Transformer",
}


def _finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _display_name(name):
    name = str(name)
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    if name == "ensemble":
        return "MoE"
    if name == "ensemble_local":
        return "MoE Local"
    if name == "ensemble_transfer":
        return "MoE Transfer"
    if name.startswith("MoE_"):
        suffix = name.split("_", 1)[1]
        return f"MoE {suffix.title()}"
    if name.startswith("ensemble_"):
        return "MoE"
    if name.startswith("vestfjorden_transfer_"):
        rest = name.replace("vestfjorden_transfer_", "", 1)
        parts = rest.split("_", 1)
        if len(parts) == 2:
            method, source = parts[1], parts[0]
            return f"{METHOD_LABELS.get(method, method.title())}_{source.title()}"
        return METHOD_LABELS.get(rest, rest.title())
    if name.startswith("localcv_"):
        method = name.replace("localcv_", "", 1)
        return METHOD_LABELS.get(method, method.title())
    if name.startswith("transfer_"):
        rest = name.replace("transfer_", "", 1)
        parts = rest.split("_", 1)
        method = parts[1] if len(parts) == 2 else rest
        return METHOD_LABELS.get(method, method.title())
    return METHOD_LABELS.get(name, name)


def _color_for_name(name):
    display = _display_name(name)
    base = display.split("_", 1)[0]
    return METHOD_COLORS.get(display, METHOD_COLORS.get(base, "#333333"))


def _method_kind(name):
    name = str(name)
    if name.startswith("vestfjorden_transfer_"):
        rest = name.replace("vestfjorden_transfer_", "", 1)
        parts = rest.split("_", 1)
        base = parts[1] if len(parts) == 2 else rest
    elif name.startswith("localcv_"):
        base = name.replace("localcv_", "", 1)
    elif name.startswith("transfer_"):
        rest = name.replace("transfer_", "", 1)
        parts = rest.split("_", 1)
        base = parts[1] if len(parts) == 2 else rest
    else:
        display = _display_name(name)
        base = display.split("_", 1)[0]
    base = base.lower()
    if base in CLASSICAL_METHODS:
        return "classical"
    if base in ML_METHODS:
        return "ml"
    return None


def _q99_rmse(model, obs):
    model = np.asarray(model, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if model.shape != obs.shape:
        n = min(model.size, obs.size)
        model = model[:n]
        obs = obs[:n]
    mask = np.isfinite(model) & np.isfinite(obs)
    model = model[mask]
    obs = obs[mask]
    if len(obs) < 20:
        return np.inf

    threshold = np.nanquantile(obs, 0.99)
    tail = obs >= threshold
    if np.sum(tail) < 20:
        return np.inf

    return float(np.sqrt(np.nanmean((model[tail] - obs[tail]) ** 2)))


def _best_by_q99(obs, series_dict, names):
    best_name = None
    best_score = np.inf
    for name in names:
        if name not in series_dict:
            continue
        score = _q99_rmse(series_dict[name], obs)
        if score < best_score:
            best_name = name
            best_score = score
    return best_name


def _display_group_name(name):
    mapping = {
        "ensemble": "MoE",
        "local": "Local",
        "transfer": "Transfer",
    }
    return mapping.get(str(name), str(name).title())


def _selected_distribution_series(obs, series_dict, include_all_ensembles=False):
    selected = {}

    if "raw" in series_dict:
        selected["raw"] = series_dict["raw"]

    ensemble_names = [
        name for name in series_dict
        if (
            str(name) == "ensemble"
            or str(name).startswith("ensemble_")
            or str(name).startswith("MoE_")
        )
    ]
    if include_all_ensembles:
        for name in ensemble_names:
            selected[name] = series_dict[name]
    else:
        best_ensemble = _best_by_q99(obs, series_dict, ensemble_names)
        if best_ensemble is not None:
            selected[best_ensemble] = series_dict[best_ensemble]

    classical_names = [
        name for name in series_dict
        if _method_kind(name) == "classical"
    ]
    best_classical = _best_by_q99(obs, series_dict, classical_names)
    if best_classical is not None:
        selected[best_classical] = series_dict[best_classical]

    ml_names = [
        name for name in series_dict
        if _method_kind(name) == "ml"
    ]
    best_ml = _best_by_q99(obs, series_dict, ml_names)
    if best_ml is not None:
        selected[best_ml] = series_dict[best_ml]

    return selected


def _is_ensemble_name(name) -> bool:
    name = str(name)
    return (
        name == "ensemble"
        or name.startswith("ensemble_")
        or name.startswith("MoE_")
    )


def _without_ensemble_series(series_dict):
    return {
        name: values
        for name, values in series_dict.items()
        if not _is_ensemble_name(name)
    }


def _has_ensemble_series(series_dict) -> bool:
    return any(_is_ensemble_name(name) for name in series_dict)


def _pdf_cdf_style(name, linewidth=1.8):
    # All solid lines for readability in PDF/CDF comparisons.
    style = {
        "linestyle": "-",
        "linewidth": linewidth,
        "alpha": 0.90,
        "color": _color_for_name(name),
    }

    return style


def _opposite_location(location: str):
    mapping = {
        "fauskane": "fedjeosen",
        "fedjeosen": "fauskane",
    }
    return mapping.get(location)


def _moe_name_for_ensemble(name: str) -> str:
    return str(name).replace("ensemble_", "MoE_", 1)


def _split_series_for_eval_plots(series_dict, location: str):
    local = {}
    transfer = {}
    ensemble = {}

    if location == "vestfjorden":
        for name, values in series_dict.items():
            if name == "raw":
                transfer[name] = values
                ensemble[name] = values
                continue

            if name.startswith("transfer_"):
                transfer[f"vestfjorden_{name}"] = values
                continue

            if name.startswith("ensemble_"):
                moe_name = _moe_name_for_ensemble(name)
                ensemble[moe_name] = values
                if name == "ensemble_combined":
                    transfer["ensemble"] = values

        return {"local": local, "transfer": transfer, "ensemble": ensemble}

    local_ensemble = f"ensemble_{location}"
    opposite = _opposite_location(location)
    transfer_ensemble = f"ensemble_{opposite}" if opposite else None

    for name, values in series_dict.items():
        if name == "raw":
            local[name] = values
            transfer[name] = values
            ensemble[name] = values
            continue

        if name.startswith("localcv_"):
            local[name] = values

        if name == local_ensemble:
            local["ensemble"] = values
            ensemble["ensemble_local"] = values
            continue

        if transfer_ensemble and name == transfer_ensemble:
            transfer["ensemble"] = values
            ensemble["ensemble_transfer"] = values
            continue

        if name.startswith("transfer_"):
            transfer[name] = values

        if name.startswith("ensemble_"):
            ensemble[name] = values

    return {"local": local, "transfer": transfer, "ensemble": ensemble}


def _plot_pdf_single(obs, series_dict, out_path, title_suffix):
    plt.figure(figsize=(7, 4))

    obs = _finite(obs)
    if len(obs) > 10:
        h, edges = np.histogram(obs, bins="fd", density=True)
        c = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(c, h, label="Observed", linewidth=1.8, alpha=0.90, linestyle="-", color=_color_for_name("buoy"))

    for name, values in series_dict.items():
        x = _finite(values)
        if len(x) < 10:
            continue

        h, edges = np.histogram(x, bins="fd", density=True)
        c = 0.5 * (edges[:-1] + edges[1:])

        style = _pdf_cdf_style(name)
        plt.plot(c, h, label=_display_name(name), **style)

    plt.xlabel("Hs (m)")
    plt.ylabel("Density")
    plt.title(f"PDF comparison ({_display_group_name(title_suffix)})")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_cdf_single(obs, series_dict, out_path, title_suffix):
    plt.figure(figsize=CDF_QQ_FIGSIZE)

    obs_values = _finite(obs)
    obs_sorted = np.sort(obs_values)
    if len(obs_sorted) == 0:
        plt.close()
        return

    p = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
    plt.plot(obs_sorted, p, label="Observed", linewidth=2.4, alpha=0.95, linestyle="-", color=_color_for_name("buoy"))

    selected = _selected_distribution_series(
        obs_values,
        series_dict,
        include_all_ensembles=title_suffix == "ensemble",
    )
    plotted_values = [obs_sorted]
    for name, values in selected.items():
        x = np.sort(_finite(values))
        if len(x) == 0:
            continue

        plotted_values.append(x)
        p = np.arange(1, len(x) + 1) / len(x)
        style = _pdf_cdf_style(name, linewidth=1.95)
        plt.plot(x, p, label=_display_name(name), **style)

    plt.xlabel("Hs (m)")
    plt.ylabel("Empirical CDF")
    q_low = [
        float(np.nanquantile(values, CDF_XMIN_QUANTILE))
        for values in plotted_values
        if len(values) > 0
    ]
    q_high = [
        float(np.nanquantile(values, CDF_XMAX_QUANTILE))
        for values in plotted_values
        if len(values) > 0
    ]
    if q_low and q_high:
        x_min = max(0.0, min(q_low) * 0.98)
        x_max = max(q_high) * 1.02
        if x_max > x_min:
            plt.xlim(x_min, x_max)
    plt.ylim(CDF_YMIN, 1.002)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _residual_style(name):
    style = {"alpha": 0.45, "s": 8, "color": _color_for_name(name)}

    if name == "raw":
        style.update({"alpha": 0.9, "s": 12})
    elif name.startswith("ensemble_") or name == "ensemble":
        style.update({"alpha": 0.9, "s": 12})

    return style


def _plot_qq_single(obs, series_dict, out_path, title_suffix):
    q = np.linspace(QQ_MIN_QUANTILE, 0.995, 200)

    obs = _finite(obs)
    if len(obs) == 0:
        return

    obs_q = np.quantile(obs, q)

    plt.figure(figsize=CDF_QQ_FIGSIZE)

    selected = _selected_distribution_series(
        obs,
        series_dict,
        include_all_ensembles=title_suffix == "ensemble",
    )
    for name, values in selected.items():
        x = _finite(values)
        if len(x) == 0:
            continue

        xq = np.quantile(x, q)
        style = _pdf_cdf_style(name, linewidth=1.95)
        plt.plot(obs_q, xq, label=_display_name(name), **style)

    lo = float(np.nanmin(obs_q))
    hi = float(np.nanmax(obs_q))
    plt.plot([lo, hi], [lo, hi], color=_color_for_name("1:1"), linestyle="-", linewidth=1.1, label="1:1")

    plt.xlabel("Observed quantiles")
    plt.ylabel("Model quantiles")
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
        plt.scatter(y[m], r, label=_display_name(name), **style)

    plt.axhline(0, color="k", linestyle="-", linewidth=1)

    plt.xlabel("Observed Hs (m)")
    plt.ylabel("Residual (model − observed)")
    plt.title(f"Residuals ({_display_group_name(title_suffix)})")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pdf(obs, series_dict, out_dir):
    location = Path(out_dir).name
    groups = _split_series_for_eval_plots(series_dict, location)

    if groups["local"]:
        _plot_pdf_single(obs, groups["local"], f"{out_dir}/pdf_local.png", "local")

    if groups["transfer"]:
        _plot_pdf_single(obs, groups["transfer"], f"{out_dir}/pdf_transfer.png", "transfer")

    if groups["ensemble"]:
        _plot_pdf_single(obs, groups["ensemble"], f"{out_dir}/pdf_ensemble.png", "ensemble")


def plot_cdf(obs, series_dict, out_dir):
    location = Path(out_dir).name
    groups = _split_series_for_eval_plots(series_dict, location)

    if groups["local"]:
        _plot_cdf_single(obs, _without_ensemble_series(groups["local"]), f"{out_dir}/cdf_local.png", "local")
        if _has_ensemble_series(groups["local"]):
            _plot_cdf_single(obs, groups["local"], f"{out_dir}/cdf_local_ensemble.png", "local")

    if groups["transfer"]:
        _plot_cdf_single(obs, _without_ensemble_series(groups["transfer"]), f"{out_dir}/cdf_transfer.png", "transfer")
        if _has_ensemble_series(groups["transfer"]):
            _plot_cdf_single(obs, groups["transfer"], f"{out_dir}/cdf_transfer_ensemble.png", "transfer")

    if groups["ensemble"]:
        _plot_cdf_single(obs, groups["ensemble"], f"{out_dir}/cdf_ensemble.png", "ensemble")


def plot_qq(obs, series_dict, out_dir):
    location = Path(out_dir).name
    groups = _split_series_for_eval_plots(series_dict, location)

    if groups["local"]:
        _plot_qq_single(obs, _without_ensemble_series(groups["local"]), f"{out_dir}/qq_local.png", "local")
        if _has_ensemble_series(groups["local"]):
            _plot_qq_single(obs, groups["local"], f"{out_dir}/qq_local_ensemble.png", "local")

    if groups["transfer"]:
        _plot_qq_single(obs, _without_ensemble_series(groups["transfer"]), f"{out_dir}/qq_transfer.png", "transfer")
        if _has_ensemble_series(groups["transfer"]):
            _plot_qq_single(obs, groups["transfer"], f"{out_dir}/qq_transfer_ensemble.png", "transfer")

    if groups["ensemble"]:
        _plot_qq_single(obs, groups["ensemble"], f"{out_dir}/qq_ensemble.png", "ensemble")


def plot_residuals(obs, series_dict, out_dir):
    location = Path(out_dir).name
    groups = _split_series_for_eval_plots(series_dict, location)

    if groups["local"]:
        _plot_residuals_single(obs, groups["local"], f"{out_dir}/residuals_local.png", "local")

    if groups["transfer"]:
        _plot_residuals_single(obs, groups["transfer"], f"{out_dir}/residuals_transfer.png", "transfer")

    if groups["ensemble"]:
        _plot_residuals_single(obs, groups["ensemble"], f"{out_dir}/residuals_ensemble.png", "ensemble")
