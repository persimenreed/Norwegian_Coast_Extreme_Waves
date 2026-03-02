import numpy as np
import matplotlib.pyplot as plt


def _finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def plot_pdf(df, obs_col, methods, out_dir):
    plt.figure(figsize=(7, 4))
    styles = {
        "buoy": "-",
        "raw": "--",
        "linear": "-.",
        "qm": ":",
        "rf": (0, (3, 1, 1, 1)),
    }

    def draw_density_line(values, label, lw):
        h, edges = np.histogram(values, bins="fd", density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, h, linestyle=styles.get(label, "-"), linewidth=lw, label=label)

    obs = _finite(df[obs_col].values)
    if len(obs) > 10:
        draw_density_line(obs, "buoy", 2.0)

    for name, col in methods.items():
        if col not in df.columns:
            continue
        x = _finite(df[col].values)
        if len(x) > 10:
            draw_density_line(x, name, 1.2)

    plt.xlabel("Hs (m)")
    plt.ylabel("Density")
    plt.title("PDF")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pdf.png", dpi=150)
    plt.close()


def plot_cdf(df, obs_col, methods, out_dir):
    plt.figure(figsize=(7, 4))
    obs = np.sort(_finite(df[obs_col].values))
    if len(obs) > 0:
        p = np.arange(1, len(obs) + 1) / len(obs)
        plt.plot(obs, p, label="buoy", linewidth=2)

    for name, col in methods.items():
        if col not in df.columns:
            continue
        x = np.sort(_finite(df[col].values))
        if len(x) == 0:
            continue
        p = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, p, label=name)

    plt.xlabel("Hs (m)")
    plt.ylabel("Empirical CDF")
    plt.title("CDF")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cdf.png", dpi=150)
    plt.close()


def plot_qq(df, obs_col, methods, out_dir):
    q = np.linspace(0.01, 0.995, 200)
    obs = _finite(df[obs_col].values)
    if len(obs) == 0:
        return
    obs_q = np.quantile(obs, q)

    plt.figure(figsize=(6, 6))
    for name, col in methods.items():
        if col not in df.columns:
            continue
        x = _finite(df[col].values)
        if len(x) == 0:
            continue
        xq = np.quantile(x, q)
        plt.plot(obs_q, xq, label=name)

    lo = float(np.nanmin(obs_q))
    hi = float(np.nanmax(obs_q))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="1:1")
    plt.xlabel("Buoy quantiles")
    plt.ylabel("Model quantiles")
    plt.title("QQ")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/qq_plot.png", dpi=150)
    plt.close()


def plot_residuals(df, obs_col, methods, out_dir):
    plt.figure(figsize=(7, 4))
    y = df[obs_col].values
    for name, col in methods.items():
        if col not in df.columns:
            continue
        r = df[col].values - y
        m = np.isfinite(y) & np.isfinite(r)
        if np.sum(m) == 0:
            continue
        plt.scatter(y[m], r[m], s=4, alpha=0.35, label=name)

    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Buoy Hs (m)")
    plt.ylabel("Residual (model - buoy)")
    plt.title("Residuals")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/residuals.png", dpi=150)
    plt.close()
