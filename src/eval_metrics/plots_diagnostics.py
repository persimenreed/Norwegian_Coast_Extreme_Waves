import numpy as np
import matplotlib.pyplot as plt


def _finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def plot_pdf(obs, series_dict, out_dir):

    plt.figure(figsize=(7,4))

    styles = {
        "raw": "--",
        "local_linear": "-.",
        "local_qm": ":",
        "local_rf": (0,(3,1,1,1)),
        "pooled_linear": "-",
        "pooled_qm": "-",
        "pooled_rf": "-"
    }

    obs = _finite(obs)
    if len(obs) > 10:
        h, edges = np.histogram(obs, bins="fd", density=True)
        c = 0.5*(edges[:-1]+edges[1:])
        plt.plot(c, h, label="buoy", linewidth=2)

    for name, values in series_dict.items():

        x = _finite(values)

        if len(x) < 10:
            continue

        h, edges = np.histogram(x, bins="fd", density=True)
        c = 0.5*(edges[:-1]+edges[1:])

        plt.plot(c, h, linestyle=styles.get(name,"-"), label=name)

    plt.xlabel("Hs (m)")
    plt.ylabel("Density")
    plt.title("PDF comparison")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pdf.png", dpi=150)
    plt.close()


def plot_cdf(obs, series_dict, out_dir):

    plt.figure(figsize=(7,4))

    obs = np.sort(_finite(obs))
    p = np.arange(1,len(obs)+1)/len(obs)

    plt.plot(obs,p,label="buoy",linewidth=2)

    for name, values in series_dict.items():

        x = np.sort(_finite(values))
        if len(x)==0:
            continue

        p = np.arange(1,len(x)+1)/len(x)
        plt.plot(x,p,label=name)

    plt.xlabel("Hs (m)")
    plt.ylabel("Empirical CDF")
    plt.title("CDF comparison")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cdf.png", dpi=150)
    plt.close()


def plot_qq(obs, series_dict, out_dir):

    q = np.linspace(0.01,0.995,200)

    obs = _finite(obs)
    obs_q = np.quantile(obs,q)

    plt.figure(figsize=(6,6))

    for name, values in series_dict.items():

        x = _finite(values)

        if len(x)==0:
            continue

        xq = np.quantile(x,q)

        plt.plot(obs_q,xq,label=name)

    lo = float(np.nanmin(obs_q))
    hi = float(np.nanmax(obs_q))

    plt.plot([lo,hi],[lo,hi],"k--",linewidth=1,label="1:1")

    plt.xlabel("Buoy quantiles")
    plt.ylabel("Model quantiles")
    plt.title("QQ plot")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/qq_plot.png",dpi=150)
    plt.close()


def plot_residuals(obs, series_dict, out_dir):

    plt.figure(figsize=(7,4))

    for name, values in series_dict.items():

        x = np.asarray(values,float)
        y = np.asarray(obs,float)

        m = np.isfinite(x) & np.isfinite(y)

        if np.sum(m)==0:
            continue

        r = x[m] - y[m]

        plt.scatter(y[m], r, s=4, alpha=0.35, label=name)

    plt.axhline(0,color="k",linestyle="--")

    plt.xlabel("Buoy Hs (m)")
    plt.ylabel("Residual (model − buoy)")
    plt.title("Residuals")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/residuals.png",dpi=150)
    plt.close()