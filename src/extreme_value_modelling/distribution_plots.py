import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, genpareto, probplot


def gev_plots(data, shape, loc, scale, out_dir, dataset):

    x_min = min(data) * 0.9
    x_max = max(data) * 1.2
    x = np.linspace(x_min, x_max, 500)

    pdf = genextreme.pdf(x, shape, loc=loc, scale=scale)
    cdf = genextreme.cdf(x, shape, loc=loc, scale=scale)
    sf = genextreme.sf(x, shape, loc=loc, scale=scale)

    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n
    esf = 1 - ecdf

    # QQ
    plt.figure()
    probplot(data, dist=genextreme, sparams=(shape, loc, scale), plot=plt)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Observed Annual Maxima Hs (m)")
    plt.title("GEV QQ Plot")
    plt.grid()
    plt.savefig(out_dir / f"gev_qq_{dataset}.png", dpi=300)
    plt.close()

    # PDF
    plt.figure()
    plt.hist(data, bins="auto", density=True, alpha=0.4, label="Empirical")
    plt.plot(x, pdf, label="GEV PDF")
    plt.xlabel("Hs (m)")
    plt.ylabel("Density")
    plt.title("GEV Probability Density")
    plt.grid()
    plt.legend()
    plt.savefig(out_dir / f"gev_pdf_{dataset}.png", dpi=300)
    plt.close()

    # CDF
    plt.figure()
    plt.step(sorted_data, ecdf, where="post", label="Empirical CDF")
    plt.plot(x, cdf, label="GEV CDF")
    plt.xlabel("Hs (m)")
    plt.ylabel("Cumulative Probability")
    plt.title("GEV Cumulative Distribution")
    plt.grid()
    plt.legend()
    plt.savefig(out_dir / f"gev_cdf_{dataset}.png", dpi=300)
    plt.close()

    # Survival
    plt.figure()
    plt.step(sorted_data, esf, where="post", label="Empirical Survival")
    plt.semilogy(x, sf, label="GEV Survival")
    plt.xlabel("Hs (m)")
    plt.ylabel("Exceedance Probability")
    plt.title("GEV Survival Function")
    plt.grid()
    plt.legend()
    plt.savefig(out_dir / f"gev_survival_{dataset}.png", dpi=300)
    plt.close()


def gpd_plots(excess, shape, scale, threshold, out_dir, dataset):

    x_min = 0
    x_max = max(excess) * 1.2
    x = np.linspace(x_min, x_max, 500)

    pdf = genpareto.pdf(x, shape, scale=scale)
    cdf = genpareto.cdf(x, shape, scale=scale)
    sf = genpareto.sf(x, shape, scale=scale)

    sorted_excess = np.sort(excess)
    n = len(sorted_excess)
    ecdf = np.arange(1, n + 1) / n
    esf = 1 - ecdf

    # QQ
    plt.figure()
    probplot(excess, dist=genpareto, sparams=(shape, 0, scale), plot=plt)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Observed Excess (m)")
    plt.title("GPD QQ Plot")
    plt.grid()
    plt.savefig(out_dir / f"gpd_qq_{dataset}.png", dpi=300)
    plt.close()

    # PDF
    plt.figure()
    plt.hist(excess, bins="auto", density=True, alpha=0.4, label="Empirical")
    plt.plot(x, pdf, label="GPD PDF")
    plt.xlabel("Excess (m)")
    plt.ylabel("Density")
    plt.title("GPD Probability Density")
    plt.grid()
    plt.legend()
    plt.savefig(out_dir / f"gpd_pdf_{dataset}.png", dpi=300)
    plt.close()

    # CDF
    plt.figure()
    plt.step(sorted_excess, ecdf, where="post", label="Empirical CDF")
    plt.plot(x, cdf, label="GPD CDF")
    plt.xlabel("Excess (m)")
    plt.ylabel("Cumulative Probability")
    plt.title("GPD Cumulative Distribution")
    plt.grid()
    plt.legend()
    plt.savefig(out_dir / f"gpd_cdf_{dataset}.png", dpi=300)
    plt.close()

    # Survival
    plt.figure()
    plt.step(sorted_excess, esf, where="post", label="Empirical Survival")
    plt.semilogy(x, sf, label="GPD Survival")
    plt.xlabel("Excess (m)")
    plt.ylabel("Exceedance Probability")
    plt.title("GPD Survival Function")
    plt.grid()
    plt.legend()
    plt.savefig(out_dir / f"gpd_survival_{dataset}.png", dpi=300)
    plt.close()