import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.extreme_value_modelling.extreme_preprocessing import load_data
from src.extreme_value_modelling.paths import resolve_input_path

SUMMARY_ROOT = Path("results/extreme_value_modelling")
OUT_ROOT = Path("results/extreme_value_analysis/tail_comparison")


def _dataset_path(location: str, dataset: str) -> Path:
	name = str(dataset).strip()
	if name == "raw":
		return resolve_input_path(location=location, mode="raw")
	return Path(f"data/output/{location}/hindcast_corrected_{name}.csv")


def _display_name(dataset: str) -> str:
	name = str(dataset).strip()
	if name == "raw":
		return "raw"
	if name.startswith("local_"):
		return name.replace("local_", "local: ")
	if name.startswith("transfer_"):
		return name.replace("transfer_", "transfer: ")
	if name.startswith("ensemble_"):
		return name.replace("ensemble_", "ensemble: ")
	return name


def _dataset_sort_key(dataset: str):
	name = str(dataset)
	if name == "raw":
		return (0, name)
	if name.startswith("ensemble_"):
		return (1, name)
	if name.startswith("local_"):
		return (2, name)
	if name.startswith("transfer_"):
		return (3, name)
	return (4, name)


def _load_evt_summary(location: str) -> pd.DataFrame:
	path = SUMMARY_ROOT / location / "evt_parameter_summary.csv"
	if not path.exists():
		raise FileNotFoundError(path)

	df = pd.read_csv(path)
	required = {"dataset", "model", "xi", "sigma", "threshold"}
	missing = required.difference(df.columns)
	if missing:
		raise KeyError(f"Missing required EVT columns in {path}: {sorted(missing)}")

	return df.copy()


def _select_datasets(evt_df: pd.DataFrame, methods):
	available = sorted(evt_df["dataset"].dropna().astype(str).unique().tolist(), key=_dataset_sort_key)

	if methods:
		requested = [str(m).strip() for m in methods]
		selected = [m for m in requested if m in available]
		missing = [m for m in requested if m not in available]
		for m in missing:
			print(f"Warning: dataset '{m}' not found in EVT summary and will be skipped.")
		return selected

	return available


def _model_row(evt_df: pd.DataFrame, dataset: str, model: str):
	sub = evt_df[(evt_df["dataset"] == dataset) & (evt_df["model"] == model)]
	if sub.empty:
		return None
	return sub.iloc[0]


def _conditional_empirical_survival(excess: np.ndarray):
	z = np.asarray(excess, dtype=float)
	z = z[np.isfinite(z) & (z >= 0)]
	if z.size == 0:
		return np.array([]), np.array([])

	z = np.sort(z)
	n = z.size
	# Weibull plotting position for a stable empirical survival in log-space.
	surv = (n - np.arange(1, n + 1) + 1) / (n + 1)
	return z, surv


def _gpd_survival_curve(xi: float, sigma: float, max_excess: float):
	if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
		return np.array([]), np.array([])

	z_max = float(max_excess)
	if z_max <= 0:
		return np.array([]), np.array([])

	if np.isclose(xi, 0.0):
		z = np.linspace(0.0, z_max, 300)
		surv = np.exp(-z / sigma)
		return z, surv

	if xi < 0:
		endpoint = -sigma / xi
		z_max = min(z_max, endpoint * 0.999)
		if z_max <= 0:
			return np.array([]), np.array([])

	z = np.linspace(0.0, z_max, 300)
	inside = 1.0 + xi * z / sigma
	inside = np.maximum(inside, 1e-12)
	surv = inside ** (-1.0 / xi)
	return z, surv


def _prepare_tail_records(location: str, evt_df: pd.DataFrame, datasets):
	records = []

	for dataset in datasets:
		gpd = _model_row(evt_df, dataset, "GPD")
		gev = _model_row(evt_df, dataset, "GEV")

		if gpd is None:
			print(f"Warning: no GPD row found for '{dataset}', skipping.")
			continue

		threshold = float(gpd["threshold"]) if pd.notna(gpd["threshold"]) else np.nan
		xi_gpd = float(gpd["xi"]) if pd.notna(gpd["xi"]) else np.nan
		sigma_gpd = float(gpd["sigma"]) if pd.notna(gpd["sigma"]) else np.nan

		xi_gev = np.nan
		if gev is not None and pd.notna(gev["xi"]):
			xi_gev = float(gev["xi"])

		path = _dataset_path(location, dataset)
		if not path.exists():
			print(f"Warning: data file for '{dataset}' not found at {path}, skipping.")
			continue

		if not np.isfinite(threshold):
			print(f"Warning: threshold missing for '{dataset}', skipping.")
			continue

		df = load_data(str(path))
		hs = df["hs"].to_numpy(dtype=float)
		excess = hs[hs > threshold] - threshold
		excess = excess[np.isfinite(excess) & (excess >= 0)]

		if excess.size < 20:
			print(
				f"Warning: too few exceedances for '{dataset}' "
				f"(n={excess.size}), skipping."
			)
			continue

		endpoint = np.nan
		if np.isfinite(xi_gpd) and np.isfinite(sigma_gpd) and xi_gpd < 0 and sigma_gpd > 0:
			endpoint = threshold + (-sigma_gpd / xi_gpd)

		records.append(
			{
				"dataset": dataset,
				"display": _display_name(dataset),
				"threshold": float(threshold),
				"xi_gpd": float(xi_gpd),
				"sigma_gpd": float(sigma_gpd),
				"xi_gev": float(xi_gev) if np.isfinite(xi_gev) else np.nan,
				"n_exceed": int(excess.size),
				"excess": excess,
				"endpoint_hs": float(endpoint) if np.isfinite(endpoint) else np.nan,
			}
		)

	return records


def _plot_tail_survival(location: str, records):
	out_dir = OUT_ROOT / location
	out_dir.mkdir(parents=True, exist_ok=True)

	fig, ax_tail = plt.subplots(figsize=(9.2, 6.0))

	cmap = plt.get_cmap("tab10")
	dataset_handles = []

	for idx, r in enumerate(records):
		color = cmap(idx % 10)
		z_emp, s_emp = _conditional_empirical_survival(r["excess"])
		if z_emp.size == 0:
			continue

		z_fit, s_fit = _gpd_survival_curve(r["xi_gpd"], r["sigma_gpd"], float(np.max(z_emp)))

		ax_tail.step(
			z_emp,
			s_emp,
			where="post",
			color=color,
			linewidth=1.7,
			alpha=0.75,
		)
		if z_fit.size > 0:
			ax_tail.plot(z_fit, s_fit, color=color, linewidth=2.1, linestyle="--")

		dataset_handles.append(
			Line2D(
				[0],
				[0],
				color=color,
				linewidth=2.0,
				label=f"{r['display']} (u={r['threshold']:.2f} m)",
			)
		)

	ax_tail.set_yscale("log")
	ax_tail.set_xlabel("Excess over threshold z = Hs - u (m)")
	ax_tail.set_ylabel("Conditional survival P(Hs - u > z | Hs > u)")
	ax_tail.set_title(f"{location} - Tail Survival (Empirical vs GPD)")
	ax_tail.grid(alpha=0.35, which="both")

	style_handles = [
		Line2D(
			[0],
			[0],
			color="black",
			linewidth=1.8,
			linestyle="-",
			label="Empirical exceedance (non-parametric)",
		),
		Line2D(
			[0],
			[0],
			color="black",
			linewidth=2.0,
			linestyle="--",
			label="GPD model fit (parametric)",
		),
	]
	leg1 = ax_tail.legend(handles=style_handles, loc="upper right", frameon=True)
	ax_tail.add_artist(leg1)
	ax_tail.legend(handles=dataset_handles, loc="lower left", frameon=True, fontsize=9)

	fig.suptitle(f"Tail survival comparison - {location}", fontsize=12)
	plt.tight_layout(rect=(0, 0, 1, 0.95))

	dataset_tag = "-".join(r["dataset"] for r in records)
	out_path = out_dir / f"tail_survival_{dataset_tag}.png"
	plt.savefig(out_path, dpi=300)
	plt.close()

	print(f"Saved {out_path}")


def _plot_shape_comparison(location: str, records):
	out_dir = OUT_ROOT / location
	out_dir.mkdir(parents=True, exist_ok=True)

	fig, ax = plt.subplots(figsize=(10.0, 5.8))

	xlabels = [r["display"] for r in records]
	xi_gpd_vals = [r["xi_gpd"] for r in records]
	xi_gev_vals = [r["xi_gev"] for r in records]
	x = np.arange(len(xlabels), dtype=float)

	ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
	ax.scatter(x - 0.09, xi_gpd_vals, marker="o", s=52, label="xi (GPD)")
	ax.scatter(x + 0.09, xi_gev_vals, marker="x", s=62, label="xi (GEV)")

	for i, xi_val in enumerate(xi_gpd_vals):
		if np.isfinite(xi_val):
			ax.annotate(
				f"{xi_val:.03f}",
				xy=(x[i] - 0.09, xi_val),
				xytext=(0, 6),
				textcoords="offset points",
				fontsize=8,
				ha="center",
				va="bottom",
			)

	for i, xi_val in enumerate(xi_gev_vals):
		if np.isfinite(xi_val):
			ax.annotate(
				f"{xi_val:.03f}",
				xy=(x[i] + 0.09, xi_val),
				xytext=(0, 6),
				textcoords="offset points",
				fontsize=8,
				ha="center",
				va="bottom",
			)

	ax.set_xticks(x)
	ax.set_xticklabels(xlabels, rotation=35, ha="right")
	ax.set_ylabel("Shape parameter xi")
	ax.set_title(f"Shape comparison - {location}")
	ax.grid(alpha=0.3)
	ax.legend(frameon=True)

	plt.tight_layout()

	dataset_tag = "-".join(r["dataset"] for r in records)
	out_path = out_dir / f"shape_comparison_{dataset_tag}.png"
	plt.savefig(out_path, dpi=300)
	plt.close()

	print(f"Saved {out_path}")


def plot(location: str, methods):
	evt_df = _load_evt_summary(location)
	datasets = _select_datasets(evt_df, methods)
	if not datasets:
		raise ValueError("No datasets available after filtering.")

	records = _prepare_tail_records(location, evt_df, datasets)
	if len(records) < 2:
		raise ValueError(
			"Need at least two valid datasets with EVT parameters and data files for comparison."
		)

	_plot_tail_survival(location, records)
	_plot_shape_comparison(location, records)


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Plot thesis-ready tail comparison using empirical exceedance survival and EVT shape "
			"parameters."
		)
	)
	parser.add_argument("--location", required=True, help="Target location.")
	parser.add_argument(
		"--method",
		nargs="+",
		metavar="DATASET",
		help=(
			"Datasets to compare, e.g. raw ensemble_fauskane transfer_fedjeosen_xgboost. "
			"If omitted, all datasets in evt_parameter_summary.csv are used."
		),
	)

	args = parser.parse_args()
	plot(args.location, args.method)


if __name__ == "__main__":
	main()
