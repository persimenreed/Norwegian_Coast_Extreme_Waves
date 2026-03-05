import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


SUMMARY_ROOT = Path("results/extreme_value_modelling")
RESULT_DIR = Path("results/extreme_value_analysis")
RETURN_PERIODS = [10.0, 20.0, 50.0]


def load_summary(location):

	path = SUMMARY_ROOT / location / "summary_return_levels.csv"

	if not path.exists():
		raise FileNotFoundError(f"Summary file not found: {path}")

	return pd.read_csv(path)


def discover_locations():

	locations = []

	for entry in sorted(SUMMARY_ROOT.iterdir()):
		if not entry.is_dir():
			continue
		if (entry / "summary_return_levels.csv").exists():
			locations.append(entry.name)

	if not locations:
		raise FileNotFoundError(
			f"No location folders with summary_return_levels.csv found under {SUMMARY_ROOT}"
		)

	return locations


def discover_datasets(locations):

	available = set()

	for location in locations:
		df = load_summary(location)
		available.update(df["dataset"].dropna().unique().tolist())

	def sort_key(name):
		if name == "raw":
			return (0, name)
		if name.startswith("local_"):
			return (1, name)
		if name.startswith("pooled_"):
			return (2, name)
		return (3, name)

	return sorted(available, key=sort_key)


def dataset_tag(datasets):
	"""
	Create short readable filename tag
	"""

	tags = []

	for dataset_name in datasets:

		if dataset_name == "raw":
			tags.append("raw")

		elif dataset_name.startswith("local_"):
			tags.append(dataset_name.replace("local_", ""))

		elif dataset_name.startswith("pooled_"):
			tags.append(dataset_name.replace("pooled_", "") + "-pool")

		else:
			tags.append(dataset_name)

	return "-".join(tags)


def dataset_label(dataset_name):
	if dataset_name == "raw":
		return "raw"
	if dataset_name.startswith("local_"):
		return dataset_name.replace("local_", "")
	if dataset_name.startswith("pooled_"):
		return dataset_name.replace("pooled_", "") + "-pool"
	if dataset_name.startswith("transfer_"):
		return dataset_name.replace("transfer_", "") + "-transfer"
	return dataset_name


def dataset_group(dataset_name):
	if dataset_name == "raw":
		return "raw"
	for prefix in ("local_", "pooled_", "transfer_"):
		if dataset_name.startswith(prefix):
			return dataset_name[len(prefix):]
	return dataset_name


def group_sort_key(group_name):
	if group_name == "raw":
		return (0, group_name)
	return (1, group_name)


def variant_sort_key(dataset_name):
	if dataset_name == "raw":
		return 0
	if dataset_name.startswith("local_"):
		return 1
	if dataset_name.startswith("pooled_"):
		return 2
	if dataset_name.startswith("transfer_"):
		return 3
	return 4


def extract_return_level(df, dataset_name, model, period):

	subset = df[(df["dataset"] == dataset_name) & (df["model"] == model)].copy()

	if subset.empty:
		return np.nan

	row = subset[np.isclose(subset["return_period"].astype(float), float(period))]

	if row.empty:
		return np.nan

	return float(row.iloc[0]["return_level"])


def plot_model_on_axis(ax, location, datasets, model, ymin):

	colors = {
		"10": "#4C78A8",
		"20": "#F58518",
		"50": "#54A24B",
	}

	bars = []
	bar_width = 0.60
	group_gap = 0.85
	cursor = 0.0
	max_rl50 = 0.0

	df_loc = load_summary(location)

	grouped = {}
	for dataset_name in datasets:
		grouped.setdefault(dataset_group(dataset_name), []).append(dataset_name)

	for group_name in sorted(grouped.keys(), key=group_sort_key):
		group_has_bars = False
		for dataset_name in sorted(grouped[group_name], key=variant_sort_key):
			rl10 = extract_return_level(df_loc, dataset_name, model, RETURN_PERIODS[0])
			rl20 = extract_return_level(df_loc, dataset_name, model, RETURN_PERIODS[1])
			rl50 = extract_return_level(df_loc, dataset_name, model, RETURN_PERIODS[2])

			if np.isnan(rl10) or np.isnan(rl20) or np.isnan(rl50):
				continue

			x = cursor
			cursor += bar_width

			bars.append(
				{
					"x": x,
					"dataset": dataset_name,
					"group": group_name,
					"rl10": rl10,
					"rl20": rl20,
					"rl50": rl50,
				}
			)
			max_rl50 = max(max_rl50, rl50)
			group_has_bars = True

		if group_has_bars:
			cursor += group_gap

	if not bars:
		print(f"No valid bars to plot for model {model} at {location}")
		return False

	label_offset = max(max_rl50 * 0.02, 0.12)

	def segment_text_y(bottom, height):
		top = bottom + height
		y = top - label_offset
		min_inside = bottom + max(0.15 * height, 0.1)
		if y < min_inside:
			y = bottom + 0.55 * height
		return y

	bar_x = []
	bar_labels = []
	group_centers = []
	current_group = None
	current_group_x = []

	for item in bars:
		x = item["x"]
		group_name = item["group"]
		rl10 = item["rl10"]
		rl20 = item["rl20"]
		rl50 = item["rl50"]
		seg10 = rl10
		seg20 = max(rl20 - rl10, 0.0)
		seg50 = max(rl50 - rl20, 0.0)

		ax.bar(
			x,
			seg10,
			width=bar_width,
			color=colors["10"],
			edgecolor="black",
			linewidth=0.6,
			zorder=3,
		)
		ax.bar(
			x,
			seg20,
			width=bar_width,
			bottom=seg10,
			color=colors["20"],
			edgecolor="black",
			linewidth=0.6,
			zorder=3,
		)
		ax.bar(
			x,
			seg50,
			width=bar_width,
			bottom=seg10 + seg20,
			color=colors["50"],
			edgecolor="black",
			linewidth=0.6,
			zorder=3,
		)

		ax.text(
			x,
			segment_text_y(0.0, rl10),
			f"{rl10:.1f}",
			ha="center",
			va="top",
			fontsize=7,
			zorder=4,
		)
		ax.text(
			x,
			segment_text_y(seg10, seg20),
			f"{rl20:.1f}",
			ha="center",
			va="top",
			fontsize=7,
			zorder=4,
		)
		ax.text(
			x,
			segment_text_y(seg10 + seg20, seg50),
			f"{rl50:.1f}",
			ha="center",
			va="top",
			fontsize=7,
			zorder=4,
		)

		bar_x.append(x)
		bar_labels.append(dataset_label(item["dataset"]))

		if group_name != current_group:
			if current_group is not None and current_group_x:
				group_centers.append((np.mean(current_group_x), current_group))
			current_group = group_name
			current_group_x = [x]
		else:
			current_group_x.append(x)

	if current_group is not None and current_group_x:
		group_centers.append((np.mean(current_group_x), current_group))

	ax.set_xticks(bar_x)
	ax.set_xticklabels(bar_labels, rotation=30, ha="right")
	ax.set_ylabel("Return levels Hs (m)")
	ax.set_title(f"{model}")
	ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

	# for center, group_name in group_centers:
	# 	ax.text(
	# 		center,
	# 		-0.14,
	# 		group_name,
	# 		transform=ax.get_xaxis_transform(),
	# 		ha="center",
	# 		va="top",
	# 		fontsize=9,
	# 		fontweight="bold",
	# 	)

	y_top = max_rl50 * 1.08
	if ymin is not None:
		ax.set_ylim(bottom=ymin, top=y_top)
	else:
		ax.set_ylim(bottom=0.0, top=y_top)

	period_handles = [
		Patch(facecolor=colors["10"], edgecolor="black", label="10-year"),
		Patch(facecolor=colors["20"], edgecolor="black", label="20-year"),
		Patch(facecolor=colors["50"], edgecolor="black", label="50-year"),
	]
	ax.legend(
		handles=period_handles,
		title="Return period",
		loc="upper left",
	)

	return True


def plot_location(location, datasets, ymin):

	fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

	gev_ok = plot_model_on_axis(axes[0], location, datasets, "GEV", ymin)
	gpd_ok = plot_model_on_axis(axes[1], location, datasets, "GPD", ymin)

	if not gev_ok and not gpd_ok:
		plt.close()
		print(f"No valid bars to plot for location {location}")
		return

	fig.suptitle(f"Return levels by dataset — {location}")

	out_path = (
		RESULT_DIR
		/ "return_level"
		/ {location}
		/ f"return_levels_{location}_bar.png"
	)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	fig.tight_layout(rect=(0, 0, 1, 0.95))
	plt.savefig(out_path, dpi=300)
	plt.close()

	print(f"Saved: {out_path}")


def main():

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--locations",
		nargs="+",
		default=None,
		help="Locations to include. Default: all discovered locations.",
	)
	parser.add_argument(
		"--datasets",
		nargs="+",
		default=None,
		help="Datasets to include. Default: all discovered datasets across locations.",
	)
	parser.add_argument(
		"--ymin",
		type=float,
		default=10.0,
		help="Lower y-axis bound in meters (default: 10.0).",
	)

	args = parser.parse_args()

	locations = args.locations if args.locations is not None else discover_locations()
	datasets = args.datasets if args.datasets is not None else discover_datasets(locations)

	for location in locations:
		plot_location(location, datasets, args.ymin)


if __name__ == "__main__":
	main()
