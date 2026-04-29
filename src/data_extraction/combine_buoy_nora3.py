from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

# add location here: "vestfjorden", "fauskane", "fedjeosen", "bergen", "stavanger", "kristiansund"
LOCATION = "fauskane"


BUOY_COLUMNS = [
	"time",
	"longitude",
	"latitude",
	"Long_Crestedness_Parameters",
	"First_Order_Spread",
	"Mean_Spreading_Angle",
	"Wave_Period_Tz",
	"Wave_Period_Tmax",
	"Wave_Height_Trough",
	"Wave_Height_Crest",
	"Wave_Height_Hmax",
	"Wave_Height_Wind_Hm0",
	"Wave_Height_Swell_Hm0",
	"Wave_Peak_Period_Wind",
	"Wave_Peak_Period_Swell",
	"Wave_Peak_Period",
	"Wave_Mean_Period_Tm02",
	"Wave_Peak_Direction_Wind",
	"Wave_Mean_Direction",
	"Wave_Peak_Direction_Swell",
	"Wave_Peak_Direction",
	"Significant_Wave_Height_Hm0",
]

NORA3_COLUMNS = [
    "hs",
    "tp",
    "fpI",
    "tm1",
    "tm2",
    "tmp",
    "Pdir",
    "thq",
    "hs_sea",
    "tp_sea",
    "thq_sea",
    "hs_swell",
    "tp_swell",
    "thq_swell",
    "wind_speed_10m",
    "wind_speed_20m",
    "wind_speed_50m",
    "wind_speed_100m",
    "wind_speed_250m",
    "wind_speed_500m",
    "wind_speed_750m",
    "wind_direction_10m",
    "wind_direction_20m",
    "wind_direction_50m",
    "wind_direction_100m",
    "wind_direction_250m",
    "wind_direction_500m",
    "wind_direction_750m",
]

OUTPUT_COLUMNS = BUOY_COLUMNS + NORA3_COLUMNS


def parse_args() -> argparse.Namespace:
	script_dir = Path(__file__).resolve().parent
	project_root = script_dir.parent

	default_buoy = project_root / "data" / "input" / "buoys_data" / f"buoy_{LOCATION}_max.csv"
	default_nora3 = (
		project_root
		/ "data"
		/ "input"
		/ "nora3_locations"
		/ f"NORA3_wind_wave_{LOCATION}_1959_2025.csv"
	)
	default_output = (
		project_root
		/ "data"
		/ "input"
		/ "nora3_buoy_combined"
		/ f"NORA3_{LOCATION}_pairs.csv"
	)

	parser = argparse.ArgumentParser(
		description="Merge buoy hourly max data with NORA3 data on timestamp."
	)
	parser.add_argument(
		"--buoy-file",
		type=Path,
		default=default_buoy,
		help="Path to buoy CSV (hourly max format).",
	)
	parser.add_argument(
		"--nora3-file",
		type=Path,
		default=default_nora3,
		help="Path to NORA3 CSV (with comment lines starting with '#').",
	)
	parser.add_argument(
		"--output-file",
		type=Path,
		default=default_output,
		help="Path to write merged output CSV.",
	)
	return parser.parse_args()


def require_columns(fieldnames: list[str] | None, required: list[str], file_path: Path) -> None:
	if not fieldnames:
		raise ValueError(f"Missing header row in: {file_path}")

	missing = [column for column in required if column not in fieldnames]
	if missing:
		raise ValueError(
			f"Missing columns in {file_path}: {', '.join(missing)}"
		)


def load_nora3_by_time(nora3_file: Path) -> dict[str, dict[str, str]]:
	if not nora3_file.exists():
		raise FileNotFoundError(f"NORA3 file not found: {nora3_file}")

	with nora3_file.open("r", newline="", encoding="utf-8") as handle:
		filtered_lines = (line for line in handle if not line.startswith("#"))
		reader = csv.DictReader(filtered_lines)
		require_columns(reader.fieldnames, ["time"] + NORA3_COLUMNS, nora3_file)

		data: dict[str, dict[str, str]] = {}
		for row in reader:
			timestamp = (row.get("time") or "").strip()
			if not timestamp:
				continue
			data[timestamp] = {column: (row.get(column) or "") for column in NORA3_COLUMNS}
	return data


def merge_rows(
	buoy_file: Path, nora3_by_time: dict[str, dict[str, str]]
) -> tuple[list[dict[str, str]], int, int]:
	if not buoy_file.exists():
		raise FileNotFoundError(f"Buoy file not found: {buoy_file}")

	merged_rows: list[dict[str, str]] = []
	skipped_missing_hs = 0
	skipped_missing_nora3 = 0

	with buoy_file.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		require_columns(reader.fieldnames, BUOY_COLUMNS, buoy_file)

		for row in reader:
			buoy_hs = (row.get("Significant_Wave_Height_Hm0") or "").strip()
			if not buoy_hs:
				skipped_missing_hs += 1
				continue

			timestamp = (row.get("time") or "").strip()
			if not timestamp:
				skipped_missing_hs += 1
				continue

			nora3_row = nora3_by_time.get(timestamp)
			if nora3_row is None:
				skipped_missing_nora3 += 1
				continue

			out_row = {column: (row.get(column) or "") for column in BUOY_COLUMNS}
			out_row.update(nora3_row)
			merged_rows.append(out_row)

	return merged_rows, skipped_missing_hs, skipped_missing_nora3


def write_output(output_file: Path, rows: list[dict[str, str]]) -> None:
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	args = parse_args()
	nora3_by_time = load_nora3_by_time(args.nora3_file)
	merged_rows, skipped_missing_hs, skipped_missing_nora3 = merge_rows(
		args.buoy_file, nora3_by_time
	)
	write_output(args.output_file, merged_rows)

	timestamps = [datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S") for r in merged_rows]
	total_gaps = sum(
		1 for a, b in zip(timestamps, timestamps[1:]) if (b - a).total_seconds() > 3600
	)

	print(f"Wrote merged dataset: {args.output_file}")
	print(f"Rows written: {len(merged_rows)}")
	print(f"Skipped buoy rows with missing Significant_Wave_Height_Hm0/time: {skipped_missing_hs}")
	print(f"Skipped buoy rows without matching NORA3 timestamp: {skipped_missing_nora3}")
	print(f"Total gaps: {total_gaps}")


if __name__ == "__main__":
	main()
