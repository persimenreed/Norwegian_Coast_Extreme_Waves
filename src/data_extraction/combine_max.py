from __future__ import annotations

import csv
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path


OUTPUT_COLUMNS = [
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

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_time(value: str) -> datetime:
	return datetime.strptime(value.strip(), TIME_FORMAT)


def parse_hs(row: dict[str, str]) -> float | None:
	raw = (row.get("Significant_Wave_Height_Hm0") or "").strip()
	if not raw:
		return None
	try:
		return float(raw)
	except ValueError:
		return None


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	input_dir = script_dir / "out" / "vestfjorden"
	output_dir = script_dir / "out" / "combined"
	output_file = output_dir / "buoy_vestfjorden_max.csv"

	input_files = sorted(input_dir.glob("*.csv"))
	if not input_files:
		raise FileNotFoundError(f"No CSV files found in: {input_dir}")

	# hour -> (max_hs, selected_row_values)
	hourly_max: OrderedDict[datetime, tuple[float, dict[str, str]]] = OrderedDict()

	for csv_file in input_files:
		with csv_file.open("r", newline="", encoding="utf-8") as handle:
			reader = csv.DictReader(handle)

			if not reader.fieldnames:
				continue
			missing_columns = [c for c in OUTPUT_COLUMNS if c not in reader.fieldnames]
			if missing_columns:
				raise ValueError(
					f"Missing expected columns in {csv_file.name}: {', '.join(missing_columns)}"
				)

			for row in reader:
				time_raw = (row.get("time") or "").strip()
				if not time_raw:
					continue

				try:
					row_time = parse_time(time_raw)
				except ValueError:
					continue

				hs = parse_hs(row)
				if hs is None:
					continue

				hour = row_time.replace(minute=0, second=0, microsecond=0)
				current = hourly_max.get(hour)
				if current is None or hs > current[0]:
					selected_row = {column: (row.get(column) or "") for column in OUTPUT_COLUMNS}
					# Force hourly timestamp for output series.
					selected_row["time"] = hour.strftime(TIME_FORMAT)
					hourly_max[hour] = (hs, selected_row)

	if not hourly_max:
		raise ValueError("No valid rows with Significant_Wave_Height_Hm0 were found.")

	output_dir.mkdir(parents=True, exist_ok=True)

	min_hour = min(hourly_max)
	max_hour = max(hourly_max)

	with output_file.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
		writer.writeheader()

		current_hour = min_hour
		while current_hour <= max_hour:
			selected = hourly_max.get(current_hour)
			if selected is not None:
				writer.writerow(selected[1])
			else:
				gap_row = {column: "" for column in OUTPUT_COLUMNS}
				gap_row["time"] = current_hour.strftime(TIME_FORMAT)
				writer.writerow(gap_row)
			current_hour += timedelta(hours=1)

	print(f"Wrote combined hourly max file: {output_file}")


if __name__ == "__main__":
	main()
