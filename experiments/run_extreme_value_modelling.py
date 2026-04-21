import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extreme_value_modelling.extreme_preprocessing import run as preprocess
from src.extreme_value_modelling.fit_gev import run as run_gev
from src.extreme_value_modelling.fit_gpd import run as run_gpd
from src.extreme_value_modelling.diagnostics import run as run_diagnostics

from src.extreme_value_modelling.common import (
    append_return_level_summary,
    dataset_name,
)
from src.extreme_value_modelling.paths import resolve_input_path
from src.settings import (
    get_all_locations,
    get_core_buoy_locations,
    get_external_validation_buoys,
    get_methods,
    get_study_area_locations,
)


def _location_role(location: str) -> str:
    if location in get_core_buoy_locations():
        return "core"
    if location in get_external_validation_buoys():
        return "external"
    if location in get_study_area_locations():
        return "study_area"
    raise ValueError(f"Unknown location role for '{location}'")


def _validate_method(method: str) -> str:
    methods = [*get_methods(), "ensemble"]
    if method in methods:
        return method
    raise ValueError(f"Unknown correction method '{method}'. Available methods: {methods}")


def _spec(mode: str, corr_method: str = "pqm", transfer_source: str | None = None):
    return mode, corr_method, transfer_source


def _input_exists(location: str, mode: str, corr_method: str = "pqm", transfer_source: str | None = None) -> bool:
    return resolve_input_path(
        location=location,
        mode=mode,
        corr_method=corr_method,
        transfer_source=transfer_source,
    ).exists()


def _ensemble_output_names(location: str):
    core_buoys = get_core_buoy_locations()
    names = [f"ensemble_{source}" for source in core_buoys]
    if _location_role(location) in {"external", "study_area"} and len(core_buoys) > 1:
        names.append("ensemble_combined")
    return names


def _standard_specs(location: str, method: str):
    role = _location_role(location)
    sources = get_core_buoy_locations()
    specs = []

    if role != "study_area":
        specs.append(_spec("corrected", method))

    transfer_sources = [source for source in sources if role != "core" or source != location]
    specs.extend(_spec("corrected", method, source) for source in transfer_sources)
    return specs


def _dataset_specs(location: str, method: str | None = None):
    specs = [_spec("raw")]
    if method == "ensemble":
        specs.extend(_spec("corrected", name) for name in _ensemble_output_names(location))
    elif method is not None:
        specs.extend(_standard_specs(location, method))
    else:
        for name in get_methods():
            specs.extend(_standard_specs(location, name))
        specs.extend(_spec("corrected", name) for name in _ensemble_output_names(location))

    seen = set()
    ordered = []
    for spec in specs:
        if spec in seen:
            continue
        seen.add(spec)
        ordered.append(spec)
    return ordered


def run_dataset(location: str, mode: str, corr_method: str = "pqm", transfer_source: str | None = None, diagnostics: bool = False):
    dataset = dataset_name(mode, corr_method=corr_method, transfer_source=transfer_source)

    if not _input_exists(location, mode, corr_method=corr_method, transfer_source=transfer_source):
        print(f"\nSkipping EVT for {location} : {dataset} (input file not found)")
        return None

    print(f"\nRunning EVT for {location} : {dataset}")

    preprocess(location=location, mode=mode, corr_method=corr_method, transfer_source=transfer_source)
    append_return_level_summary(location, dataset, "GEV", run_gev(location, mode, corr_method, transfer_source))
    append_return_level_summary(location, dataset, "GPD", run_gpd(location, mode, corr_method, transfer_source))

    if diagnostics:
        run_diagnostics(location=location, mode=mode, corr_method=corr_method, transfer_source=transfer_source)

    return dataset


def run_location(location: str, method: str | None = None, diagnostics: bool = False):
    method = _validate_method(method) if method is not None else None
    print(f"\n==============================")
    print(f"LOCATION: {location}")
    print(f"METHOD:   {method or 'all'}")
    print(f"==============================")

    for mode, corr_method, transfer_source in _dataset_specs(location, method):
        run_dataset(
            location=location,
            mode=mode,
            corr_method=corr_method,
            transfer_source=transfer_source,
            diagnostics=diagnostics,
        )


def run_all_for_method(method: str, diagnostics: bool = False):
    for location in get_all_locations():
        run_location(location, method=method, diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser(
        description="Run EVT for one location, one method, or one method across all locations."
    )
    parser.add_argument("--location", help="Optional target location.")
    parser.add_argument("--method", help="Optional correction method. Use 'ensemble' for ensemble datasets.")
    parser.add_argument("--diagnostics", action="store_true", help="Run EVT diagnostic plots.")

    args = parser.parse_args()

    if args.location is None and args.method is None:
        parser.error("You must provide at least one of --location or --method.")

    if args.location is not None:
        run_location(location=args.location, method=args.method, diagnostics=args.diagnostics)
        return

    run_all_for_method(method=args.method, diagnostics=args.diagnostics)


if __name__ == "__main__":
    main()
