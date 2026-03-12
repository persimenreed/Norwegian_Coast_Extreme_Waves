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
    dataset_name,
    append_return_level_summary,
    build_evt_summary_metrics,
)
from src.extreme_value_modelling.paths import resolve_input_path
from src.settings import (
    get_methods,
    get_all_locations,
    get_core_buoy_locations,
    get_external_validation_buoys,
    get_study_area_locations,
)


def _validate_method(method: str) -> str:
    methods = list(get_methods()) + ["ensemble"]
    if method not in methods:
        raise ValueError(
            f"Unknown correction method '{method}'. Available methods: {methods}"
        )

    return method


def _input_exists(
    location: str,
    mode: str,
    corr_method: str = "pqm",
    pooling: bool = False,
    transfer_source: str | None = None,
) -> bool:
    path = resolve_input_path(
        location=location,
        mode=mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer_source=transfer_source,
    )
    return Path(path).exists()


def run_single(
    location: str,
    mode: str,
    corr_method: str = "pqm",
    pooling: bool = False,
    transfer_source: str | None = None,
    diagnostics: bool = False,
):
    dataset = dataset_name(
        mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer_source=transfer_source,
    )

    if not _input_exists(
        location=location,
        mode=mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer_source=transfer_source,
    ):
        print(f"\nSkipping EVT for {location} : {dataset} (input file not found)")
        return None

    print(f"\nRunning EVT for {location} : {dataset}")

    preprocess(
        location=location,
        mode=mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer_source=transfer_source,
    )

    gev_df = run_gev(
        location=location,
        mode=mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer_source=transfer_source,
    )

    gpd_df = run_gpd(
        location=location,
        mode=mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer_source=transfer_source,
    )

    append_return_level_summary(
        location=location,
        dataset=dataset,
        model="GEV",
        table=gev_df
    )

    append_return_level_summary(
        location=location,
        dataset=dataset,
        model="GPD",
        table=gpd_df
    )

    if diagnostics:
        run_diagnostics(
            location=location,
            mode=mode,
            corr_method=corr_method,
            pooling=pooling,
            transfer_source=transfer_source,
        )

    return dataset


def run_location_all_methods(location: str, diagnostics: bool = False):
    core_buoys = set(get_core_buoy_locations())
    external_buoys = set(get_external_validation_buoys())
    study_areas = set(get_study_area_locations())
    methods = get_methods()

    print(f"\nRunning full EVT pipeline for location: {location}")

    run_single(
        location=location,
        mode="raw",
        diagnostics=diagnostics,
    )

    if location in core_buoys:
        for method in methods:
            run_single(
                location=location,
                mode="corrected",
                corr_method=method,
                pooling=False,
                diagnostics=diagnostics,
            )

        for source in get_core_buoy_locations():
            if source == location:
                continue
            for method in methods:
                run_single(
                    location=location,
                    mode="corrected",
                    corr_method=method,
                    transfer_source=source,
                    diagnostics=diagnostics,
                )

    elif location in external_buoys:
        for source in get_core_buoy_locations():
            for method in methods:
                run_single(
                    location=location,
                    mode="corrected",
                    corr_method=method,
                    transfer_source=source,
                    diagnostics=diagnostics,
                )

        for method in methods:
            run_single(
                location=location,
                mode="corrected",
                corr_method=method,
                pooling=True,
                diagnostics=diagnostics,
            )

    elif location in study_areas:
        for method in methods:
            run_single(
                location=location,
                mode="corrected",
                corr_method=method,
                pooling=True,
                diagnostics=diagnostics,
            )

    else:
        raise ValueError(f"Unknown location role for '{location}'")

    build_evt_summary_metrics(location)
    print("\nEVT pipeline complete.")

def run_all_for_method(method: str, diagnostics: bool = False):
    method = _validate_method(method)

    if method == "ensemble":

        location = "vestfjorden"

        print(f"\n==============================")
        print(f"LOCATION: {location}")
        print(f"METHOD:   ensemble")
        print(f"==============================")

        run_single(
            location=location,
            mode="corrected",
            corr_method="ensemble",
            diagnostics=diagnostics,
        )

        build_evt_summary_metrics(location)
        return

    for location in get_all_locations():

        print(f"\n==============================")
        print(f"LOCATION: {location}")
        print(f"METHOD:   {method}")
        print(f"==============================")

        # raw baseline
        run_single(
            location=location,
            mode="raw",
            diagnostics=diagnostics
        )

        if location in set(get_core_buoy_locations()):

            run_single(
                location=location,
                mode="corrected",
                corr_method=method,
                diagnostics=diagnostics
            )

            for source in get_core_buoy_locations():
                if source == location:
                    continue

                run_single(
                    location=location,
                    mode="corrected",
                    corr_method=method,
                    transfer_source=source,
                    diagnostics=diagnostics
                )

        elif location in set(get_external_validation_buoys()):

            for source in get_core_buoy_locations():

                run_single(
                    location=location,
                    mode="corrected",
                    corr_method=method,
                    transfer_source=source,
                    diagnostics=diagnostics
                )

            run_single(
                location=location,
                mode="corrected",
                corr_method=method,
                pooling=True,
                diagnostics=diagnostics
            )

        elif location in set(get_study_area_locations()):

            run_single(
                location=location,
                mode="corrected",
                corr_method=method,
                pooling=True,
                diagnostics=diagnostics
            )

        build_evt_summary_metrics(location)


def main():
    parser = argparse.ArgumentParser(
        description="Run EVT either for one location (all methods) or one method (all locations)."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--location",
        help="Run EVT for this location across all methods.",
    )
    group.add_argument(
        "--method",
        help="Run EVT for this method across all locations.",
    )

    parser.add_argument("--diagnostics", action="store_true", help="Run EVT diagnostic plots.")

    args = parser.parse_args()

    if args.location:
        run_location_all_methods(
            location=args.location,
            diagnostics=args.diagnostics,
        )
        return

    run_all_for_method(
        method=args.method,
        diagnostics=args.diagnostics,
    )


if __name__ == "__main__":
    main()