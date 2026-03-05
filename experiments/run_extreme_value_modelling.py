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

from src.extreme_value_modelling.common import dataset_name, append_return_level_summary, build_evt_summary_metrics
from src.settings import get_methods, get_locations, get_study_area_locations


# ==========================================================
# RUN SINGLE DATASET
# ==========================================================

def run_single(location: str,
               mode: str,
               corr_method: str = "qm",
               pooling: bool = False,
               diagnostics: bool = False,
               transfer: bool = False):

    dataset = dataset_name(mode, corr_method=corr_method, pooling=pooling, transfer=transfer)

    print(f"\nRunning EVT for {location} : {dataset}")

    preprocess(
        location=location,
        mode=mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer=transfer
    )

    gev_df = run_gev(
        location=location,
        mode=mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer=transfer
    )

    gpd_df = run_gpd(
        location=location,
        mode=mode,
        corr_method=corr_method,
        pooling=pooling,
        transfer=transfer
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
            transfer=transfer
        )


# ==========================================================
# RUN ALL LOCATIONS FOR ONE CORRECTION METHOD
# ==========================================================

def run_all_for_corr_method(corr_method: str, diagnostics: bool = False):

    methods = get_methods()

    if corr_method not in methods:
        print(f"WARNING: corr_method='{corr_method}' not in get_methods()={methods}. Continuing anyway.")

    locations = get_locations()
    study_areas = set(get_study_area_locations())

    for location in locations:

        print(f"\n==============================")
        print(f"LOCATION: {location}")
        print(f"==============================")

        # RAW
        run_single(
            location=location,
            mode="raw",
            diagnostics=diagnostics
        )

        # corrected local
        if location not in study_areas:
            run_single(
                location=location,
                mode="corrected",
                corr_method=corr_method,
                pooling=False,
                diagnostics=diagnostics
            )

        # corrected pooled
        run_single(
            location=location,
            mode="corrected",
            corr_method=corr_method,
            pooling=True,
            diagnostics=diagnostics
        )

        # OBSERVED
        if location not in study_areas:
            run_single(
                location=location,
                mode="observed",
                diagnostics=diagnostics
            )


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(description="Run Extreme Value Modelling pipeline.")

    parser.add_argument("--location", help="Location name (e.g. fauskane). If omitted, runs all.")

    parser.add_argument(
        "--mode",
        choices=["raw", "corrected"],
        help="If set, run only this mode (single location)."
    )

    parser.add_argument(
        "--corr-method",
        default="qm",
        help="Correction method."
    )

    parser.add_argument(
        "--pooling",
        action="store_true",
        help="Use pooled corrected dataset (single run)."
    )

    parser.add_argument(
        "--all-for-corr-method",
        action="store_true",
        help="Run all locations for the given --corr-method."
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run EVT diagnostic plots."
    )

    args = parser.parse_args()

    # ======================================================
    # RUN ALL LOCATIONS
    # ======================================================

    if args.all_for_corr_method:

        run_all_for_corr_method(
            corr_method=args.corr_method,
            diagnostics=args.diagnostics
        )

        return

    # ======================================================
    # SINGLE LOCATION REQUIRED
    # ======================================================

    if not args.location:
        raise ValueError("Either provide --location or use --all-for-corr-method.")

    # ======================================================
    # SINGLE MODE RUN
    # ======================================================

    if args.mode:

        run_single(
            location=args.location,
            mode=args.mode,
            corr_method=args.corr_method,
            pooling=args.pooling,
            diagnostics=args.diagnostics
        )

        return

    # ======================================================
    # DEFAULT FULL PIPELINE
    # ======================================================

    location = args.location
    methods = get_methods()
    study_areas = set(get_study_area_locations())

    print(f"\nRunning full EVT pipeline for location: {location}")

    # RAW
    run_single(
        location=location,
        mode="raw",
        diagnostics=args.diagnostics
    )

    # CORRECTED METHODS
    for method in methods:

        if location not in study_areas:

            # local
            run_single(
                location=location,
                mode="corrected",
                corr_method=method,
                pooling=False,
                diagnostics=args.diagnostics
            )

            # transfer
            run_single(
                location=location,
                mode="corrected",
                corr_method=method,
                pooling=False,
                transfer=True,
                diagnostics=args.diagnostics
            )

        run_single(
            location=location,
            mode="corrected",
            corr_method=method,
            pooling=True,
            diagnostics=args.diagnostics
        )

    from src.settings import get_buoy_locations

    if args.location:
        if args.location in get_buoy_locations():
            build_evt_summary_metrics(args.location)

    print("\nEVT pipeline complete.")


if __name__ == "__main__":
    main()