import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.settings import (
    get_core_buoy_locations,
    get_external_validation_buoys,
    get_study_area_locations,
)

ENSEMBLE_MODELS = ["linear", "pqm", "gpr", "xgboost", "transformer", "dagqm"]

def _print_paths(title, paths):
    if not paths:
        print(f"\n{title}: none")
        return

    print(f"\n{title}:")
    for location, path in sorted(paths.items()):
        print(f"  {location}: {path}")


def _print_summary(res):
    print(f"\nEnsemble completed: {res['name']}")
    print(f"Training cases: {', '.join(res['training_labels'])}")
    print(f"Apply member family: {res['application_member_family']}")

    if res["top_features"]:
        print("\nTop XGBoost features:")
        if isinstance(res["top_features"], dict):
            for section, values in res["top_features"].items():
                print(f"  {section}:")
                for name, score in values[:10]:
                    print(f"    {name}: {score:.6f}")
        else:
            for name, score in res["top_features"][:10]:
                print(f"  {name}: {score:.6f}")

    print(f"\nReport: {res['report_path']}")
    _print_paths("Validation outputs", res["validation_paths"])
    _print_paths("Hindcast outputs", res["hindcast_paths"])
    _print_paths("Weight summaries", res.get("weight_summary_paths", {}))


def _ensemble_jobs_for_location(location):
    core_buoys = get_core_buoy_locations()
    external_buoys = set(get_external_validation_buoys())
    study_areas = set(get_study_area_locations())

    if location in core_buoys:
        jobs = [
            {
                "source": location,
                "combined": False,
                "output_name": f"ensemble_{location}",
            }
        ]
        jobs.extend(
            [
                {
                    "source": source,
                    "combined": False,
                    "output_name": f"ensemble_{source}",
                }
                for source in core_buoys
                if source != location
            ]
        )
        return jobs

    if location in external_buoys or location in study_areas:
        jobs = [
            {
                "source": source,
                "combined": False,
                "output_name": f"ensemble_{source}",
            }
            for source in core_buoys
        ]
        if len(core_buoys) > 1:
            jobs.append(
                {
                    "source": None,
                    "combined": True,
                    "output_name": "ensemble_combined",
                }
            )
        return jobs

    raise ValueError(f"Unknown location role for '{location}'")


def _all_locations():
    out = []
    for location in (
        get_core_buoy_locations()
        + get_external_validation_buoys()
        + get_study_area_locations()
    ):
        if location not in out:
            out.append(location)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run the XGBoost ensemble datasets used in the thesis pipeline."
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Optional target location.",
    )
    args = parser.parse_args()

    from src.ensemble.xgboost_ensemble_transfer import run as run_ensemble

    locations = [args.location] if args.location else _all_locations()

    for location in locations:
        print(f"\n==============================")
        print(f"LOCATION: {location}")
        print(f"==============================")
        print(f"Ensemble members: {', '.join(ENSEMBLE_MODELS)}")

        for job in _ensemble_jobs_for_location(location):
            res = run_ensemble(
                location=location,
                methods=ENSEMBLE_MODELS,
                source=job["source"],
                combined=job["combined"],
                output_name=job["output_name"],
            )
            _print_summary(res)


if __name__ == "__main__":
    main()
