import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ensemble.weighted import run as run_simple
from src.ensemble.xgboost_blend import run as run_xgboost
from src.settings import get_external_validation_buoys, get_methods, get_study_area_locations


def _default_locations():
    out = []
    for location in get_external_validation_buoys() + get_study_area_locations():
        if location not in out:
            out.append(location)
    return out


def _print_paths(title, paths):
    if not paths:
        print(f"\n{title}: none")
        return

    print(f"\n{title}:")
    for location, path in sorted(paths.items()):
        print(f"  {location}: {path}")


def _print_simple_summary(res):
    print(f"\nSimple ensemble completed: {res['name']}")
    print(f"Training locations: {', '.join(res['train_locations'])}")
    print(f"Selected members: {', '.join(res['selected_methods'])}")

    print("\nSimple member weights:")
    for name, weight in sorted(res["weights"].items()):
        print(f"  {name}: {weight:.6f}")

    metrics = res["member_metrics"][["rank_score", "rmse", "twrmse", "tail_rmse_95"]]
    print("\nSimple member ranking:")
    print(metrics.round(6).to_string())

    _print_paths("Validation outputs", res["validation_paths"])
    _print_paths("Hindcast outputs", res["hindcast_paths"])


def _print_xgboost_summary(res):
    print(f"\nXGBoost ensemble completed: {res['name']}")
    print(f"Training locations: {', '.join(res['train_locations'])}")

    print("\nWinner counts in training labels:")
    for name, count in sorted(res["class_counts"].items()):
        print(f"  {name}: {count}")

    if res["top_features"]:
        print("\nTop XGBoost features:")
        for name, score in res["top_features"][:10]:
            print(f"  {name}: {score:.6f}")

    _print_paths("Validation outputs", res["validation_paths"])
    _print_paths("Hindcast outputs", res["hindcast_paths"])


def main():
    parser = argparse.ArgumentParser(
        description="Run simple and/or XGBoost pooled ensembles."
    )

    parser.add_argument(
        "--locations",
        nargs="*",
        default=None,
        help=(
            "Locations to apply the ensemble to. "
            f"Defaults to external validation + study areas: {_default_locations()}"
        ),
    )

    parser.add_argument(
        "--train-locations",
        nargs="*",
        default=None,
        help=(
            "Locations with pooled validation members used for training. "
            f"Defaults to available external validation buoys: {get_external_validation_buoys()}"
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=f"Subset of input members (default = all non-ensemble methods: {get_methods()})",
    )

    parser.add_argument(
        "--which",
        choices=["simple", "xgboost", "both"],
        default="both",
        help="Which ensemble implementation to run.",
    )

    parser.add_argument(
        "--simple-top-k",
        type=int,
        default=3,
        help="Number of members to keep in the simple average.",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of contiguous folds used to create validation ensemble outputs.",
    )

    args = parser.parse_args()
    locations = args.locations or _default_locations()

    if args.which in {"simple", "both"}:
        res_simple = run_simple(
            train_locations=args.train_locations,
            target_locations=locations,
            methods=args.methods,
            top_k=args.simple_top_k,
            cv_folds=args.cv_folds,
        )
        _print_simple_summary(res_simple)

    if args.which in {"xgboost", "both"}:
        res_xgb = run_xgboost(
            train_locations=args.train_locations,
            target_locations=locations,
            methods=args.methods,
            cv_folds=args.cv_folds,
        )
        _print_xgboost_summary(res_xgb)


if __name__ == "__main__":
    main()
