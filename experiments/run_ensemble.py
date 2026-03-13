import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ensemble.xgboost_blend import run as run_pooled_ensemble


def _print_paths(title, paths):
    if not paths:
        print(f"\n{title}: none")
        return

    print(f"\n{title}:")
    for location, path in sorted(paths.items()):
        print(f"  {location}: {path}")


def _print_summary(label, res):
    print(f"\n{label} completed: {res['name']}")
    print(f"Training cases: {', '.join(res['training_labels'])}")
    print(f"Apply member family: {res['application_member_family']}")

    print("\nWinner counts in training labels:")
    for name, count in sorted(res["class_counts"].items()):
        print(f"  {name}: {count}")

    if res["top_features"]:
        print("\nTop XGBoost features:")
        for name, score in res["top_features"][:10]:
            print(f"  {name}: {score:.6f}")

    print(f"\nReport: {res['report_path']}")
    _print_paths("Validation outputs", res["validation_paths"])
    _print_paths("Hindcast outputs", res["hindcast_paths"])


def main():
    parser = argparse.ArgumentParser(
        description="Run the pooled ensemble with the default fair training setup."
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Optional target location. Default: run vestfjorden and all study areas.",
    )
    args = parser.parse_args()

    res = run_pooled_ensemble(location=args.location)
    _print_summary("Pooled ensemble", res)


if __name__ == "__main__":
    main()
