import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ensemble.weighted import run
from src.settings import get_methods


def main():

    parser = argparse.ArgumentParser(
        description="Run pooled weighted ensemble."
    )

    parser.add_argument(
        "--location",
        required=True,
        help="Target location (e.g. vestfjorden).",
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=f"Subset of methods (default = all: {get_methods()})",
    )

    args = parser.parse_args()

    res = run(location=args.location, methods=args.methods)

    print(f"\nWeighted ensemble completed for {res['location']}")

    print("\nWeights:")
    for name, weight in sorted(res["weights"].items()):
        print(f"  {name}: {weight:.6f}")

    print(f"\nHindcast saved to: {res['hindcast_path']}")


if __name__ == "__main__":
    main()