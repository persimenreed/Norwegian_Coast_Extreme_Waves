import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bias_correction.pipeline import run_bias_correction
from src.settings import get_all_locations


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run bias correction for one location, one method, "
            "or one method at one location."
        )
    )

    parser.add_argument(
        "--location",
        help="Run for this location.",
    )
    parser.add_argument(
        "--method",
        help="Run this method.",
    )

    args = parser.parse_args()

    # Require at least one selector
    if args.location is None and args.method is None:
        parser.error("You must provide at least one of --location or --method.")

    # Case 1: one location + one method
    if args.location is not None and args.method is not None:
        print(f"Running bias correction for {args.location} with method={args.method}")
        run_bias_correction(args.location, method=args.method)
        return

    # Case 2: one location, all methods
    if args.location is not None:
        print(f"Running bias correction for {args.location} with all methods")
        run_bias_correction(args.location, method=None)
        return

    # Case 3: one method, all locations
    for loc in get_all_locations():
        print(f"\nRunning bias correction for {loc} with method={args.method}")
        run_bias_correction(loc, method=args.method)


if __name__ == "__main__":
    main()