import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.settings import get_all_locations


def main():
    parser = argparse.ArgumentParser(description="Run bias correction by method, location, or both.")
    parser.add_argument("--location", help="Run for this location.")
    parser.add_argument("--method", help="Run this method.")
    args = parser.parse_args()

    if args.location is None and args.method is None:
        parser.error("You must provide at least one of --location or --method.")

    from src.bias_correction.pipeline import run_bias_correction

    locations = [args.location] if args.location else get_all_locations()
    methods = [args.method] if args.method else [None]

    for location in locations:
        for method in methods:
            label = method or "all methods"
            print(f"Running bias correction for {location} with {label}")
            run_bias_correction(location, method=method)


if __name__ == "__main__":
    main()
