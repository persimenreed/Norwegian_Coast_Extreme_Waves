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
        description="Run bias correction for one or all locations."
    )

    parser.add_argument("--location")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:

        for loc in get_all_locations():
            print(f"\nRunning bias correction for {loc}")
            run_bias_correction(loc)

        return

    if not args.location:
        raise ValueError("Provide --location or use --all")

    run_bias_correction(args.location)


if __name__ == "__main__":
    main()