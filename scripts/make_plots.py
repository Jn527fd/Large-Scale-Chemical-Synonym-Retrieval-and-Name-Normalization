from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chem_synonymizer.plots import make_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summary plots from saved run folders.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = make_all_plots(args.results_dir, args.output_dir)
    if not created:
        print("No plots created. Run one or more experiments first.")
        return
    print("Created plots:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
