"""CLI entry point: `python -m src.eval configs/baseline.yaml`."""

from __future__ import annotations

import sys

from src.eval.runner import run_experiment


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m src.eval <path-to-experiment.yaml>", file=sys.stderr)
        raise SystemExit(2)

    results_path = run_experiment(sys.argv[1])
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
