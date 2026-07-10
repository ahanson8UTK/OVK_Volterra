#!/usr/bin/env python3
"""Add route-comparison rotation diagnostics to monetary LP memory-target outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from route_rotation import RouteRotationConfig, run_rotation_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=ROOT / "outputs" / "monetary_lp_memory_targets" / "comparison")
    parser.add_argument("--targets-dir", type=Path, default=ROOT / "outputs" / "monetary_lp_memory_targets")
    parser.add_argument(
        "--routes",
        nargs=3,
        default=["diagonal_old", "hac_filtered_L12", "hilbert_volterra_L12_gamma005_memory_3_12_36"],
    )
    parser.add_argument(
        "--rotation-reference",
        choices=["pooled", "diagonal", "hac", "hilbert_volterra"],
        default="pooled",
    )
    parser.add_argument("--rotation-lambda-min", type=float, default=1e-2)
    parser.add_argument("--rotation-lambda-max", type=float, default=1e2)
    parser.add_argument("--rotation-lambda-count", type=int, default=41)
    parser.add_argument("--min-rotation-anisotropy", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_rotation_diagnostics(
        RouteRotationConfig(
            targets_dir=args.targets_dir,
            comparison_dir=args.comparison_dir,
            routes=tuple(args.routes),
            rotation_reference=args.rotation_reference,
            lambda_min=float(args.rotation_lambda_min),
            lambda_max=float(args.rotation_lambda_max),
            lambda_count=int(args.rotation_lambda_count),
            min_anisotropy=float(args.min_rotation_anisotropy),
        )
    )
    print(f"Wrote route rotation diagnostics to {args.comparison_dir}")
    print(f"Common dates: {metadata['number_of_common_dates']}; p={metadata['p']}")


if __name__ == "__main__":
    main()

