"""Run and plot the one-dimensional Sod shock-tube problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from shocktube import PrimitiveState, solve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the 1D Sod shock-tube problem with a Rusanov flux."
    )
    parser.add_argument("--cells", type=int, default=500, help="number of grid cells")
    parser.add_argument("--time", type=float, default=0.2, help="final simulation time")
    parser.add_argument("--cfl", type=float, default=0.5, help="CFL number")
    parser.add_argument(
        "--save",
        type=Path,
        metavar="PATH",
        help="save the plot instead of opening a window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save:
        plt.switch_backend("Agg")

    result = solve(
        left=PrimitiveState(density=1.0, pressure=1.0, velocity=0.0),
        right=PrimitiveState(density=0.125, pressure=0.1, velocity=0.0),
        cells=args.cells,
        final_time=args.time,
        cfl=args.cfl,
    )

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fields = (
        ("Density", result.density),
        ("Velocity", result.velocity),
        ("Pressure", result.pressure),
    )

    for axis, (label, values) in zip(axes, fields):
        axis.plot(result.x, values, linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Position")
    fig.suptitle(f"Sod shock tube at t = {result.time:.3f}")
    fig.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
