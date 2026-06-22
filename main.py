"""Run and plot the one-dimensional Sod shock-tube problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from shocktube import PrimitiveState, solve, solve_history


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
    parser.add_argument(
        "--animate",
        type=Path,
        metavar="GIF",
        help="render the simulation history to an animated GIF",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=90,
        help="number of animation frames (default: 90)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="animation frames per second (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save and args.animate:
        raise SystemExit("Choose either --save or --animate, not both.")
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than zero.")
    if args.save or args.animate:
        plt.switch_backend("Agg")

    left = PrimitiveState(density=1.0, pressure=1.0, velocity=0.0)
    right = PrimitiveState(density=0.125, pressure=0.1, velocity=0.0)

    if args.animate:
        from visualization import animate_history

        history = solve_history(
            left,
            right,
            cells=args.cells,
            final_time=args.time,
            cfl=args.cfl,
            frames=args.frames,
        )
        animate_history(history, output=args.animate, fps=args.fps)
        print(f"Saved animation to {args.animate}")
        return

    from visualization import plot_result

    result = solve(
        left,
        right,
        cells=args.cells,
        final_time=args.time,
        cfl=args.cfl,
    )
    fig = plot_result(result)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
