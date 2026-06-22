"""Plotting and animation helpers for shock-tube simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from shocktube import SimulationHistory, SimulationResult

COLORS = ("#4cc9f0", "#f72585", "#fca311")
BACKGROUND = "#0b1020"
PANEL = "#11182b"
TEXT = "#e8edf7"
GRID = "#65708a"


def _style_figure():
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)
    for axis in axes:
        axis.set_facecolor(PANEL)
        axis.tick_params(colors=TEXT)
        axis.grid(color=GRID, alpha=0.18, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_color(GRID)
            spine.set_alpha(0.35)
    return fig, axes


def plot_result(result: SimulationResult):
    """Create a polished static plot of a simulation result."""
    fig, axes = _style_figure()
    fields = (
        ("Density", result.density),
        ("Velocity", result.velocity),
        ("Pressure", result.pressure),
    )

    for axis, color, (label, values) in zip(axes, COLORS, fields):
        axis.plot(result.x, values, color=color, linewidth=2)
        axis.fill_between(result.x, values, color=color, alpha=0.12)
        axis.set_ylabel(label, color=TEXT)

    axes[-1].set_xlabel("Position", color=TEXT)
    fig.suptitle(
        f"Sod Shock Tube  |  t = {result.time:.3f}",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def animate_history(
    history: SimulationHistory,
    *,
    output: Path | None = None,
    fps: int = 30,
):
    """Animate sampled simulation history and optionally save it as a GIF."""
    fig, axes = _style_figure()
    histories = (history.density, history.velocity, history.pressure)
    labels = ("Density", "Velocity", "Pressure")
    lines = []
    fills = [None, None, None]

    for axis, color, label, values in zip(axes, COLORS, labels, histories):
        padding = max(0.05 * (values.max() - values.min()), 0.02)
        axis.set_xlim(history.x[0], history.x[-1])
        axis.set_ylim(values.min() - padding, values.max() + padding)
        axis.set_ylabel(label, color=TEXT)
        (line,) = axis.plot([], [], color=color, linewidth=2.2)
        lines.append(line)

    axes[-1].set_xlabel("Position", color=TEXT)
    title = fig.suptitle("", color=TEXT, fontsize=15, fontweight="bold")
    fig.tight_layout()

    def update(frame: int):
        artists = [title]
        for index, (axis, line, color, values) in enumerate(
            zip(axes, lines, COLORS, histories)
        ):
            line.set_data(history.x, values[frame])
            if fills[index] is not None:
                fills[index].remove()
            fills[index] = axis.fill_between(
                history.x, values[frame], color=color, alpha=0.14
            )
            artists.extend((line, fills[index]))

        title.set_text(
            f"Sod Shock Tube  |  t = {history.times[frame]:.3f}"
            f" / {history.times[-1]:.3f}"
        )
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history.times),
        interval=1000 / fps,
        repeat=True,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output, writer=PillowWriter(fps=fps), dpi=110)

    return fig, animation
