"""A small finite-volume solver for the one-dimensional Euler equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PrimitiveState:
    """Density, pressure, and velocity at one side of the diaphragm."""

    density: float
    pressure: float
    velocity: float


@dataclass(frozen=True)
class SimulationResult:
    """Primitive variables evaluated at the cell centers."""

    x: Array
    density: Array
    velocity: Array
    pressure: Array
    time: float


def primitive_to_conserved(state: PrimitiveState, gamma: float) -> Array:
    """Convert a primitive state to density, momentum, and total energy."""
    total_energy = (
        state.pressure / (gamma - 1.0)
        + 0.5 * state.density * state.velocity**2
    )
    return np.array(
        [state.density, state.density * state.velocity, total_energy],
        dtype=float,
    )


def conserved_to_primitive(
    conserved: Array, gamma: float
) -> tuple[Array, Array, Array]:
    """Convert conserved variables to density, velocity, and pressure."""
    density = conserved[:, 0]
    velocity = conserved[:, 1] / density
    pressure = (gamma - 1.0) * (
        conserved[:, 2] - 0.5 * density * velocity**2
    )
    return density, velocity, pressure


def rusanov_flux(conserved: Array, gamma: float) -> Array:
    """Compute the local Lax-Friedrichs (Rusanov) flux at every interface."""
    density, velocity, pressure = conserved_to_primitive(conserved, gamma)
    physical_flux = np.column_stack(
        (
            density * velocity,
            density * velocity**2 + pressure,
            (conserved[:, 2] + pressure) * velocity,
        )
    )

    sound_speed = np.sqrt(gamma * pressure / density)
    wave_speed = np.maximum(
        np.abs(velocity[:-1]) + sound_speed[:-1],
        np.abs(velocity[1:]) + sound_speed[1:],
    )

    return (
        0.5 * (physical_flux[:-1] + physical_flux[1:])
        - 0.5 * wave_speed[:, None] * (conserved[1:] - conserved[:-1])
    )


def solve(
    left: PrimitiveState = PrimitiveState(1.0, 1.0, 0.0),
    right: PrimitiveState = PrimitiveState(0.125, 0.1, 0.0),
    *,
    cells: int = 500,
    final_time: float = 0.2,
    cfl: float = 0.5,
    gamma: float = 1.4,
    diaphragm_position: float = 0.5,
) -> SimulationResult:
    """Solve a shock-tube problem on the unit interval."""
    if cells < 2:
        raise ValueError("cells must be at least 2")
    if final_time < 0:
        raise ValueError("final_time cannot be negative")
    if not 0 < cfl <= 1:
        raise ValueError("cfl must be in the interval (0, 1]")
    if gamma <= 1:
        raise ValueError("gamma must be greater than 1")
    if not 0 < diaphragm_position < 1:
        raise ValueError("diaphragm_position must be between 0 and 1")

    dx = 1.0 / cells
    x = (np.arange(cells, dtype=float) + 0.5) * dx
    left_conserved = primitive_to_conserved(left, gamma)
    right_conserved = primitive_to_conserved(right, gamma)

    # One ghost cell on each side supplies transmissive boundaries.
    conserved = np.empty((cells + 2, 3), dtype=float)
    conserved[1:-1] = np.where(
        (x < diaphragm_position)[:, None],
        left_conserved,
        right_conserved,
    )

    time = 0.0
    while time < final_time:
        conserved[0] = conserved[1]
        conserved[-1] = conserved[-2]

        density, velocity, pressure = conserved_to_primitive(conserved, gamma)
        sound_speed = np.sqrt(gamma * pressure / density)
        dt = cfl * dx / np.max(np.abs(velocity) + sound_speed)
        dt = min(dt, final_time - time)

        flux = rusanov_flux(conserved, gamma)
        conserved[1:-1] -= dt / dx * (flux[1:] - flux[:-1])
        time += dt

    density, velocity, pressure = conserved_to_primitive(conserved[1:-1], gamma)
    return SimulationResult(x, density, velocity, pressure, time)
