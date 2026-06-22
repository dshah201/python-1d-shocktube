# Python 1D Shock Tube

A compact finite-volume solver for the one-dimensional Sod shock-tube
problem. It evolves the compressible Euler equations using a Rusanov
(local Lax-Friedrichs) numerical flux and transmissive boundary conditions.

The default initial state is:

| Region | Density | Pressure | Velocity |
| --- | ---: | ---: | ---: |
| Left | 1.0 | 1.0 | 0.0 |
| Right | 0.125 | 0.1 | 0.0 |

## Run it

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies and run the simulation:

```bash
pip install -r requirements.txt
python main.py
```

To save the result without opening a plot window:

```bash
python main.py --save output/shock-tube.png
```

The grid size, final time, and CFL number can also be changed:

```bash
python main.py --cells 1000 --time 0.2 --cfl 0.5
```

Run `python main.py --help` for all command-line options.

## Project layout

```text
.
├── main.py          # Command-line interface and plotting
├── shocktube.py     # Numerical solver
├── requirements.txt
└── README.md
```

## Numerical method

The solver uses a first-order finite-volume discretization of the 1D Euler
equations. A Rusanov flux handles the discontinuity robustly, while the time
step is selected from the CFL condition. The method is intentionally simple
and educational rather than optimized for high-accuracy CFD work.
