# Solar System N-Body Simulation

Interactive, high-accuracy N-body solar system simulator with Python, Julia, and C++ backends.
Includes real-time 3D visualization, three symplectic integrators, and 8 major moons.
A notable interactive feature: adjust Pluto's orbital inclination live (0–90°) to explore
what the solar system would look like with a tilted Pluto.

## Features

- Real-time 3D visualization (Julia/GLMakie, GPU-accelerated orbital trails)
- Three implementations: Python (NumPy + Numba JIT), Julia (native), C++ (header-only, -O3)
- Three symplectic integrators: Velocity-Verlet, Ruth-Forest 4th-order, Yoshida 4th-order
- Solar system, solar system + Pluto/Ceres/Vesta, and full moons initial conditions
- NASA JPL Horizons ephemeris data (2024-01-01 TDB barycentric ICRF)

## Integrators

| Integrator | Order | Force evals/step | Notes |
|---|---|---|---|
| Velocity-Verlet | 2nd | 1 | Fast, good energy conservation |
| Ruth-Forest | 4th | 3 | Better long-term accuracy, 3× cost |
| Yoshida-4 | 4th | 4 | Widely-used 4th-order symplectic, 4× cost |

All three are **symplectic** — they preserve the symplectic structure of Hamiltonian mechanics,
which means energy oscillates around a fixed value rather than drifting over long integrations.
This makes them far superior to non-symplectic methods (e.g., Euler, RK4) for orbital mechanics.

## Benchmark Results

*Run `python benchmark_all.py` to populate with real numbers.*

### Steps / second

| Integrator | Python+Numba | Julia | C++ |
|---|---|---|---|
| Velocity-Verlet | TBD | TBD | TBD |
| Ruth-Forest | TBD | TBD | TBD |
| Yoshida-4 | TBD | TBD | TBD |

### Energy conservation error \|ΔE/E₀\|

| Integrator | Python+Numba | Julia | C++ |
|---|---|---|---|
| Velocity-Verlet | TBD | TBD | TBD |
| Ruth-Forest | TBD | TBD | TBD |
| Yoshida-4 | TBD | TBD | TBD |

*9 bodies, 10 000 steps, dt = 5 days.*

## Moons

The `solar_system_moons` initial condition adds 8 major moons:

| Moon | Parent | Orbital period |
|---|---|---|
| Moon | Earth | 27.3 days |
| Io | Jupiter | 1.77 days |
| Europa | Jupiter | 3.55 days |
| Ganymede | Jupiter | 7.15 days |
| Callisto | Jupiter | 16.7 days |
| Titan | Saturn | 15.9 days |
| Triton | Neptune | 5.88 days |
| Charon | Pluto | 6.39 days |

**Important:** The default dt = 5 days is too coarse for Io (period 1.77 days) and the Moon
(period 27.3 days). Use `dt = 0.1` days when running with moons.
`RECOMMENDED_DT["solar_system_moons"] = 0.1` is available in `python/common.py`.

## Installation

### Python

```
pip install numpy matplotlib numba
```

Run interactive visualization:
```
python python/solarsystem
```

Run benchmarks:
```
python python/benchmark_python.py
```

### Julia

```
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/interactive.jl
```

Run Julia benchmark:
```
julia --project=julia julia/scripts/benchmark.jl
```

### C++

Requires CMake 3.15+ and a C++17 compiler (GCC/Clang/MSVC).

```
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./solar_benchmark          # Linux/macOS
.\Release\solar_benchmark  # Windows
```

## Unified benchmark

Runs all 3 languages × 3 integrators and produces a table, JSON, and PNG chart:

```
python benchmark_all.py
```

The C++ binary must be built first (see above). Julia must be installed and the
`julia/` project instantiated.

## Technologies

- **Python**: NumPy, Matplotlib, Numba JIT
- **Julia**: GLMakie (GPU rendering), native JIT compilation
- **C++**: C++17, header-only, no external dependencies
- **Physics**: Symplectic integrators (VV / Ruth-Forest / Yoshida-4), N-body gravity
- **Data**: NASA JPL Horizons DE440/441 ephemeris, 2024-01-01 TDB barycentric ICRF
