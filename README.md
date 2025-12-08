# Pluto Orbital Simulation

Interactive N-body solar system simulator comparing Python and Julia implementations.

## Features
- 🪐 Real-time 3D visualization of solar system with adjustable Pluto inclination
- ⚡ Dual implementation: Python (NumPy/Numba) vs Julia (native JIT)
- 📊 **2.3x performance improvement** with Julia
- 🎮 Interactive controls: orbit inclination (0-90°), simulation speed, pause/reset
- 🌌 GPU-accelerated rendering with orbital trails

## Performance Comparison

| Metric | Python + Numba | Julia | Speedup |
|--------|----------------|-------|---------|
| Steps/second | 437,180 | 1,000,073 | **2.3x** |
| Energy conservation | 21.87% error | 0.0036% error | **6075x better** |
| Visualization | Matplotlib (CPU) | Makie.jl (GPU) | Smoother |

## Installation

### Python
pip install numpy matplotlib numba
python interactive

text

### Julia
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/interactive.jl

text

## Benchmarks
Julia
julia --project=. scripts/benchmark.jl

Python
python benchmark_python.py

text

## Technologies
- **Python**: NumPy, Matplotlib, Numba JIT
- **Julia**: Makie.jl, GLMakie, native compilation
- **Physics**: Velocity-Verlet integrator, N-body gravitational dynamics
- **Data**: NASA JPL Horizons ephemeris (2024-01-01)