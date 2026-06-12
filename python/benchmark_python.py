import os
import sys
import pathlib
import shutil
import time

import numpy as np

# ── Clear Numba cache so any common.py changes take effect ───────────────────
_here = pathlib.Path(__file__).parent
for _p in list(_here.rglob("__pycache__")) + list(pathlib.Path(".").rglob("__pycache__")):
    shutil.rmtree(_p, ignore_errors=True)
for _ext in ["*.nbi", "*.nbc"]:
    for _p in list(_here.rglob(_ext)) + list(pathlib.Path(".").rglob(_ext)):
        _p.unlink(missing_ok=True)

# Must import after cache wipe so Numba recompiles from source
sys.path.insert(0, str(_here))
import common


# ── Energy helper ─────────────────────────────────────────────────────────────

def total_energy(system):
    KE = 0.5 * np.sum(system.m[:, np.newaxis] * system.v**2)
    PE = 0.0
    N = system.num_particles
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(system.x[j] - system.x[i])
            PE -= system.G * system.m[i] * system.m[j] / r
    return KE + PE


# ── Per-integrator benchmark ──────────────────────────────────────────────────

def bench_integrator(name, step_fn, force_evals_per_step, ic, num_steps, dt,
                     warmup_steps=10):
    # Fresh initial conditions
    system, _, _, _ = common.get_initial_conditions(ic)
    a = np.zeros((system.num_particles, 3))
    common.acceleration_numba(a, system.x, system.m, system.G, system.num_particles)

    # Numba JIT warmup (compiles on first call)
    print(f"  [{name}] warming up ({warmup_steps} steps)...", flush=True)
    for _ in range(warmup_steps):
        step_fn(a, system.x, system.v, system.m, system.G, system.num_particles, dt)

    # Fresh reset after warmup
    system, _, _, _ = common.get_initial_conditions(ic)
    a = np.zeros((system.num_particles, 3))
    common.acceleration_numba(a, system.x, system.m, system.G, system.num_particles)
    E0 = total_energy(system)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(num_steps):
        step_fn(a, system.x, system.v, system.m, system.G, system.num_particles, dt)
    wall = time.perf_counter() - t0

    Ef = total_energy(system)
    energy_error = abs((Ef - E0) / E0)
    sps = num_steps / wall
    mean_ms = wall / num_steps * 1000

    print(f"  [{name}] done: {sps:.1f} steps/s, |dE/E0|={energy_error:.3e}, "
          f"mean={mean_ms:.4f} ms/step")

    return {
        "name": name,
        "steps_per_sec": sps,
        "mean_ms_per_step": mean_ms,
        "energy_error": energy_error,
        "force_evals_per_step": force_evals_per_step,
        "wall_s": wall,
        "E0": E0,
        "Ef": Ef,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    IC = "solar_system_plus"
    NUM_STEPS = 1000
    DT = 5.0

    print()
    print("=" * 60)
    print("PYTHON N-BODY BENCHMARK")
    print(f"Initial condition : {IC}  ({NUM_STEPS} steps, dt={DT} days)")
    print("=" * 60)

    integrators = [
        ("Velocity-Verlet", common.velocity_verlet_numba, 2),
        ("Ruth-Forest",     common.ruth_forest_numba,     3),
        ("Yoshida-4",       common.yoshida4_numba,        4),
    ]

    results = []
    for name, fn, feval in integrators:
        r = bench_integrator(name, fn, feval, IC, NUM_STEPS, DT)
        results.append(r)

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"{'Integrator':<20} {'Steps/sec':>12} {'|dE/E0|':>14} {'Force evals/step':>18}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['steps_per_sec']:>12.1f} "
              f"{r['energy_error']:>14.3e} {r['force_evals_per_step']:>18d}")
    print("=" * 70)
    print()

    # ── Memory estimate ────────────────────────────────────────────────────────
    system, _, _, _ = common.get_initial_conditions(IC)
    a = np.zeros((system.num_particles, 3))
    mem = system.x.nbytes + system.v.nbytes + system.m.nbytes + a.nbytes
    print(f"Memory (state arrays): {mem} bytes ({mem/1024:.1f} KB), "
          f"{mem/system.num_particles:.0f} bytes/particle")
    print()
