"""
benchmark_all.py — Unified 3-language x 3-integrator N-body benchmark.

Outputs:
  - Formatted ASCII table to stdout
  - benchmark_results.json
  - benchmark_results.png  (bar charts: steps/sec and energy error)
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
IC   = "solar_system_plus"   # 12 bodies, used by all three languages

# ---------------------------------------------------------------------------
# Python benchmarks
# ---------------------------------------------------------------------------

def _energy_python(sys_obj):
    x, v, m, G = sys_obj.x, sys_obj.v, sys_obj.m, sys_obj.G
    KE = 0.5 * np.sum(m[:, None] * v ** 2)
    PE = 0.0
    N = sys_obj.num_particles
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(x[j] - x[i])
            PE -= G * m[i] * m[j] / r
    return KE + PE


def _bench_python_integrator(step_fn, dt, steps, get_ic, accel_fn):
    # Warmup (triggers Numba JIT)
    sys_obj, _, _, _ = get_ic(IC)
    a = np.zeros((sys_obj.num_particles, 3))
    accel_fn(a, sys_obj.x, sys_obj.m, sys_obj.G, sys_obj.num_particles)
    for _ in range(10):
        step_fn(a, sys_obj.x, sys_obj.v, sys_obj.m, sys_obj.G, sys_obj.num_particles, dt)

    # Fresh timed run
    sys_obj, _, _, _ = get_ic(IC)
    a = np.zeros((sys_obj.num_particles, 3))
    accel_fn(a, sys_obj.x, sys_obj.m, sys_obj.G, sys_obj.num_particles)
    E0 = _energy_python(sys_obj)

    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn(a, sys_obj.x, sys_obj.v, sys_obj.m, sys_obj.G, sys_obj.num_particles, dt)
    wall = time.perf_counter() - t0

    Ef = _energy_python(sys_obj)
    return wall, steps / wall, abs((Ef - E0) / E0)


def bench_python(steps=1000, dt=5.0):
    sys.path.insert(0, str(ROOT))
    from python.common import (
        velocity_verlet_numba,
        ruth_forest_numba,
        yoshida4_numba,
        get_initial_conditions,
        acceleration_numba,
    )

    results = {}
    for label, fn in [
        ("Velocity-Verlet", velocity_verlet_numba),
        ("Ruth-Forest",     ruth_forest_numba),
        ("Yoshida-4",       yoshida4_numba),
    ]:
        print(f"  Python / {label} ...", flush=True)
        wall, sps, err = _bench_python_integrator(
            fn, dt, steps, get_initial_conditions, acceleration_numba
        )
        print(f"    done: {sps:.1f} steps/s, |dE/E0|={err:.3e}")
        results[label] = {"wall_s": wall, "steps_per_sec": sps, "energy_error": err}
    return results


# ---------------------------------------------------------------------------
# Julia benchmarks (subprocess, --json flag)
# ---------------------------------------------------------------------------

def bench_julia(steps=1000, dt=5.0):
    julia_script = ROOT / "julia" / "scripts" / "benchmark.jl"
    if not julia_script.exists():
        print("  Julia benchmark script not found, skipping.")
        return {}

    # Find julia executable
    julia_exe = "julia"
    try:
        subprocess.run([julia_exe, "--version"], capture_output=True, check=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  'julia' not found in PATH, skipping.")
        return {}

    print("  Running julia/scripts/benchmark.jl --json ...", flush=True)
    proc = subprocess.run(
        [julia_exe, f"--project={ROOT / 'julia'}", str(julia_script), "--json"],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT)
    )

    if proc.returncode != 0:
        print(f"  Julia exited with code {proc.returncode}")
        print(f"  stderr (last 800 chars): {proc.stderr[-800:]}")
        return {}

    # The JSON line is the only line starting with '{'
    json_lines = [l.strip() for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if not json_lines:
        print("  Julia produced no JSON output.")
        print(f"  stdout: {proc.stdout[-400:]}")
        print(f"  stderr: {proc.stderr[-400:]}")
        return {}

    try:
        data = json.loads(json_lines[-1])
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Raw line: {json_lines[-1][:200]}")
        return {}

    results = {}
    for entry in data.get("results", []):
        name = entry["integrator"]
        results[name] = {
            "steps_per_sec": entry["steps_per_sec"],
            "energy_error":  entry["energy_error"],
            "wall_s":        entry.get("wall_s", 0.0),
        }
        print(f"  Julia / {name}: {entry['steps_per_sec']:.1f} steps/s, "
              f"|dE/E0|={entry['energy_error']:.3e}")
    return results


# ---------------------------------------------------------------------------
# C++ benchmarks (subprocess)
# ---------------------------------------------------------------------------

def bench_cpp():
    # Prefer Release build (MSVC layout); fall back to flat build dir (GCC/Clang)
    candidates = [
        ROOT / "cpp" / "build" / "Release" / "solar_benchmark.exe",
        ROOT / "cpp" / "build" / "solar_benchmark.exe",
        ROOT / "cpp" / "build" / "Release" / "solar_benchmark",
        ROOT / "cpp" / "build" / "solar_benchmark",
    ]
    binary = next((p for p in candidates if p.exists()), None)

    if binary is None:
        print(
            "  C++ binary not found. Build with:\n"
            "    cd cpp && mkdir build && cd build\n"
            "    cmake .. -DCMAKE_BUILD_TYPE=Release\n"
            "    cmake --build . --config Release"
        )
        return {}

    print(f"  Running {binary.name} ...", flush=True)
    proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)

    if proc.returncode != 0:
        print(f"  C++ binary exited with code {proc.returncode}")
        print(f"  stderr: {proc.stderr[-400:]}")
        return {}

    # Parse the summary table that looks like:
    #   Velocity-Verlet       1042752.9    3.602e-05              2
    # Columns: name  steps_per_sec  energy_error  force_evals
    results = {}
    in_table = False
    known_names = {"Velocity-Verlet", "Ruth-Forest", "Yoshida-4"}
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("---") or stripped.startswith("==="):
            in_table = True
            continue
        if not in_table or not stripped:
            continue
        # Try to parse: last two tokens are floats (sps, err), third-last is int (fevals)
        # Name is everything before those three tokens
        parts = stripped.split()
        if len(parts) < 4:
            continue
        try:
            force_evals = int(parts[-1])
            err = float(parts[-2])
            sps = float(parts[-3])
            name = " ".join(parts[:-3]).strip()
            if name in known_names:
                results[name] = {"steps_per_sec": sps, "energy_error": err, "wall_s": 0.0}
                print(f"  C++ / {name}: {sps:.1f} steps/s, |dE/E0|={err:.3e}")
        except (ValueError, IndexError):
            continue

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

INTEGRATORS = ["Velocity-Verlet", "Ruth-Forest", "Yoshida-4"]
LANGUAGES   = ["Python", "Julia", "C++"]
N_BODIES    = 12
N_STEPS     = 1000


def print_table(all_results):
    col_w = 16
    title = f"N-Body Benchmark: {N_BODIES} bodies, {N_STEPS} steps, dt=5 days"
    width = 80
    print()
    print("=" * width)
    print(title)
    print("=" * width)

    header = f"{'Integrator':<20}" + "".join(f"{lang:>{col_w}}" for lang in LANGUAGES)
    sep    = "-" * len(header)

    print(f"\nSteps / second:")
    print(header)
    print(sep)
    for integ in INTEGRATORS:
        row = f"{integ:<20}"
        for lang in LANGUAGES:
            val = all_results.get(lang, {}).get(integ, {}).get("steps_per_sec")
            row += f"{val:>{col_w}.1f}" if val is not None else f"{'N/A':>{col_w}}"
        print(row)

    print(f"\nEnergy error |dE/E0|:")
    print(header)
    print(sep)
    for integ in INTEGRATORS:
        row = f"{integ:<20}"
        for lang in LANGUAGES:
            val = all_results.get(lang, {}).get(integ, {}).get("energy_error")
            row += f"{val:>{col_w}.3e}" if val is not None else f"{'N/A':>{col_w}}"
        print(row)

    print()


def save_json(all_results):
    out = ROOT / "benchmark_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"Results saved to {out}")


def save_chart(all_results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping chart.")
        return

    x = np.arange(len(INTEGRATORS))
    width = 0.25
    title = f"N-Body Benchmark: {N_BODIES} bodies, {N_STEPS} steps, dt=5 days"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for idx, (lang, color) in enumerate(zip(LANGUAGES, colors)):
        sps_vals = [
            all_results.get(lang, {}).get(integ, {}).get("steps_per_sec") or 0
            for integ in INTEGRATORS
        ]
        ax1.bar(x + idx * width, sps_vals, width, label=lang, color=color)

    ax1.set_xlabel("Integrator")
    ax1.set_ylabel("Steps / second")
    ax1.set_title("Throughput")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(INTEGRATORS, rotation=10)
    ax1.legend()

    for idx, (lang, color) in enumerate(zip(LANGUAGES, colors)):
        err_vals = [
            all_results.get(lang, {}).get(integ, {}).get("energy_error") or 1e-20
            for integ in INTEGRATORS
        ]
        ax2.bar(x + idx * width, err_vals, width, label=lang, color=color)

    ax2.set_yscale("log")
    ax2.set_xlabel("Integrator")
    ax2.set_ylabel("|dE/E0|")
    ax2.set_title("Energy Conservation Error")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(INTEGRATORS, rotation=10)
    ax2.legend()

    plt.tight_layout()
    out = ROOT / "benchmark_results.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Chart saved to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = {}

    print(f"\n--- Python benchmarks ({IC}, {N_STEPS} steps) ---")
    try:
        all_results["Python"] = bench_python(N_STEPS, 5.0)
    except Exception as e:
        print(f"  Failed: {e}")
        all_results["Python"] = {}

    print(f"\n--- Julia benchmarks ({IC}, {N_STEPS} steps) ---")
    all_results["Julia"] = bench_julia(N_STEPS, 5.0)

    print(f"\n--- C++ benchmarks ({IC}, {N_STEPS} steps) ---")
    all_results["C++"] = bench_cpp()

    print_table(all_results)
    save_json(all_results)
    save_chart(all_results)
