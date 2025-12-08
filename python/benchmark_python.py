import numpy as np
import time
import common
from statistics import mean, stdev

def benchmark_python_physics(num_steps, dt):
    print("="*60)
    print("PYTHON BENCHMARK")
    print("="*60)
    
    # Setup
    system, _, _, _ = common.get_initial_conditions("solar_system_plus")
    a = np.zeros((system.num_particles, 3))
    
    # Initial energy
    E0 = compute_total_energy(system)
    
    # Warmup
    print("Warming up Numba JIT compiler...")
    for _ in range(100):
        common.velocity_verlet_numba(a, system.x, system.v, system.m, 
                                     system.G, system.num_particles, dt)
    
    # Reset
    system, _, _, _ = common.get_initial_conditions("solar_system_plus")
    common.acceleration_numba(a, system.x, system.m, system.G, system.num_particles)
    
    # Benchmark
    print(f"Running {num_steps} integration steps...")
    times = []
    
    for i in range(num_steps):
        t_start = time.time()
        common.velocity_verlet_numba(a, system.x, system.v, system.m,
                                     system.G, system.num_particles, dt)
        t_end = time.time()
        times.append((t_end - t_start) * 1000)  # Convert to ms
    
    # Final energy
    E_final = compute_total_energy(system)
    energy_error = abs((E_final - E0) / E0) * 100
    
    # Statistics
    mean_time = mean(times)
    std_time = stdev(times)
    min_time = min(times)
    max_time = max(times)
    total_time = sum(times) / 1000  # Convert to seconds
    
    print("\nResults:")
    print(f"  Total time:          {total_time:.3f} seconds")
    print(f"  Mean time per step:  {mean_time:.4f} ms")
    print(f"  Std deviation:       {std_time:.4f} ms")
    print(f"  Min time:            {min_time:.4f} ms")
    print(f"  Max time:            {max_time:.4f} ms")
    print(f"  Steps per second:    {1000/mean_time:.1f}")
    print("\nEnergy Conservation:")
    print(f"  Initial energy:      {E0:.6e}")
    print(f"  Final energy:        {E_final:.6e}")
    print(f"  Relative error:      {energy_error:.6f}%")
    
    return {
        "mean_time_ms": mean_time,
        "total_time_s": total_time,
        "energy_error_percent": energy_error,
        "steps_per_second": 1000/mean_time
    }

def compute_total_energy(system):
    # Kinetic energy
    KE = 0.5 * np.sum(system.m[:, np.newaxis] * np.sum(system.v**2, axis=1))
    
    # Potential energy
    PE = 0.0
    for i in range(system.num_particles):
        for j in range(i+1, system.num_particles):
            r_ij = system.x[j] - system.x[i]
            r = np.linalg.norm(r_ij)
            PE -= system.G * system.m[i] * system.m[j] / r
    
    return KE + PE

def memory_benchmark():
    print("\n" + "="*60)
    print("MEMORY USAGE")
    print("="*60)
    
    system, _, _, _ = common.get_initial_conditions("solar_system_plus")
    a = np.zeros((system.num_particles, 3))
    
    # Estimate memory usage
    system_size = system.x.nbytes + system.v.nbytes + system.m.nbytes + a.nbytes
    print(f"  System state:        {system_size} bytes ({system_size/1024:.1f} KB)")
    print(f"  Per particle:        {system_size/system.num_particles:.1f} bytes")
    
    return system_size

# Main benchmark
if __name__ == "__main__":
    print("\nStarting Python benchmarks...")
    print("Configuration: 12 bodies, dt=5.0 days")
    
    # Run benchmarks
    results = benchmark_python_physics(1000, 5.0)
    mem_usage = memory_benchmark()
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print("\nPython Results Summary:")
    print(f"  - {results['steps_per_second']:.1f} steps/second")
    print(f"  - {results['energy_error_percent']:.6f}% energy error")
    print(f"  - {results['total_time_s']:.3f} seconds total")
