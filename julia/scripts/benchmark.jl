using Printf
using Statistics
include("../src/PlutoSim.jl")
using .PlutoSim

function benchmark_julia_physics(num_steps::Int, dt::Float64)
    println("="^60)
    println("JULIA BENCHMARK")
    println("="^60)
    
    # Setup
    system, _, _ = PlutoSim.get_initial_conditions()
    a = zeros(system.num_particles, 3)
    PlutoSim.acceleration!(a, system)
    
    # Initial energy
    E0 = compute_total_energy(system)
    
    # Warmup (JIT compilation)
    println("Warming up Julia JIT compiler...")
    for _ in 1:100
        PlutoSim.velocity_verlet!(system, a, dt)
    end
    
    # Reset
    system, _, _ = PlutoSim.get_initial_conditions()
    PlutoSim.acceleration!(a, system)
    
    # Benchmark
    println("Running $(num_steps) integration steps...")
    times = Float64[]
    
    for i in 1:num_steps
        t_start = time()
        PlutoSim.velocity_verlet!(system, a, dt)
        t_end = time()
        push!(times, (t_end - t_start) * 1000)  # Convert to ms
    end
    
    # Final energy
    E_final = compute_total_energy(system)
    energy_error = abs((E_final - E0) / E0) * 100
    
    # Statistics
    mean_time = mean(times)
    std_time = std(times)
    min_time = minimum(times)
    max_time = maximum(times)
    total_time = sum(times) / 1000  # Convert to seconds
    
    println("\nResults:")
    println("  Total time:          $(Printf.@sprintf("%.3f", total_time)) seconds")
    println("  Mean time per step:  $(Printf.@sprintf("%.4f", mean_time)) ms")
    println("  Std deviation:       $(Printf.@sprintf("%.4f", std_time)) ms")
    println("  Min time:            $(Printf.@sprintf("%.4f", min_time)) ms")
    println("  Max time:            $(Printf.@sprintf("%.4f", max_time)) ms")
    println("  Steps per second:    $(Printf.@sprintf("%.1f", 1000/mean_time))")
    println("\nEnergy Conservation:")
    println("  Initial energy:      $(Printf.@sprintf("%.6e", E0))")
    println("  Final energy:        $(Printf.@sprintf("%.6e", E_final))")
    println("  Relative error:      $(Printf.@sprintf("%.6f", energy_error))%")
    
    return Dict(
        "mean_time_ms" => mean_time,
        "total_time_s" => total_time,
        "energy_error_percent" => energy_error,
        "steps_per_second" => 1000/mean_time
    )
end

function compute_total_energy(system::System)
    # Kinetic energy
    KE = 0.5 * sum(system.m[i] * sum(system.v[i, :].^2) for i in 1:system.num_particles)
    
    # Potential energy
    PE = 0.0
    for i in 1:system.num_particles
        for j in (i+1):system.num_particles
            r_ij = system.x[j, :] - system.x[i, :]
            r = sqrt(sum(r_ij.^2))
            PE -= system.G * system.m[i] * system.m[j] / r
        end
    end
    
    return KE + PE
end

function memory_benchmark()
    println("\n" * "="^60)
    println("MEMORY USAGE")
    println("="^60)
    
    system, _, _ = PlutoSim.get_initial_conditions()
    a = zeros(system.num_particles, 3)
    
    # Estimate memory usage
    system_size = sizeof(system.x) + sizeof(system.v) + sizeof(system.m) + sizeof(a)
    println("  System state:        $(system_size) bytes ($(system_size/1024) KB)")
    println("  Per particle:        $(system_size/system.num_particles) bytes")
    
    return system_size
end

# Main benchmark
println("\nStarting Julia benchmarks...")
println("Configuration: 12 bodies, dt=5.0 days")

# Run benchmarks
results = benchmark_julia_physics(1000, 5.0)
mem_usage = memory_benchmark()

println("\n" * "="^60)
println("BENCHMARK COMPLETE")
println("="^60)
println("\nTo compare with Python:")
println("  1. Run: python benchmark_python.py")
println("  2. Compare the results")
println("\nJulia Results Summary:")
println("  - $(Printf.@sprintf("%.1f", results["steps_per_second"])) steps/second")
println("  - $(Printf.@sprintf("%.6f", results["energy_error_percent"]))% energy error")
println("  - $(Printf.@sprintf("%.3f", results["total_time_s"])) seconds total")
