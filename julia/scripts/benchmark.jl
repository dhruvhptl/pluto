using Printf
include("../src/PlutoSim.jl")
using .PlutoSim

function compute_total_energy(system::System)
    KE = 0.5 * sum(system.m[i] * sum(system.v[i, :].^2) for i in 1:system.num_particles)
    PE = 0.0
    for i in 1:system.num_particles
        for j in (i+1):system.num_particles
            r = sqrt(sum((system.x[j, :] .- system.x[i, :]).^2))
            PE -= system.G * system.m[i] * system.m[j] / r
        end
    end
    return KE + PE
end

function bench_integrator(name::String, integrator_fn!, force_evals::Int,
                           ic::String, num_steps::Int, dt::Float64,
                           warmup_steps::Int=10; quiet::Bool=false)
    # Warmup: fresh system, run warmup_steps to trigger JIT
    system, _, _ = PlutoSim.get_initial_conditions(ic)
    a = zeros(system.num_particles, 3)
    PlutoSim.acceleration!(a, system)
    if !quiet
        print("  [$name] warming up ($warmup_steps steps)... ")
        flush(stdout)
    end
    for _ in 1:warmup_steps
        integrator_fn!(system, a, dt)
    end
    if !quiet
        println("done")
    end

    # Fresh system for timed run
    system, _, _ = PlutoSim.get_initial_conditions(ic)
    a = zeros(system.num_particles, 3)
    PlutoSim.acceleration!(a, system)
    E0 = compute_total_energy(system)

    t0 = time()
    for _ in 1:num_steps
        integrator_fn!(system, a, dt)
    end
    wall = time() - t0

    Ef = compute_total_energy(system)
    energy_error = abs((Ef - E0) / E0)
    sps = num_steps / wall
    mean_ms = wall / num_steps * 1000

    if !quiet
        @printf("  [%s] done: %.1f steps/s, |dE/E0|=%.3e, mean=%.4f ms/step\n",
                name, sps, energy_error, mean_ms)
    end

    return Dict(
        "name"                 => name,
        "steps_per_sec"        => sps,
        "mean_ms_per_step"     => mean_ms,
        "energy_error"         => energy_error,
        "force_evals_per_step" => force_evals,
        "wall_s"               => wall,
        "E0"                   => E0,
        "Ef"                   => Ef,
    )
end

function run_benchmarks(; json_mode::Bool=false)
    IC        = "solar_system_plus"
    NUM_STEPS = 1000
    DT        = 5.0

    if !json_mode
        println()
        println("=" ^ 60)
        println("JULIA N-BODY BENCHMARK")
        @printf("Initial condition : %s  (%d steps, dt=%.1f days)\n", IC, NUM_STEPS, DT)
        println("=" ^ 60)
    end

    integrators = [
        ("Velocity-Verlet", PlutoSim.velocity_verlet!, 2),
        ("Ruth-Forest",     PlutoSim.ruth_forest!,     3),
        ("Yoshida-4",       PlutoSim.yoshida4!,        4),
    ]

    results = []
    for (name, fn!, fevals) in integrators
        r = bench_integrator(name, fn!, fevals, IC, NUM_STEPS, DT; quiet=json_mode)
        push!(results, r)
    end

    if json_mode
        # Emit compact JSON to stdout for benchmark_all.py to parse
        entries = join([
            string("{\"integrator\":\"", r["name"],
                   "\",\"steps_per_sec\":", r["steps_per_sec"],
                   ",\"energy_error\":", r["energy_error"], "}")
            for r in results
        ], ",")
        println("{\"results\":[", entries, "]}")
    else
        # Human-readable table
        println()
        println("=" ^ 70)
        @printf("%-20s  %12s  %14s  %18s\n",
                "Integrator", "Steps/sec", "|dE/E0|", "Force evals/step")
        println("-" ^ 70)
        for r in results
            @printf("%-20s  %12.1f  %14.3e  %18d\n",
                    r["name"], r["steps_per_sec"], r["energy_error"],
                    r["force_evals_per_step"])
        end
        println("=" ^ 70)

        # Memory estimate
        system, _, _ = PlutoSim.get_initial_conditions(IC)
        a = zeros(system.num_particles, 3)
        mem_bytes = sizeof(system.x) + sizeof(system.v) + sizeof(system.m) + sizeof(a)
        @printf("\nMemory (state arrays): %d bytes (%.1f KB), %.0f bytes/particle\n\n",
                mem_bytes, mem_bytes/1024, mem_bytes/system.num_particles)
    end

    return results
end

# Detect --json flag
json_mode = "--json" in ARGS
run_benchmarks(; json_mode=json_mode)
