#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "initial_conditions.hpp"
#include "integrators.hpp"

static double total_energy(const System& sys) {
    double KE = 0.0;
    for (int i = 0; i < sys.N; ++i)
        KE += 0.5 * sys.m[i] * sys.v[i].norm2();

    double PE = 0.0;
    for (int i = 0; i < sys.N; ++i)
        for (int j = i + 1; j < sys.N; ++j) {
            double r = (sys.x[j] - sys.x[i]).norm();
            PE -= sys.G * sys.m[i] * sys.m[j] / r;
        }
    return KE + PE;
}

struct BenchResult {
    std::string name;
    double steps_per_sec;
    double energy_error;
    double wall_time_s;
    int force_evals;
};

static BenchResult run_bench(
    const std::string& name,
    int force_evals,
    int steps,
    int warmup_steps,
    double dt,
    std::function<void(System&, std::vector<Vec3>&, double)> step_fn)
{
    // Warmup pass (ensures branch predictors / caches are warm)
    {
        System sys = make_solar_system();
        std::vector<Vec3> a = compute_acceleration(sys);
        for (int s = 0; s < warmup_steps; ++s)
            step_fn(sys, a, dt);
    }

    // Timed pass with fresh initial conditions
    System sys = make_solar_system();
    std::vector<Vec3> a = compute_acceleration(sys);
    const double E0 = total_energy(sys);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s)
        step_fn(sys, a, dt);
    auto t1 = std::chrono::high_resolution_clock::now();

    double wall = std::chrono::duration<double>(t1 - t0).count();
    double Ef = total_energy(sys);
    double err = std::abs((Ef - E0) / E0);

    std::printf("  [%s] done: %.1f steps/s, |dE/E0|=%.3e, mean=%.4f ms/step\n",
                name.c_str(), steps / wall, err, wall / steps * 1000.0);

    return {name, steps / wall, err, wall, force_evals};
}

int main() {
    constexpr int STEPS        = 1000;
    constexpr int WARMUP_STEPS = 10;
    constexpr double DT        = 5.0;

    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("C++ N-BODY BENCHMARK\n");
    std::printf("Initial condition : solar_system_plus  (%d steps, dt=%.1f days)\n",
                STEPS, DT);
    std::printf("============================================================\n");

    std::vector<BenchResult> results;
    results.push_back(run_bench("Velocity-Verlet", 2, STEPS, WARMUP_STEPS, DT, velocity_verlet));
    results.push_back(run_bench("Ruth-Forest",     3, STEPS, WARMUP_STEPS, DT, ruth_forest));
    results.push_back(run_bench("Yoshida-4",       4, STEPS, WARMUP_STEPS, DT, yoshida4));

    std::printf("\n");
    std::printf("======================================================================\n");
    std::printf("%-20s  %12s  %14s  %18s\n",
                "Integrator", "Steps/sec", "|dE/E0|", "Force evals/step");
    std::printf("----------------------------------------------------------------------\n");
    for (auto& r : results) {
        std::printf("%-20s  %12.1f  %14.3e  %18d\n",
                    r.name.c_str(), r.steps_per_sec, r.energy_error, r.force_evals);
    }
    std::printf("======================================================================\n\n");

    return 0;
}
