#pragma once
#include "system.hpp"
#include "physics.hpp"

inline void velocity_verlet(System& sys, std::vector<Vec3>& a, double dt) {
    auto a_old = a;
    for (int i = 0; i < sys.N; ++i)
        sys.x[i] += sys.v[i] * dt + a_old[i] * (0.5 * dt * dt);
    a = compute_acceleration(sys);
    for (int i = 0; i < sys.N; ++i)
        sys.v[i] += (a_old[i] + a[i]) * (0.5 * dt);
}

inline void ruth_forest(System& sys, std::vector<Vec3>& a, double dt) {
    constexpr double c[3] = { 7.0/24.0, 3.0/4.0, -1.0/24.0 };
    constexpr double d[3] = { 2.0/3.0, -2.0/3.0, 1.0 };

    for (int s = 0; s < 3; ++s) {
        for (int i = 0; i < sys.N; ++i)
            sys.x[i] += sys.v[i] * (c[s] * dt);
        a = compute_acceleration(sys);
        for (int i = 0; i < sys.N; ++i)
            sys.v[i] += a[i] * (d[s] * dt);
    }
}

inline void yoshida4(System& sys, std::vector<Vec3>& a, double dt) {
    const double cr = std::cbrt(2.0);
    const double w0 = -cr / (2.0 - cr);
    const double w1 = 1.0 / (2.0 - cr);
    const double c[3] = { w1, w0, w1 };
    const double d[4] = { w1/2.0, (w0+w1)/2.0, (w0+w1)/2.0, w1/2.0 };

    for (int s = 0; s < 3; ++s) {
        for (int i = 0; i < sys.N; ++i)
            sys.x[i] += sys.v[i] * (d[s] * dt);
        a = compute_acceleration(sys);
        for (int i = 0; i < sys.N; ++i)
            sys.v[i] += a[i] * (c[s] * dt);
    }
    for (int i = 0; i < sys.N; ++i)
        sys.x[i] += sys.v[i] * (d[3] * dt);
}
