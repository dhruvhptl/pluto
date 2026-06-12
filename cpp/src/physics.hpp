#pragma once
#include "system.hpp"

inline std::vector<Vec3> compute_acceleration(const System& sys) {
    constexpr double eps2 = 1e-12;
    std::vector<Vec3> a(sys.N);

    for (int i = 0; i < sys.N; ++i) {
        for (int j = i + 1; j < sys.N; ++j) {
            Vec3 dr = sys.x[j] - sys.x[i];
            double r2 = dr.norm2() + eps2;
            double inv_r = 1.0 / std::sqrt(r2);
            double inv_r3 = inv_r / r2;
            double fac = sys.G * inv_r3;
            Vec3 force = dr * fac;
            a[i] += force * sys.m[j];
            a[j] -= force * sys.m[i];
        }
    }
    return a;
}
