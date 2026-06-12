#pragma once
#include <vector>
#include <cmath>

struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)       const { return {x*s, y*s, z*s}; }
    Vec3 operator/(double s)       const { return {x/s, y/s, z/s}; }
    Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }

    double norm2() const { return x*x + y*y + z*z; }
    double norm()  const { return std::sqrt(norm2()); }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

struct System {
    std::vector<Vec3> x;
    std::vector<Vec3> v;
    std::vector<double> m;
    int N;
    double G;
};
