cmake_minimum_required(VERSION 3.15)
project(solar_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(solar_benchmark src/benchmark.cpp)
target_include_directories(solar_benchmark PRIVATE src)

if(MSVC)
    target_compile_options(solar_benchmark PRIVATE /O2 /arch:AVX2 /fp:fast)
else()
    target_compile_options(solar_benchmark PRIVATE -O3 -march=native -ffast-math)
endif()
