add_requires("openblas", { configs = { fortran = false, openmp = true } })
add_requires("catch2", "nanobench")

set_exceptions("cxx") -- Required for nanobench

target("test_multiply")
    set_default(false)
    add_packages("catch2")
    add_deps("core")
    add_files("src/test_multiply.cpp")

target("benchmark_multiply")
    set_default(false)
    add_packages("nanobench")
    add_deps("core")
    add_files("src/bench_multiply.cpp")

target("benchmark_multiply_openblas")
    set_default(false)
    add_packages("nanobench", "openblas")
    add_deps("core")
    add_files("src/bench_multiply_openblas.cpp")

target("cache_test_conv")
    set_default(false)
    add_packages("nanobench")
    add_deps("core")
    add_files("src/cache_test_conv.cpp")

target("cache_test_conv_packed")
    set_default(false)
    add_packages("nanobench")
    add_deps("core")
    add_defines("USE_PACKED_CONV")
    add_files("src/cache_test_conv.cpp")

target("cache_test_multiply")
    set_default(false)
    add_packages("nanobench")
    add_deps("core")
    add_files("src/cache_test_multiply.cpp")
