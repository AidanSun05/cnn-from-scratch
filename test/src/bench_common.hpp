#pragma once

#include <chrono>
#include <cstddef>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <nanobench.h>

#include "test_common.hpp"

template <std::size_t N, class Fn>
void bench(ankerl::nanobench::Bench* b, RandomFiller& filler, Fn f) {
    constexpr std::size_t squareSize = N * N;

    std::vector<float> A(squareSize, 0);
    std::vector<float> B(squareSize, 0);
    std::vector<float> C(squareSize, 0);

    filler.fill(A);
    filler.fill(B);
    filler.fill(C);

    std::string benchName = std::format("{}x{}", N, N);
    b->run(benchName.data(), [&]() {
        f(A.data(), B.data(), C.data(), N);
        ankerl::nanobench::doNotOptimizeAway(C);
    });
}

template <class Fn>
void runBenchmark(std::string_view title, Fn f) {
    using namespace std::literals;

    RandomFiller filler;
    ankerl::nanobench::Bench b;
    b.title(title.data()).warmup(3).performanceCounters(true).minEpochTime(200ms);

    bench<64>(&b, filler, f);
    bench<128>(&b, filler, f);
    bench<256>(&b, filler, f);
    bench<512>(&b, filler, f);
    bench<1024>(&b, filler, f);
}
