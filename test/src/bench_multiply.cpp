#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>

#include "bench_common.hpp"
#include "core/gemm.h"

void testFn(const float* RESTRICT A, const float* RESTRICT B, float* RESTRICT C, std::size_t N) {
    multiply(A, B, C, N, N, N, false, false, N, N);
}

int main() {
    gemmInit();

    runBenchmark("multiply", testFn);

    gemmCleanup();
}
