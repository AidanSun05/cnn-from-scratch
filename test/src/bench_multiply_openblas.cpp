#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>

#include "bench_common.hpp"
#include "core/internal/common.h"

extern "C" {
#include <cblas.h>
}

void testFn(const float* RESTRICT A, const float* RESTRICT B, float* RESTRICT C, std::size_t N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}

int main() {
    runBenchmark("multiply", testFn);
}
