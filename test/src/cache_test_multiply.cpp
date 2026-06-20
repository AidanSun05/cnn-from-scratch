#include <vector>

#include <nanobench.h>

#include "test_common.hpp"
#include "core/gemm.h"

constexpr int matrixSize = 512;

int main() {
    gemmInit();

    std::vector<float> a(matrixSize * matrixSize, 0);
    std::vector<float> b(matrixSize * matrixSize, 0);
    std::vector<float> c(matrixSize * matrixSize, 0);

    RandomFiller filler;
    filler.fill(a);
    filler.fill(b);

    multiply(a.data(), b.data(), c.data(), matrixSize, matrixSize, matrixSize, false, false, matrixSize, matrixSize);

    ankerl::nanobench::doNotOptimizeAway(c);
    gemmCleanup();
}
