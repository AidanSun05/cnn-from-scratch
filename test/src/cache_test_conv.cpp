#include <vector>

#include <nanobench.h>

#include "test_common.hpp"
#include "core/conv.h"

#ifdef USE_PACKED_CONV
#define CONV_FN conv2dPacked
#else
#define CONV_FN conv2d
#endif

constexpr int matrixSize = 1024;
constexpr int kernelSize = 3;
constexpr int outSize = matrixSize - kernelSize + 1;

constexpr int numChannels = 32;
constexpr int numKernels = 128;

int main() {
    std::vector<float> x(matrixSize * matrixSize * numChannels, 0);
    std::vector<float> w(kernelSize * kernelSize * numChannels * numKernels, 0);
    std::vector<float> b(numKernels, 0);
    std::vector<float> out(outSize * outSize * numKernels, 0);

    // Fill input image and weights with random data, keep biases at 0
    RandomFiller filler;
    filler.fill(x);
    filler.fill(w);

    CONV_FN(x.data(), w.data(), b.data(), out.data(), outSize, outSize, 1, matrixSize, matrixSize, numChannels,
        numKernels, kernelSize, kernelSize);
    ankerl::nanobench::doNotOptimizeAway(out);
}
