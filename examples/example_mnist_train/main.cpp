#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "core/ops.h"
#include "inference/nn.h"
#include "training/nn.hpp"
#include "training/optimizer.hpp"

static constexpr auto trainDataPath = "mnist_train/mnist_train.csv";
static constexpr auto testDataPath = "mnist_test/mnist_test.csv";

template <std::size_t N, typename T, std::size_t... Is>
constexpr auto makeConsecutiveArrayImpl(T start, std::index_sequence<Is...>) {
    return std::array<T, N>{ { (start + static_cast<T>(Is))... } };
}

template <std::size_t N, typename T>
constexpr auto makeConsecutiveArray(T start) {
    return makeConsecutiveArrayImpl<N>(start, std::make_index_sequence<N>{});
}

static std::pair<Matrix, Matrix> loadMnist(std::string_view path, int maxLines) {
    const int dims = 28 * 28;
    const char delimiter = ',';

    Matrix data{ maxLines, dims };
    Matrix labels{ maxLines, 10 };

    std::ifstream file{ path.data() };

    int numLines = 0;
    std::string line;
    while (numLines < maxLines && std::getline(file, line)) {
        std::istringstream lineStream{ line };

        std::string labelStr;
        std::getline(lineStream, labelStr, delimiter);
        labels(numLines, std::stoi(labelStr)) = 1; // One-hot encoding

        int numPixels = 0;
        std::string pixelVal;
        while (std::getline(lineStream, pixelVal, delimiter)) {
            data(numLines, numPixels) = std::stof(pixelVal) / 255.0f;
            numPixels++;
        }

        assert(numPixels == dims);
        numLines++;
    }

    return { data, labels };
}

static void train() {
    const auto [x, y] = loadMnist(trainDataPath, 60000);

    const int BATCH_SIZE = 32;
    Sequential nn{ makeSizeInfoCNN(BATCH_SIZE, 28, 28, 1) };
    nn.add<Conv2D>(Activation::ReLU, 16, 3, 3);
    nn.add<MaxPool2D>(2, 2);
    nn.add<Conv2D>(Activation::ReLU, 32, 3, 3);
    nn.add<MaxPool2D>(2, 2);
    nn.add<Conv2D>(Activation::ReLU, 64, 3, 3);
    nn.add<Flatten>();
    nn.add<Linear>(Activation::ReLU, 128);
    nn.add<Linear>(Activation::Softmax, 10);

    SGD opt;
    nn.sgd(x, y, 2, opt);
    nn.save("model-mnist.bin");
}

static void test() {
    constexpr int SAMPLES = 10000;
    constexpr int BATCH_SIZE = 10;
    static_assert(SAMPLES % BATCH_SIZE == 0, "Sample count must be divisible by batch size");

    const auto [x, y] = loadMnist(testDataPath, SAMPLES);

    NNModel* model;
    if (int rc = nnLoad("model-mnist.bin", &model, BATCH_SIZE); rc != NN_OK) {
        std::cerr << "Model load failed: " << rc << "\n";
        return;
    }

    int numCorrect = 0;
    Matrix sampleX{ BATCH_SIZE, 784 };
    Matrix sampleY{ BATCH_SIZE, 10 };

    for (int i = 0; i < SAMPLES; i += BATCH_SIZE) {
        std::array indices = makeConsecutiveArray<BATCH_SIZE>(i);
        x.sliceRowsTo(sampleX, indices);
        y.sliceRowsTo(sampleY, indices);

        // Example of batch inference (running forward pass on BATCH_SIZE samples at a time)
        const float* pred = nnPredict(model, sampleX.data().data(), 784, nullptr);
        numCorrect += countSame(sampleY.data().data(), pred, BATCH_SIZE, 10);
    }

    std::cout << "Test accuracy: " << static_cast<float>(numCorrect) / SAMPLES << "\n";
    nnFree(model);
}

int main() {
    MemoryManager manager;

    train();
    test();
}
