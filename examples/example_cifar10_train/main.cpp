#include <cassert>
#include <fstream>
#include <iostream>
#include <span>
#include <utility>

#include "core/ops.h"
#include "inference/nn.h"
#include "training/nn.hpp"
#include "training/optimizer.hpp"

static constexpr std::array trainDataPaths{
    "cifar-10-batches-bin/data_batch_1.bin",
    "cifar-10-batches-bin/data_batch_2.bin",
    "cifar-10-batches-bin/data_batch_3.bin",
    "cifar-10-batches-bin/data_batch_4.bin",
    "cifar-10-batches-bin/data_batch_5.bin",
};
static constexpr std::array testDataPaths{
    "cifar-10-batches-bin/test_batch.bin",
};

static std::pair<Matrix, Matrix> loadCifar10(std::span<const char* const> paths) {
    const int maxImages = 10000 * paths.size();
    const int H = 32;
    const int W = 32;
    const int C = 3;
    const int dims = H * W * C;
    const int recordSize = 1 + dims;

    Matrix data{ maxImages, dims };
    Matrix labels{ maxImages, 10 };

    int n = 0;
    for (const auto& path : paths) {
        std::ifstream file{ path, std::ios::binary };

        unsigned char buffer[recordSize];
        while (n < maxImages && file.read(reinterpret_cast<char*>(buffer), recordSize)) {
            unsigned char label = buffer[0];
            labels(n, label) = 1.0f;

            // CIFAR-10 stores: R(1024), G(1024), B(1024)
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int px = h * W + w;

                    // NHWC flattened index
                    int base = (h * W + w) * C;

                    // Store normalized values using mean/standard deviation of dataset
                    data(n, base + 0) = (buffer[1 + px] / 255.0f - 0.4914f) / 0.2470f; // R
                    data(n, base + 1) = (buffer[1 + 1024 + px] / 255.0f - 0.4822f) / 0.2435f; // G
                    data(n, base + 2) = (buffer[1 + 2048 + px] / 255.0f - 0.4465f) / 0.2616f; // B
                }
            }

            n++;
        }
    }

    assert(n == maxImages);
    return { data, labels };
}

static void train() {
    auto [x, y] = loadCifar10(trainDataPaths);

    const int BATCH_SIZE = 64;
    Sequential nn{ makeSizeInfoCNN(BATCH_SIZE, 32, 32, 3) };
    nn.add<Conv2D>(Activation::ReLU, 32, 3, 3);
    nn.add<MaxPool2D>(2, 2);
    nn.add<Conv2D>(Activation::ReLU, 64, 3, 3);
    nn.add<MaxPool2D>(2, 2);
    nn.add<Conv2D>(Activation::ReLU, 128, 3, 3);
    nn.add<MaxPool2D>(2, 2);
    nn.add<Flatten>();
    nn.add<Linear>(Activation::ReLU, 128);
    nn.add<Linear>(Activation::Softmax, 10);

    Adam opt;
    nn.sgd(x, y, 20, opt);
    nn.save("model-cifar-10.bin");
}

static void test() {
    const int SAMPLES = 10000;
    const auto [x, y] = loadCifar10(testDataPaths);

    NNModel* model;
    if (int rc = nnLoad("model-cifar-10.bin", &model, 1); rc != NN_OK) {
        std::cerr << "Model load failed: " << rc << "\n";
        return;
    }

    int numCorrect = 0;
    Matrix sampleX{ 1, 3072 };
    Matrix sampleY{ 1, 10 };

    for (int i = 0; i < SAMPLES; i++) {
        std::array indices{ i };
        x.sliceRowsTo(sampleX, indices);
        y.sliceRowsTo(sampleY, indices);

        const float* pred = nnPredict(model, sampleX.data().data(), 3072, nullptr);
        numCorrect += countSame(sampleY.data().data(), pred, 1, 10);
    }

    std::cout << "Test accuracy: " << static_cast<float>(numCorrect) / SAMPLES << "\n";
    nnFree(model);
}

int main() {
    MemoryManager manager;

    train();
    test();
}
