#include <cassert>
#include <fstream>
#include <iostream>
#include <utility>

#include "core/ops.h"
#include "inference/nn.h"
#include "training/nn.hpp"

static constexpr auto trainDataPath = "data-train.dat";
static constexpr auto trainLabelsPath = "labels-train.dat";

static constexpr auto testDataPath = "data-test.dat";
static constexpr auto testLabelsPath = "labels-test.dat";

static std::pair<Matrix, Matrix> loadData(const char* dataPath, const char* labelsPath, int maxImages) {
    const int H = 32;
    const int W = 32;
    const int C = 3;
    const int dims = H * W * C;

    std::ifstream dataFile{ dataPath, std::ios::binary };
    Matrix data{ maxImages, dims };
    dataFile.read(reinterpret_cast<char*>(data.data().data()), data.data().size() * sizeof(float));

    std::ifstream labelsFile{ labelsPath, std::ios::binary };
    std::vector<std::uint8_t> trainLabels(maxImages, 0);
    labelsFile.read(reinterpret_cast<char*>(trainLabels.data()), trainLabels.size() * sizeof(std::uint8_t));

    Matrix labels{ maxImages, 3 };
    for (int i = 0; i < maxImages; i++) {
        labels(i, trainLabels[i]) = 1.0f;
    }

    return { data, labels };
}

static void train() {
    auto [x, y] = loadData(trainDataPath, trainLabelsPath, 20000);

    const int BATCH_SIZE = 32;
    Sequential nn{ makeSizeInfoCNN(BATCH_SIZE, 32, 32, 3) };
    nn.add<Conv2D>(Activation::ReLU, 16, 3, 3);
    nn.add<MaxPool2D>(2, 2);
    nn.add<Conv2D>(Activation::ReLU, 32, 3, 3);
    nn.add<MaxPool2D>(2, 2);
    nn.add<Conv2D>(Activation::ReLU, 64, 3, 3);
    nn.add<Flatten>();
    nn.add<Linear>(Activation::ReLU, 128);
    nn.add<Linear>(Activation::Softmax, 3);

    SGD opt;
    nn.sgd(x, y, 2, opt);
    nn.save("model-synthetic.bin");
}

static void test() {
    const int SAMPLES = 5000;
    const auto [x, y] = loadData(testDataPath, testLabelsPath, SAMPLES);

    NNModel* model;
    if (int rc = nnLoad("model-synthetic.bin", &model, 1); rc != NN_OK) {
        std::cerr << "Model load failed: " << rc << "\n";
        return;
    }

    int numCorrect = 0;
    Matrix sampleX{ 1, 3072 };
    Matrix sampleY{ 1, 3 };

    for (int i = 0; i < SAMPLES; i++) {
        std::array indices{ i };
        x.sliceRowsTo(sampleX, indices);
        y.sliceRowsTo(sampleY, indices);

        const float* pred = nnPredict(model, sampleX.data().data(), 3072, nullptr);
        numCorrect += countSame(sampleY.data().data(), pred, 1, 3);
    }

    std::cout << "Test accuracy: " << static_cast<float>(numCorrect) / SAMPLES << "\n";
    nnFree(model);
}

int main() {
    MemoryManager manager;

    train();
    test();
}
