#pragma once

#include <concepts>
#include <fstream>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "format/format.h"
#include "training/matrix.hpp"
#include "training/optimizer.hpp"

struct MemoryManager {
    MemoryManager();

    ~MemoryManager();
};

struct LayerSize {
    int n;
    int h;
    int w;
    int c;
    int rowsPerSample;
    int cols;
};

inline LayerSize makeSizeInfoCNN(int n, int h, int w, int c) {
    return { n, h, w, c, h * w, c };
}

inline LayerSize makeSizeInfoMLP(int n, int d) {
    return { n, 0, 0, 0, 1, d };
}

struct Layer {
    virtual ~Layer() = default;

    // Initializes the weights in this layer.
    // inSize specifies the dimensions of the input data to this layer.
    virtual LayerSize init(const LayerSize& inSize) = 0;

    // Appends the addresses of weight buffers of this layer to a common vector.
    // Used to keep track of per-weight state in optimizers.
    virtual void collectParams(std::vector<std::span<float>>& out) {}

    // Runs the forward pass of this layer.
    virtual void forward(const Matrix& in, Matrix& out) = 0;

    // Runs the backward pass of this layer with a specified optimizer.
    // Computes the gradient of this layer using the incoming gradient and previous input to this layer, then writes it
    // to prevIn.
    virtual void backward(Matrix& inGrad, Matrix& prevIn, Optimizer& opt) = 0;

    // Returns the descriptor for this layer, to write into the model binary file.
    virtual LayerDescriptor getDescriptorBinary(uint32_t& offset) const = 0;

    // Writes this layer's weight buffers into the specified stream.
    virtual void saveWeights(std::ofstream& f) const = 0;
};

using LayerPtr = std::unique_ptr<Layer>;

enum class Activation { ReLU, Softmax };

class Linear : public Layer {
    Activation act;

    int m = -1;
    int n;

    Matrix W;
    Matrix b;

    Matrix Z;
    Matrix dLdW;
    Matrix dLdb;

public:
    Linear(Activation act, int n) : act(act), n(n), b{ 1, n } {}

    LayerSize init(const LayerSize& inSize) override;

    void collectParams(std::vector<std::span<float>>& out) override;

    void forward(const Matrix& in, Matrix& out) override;

    void backward(Matrix& inGrad, Matrix& prevIn, Optimizer& opt) override;

    LayerDescriptor getDescriptorBinary(uint32_t& offset) const override;

    void saveWeights(std::ofstream& f) const override;
};

class Conv2D : public Layer {
    Activation act;

    int n = -1;
    int h = -1;
    int w = -1;
    int c = -1;
    int k;
    int kh;
    int kw;
    int outH = -1;
    int outW = -1;

    Matrix W;
    Matrix b;

    Matrix Z;
    Matrix dLdW;
    Matrix dLdb;

public:
    Conv2D(Activation act, int k, int kh, int kw) : act(act), k(k), kh(kh), kw(kw), b{ 1, k } {}

    LayerSize init(const LayerSize& inSize) override;

    void collectParams(std::vector<std::span<float>>& out) override;

    void forward(const Matrix& in, Matrix& out) override;

    void backward(Matrix& inGrad, Matrix& prevIn, Optimizer& opt) override;

    LayerDescriptor getDescriptorBinary(uint32_t& offset) const override;

    void saveWeights(std::ofstream& f) const override;
};

class MaxPool2D : public Layer {
    int n = -1;
    int h = -1;
    int w = -1;
    int c = -1;
    int kh;
    int kw;
    int outH = -1;
    int outW = -1;

    std::vector<uint32_t> locs;

public:
    MaxPool2D(int kh, int kw) : kh(kh), kw(kw) {}

    LayerSize init(const LayerSize& inSize) override;

    void forward(const Matrix& in, Matrix& out) override;

    void backward(Matrix& inGrad, Matrix& prevIn, Optimizer& opt) override;

    LayerDescriptor getDescriptorBinary(uint32_t& offset) const override;

    void saveWeights(std::ofstream& f) const override {}
};

class Flatten : public Layer {
    int n = -1;
    int h = -1;
    int w = -1;
    int c = -1;

public:
    Flatten() {}

    LayerSize init(const LayerSize& inSize) override;

    void forward(const Matrix& in, Matrix& out) override;

    void backward(Matrix& inGrad, Matrix& prevIn, Optimizer& opt) override;

    LayerDescriptor getDescriptorBinary(uint32_t& offset) const override;

    void saveWeights(std::ofstream& f) const override {}
};

struct NLL {
    float forward(const Matrix& Ypred, const Matrix& Y);

    void backward(Matrix& Ypred, const Matrix& Y);
};

class Sequential {
    std::vector<LayerPtr> layers;
    std::vector<Matrix> layerOutputs;
    LayerSize curSize;
    int maxBufSize;
    NLL loss;

    void forward(const Matrix& Xt);

    void backward(Optimizer& opt);

public:
    // Initializes a Sequential model. The size parameter specifies the dimensions of the model input data.
    Sequential(LayerSize size) :
        layerOutputs{ { size.n * size.rowsPerSample, size.cols } }, curSize(size),
        maxBufSize(size.rowsPerSample * size.cols) {}

    // Adds a layer to this model.
    template <class T, class... Args>
    requires std::derived_from<T, Layer> && std::constructible_from<T, Args...>
    void add(Args&&... args) {
        layers.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
        curSize = layers.back()->init(curSize);
        layerOutputs.emplace_back(curSize.n * curSize.rowsPerSample, curSize.cols);

        // Keep track of the maximum layer output buffer size used in this model
        // This is used during inference to allocate a constant number of buffers
        if (curSize.rowsPerSample * curSize.cols > maxBufSize) {
            maxBufSize = curSize.rowsPerSample * curSize.cols;
        }
    }

    // Performs stochastic gradient descent on this model with a specified optimizer.
    void sgd(const Matrix& X, const Matrix& Y, int epochs, Optimizer& opt);

    // Writes this model's layers and weights to a file path.
    void save(std::string_view path) const;
};
