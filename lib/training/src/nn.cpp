#include "training/nn.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include "core/gemm.h"
#include "format/format.h"
#include "training/matrix.hpp"
#include "training/optimizer.hpp"

static uint8_t encodeActivation(Activation act) {
    switch (act) {
        case Activation::ReLU:
            return ActivationType_ReLU;
        case Activation::Softmax:
            return ActivationType_Softmax;
        default:
            return ActivationType_None;
    }
}

static void activationForward(Activation act, Matrix& out) {
    switch (act) {
        case Activation::ReLU:
            out.reluInPlace();
            break;
        case Activation::Softmax:
            out.softmaxInPlace();
    }
}

static void activationBackward(Activation act, const Matrix& Z, Matrix& in) {
    if (act == Activation::ReLU) {
        in.reluBackwardInPlace(Z);
    }
}

MemoryManager::MemoryManager() {
    gemmInit();
}

MemoryManager::~MemoryManager() {
    gemmCleanup();
}

LayerSize Linear::init(const LayerSize& inSize) {
    m = inSize.cols;

    W = { m, n };

    // Training
    Z = { inSize.n * inSize.rowsPerSample, n };
    dLdW = { m, n };
    dLdb = { 1, n };

    std::random_device rd{};
    std::mt19937 gen{ rd() };

    // He initialization for weights
    std::normal_distribution d{ 0.0f, std::sqrt(2.0f / static_cast<float>(m)) };
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            W(i, j) = d(gen);
        }
    }

    LayerSize outSize = inSize;
    outSize.cols = n;
    return outSize;
}

void Linear::collectParams(std::vector<std::span<float>>& out) {
    out.push_back(W.data());
    out.push_back(b.data());
}

void Linear::forward(const Matrix& in, Matrix& out) {
    in.gemm(W, out, false, false);
    out.rvAdd(b);
    out.copyTo(Z);
    activationForward(act, out);
}

void Linear::backward(Matrix& inGrad, Matrix& prevIn, Optimizer& opt) {
    activationBackward(act, Z, inGrad);

    prevIn.gemm(inGrad, dLdW, true, false);
    inGrad.sumColsTo(dLdb);
    inGrad.gemm(W, prevIn, false, true);

    opt.step(W, dLdW);
    opt.step(b, dLdb);
}

LayerDescriptor Linear::getDescriptorBinary(uint32_t& offset) const {
    LayerDescriptor desc{};
    desc.type = LayerType_Linear;
    desc.activation = encodeActivation(act);
    desc.dim[0] = m;
    desc.dim[1] = n;

    desc.weightOffset = offset;
    desc.weightSize = m * n;
    offset += desc.weightSize * sizeof(float);

    desc.biasOffset = offset;
    desc.biasSize = n;
    offset += desc.biasSize * sizeof(float);

    return desc;
}

void Linear::saveWeights(std::ofstream& f) const {
    W.writeTo(f);
    b.writeTo(f);
}

LayerSize Conv2D::init(const LayerSize& inSize) {
    n = inSize.n;
    h = inSize.h;
    w = inSize.w;
    c = inSize.c;

    W = { kh * kw * c, k };
    outH = h - kh + 1;
    outW = w - kw + 1;

    const int outC = k;
    const int outRows = outH * outW;
    const int outCols = outC;

    // Training
    Z = { n * outRows, outCols };
    dLdW = { kh * kw * c, k };
    dLdb = { 1, k };

    const int in = c * kh * kw;

    std::random_device rd{};
    std::mt19937 gen{ rd() };

    // He initialization for weights
    std::normal_distribution d{ 0.0f, std::sqrt(2.0f / static_cast<float>(in)) };
    const auto [Wm, Wn] = W.shape();
    for (int i = 0; i < Wm; i++) {
        for (int j = 0; j < Wn; j++) {
            W(i, j) = d(gen);
        }
    }

    return { n, outH, outW, outC, outRows, outCols };
}

void Conv2D::collectParams(std::vector<std::span<float>>& out) {
    out.push_back(W.data());
    out.push_back(b.data());
}

void Conv2D::forward(const Matrix& in, Matrix& out) {
    in.conv2dTo(out, W, b, outH, outW, n, h, w, c, k, kh, kw);
    out.copyTo(Z);

    activationForward(act, out);
}

void Conv2D::backward(Matrix& inGrad, Matrix& prevIn, Optimizer& opt) {
    activationBackward(act, Z, inGrad);

    inGrad.sumColsTo(dLdb);
    prevIn.conv2dBackwardWTo(dLdW, inGrad, outH, outW, n, h, w, c, k, kh, kw);
    inGrad.conv2dBackwardTo(prevIn, W, outH, outW, n, h, w, c, k, kh, kw);

    opt.step(W, dLdW);
    opt.step(b, dLdb);
}

LayerDescriptor Conv2D::getDescriptorBinary(uint32_t& offset) const {
    LayerDescriptor desc{};
    desc.type = LayerType_Conv2D;
    desc.activation = encodeActivation(act);
    desc.dim[0] = h;
    desc.dim[1] = w;
    desc.dim[2] = c;
    desc.dim[3] = k;
    desc.dim[4] = kh;
    desc.dim[5] = kw;
    desc.dim[6] = outH;
    desc.dim[7] = outW;

    const auto [Wm, Wn] = W.shape();
    desc.weightOffset = offset;
    desc.weightSize = Wm * Wn;
    offset += desc.weightSize * sizeof(float);

    desc.biasOffset = offset;
    desc.biasSize = k;
    offset += desc.biasSize * sizeof(float);

    return desc;
}

void Conv2D::saveWeights(std::ofstream& f) const {
    W.writePackedWTo(f, k, kh, kw, c);
    b.writeTo(f);
}

LayerSize MaxPool2D::init(const LayerSize& inSize) {
    n = inSize.n;
    h = inSize.h;
    w = inSize.w;
    c = inSize.c;

    outH = h / kh;
    outW = w / kw;

    locs.resize(n * outH * outW * c);

    LayerSize outSize = inSize;
    outSize.h = outH;
    outSize.w = outW;
    outSize.rowsPerSample = outH * outW;
    return outSize;
}

void MaxPool2D::forward(const Matrix& in, Matrix& out) {
    in.maxPool2dTo(out, locs, n, h, w, c, kh, kw);
}

void MaxPool2D::backward(Matrix& inGrad, Matrix& prevIn, Optimizer&) {
    inGrad.maxPool2dBackwardTo(prevIn, locs, n, h, w, c, kh, kw);
}

LayerDescriptor MaxPool2D::getDescriptorBinary(uint32_t&) const {
    LayerDescriptor desc{};
    desc.type = LayerType_MaxPool2D;
    desc.activation = ActivationType_None;
    desc.dim[0] = h;
    desc.dim[1] = w;
    desc.dim[2] = c;
    desc.dim[3] = kh;
    desc.dim[4] = kw;
    desc.dim[5] = outH;
    desc.dim[6] = outW;

    return desc;
}

LayerSize Flatten::init(const LayerSize& inSize) {
    n = inSize.n;
    h = inSize.h;
    w = inSize.w;
    c = inSize.c;

    LayerSize outSize = inSize;
    outSize.rowsPerSample = 1;
    outSize.cols = h * w * c;
    return outSize;
}

void Flatten::forward(const Matrix& in, Matrix& out) {
    in.copyTo(out);
    out.reshape(n, h * w * c);
}

void Flatten::backward(Matrix& inGrad, Matrix& prevIn, Optimizer&) {
    inGrad.copyTo(prevIn);
    prevIn.reshape(n * h * w, c);
}

LayerDescriptor Flatten::getDescriptorBinary(uint32_t&) const {
    LayerDescriptor desc{};
    desc.type = LayerType_Flatten;
    desc.activation = ActivationType_None;

    return desc;
}

float NLL::forward(const Matrix& Ypred, const Matrix& Y) {
    return Y.nll(Ypred);
}

void NLL::backward(Matrix& Ypred, const Matrix& Y) {
    Ypred -= Y;
}

void Sequential::forward(const Matrix& Xt) {
    Xt.copyTo(layerOutputs.front());

    for (int i = 0; i < layers.size(); i++) {
        const Matrix& in = layerOutputs[i];
        Matrix& out = layerOutputs[i + 1];

        layers[i]->forward(in, out);
    }
}

void Sequential::backward(Optimizer& opt) {
    for (int i = layers.size() - 1; i > -1; i--) {
        Matrix& prevIn = layerOutputs[i];
        Matrix& inGrad = layerOutputs[i + 1];

        layers[i]->backward(inGrad, prevIn, opt);
    }
}

void Sequential::sgd(const Matrix& X, const Matrix& Y, int epochs, Optimizer& opt) {
    // Collect parameters for optimizer
    std::vector<std::span<float>> params;
    for (auto& layer : layers) {
        layer->collectParams(params);
    }
    opt.init(params);

    const int N = X.shape()[0];
    const int K = curSize.n;
    const int numBatches = N / K;

    unsigned int it = 0;
    std::vector<int> indices(N, 0);
    std::iota(indices.begin(), indices.end(), 0);
    auto rng = std::default_random_engine{};

    // Batch buffers (reused between training and accuracy evaluations)
    Matrix Xbatch{ K, X.shape()[1] };
    Matrix Ybatch{ K, Y.shape()[1] };

    std::vector<int> evalIndices(K);

    // Allocation-free training loop
    for (int i = 0; i < epochs; i++) {
        std::cout << "===== Epoch " << (i + 1) << " =====\n";

        std::ranges::shuffle(indices, rng);
        for (int j = 0; j < numBatches; j++) {
            const auto batchStart = indices.begin() + j * K;
            const auto batchEnd = indices.begin() + (j + 1) * K;
            X.sliceRowsTo(Xbatch, { batchStart, batchEnd });
            Y.sliceRowsTo(Ybatch, { batchStart, batchEnd });

            forward(Xbatch);
            Matrix& Ypred = layerOutputs.back();

            const float curLoss = loss.forward(Ypred, Ybatch);
            loss.backward(Ypred, Ybatch);

            opt.beginStep();
            backward(opt);

            // Print accuracy
            constexpr int every = 200;
            if (it != 0 && it % every == 0) {
                unsigned int numCorrect = 0;
                for (int k = 0; k < numBatches; k++) {
                    std::iota(evalIndices.begin(), evalIndices.end(), k * K);
                    X.sliceRowsTo(Xbatch, evalIndices);
                    Y.sliceRowsTo(Ybatch, evalIndices);

                    forward(Xbatch);
                    numCorrect += layerOutputs.back().countSame(Ybatch);
                }

                const float acc = static_cast<float>(numCorrect) / (numBatches * K);
                std::cout << "Iteration = " << it << "\tAcc = " << acc << "\tLoss = " << curLoss << "\n";
            }

            it++;
        }
    }
}

void Sequential::save(std::string_view path) const {
    std::ofstream f{ path.data() };

    ModelHeader header{
        { 'N', 'N', 'M', 'L' },
        1,
        static_cast<uint32_t>(layers.size()),
        static_cast<uint32_t>(maxBufSize),
        { 0 },
    };

    f.write(reinterpret_cast<const char*>(&header), sizeof(header));
    uint32_t offset = 0;

    // Write layer descriptors
    for (const auto& layer : layers) {
        LayerDescriptor desc = layer->getDescriptorBinary(offset);
        f.write(reinterpret_cast<const char*>(&desc), sizeof(desc));
    }

    // Write all weights sequentially
    for (const auto& layer : layers) {
        layer->saveWeights(f);
    }
}
