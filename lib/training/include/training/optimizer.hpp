#pragma once

#include <unordered_map>
#include <vector>

#include "training/matrix.hpp"

struct Optimizer {
    virtual ~Optimizer() = default;

    // Initializes this optimizer.
    // The params vector contains pointers to all weight buffers in the model.
    // Keeping track of weights through pointers is safe since layers never deallocate weights during training.
    virtual void init(const std::vector<std::span<float>>& params) {}

    // Prepares this optimizer for a single SGD step.
    virtual void beginStep() {}

    // Performs a step on the given matrix using the specified gradient.
    virtual void step(Matrix& param, const Matrix& grad) = 0;
};

class SGD : public Optimizer {
    float lrate;

public:
    explicit SGD(float lrate = 0.005f) : lrate(lrate) {}

    void step(Matrix& param, const Matrix& grad) override;
};

class Adam : public Optimizer {
    // Hyperparameters
    float lrate;
    float beta1;
    float beta2;
    float eps;

    // State
    unsigned int t = 0;
    float bc1 = 0.0f;
    float bc2 = 0.0f;

    struct State {
        std::vector<float> m;
        std::vector<float> v;
    };

    std::vector<State> states;
    std::unordered_map<float*, int> paramIndex;

public:
    explicit Adam(float lrate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) :
        lrate(lrate), beta1(beta1), beta2(beta2), eps(eps) {}

    void init(const std::vector<std::span<float>>& params) override;

    void beginStep() override;

    void step(Matrix& param, const Matrix& grad) override;
};
