#include "training/optimizer.hpp"

#include <cmath>

#include <omp.h>

void SGD::step(Matrix& param, const Matrix& grad) {
    auto p = param.data();
    auto g = grad.data();

#pragma omp simd
    for (int i = 0; i < p.size(); i++) {
        p[i] -= lrate * g[i];
    }
}

void Adam::init(const std::vector<std::span<float>>& params) {
    states.reserve(params.size());
    paramIndex.reserve(params.size());

    for (int i = 0; i < params.size(); i++) {
        const int n = params[i].size();
        states.push_back({ std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f) });
        paramIndex[params[i].data()] = i;
    }
}

void Adam::beginStep() {
    t++;
    bc1 = 1.0f - std::pow(beta1, t);
    bc2 = 1.0f - std::pow(beta2, t);
}

void Adam::step(Matrix& param, const Matrix& grad) {
    auto p = param.data();
    auto g = grad.data();
    State& s = states[paramIndex.at(p.data())];

    float* m = s.m.data();
    float* v = s.v.data();
    const int n = p.size();

#pragma omp simd
    for (int i = 0; i < n; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        p[i] -= lrate * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
    }
}
