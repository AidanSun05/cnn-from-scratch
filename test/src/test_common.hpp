#pragma once

#include <random>
#include <vector>

constexpr auto RANDOM_SEED = 123;

class RandomFiller {
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

public:
    RandomFiller() : rng(RANDOM_SEED), dist(-10.0f, 10.0f) {}

    void fill(std::vector<float>& out) {
        for (auto& val : out) {
            val = dist(rng);
        }
    }
};
