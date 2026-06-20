#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include "test_common.hpp"
#include "core/gemm.h"

struct InitListener : Catch::EventListenerBase {
    using Catch::EventListenerBase::EventListenerBase;

    void testRunStarting(const Catch::TestRunInfo&) override {
        gemmInit();
    }

    void testRunEnded(const Catch::TestRunStats&) override {
        gemmCleanup();
    }
};

CATCH_REGISTER_LISTENER(InitListener)

// Reference implementation using triple loop
void refMultiply(const float* a, const float* b, float* c, int m, int k, int n, bool at, bool bt, int lda, int ldb) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0;
            for (int p = 0; p < k; ++p) {
                float a_val = at ? a[p * lda + i] : a[i * lda + p];
                float b_val = bt ? b[j * ldb + p] : b[p * ldb + j];
                sum += a_val * b_val;
            }
            c[i * n + j] = sum;
        }
    }
}

TEST_CASE("Matrix multiplication - Small square matrices") {
    int m = 4, k = 4, n = 4;
    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c1(m * n), c2(m * n);

    RandomFiller filler;
    filler.fill(a);
    filler.fill(b);
    filler.fill(c1);
    c2 = c1;

    multiply(a.data(), b.data(), c1.data(), m, k, n, false, false, k, n);
    refMultiply(a.data(), b.data(), c2.data(), m, k, n, false, false, k, n);

    for (int i = 0; i < m * n; ++i) {
        REQUIRE_THAT(c1[i], Catch::Matchers::WithinAbs(c2[i], 0.01));
    }
}

TEST_CASE("Matrix multiplication - Rectangular matrices") {
    int m = 3, k = 5, n = 2;
    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c1(m * n), c2(m * n);

    RandomFiller filler;
    filler.fill(a);
    filler.fill(b);
    filler.fill(c1);
    c2 = c1;

    multiply(a.data(), b.data(), c1.data(), m, k, n, false, false, k, n);
    refMultiply(a.data(), b.data(), c2.data(), m, k, n, false, false, k, n);

    for (int i = 0; i < m * n; ++i) {
        REQUIRE_THAT(c1[i], Catch::Matchers::WithinAbs(c2[i], 0.01));
    }
}

TEST_CASE("Matrix multiplication - With transposition") {
    int m = 4, k = 3, n = 2;
    std::vector<float> a(k * m); // Storage for A^T
    std::vector<float> b(n * k); // Storage for B^T
    std::vector<float> c1(m * n), c2(m * n);

    RandomFiller filler;
    filler.fill(a);
    filler.fill(b);
    filler.fill(c1);
    c2 = c1;

    multiply(a.data(), b.data(), c1.data(), m, k, n, true, true, m, k);
    refMultiply(a.data(), b.data(), c2.data(), m, k, n, true, true, m, k);

    for (int i = 0; i < m * n; ++i) {
        REQUIRE_THAT(c1[i], Catch::Matchers::WithinAbs(c2[i], 0.01));
    }
}

TEST_CASE("Matrix multiplication - Large matrices") {
    int m = 1000, k = 2000, n = 500;
    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c1(m * n), c2(m * n);

    RandomFiller filler;
    filler.fill(a);
    filler.fill(b);
    filler.fill(c1);
    c2 = c1;

    multiply(a.data(), b.data(), c1.data(), m, k, n, false, false, k, n);
    refMultiply(a.data(), b.data(), c2.data(), m, k, n, false, false, k, n);

    for (int i = 0; i < m * n; ++i) {
        REQUIRE_THAT(c1[i], Catch::Matchers::WithinAbs(c2[i], 0.01));
    }
}
