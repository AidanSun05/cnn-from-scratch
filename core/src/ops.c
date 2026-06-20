#include "core/ops.h"

#include <math.h>

void rvAdd(float* restrict a, const float* restrict b, int m, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        float* arow = a + i * n;

#pragma omp simd
        for (int j = 0; j < n; j++) {
            arow[j] += b[j];
        }
    }
}

void sub(float* restrict a, const float* restrict b, int m, int n) {
#pragma omp simd
    for (int i = 0; i < m * n; i++) {
        a[i] -= b[i];
    }
}

void sumCols(const float* restrict in, float* restrict out, int m, int n) {
#pragma omp parallel for reduction(+ : out[ : n]) schedule(static)
    for (int i = 0; i < m; i++) {
        const float* inrow = in + i * n;
        for (int j = 0; j < n; j++) {
            out[j] += inrow[j];
        }
    }
}

void relu(float* in, int m, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m * n; i++) {
        in[i] *= (in[i] > 0.0f);
    }
}

void reluBackward(float* restrict in, const float* restrict z, int m, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m * n; i++) {
        in[i] *= (z[i] > 0.0f);
    }
}

void softmax(float* in, int m, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        float* inrow = in + i * n;

        // Stable softmax: the maximum element is subtracted from each element in the row.
        // This keeps arguments to expf nonpositive, preventing overflow.
        //
        // This requires 3 separate passes over each row. For small output dimensionality this is acceptable, but models
        // with a higher number of output classes could benefit from improved cache locality.

        float rowmax = inrow[0];
        for (int j = 1; j < n; j++) {
            if (inrow[j] > rowmax) {
                rowmax = inrow[j];
            }
        }

        float rowsum = 0.0f;
        for (int j = 0; j < n; j++) {
            rowsum += expf(inrow[j] - rowmax);
        }

        for (int j = 0; j < n; j++) {
            inrow[j] = expf(inrow[j] - rowmax) / rowsum;
        }
    }
}

float nll(const float* restrict in, const float* restrict pred, int m, int n) {
    float ret = 0.0f;

#pragma omp parallel for reduction(+ : ret) schedule(static)
    for (int i = 0; i < m * n; i++) {
        float p = fmaxf(pred[i], 1e-12f);
        ret += -in[i] * logf(p);
    }

    return ret / (float)m;
}

int argmax(const float* in, int n) {
    int idx = 0;
    float maxVal = in[0];

    for (int i = 1; i < n; i++) {
        if (in[i] > maxVal) {
            maxVal = in[i];
            idx = i;
        }
    }
    return idx;
}

unsigned int countSame(const float* restrict a, const float* restrict b, int m, int n) {
    unsigned int total = 0;
    for (int i = 0; i < m; i++) {
        const float* arow = a + i * n;
        const float* brow = b + i * n;
        total += (argmax(arow, n) == argmax(brow, n));
    }

    return total;
}
