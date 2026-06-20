#pragma once

#include "internal/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Computes the broadcasted addition of a matrix and a row vector.
void rvAdd(float* RESTRICT a, const float* RESTRICT b, int m, int n);

// Computes the elementwise difference of 2 matrices.
void sub(float* RESTRICT a, const float* RESTRICT b, int m, int n);

// Sums each column of a given matrix and stores each result in a vector.
void sumCols(const float* RESTRICT in, float* RESTRICT out, int m, int n);

// Computes the ReLU of a given preactivation.
void relu(float* in, int m, int n);

// Computes the ReLU gradient given the incoming gradient and preactivations.
void reluBackward(float* RESTRICT in, const float* RESTRICT z, int m, int n);

// Computes the per-row softmax of a given matrix.
void softmax(float* in, int m, int n);

// Computes the NLL loss of a given prediction.
float nll(const float* RESTRICT in, const float* RESTRICT pred, int m, int n);

// Computes the argmax of a given vector.
int argmax(const float* in, int n);

// Counts the number of rows in matrix a which have the same argmax as the corresponding row in matrix b.
unsigned int countSame(const float* RESTRICT a, const float* RESTRICT b, int m, int n);

#ifdef __cplusplus
}
#endif
