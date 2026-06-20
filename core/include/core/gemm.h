#pragma once

#ifndef __cplusplus
#include <stdbool.h>
#endif

#include "internal/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes resources for matrix multiplication.
void gemmInit(void);

// Frees resources allocated by gemmInit.
void gemmCleanup(void);

// Computes the matrix product of 2 given matrices.
//
// Input:
//   - a: Matrix A, of size m*k
//   - b: Matrix B, of size k*n
//   - m, k, n: Input matrix dimensions
//   - at, bt: Whether each matrix should be transposed before multiplication
//   - lda, ldb: Leading dimensions of each matrix
//
// Output:
//   - c: Result of the matrix product
//        A B,     if at = false, bt = false
//        A^T B,   if at = true,  bt = false
//        A B^T,   if at = false, bt = true
//        A^T B^T, if at = true,  bt = true
void multiply(const float* RESTRICT a, const float* RESTRICT b, float* RESTRICT c, int m, int k, int n, bool at,
    bool bt, int lda, int ldb);

#ifdef __cplusplus
}
#endif
