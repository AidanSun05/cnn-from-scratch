#include "core/gemm.h"

#include <immintrin.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "core/internal/gemmconf.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))

float* aps[NUM_THREADS];
float* bps[NUM_THREADS];

void gemmInit(void) {
    omp_set_num_threads(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        aps[i] = aligned_alloc(CACHE_LINE_SIZE, MC * KC * sizeof(float));
        bps[i] = aligned_alloc(CACHE_LINE_SIZE, KC * NC * sizeof(float));
    }
}

void gemmCleanup(void) {
    for (int i = 0; i < NUM_THREADS; i++) {
        free(aps[i]);
        free(bps[i]);
    }
}

static inline void packA(const float* restrict a, float* restrict ap, bool at, int pc, int ic, int pmax, int imax,
    int lda) {
    for (int p = 0; p < pmax; p++) {
        float* dst = ap + p * MC;
        if (at) {
            const float* src = a + (pc + p) * lda + ic;
            memcpy(dst, src, imax * sizeof(float));
        } else {
            for (int i = 0; i < imax; i++) {
                dst[i] = a[(ic + i) * lda + pc + p];
            }
        }
    }
}

static inline void packB(const float* restrict b, float* restrict bp, bool bt, int jc, int pc, int jmax, int pmax,
    int ldb) {
    for (int p = 0; p < pmax; p++) {
        float* dst = bp + p * NC;
        if (bt) {
            for (int j = 0; j < jmax; j++) {
                dst[j] = b[(jc + j) * ldb + pc + p];
            }
        } else {
            const float* src = b + (pc + p) * ldb + jc;
            memcpy(dst, src, jmax * sizeof(float));
        }
    }
}

// GEMM microkernel for AVX2
static inline void microkernel(const float* restrict ap, const float* restrict bp, float* restrict c, int ldc,
    int jrmax, int pmax, int irmax, bool firstK) {
    // Assume jrmax <= 8, irmax <= 8, NR = 8 for float (since 8 floats fit in 256 bits)

    __m256 c0, c1, c2, c3, c4, c5, c6, c7;

    // Load existing values or start from zero
    if (firstK) {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
    } else {
        // Load existing values from C
        if (irmax > 0) c0 = _mm256_loadu_ps(c + 0 * ldc);
        if (irmax > 1) c1 = _mm256_loadu_ps(c + 1 * ldc);
        if (irmax > 2) c2 = _mm256_loadu_ps(c + 2 * ldc);
        if (irmax > 3) c3 = _mm256_loadu_ps(c + 3 * ldc);
        if (irmax > 4) c4 = _mm256_loadu_ps(c + 4 * ldc);
        if (irmax > 5) c5 = _mm256_loadu_ps(c + 5 * ldc);
        if (irmax > 6) c6 = _mm256_loadu_ps(c + 6 * ldc);
        if (irmax > 7) c7 = _mm256_loadu_ps(c + 7 * ldc);
    }

    for (int p = 0; p < pmax; ++p) {
        const float* arow = ap + p * MC;
        const float* brow = bp + p * NC; // length >= jrmax
        __m256 b = _mm256_loadu_ps(brow); // 8 floats (jrmax <= 8)

        if (irmax > 0) {
            __m256 a = _mm256_broadcast_ss(arow + 0);
            c0 = _mm256_fmadd_ps(a, b, c0);
        }
        if (irmax > 1) {
            __m256 a = _mm256_broadcast_ss(arow + 1);
            c1 = _mm256_fmadd_ps(a, b, c1);
        }
        if (irmax > 2) {
            __m256 a = _mm256_broadcast_ss(arow + 2);
            c2 = _mm256_fmadd_ps(a, b, c2);
        }
        if (irmax > 3) {
            __m256 a = _mm256_broadcast_ss(arow + 3);
            c3 = _mm256_fmadd_ps(a, b, c3);
        }
        if (irmax > 4) {
            __m256 a = _mm256_broadcast_ss(arow + 4);
            c4 = _mm256_fmadd_ps(a, b, c4);
        }
        if (irmax > 5) {
            __m256 a = _mm256_broadcast_ss(arow + 5);
            c5 = _mm256_fmadd_ps(a, b, c5);
        }
        if (irmax > 6) {
            __m256 a = _mm256_broadcast_ss(arow + 6);
            c6 = _mm256_fmadd_ps(a, b, c6);
        }
        if (irmax > 7) {
            __m256 a = _mm256_broadcast_ss(arow + 7);
            c7 = _mm256_fmadd_ps(a, b, c7);
        }
    }

    // Write back
    for (int i = 0; i < irmax; ++i) {
        __m256 v = (i == 0) ? c0
            : (i == 1)      ? c1
            : (i == 2)      ? c2
            : (i == 3)      ? c3
            : (i == 4)      ? c4
            : (i == 5)      ? c5
            : (i == 6)      ? c6
                            : c7;
        float* ci = c + i * ldc;

        if (jrmax == 8) {
            _mm256_storeu_ps(ci, v);
        } else {
            // tail j < 8
            float temp[8];
            _mm256_storeu_ps(temp, v);
            memcpy(ci, temp, jrmax * sizeof(float));
        }
    }
}

void multiply(const float* restrict a, const float* restrict b, float* restrict c, int m, int k, int n, bool at,
    bool bt, int lda, int ldb) {
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        float* ap = aps[tid];
        float* bp = bps[tid];

        // Pack B
#pragma omp for schedule(static)
        for (int jc = 0; jc < n; jc += NC) {
            const int jmax = min(n - jc, NC);
            for (int pc = 0; pc < k; pc += KC) {
                const int pmax = min(k - pc, KC);
                const bool firstK = (pc == 0);

                packB(b, bp, bt, jc, pc, jmax, pmax, ldb);

                // Pack A
                for (int ic = 0; ic < m; ic += MC) {
                    const int imax = min(m - ic, MC);
                    packA(a, ap, at, pc, ic, pmax, imax, lda);

                    for (int jr = 0; jr < jmax; jr += NR) {
                        const int jrmax = min(jmax - jr, NR);
                        const float* bp0 = bp + jr;
                        for (int ir = 0; ir < imax; ir += MR) {
                            const int irmax = min(imax - ir, MR);
                            const float* ap0 = ap + ir;
                            float* cblk = c + (ic + ir) * n + (jc + jr);

                            microkernel(ap0, bp0, cblk, n, jrmax, pmax, irmax, firstK);
                        }
                    }
                }
            }
        }
    }
}
