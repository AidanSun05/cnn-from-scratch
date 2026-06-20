#include "core/conv.h"

#include <assert.h>
#include <immintrin.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define OH_TILE 2
#define OW_TILE 6

void conv2d(const float* restrict x, const float* restrict w, const float* restrict b, float* restrict out, int OH,
    int OW, int N, int H, int W, int C, int K, int Kh, int Kw) {
#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int oh0 = 0; oh0 < OH; oh0 += OH_TILE) {
            for (int ow0 = 0; ow0 < OW; ow0 += OW_TILE) {
                const int oh_t = (oh0 + OH_TILE <= OH) ? OH_TILE : OH - oh0;
                const int ow_t = (ow0 + OW_TILE <= OW) ? OW_TILE : OW - ow0;
                const float* xn = x + (size_t)n * H * W * C;
                float* outn = out + (size_t)n * OH * OW * K;

                for (int k = 0; k < K; k += 8) {
                    // 12 independent FMA chains: 2 OH rows * 6 OW cols.
                    // One weight vector is reused across all 12 positions.
                    __m256 acc[OH_TILE][OW_TILE];
                    const __m256 bv = _mm256_loadu_ps(b + k);
                    for (int th = 0; th < oh_t; th++)
                        for (int tw = 0; tw < ow_t; tw++) acc[th][tw] = bv;

                    for (int kh = 0; kh < Kh; kh++) {
                        for (int kw = 0; kw < Kw; kw++) {
                            for (int c = 0; c < C; c++) {
                                const __m256 wv = _mm256_loadu_ps(w + ((kh * Kw + kw) * C + c) * K + k);

                                for (int th = 0; th < oh_t; th++) {
                                    const float* xrow = xn + ((oh0 + th + kh) * W + ow0 + kw) * C;

                                    for (int tw = 0; tw < ow_t; tw++) {
                                        const __m256 iv = _mm256_set1_ps(xrow[tw * C + c]);
                                        acc[th][tw] = _mm256_fmadd_ps(iv, wv, acc[th][tw]);
                                    }
                                }
                            }
                        }
                    }

                    for (int th = 0; th < oh_t; th++)
                        for (int tw = 0; tw < ow_t; tw++)
                            _mm256_storeu_ps(outn + ((oh0 + th) * OW + ow0 + tw) * K + k, acc[th][tw]);
                }
            }
        }
    }
}

void conv2dPacked(const float* restrict x, const float* restrict wp, const float* restrict b, float* restrict out,
    int OH, int OW, int N, int H, int W, int C, int K, int Kh, int Kw) {
    assert(K % 8 == 0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int oh0 = 0; oh0 < OH; oh0 += OH_TILE) {
            for (int ow0 = 0; ow0 < OW; ow0 += OW_TILE) {
                const int oh_t = (oh0 + OH_TILE <= OH) ? OH_TILE : OH - oh0;
                const int ow_t = (ow0 + OW_TILE <= OW) ? OW_TILE : OW - ow0;

                const float* xn = x + (size_t)n * H * W * C;
                float* outn = out + (size_t)n * OH * OW * K;

                for (int k = 0; k < K; k += 8) {
                    // Base pointer into the packed weight array for this k-block.  All (kh, kw, c) entries follow
                    // sequentially:
                    //
                    //  wbase[((kh*Kw + kw)*C + c) * 8 .. +7]
                    const float* wbase = wp + (k / 8) * Kh * Kw * C * 8;

                    // OH_TILE*OW_TILE independent accumulator chains.
                    __m256 acc[OH_TILE][OW_TILE];
                    const __m256 bv = _mm256_loadu_ps(b + k);
                    for (int th = 0; th < oh_t; th++) {
                        for (int tw = 0; tw < ow_t; tw++) {
                            acc[th][tw] = bv;
                        }
                    }

                    for (int kh = 0; kh < Kh; kh++) {
                        for (int kw = 0; kw < Kw; kw++) {
                            // Weight pointer for this (kh, kw) row. Advances by 8 floats per c step
                            const float* wrow = wbase + (kh * Kw + kw) * C * 8;

                            if (C >= 4) {
                                for (int c = 0; c < C; c += 4) {
                                    // 4 weight vectors loaded per iteration
                                    const __m256 wv0 = _mm256_loadu_ps(wrow + (c + 0) * 8);
                                    const __m256 wv1 = _mm256_loadu_ps(wrow + (c + 1) * 8);
                                    const __m256 wv2 = _mm256_loadu_ps(wrow + (c + 2) * 8);
                                    const __m256 wv3 = _mm256_loadu_ps(wrow + (c + 3) * 8);

                                    for (int th = 0; th < oh_t; th++) {
                                        const float* xrow = xn + ((oh0 + th + kh) * W + ow0 + kw) * C;

                                        for (int tw = 0; tw < ow_t; tw++) {
                                            // Input values for channels c..c+3 at this output column
                                            const float* xp = xrow + tw * C + c;

                                            acc[th][tw] = _mm256_fmadd_ps(_mm256_set1_ps(xp[0]), wv0, acc[th][tw]);
                                            acc[th][tw] = _mm256_fmadd_ps(_mm256_set1_ps(xp[1]), wv1, acc[th][tw]);
                                            acc[th][tw] = _mm256_fmadd_ps(_mm256_set1_ps(xp[2]), wv2, acc[th][tw]);
                                            acc[th][tw] = _mm256_fmadd_ps(_mm256_set1_ps(xp[3]), wv3, acc[th][tw]);
                                        }
                                    }
                                }
                            }

                            for (int c = (C & ~3); c < C; c++) {
                                const __m256 wv = _mm256_loadu_ps(wrow + c * 8);
                                for (int th = 0; th < oh_t; th++) {
                                    const float* xrow = xn + ((oh0 + th + kh) * W + ow0 + kw) * C;
                                    for (int tw = 0; tw < ow_t; tw++) {
                                        acc[th][tw]
                                            = _mm256_fmadd_ps(_mm256_set1_ps(xrow[tw * C + c]), wv, acc[th][tw]);
                                    }
                                }
                            }
                        }
                    }

                    for (int th = 0; th < oh_t; th++) {
                        for (int tw = 0; tw < ow_t; tw++) {
                            _mm256_storeu_ps(outn + ((oh0 + th) * OW + ow0 + tw) * K + k, acc[th][tw]);
                        }
                    }
                }
            }
        }
    }
}

void conv2dBackwardW(const float* RESTRICT x, const float* RESTRICT dout, float* RESTRICT dw, int OH, int OW, int N,
    int H, int W, int C, int K, int Kh, int Kw) {
#pragma omp parallel for collapse(3) schedule(static)
    for (int kh = 0; kh < Kh; kh++) {
        for (int kw = 0; kw < Kw; kw++) {
            for (int c = 0; c < C; c++) {
                for (int k = 0; k < K; k += 8) {
                    __m256 acc = _mm256_setzero_ps();

                    for (int n = 0; n < N; n++) {
                        const float* x0 = x + n * H * W * C;
                        const float* dout0 = dout + n * OH * OW * K;

                        for (int oh = 0; oh < OH; oh++) {
                            const float* xrow = x0 + (oh + kh) * W * C;
                            const float* doutrow = dout0 + oh * OW * K;

                            for (int ow = 0; ow < OW; ow++) {
                                __m256 inBroadcast = _mm256_set1_ps(xrow[(ow + kw) * C + c]);
                                __m256 doutVec = _mm256_loadu_ps(doutrow + ow * K + k);
                                acc = _mm256_fmadd_ps(inBroadcast, doutVec, acc);
                            }
                        }
                    }

                    _mm256_storeu_ps(dw + ((kh * Kw + kw) * C + c) * K + k, acc);
                }
            }
        }
    }
}

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo); // [1,1,3,3]
    lo = _mm_add_ps(lo, shuf); // [0+1, _, 2+3, _]
    shuf = _mm_movehl_ps(shuf, lo); // high half -> low
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

void conv2dBackward(const float* RESTRICT dout, const float* RESTRICT w, float* RESTRICT dx, int OH, int OW, int N,
    int H, int W, int C, int K, int Kh, int Kw) {
#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w_ = 0; w_ < W; w_++) {
                const float* dout_n = dout + n * OH * OW * K;
                float* dx_n = dx + n * H * W * C;

                for (int c = 0; c < C; c++) {
                    __m256 acc = _mm256_setzero_ps();

                    for (int kh = 0; kh < Kh; kh++) {
                        const int oh = h - kh;
                        if ((unsigned)oh >= (unsigned)OH) {
                            continue;
                        }

                        for (int kw = 0; kw < Kw; kw++) {
                            const int ow = w_ - kw;
                            if ((unsigned)ow >= (unsigned)OW) {
                                continue;
                            }

                            const float* dout_ptr = dout_n + (oh * OW + ow) * K;

                            for (int k = 0; k < K; k += 8) {
                                int wi = ((kh * Kw + kw) * C + c) * K + k;

                                __m256 doutVec = _mm256_loadu_ps(dout_ptr + k);
                                __m256 wVec = _mm256_loadu_ps(w + wi);
                                acc = _mm256_fmadd_ps(doutVec, wVec, acc);
                            }
                        }
                    }

                    dx_n[(h * W + w_) * C + c] = hsum(acc);
                }
            }
        }
    }
}

void maxPool2d(const float* restrict x, float* restrict out, uint32_t* restrict locs, int N, int H, int W, int C,
    int Kh, int Kw) {
    const int OH = H / Kh;
    const int OW = W / Kw;

    memset(locs, 0, N * OH * OW * C * sizeof(uint32_t));

#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                const int ih0 = oh * Kh;
                const int iw0 = ow * Kw;

                const float* x0 = x + n * H * W * C;
                float* out0 = out + n * OH * OW * C;
                uint32_t* locs0 = locs + n * OH * OW * C;

                for (int c = 0; c < C; c += 8) {
                    // First element
                    __m256 maxv = _mm256_loadu_ps(x0 + (ih0 * W + iw0) * C + c);
                    __m256i idxv = _mm256_setzero_si256();

                    int idx = 1;

                    for (int kh = ih0; kh < ih0 + Kh; kh++) {
                        for (int kw = iw0 + (kh == ih0); kw < iw0 + Kw; kw++, idx++) {
                            __m256 p = _mm256_loadu_ps(x0 + (kh * W + kw) * C + c);
                            __m256 cmp = _mm256_cmp_ps(p, maxv, _CMP_GT_OQ);

                            maxv = _mm256_max_ps(maxv, p);
                            __m256i idxvp = _mm256_set1_epi32(idx);
                            idxv = _mm256_blendv_epi8(idxv, idxvp, _mm256_castps_si256(cmp));
                        }
                    }

                    _mm256_storeu_ps(out0 + (oh * OW + ow) * C + c, maxv);
                    _mm256_storeu_si256((__m256i*)(locs0 + (oh * OW + ow) * C + c), idxv);
                }
            }
        }
    }
}

void maxPool2dFwd(const float* restrict x, float* restrict out, int N, int H, int W, int C, int Kh, int Kw) {
    const int OH = H / Kh;
    const int OW = W / Kw;

#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                const int ih0 = oh * Kh;
                const int iw0 = ow * Kw;

                const float* x0 = x + n * H * W * C;
                float* out0 = out + n * OH * OW * C;

                for (int c = 0; c < C; c += 8) {
                    // First element
                    __m256 maxv = _mm256_loadu_ps(x0 + (ih0 * W + iw0) * C + c);

                    for (int kh = ih0; kh < ih0 + Kh; kh++) {
                        for (int kw = iw0 + (kh == ih0); kw < iw0 + Kw; kw++) {
                            __m256 p = _mm256_loadu_ps(x0 + (kh * W + kw) * C + c);
                            __m256 cmp = _mm256_cmp_ps(p, maxv, _CMP_GT_OQ);

                            maxv = _mm256_max_ps(maxv, p);
                        }
                    }

                    _mm256_storeu_ps(out0 + (oh * OW + ow) * C + c, maxv);
                }
            }
        }
    }
}

void maxPool2dBackward(const float* restrict dout, const uint32_t* restrict locs, float* restrict dx, int N, int H,
    int W, int C, int Kh, int Kw) {
    const int OH = H / Kh;
    const int OW = W / Kw;

    memset(dx, 0, N * H * W * C * sizeof(float));

#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                const int h0 = oh * Kh;
                const int w0 = ow * Kw;

                const int off = ((n * OH + oh) * OW + ow) * C;
                const float* dout0 = dout + off;
                const uint32_t* locs0 = locs + off;
                float* dx0 = dx + ((n * H + h0) * W + w0) * C;

                for (int c = 0; c < C; c++) {
                    const int x = locs0[c] % Kw;
                    const int y = locs0[c] / Kw;

                    // Non-overlapping windows -> can set instead of accumulate
                    dx0[(y * W + x) * C + c] = dout0[c];
                }
            }
        }
    }
}

float* packConvWeights(const float* w, int K, int Kh, int Kw, int C) {
    assert(K % 8 == 0 && "K must be a multiple of 8 for AVX packing");

    float* wp = malloc(K * Kh * Kw * C * sizeof(float));

    for (int k8 = 0; k8 < K / 8; k8++) {
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
                for (int c = 0; c < C; c++) {
                    // source: original [Kh,Kw,C,K] at (kh,kw,c,k8*8..k8*8+7)
                    const float* src = w + ((kh * Kw + kw) * C + c) * K + k8 * 8;

                    // dest: packed [K/8, Kh*Kw*C, 8] slot
                    float* dst = wp + (k8 * Kh * Kw * C + (kh * Kw + kw) * C + c) * 8;
                    memcpy(dst, src, 8 * sizeof(float));
                }
            }
        }
    }
    return wp;
}
