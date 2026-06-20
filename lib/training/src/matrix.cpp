#include "training/matrix.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "core/conv.h"

#ifndef NDEBUG
#include <iostream>
#endif

#include "core/gemm.h"
#include "core/ops.h"

#ifdef NDEBUG
#define CHECK_DIMS(desc, dim, expected)
#define CHECK_SIZE(other)
#else
void checkDims(const char* desc, int dim, int expected, const char* file, int line) {
    if (dim != expected) {
        std::cerr << "Incorrect " << desc << " dimensions:\n"
                  << "  Expected: " << expected << "\n"
                  << "  Actual  : " << dim << "\n"
                  << "  File    : " << file << "\n"
                  << "  Line    : " << line << std::endl;
        std::abort();
    }
}

#define CHECK_DIMS(desc, dim, expected) checkDims(desc, dim, expected, __FILE__, __LINE__)
#define CHECK_SIZE(other)                                                                                              \
    CHECK_DIMS("row", other.rows, rows);                                                                               \
    CHECK_DIMS("column", other.cols, cols)
#endif

void Matrix::copyTo(Matrix& other) const {
    CHECK_DIMS("size", other.rows * other.cols, rows * cols);
    std::memcpy(other.v.data(), v.data(), rows * cols * sizeof(float));
    other.rows = rows;
    other.cols = cols;
}

void Matrix::sliceRowsTo(Matrix& out, std::span<int> rowIndices) const {
    CHECK_DIMS("row", out.rows, rowIndices.size());
    CHECK_DIMS("column", out.cols, cols);
    for (int i = 0; i < rowIndices.size(); i++) {
        std::memcpy(out.v.data() + i * cols, v.data() + rowIndices[i] * cols, cols * sizeof(float));
    }
}

void Matrix::clear() {
    std::memset(v.data(), 0, rows * cols * sizeof(float));
}

void Matrix::gemm(const Matrix& other, Matrix& out, bool at, bool bt) const {
    int ma = at ? cols : rows;
    int na = at ? rows : cols;
    int mb = bt ? other.cols : other.rows;
    int nb = bt ? other.rows : other.cols;

    CHECK_DIMS("inner", mb, na);
    CHECK_DIMS("row", out.rows, ma);
    CHECK_DIMS("column", out.cols, nb);

    multiply(v.data(), other.v.data(), out.v.data(), ma, na, nb, at, bt, cols, other.cols);
}

Matrix& Matrix::operator-=(const Matrix& other) {
    CHECK_SIZE(other);
    sub(v.data(), other.v.data(), rows, cols);
    return *this;
}

void Matrix::rvAdd(const Matrix& other) {
    CHECK_DIMS("row", other.rows, 1);
    CHECK_DIMS("column", other.cols, cols);
    ::rvAdd(v.data(), other.v.data(), rows, cols);
}

void Matrix::sumColsTo(Matrix& other) const {
    CHECK_DIMS("row", other.rows, 1);
    CHECK_DIMS("column", other.cols, cols);
    other.clear();
    sumCols(v.data(), other.v.data(), rows, cols);
}

void Matrix::reluInPlace() {
    relu(v.data(), rows, cols);
}

void Matrix::reluBackwardInPlace(const Matrix& Z) {
    CHECK_SIZE(Z);
    reluBackward(v.data(), Z.v.data(), rows, cols);
}

void Matrix::softmaxInPlace() {
    softmax(v.data(), rows, cols);
}

void Matrix::conv2dTo(Matrix& out, const Matrix& w, const Matrix& b, int OH, int OW, int N, int H, int W, int C, int K,
    int Kh, int Kw) const {
    CHECK_DIMS("size", rows * cols, N * H * W * C);
    CHECK_DIMS("size", out.rows * out.cols, N * OH * OW * K);
    CHECK_DIMS("size", w.rows * w.cols, Kh * Kw * C * K);
    CHECK_DIMS("size", b.rows * b.cols, K);

    conv2d(v.data(), w.v.data(), b.v.data(), out.v.data(), OH, OW, N, H, W, C, K, Kh, Kw);
}

void Matrix::conv2dBackwardWTo(Matrix& dw, const Matrix& dout, int OH, int OW, int N, int H, int W, int C, int K,
    int Kh, int Kw) const {
    CHECK_DIMS("size", rows * cols, N * H * W * C);
    CHECK_DIMS("size", dw.rows * dw.cols, Kh * Kw * C * K);
    CHECK_DIMS("size", dout.rows * dout.cols, N * OH * OW * K);

    conv2dBackwardW(v.data(), dout.v.data(), dw.v.data(), OH, OW, N, H, W, C, K, Kh, Kw);
}

void Matrix::conv2dBackwardTo(Matrix& dx, const Matrix& w, int OH, int OW, int N, int H, int W, int C, int K, int Kh,
    int Kw) const {
    CHECK_DIMS("size", rows * cols, N * OH * OW * K);
    CHECK_DIMS("size", dx.rows * dx.cols, N * H * W * C);
    CHECK_DIMS("size", w.rows * w.cols, Kh * Kw * C * K);

    conv2dBackward(v.data(), w.v.data(), dx.v.data(), OH, OW, N, H, W, C, K, Kh, Kw);
}

void Matrix::maxPool2dTo(Matrix& out, std::vector<uint32_t>& locs, int N, int H, int W, int C, int Kh, int Kw) const {
    CHECK_DIMS("size", rows * cols, N * H * W * C);
    CHECK_DIMS("size", out.rows * out.cols, N * (H / Kh) * (W / Kw) * C);

    maxPool2d(v.data(), out.v.data(), locs.data(), N, H, W, C, Kh, Kw);
}

void Matrix::maxPool2dBackwardTo(Matrix& dx, const std::vector<uint32_t>& locs, int N, int H, int W, int C, int Kh,
    int Kw) const {
    CHECK_DIMS("size", rows * cols, N * (H / Kh) * (W / Kw) * C);
    CHECK_DIMS("size", dx.rows * dx.cols, N * H * W * C);

    maxPool2dBackward(v.data(), locs.data(), dx.v.data(), N, H, W, C, Kh, Kw);
}

float Matrix::nll(const Matrix& pred) const {
    CHECK_SIZE(pred);
    return ::nll(v.data(), pred.v.data(), rows, cols);
}

void Matrix::reshape(int newRows, int newCols) {
    int curShape = rows * cols;
    int newShape = newRows * newCols;
    CHECK_DIMS("size", newShape, curShape);

    rows = newRows;
    cols = newCols;
}

unsigned int Matrix::countSame(const Matrix& other) const {
    CHECK_SIZE(other);
    return ::countSame(v.data(), other.v.data(), rows, cols);
}

void Matrix::writeTo(std::ofstream& f) const {
    f.write(reinterpret_cast<const char*>(v.data()), rows * cols * sizeof(float));
}

void Matrix::writePackedWTo(std::ofstream& f, int K, int Kh, int Kw, int C) const {
    float* wp = packConvWeights(v.data(), K, Kh, Kw, C);
    f.write(reinterpret_cast<const char*>(wp), rows * cols * sizeof(float));
    std::free(wp);
}
