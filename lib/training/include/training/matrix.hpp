#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <vector>

class Matrix {
    std::vector<float> v;
    int rows = 0;
    int cols = 0;

public:
    Matrix() = default;

    Matrix(int rows, int cols) : v(rows * cols, 0), rows(rows), cols(cols) {}

    std::array<int, 2> shape() const {
        return { rows, cols };
    }

    std::span<float> data() {
        return { v.data(), static_cast<std::size_t>(rows * cols) };
    }

    std::span<const float> data() const {
        return { v.data(), static_cast<std::size_t>(rows * cols) };
    }

    void copyTo(Matrix& other) const;

    void sliceRowsTo(Matrix& out, std::span<int> rowIndices) const;

    void clear();

    float operator()(int row, int col) const {
        return v[row * cols + col];
    }

    float& operator()(int row, int col) {
        return v[row * cols + col];
    }

    void gemm(const Matrix& other, Matrix& out, bool at, bool bt) const;

    Matrix& operator-=(const Matrix& other);

    void rvAdd(const Matrix& other);

    void sumColsTo(Matrix& other) const;

    void reluInPlace();

    void reluBackwardInPlace(const Matrix& Z);

    void softmaxInPlace();

    void conv2dTo(Matrix& out, const Matrix& w, const Matrix& b, int OH, int OW, int N, int H, int W, int C, int K,
        int Kh, int Kw) const;

    void conv2dBackwardWTo(Matrix& dw, const Matrix& dout, int OH, int OW, int N, int H, int W, int C, int K, int Kh,
        int Kw) const;

    void conv2dBackwardTo(Matrix& dx, const Matrix& w, int OH, int OW, int N, int H, int W, int C, int K, int Kh,
        int Kw) const;

    void maxPool2dTo(Matrix& out, std::vector<uint32_t>& locs, int N, int H, int W, int C, int Kh, int Kw) const;

    void maxPool2dBackwardTo(Matrix& dx, const std::vector<uint32_t>& locs, int N, int H, int W, int C, int Kh,
        int Kw) const;

    float nll(const Matrix& pred) const;

    unsigned int countSame(const Matrix& other) const;

    void reshape(int newRows, int newCols);

    void writeTo(std::ofstream& f) const;

    void writePackedWTo(std::ofstream& f, int K, int Kh, int Kw, int C) const;
};
