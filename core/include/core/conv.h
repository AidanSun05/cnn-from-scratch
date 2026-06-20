#pragma once

#include <stdint.h>

#include "internal/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parameters:
//   - OH, OW: Output spatial dimensions
//   - N, H, W, C: Batch size, input height, input width, input channels
//   - K: Output channels (number of filters)
//   - Kh, Kw: Kernel width, kernel height

// Performs a 2D convolution forward pass over a batch of inputs.
//
// Computes the convolution of input feature maps x with filter weights w, adds bias b, and writes the result to out.
// Padding and stride are applied independently in height and width.
//
// Input:
//   - x: Input tensor of shape [N*H*W][C]
//   - w: Filter tensor of shape [Kh*Kw*C][K]
//   - b: Bias vector of shape [K]
//
// Output:
//   - out: Output tensor of shape [N*OH*OW][K]
//
// Preconditions:
//   - K % 8 == 0
void conv2d(const float* RESTRICT x, const float* RESTRICT w, const float* RESTRICT b, float* RESTRICT out, int OH,
    int OW, int N, int H, int W, int C, int K, int Kh, int Kw);

void conv2dPacked(const float* RESTRICT x, const float* RESTRICT w, const float* RESTRICT b, float* RESTRICT out, int OH,
    int OW, int N, int H, int W, int C, int K, int Kh, int Kw);

// Computes the gradient of the convolution weights during backpropagation.
//
// Accumulates gradients into dw using the input tensor x and upstream gradients dout.
//
// Input:
//   - x: Input tensor of shape [N*H*W][C]
//   - dout: Upstream gradient tensor of shape [N*OH*OW][K]
//
// Output:
//   - dw: Weight gradient tensor of shape [Kh*Kw*C][K]
//
// Preconditions:
//   - K % 8 == 0
void conv2dBackwardW(const float* RESTRICT x, const float* RESTRICT dout, float* RESTRICT dw, int OH, int OW, int N,
    int H, int W, int C, int K, int Kh, int Kw);

// Computes the gradient of the input tensor during backpropagation.
//
// Propagates gradients from dout through the convolution operation using the filter weights w.
//
// Input:
//   - dout: Upstream gradient tensor of shape [N*OH*OW][K]
//   - w: Filter tensor of shape [Kh*Kw*C][K]
//
// Output:
//   - dx: Gradient tensor with respect to the input, shape [N*H*W][C]
//
// Preconditions:
//   - K % 8 == 0
void conv2dBackward(const float* RESTRICT dout, const float* RESTRICT w, float* RESTRICT dx, int OH, int OW, int N,
    int H, int W, int C, int K, int Kh, int Kw);

// Performs 2D max pooling over spatial dimensions.
//
// For each pooling window, selects the maximum value and records its location for use in backpropagation.
//
// Input:
//   - x: Input tensor of shape [N*H*W][C]
//
// Output:
//   - out: Pooled output tensor of shape [N*OH*OW][C]
//   - locs: Indices of maximal values, shape [N*OH*OW*C]
//
// Preconditions:
//   - C % 8 == 0
void maxPool2d(const float* RESTRICT x, float* RESTRICT out, uint32_t* RESTRICT locs, int N, int H, int W, int C,
    int Kh, int Kw);

void maxPool2dFwd(const float* RESTRICT x, float* RESTRICT out, int N, int H, int W, int C, int Kh, int Kw);

// Computes the gradient of the input tensor for max pooling.
//
// Routes gradients from dout back to the input positions recorded in locs.
//
// Input:
//   - dout: Upstream gradient tensor of shape [N*OH*OW][C]
//   - locs: locs: Max index locations from the forward pass, shape [N*OH*OW*C]
//
// Output:
//   - dx: Gradient with respect to the input, shape [N*H*W][C]
//
// Preconditions:
//   - C % 8 == 0
//   - H % Kh == 0
//   - W % Kw == 0
void maxPool2dBackward(const float* RESTRICT dout, const uint32_t* RESTRICT locs, float* RESTRICT dx, int N, int H,
    int W, int C, int Kh, int Kw);

// Converts the weight layout [Kh, Kw, C, K] used during training into [K/8, Kh*Kw*C, 8].
//
// This lets the forward pass during inference have better cache use.
//
// Preconditions:
//   - K % 8 == 0
float* packConvWeights(const float* w, int K, int Kh, int Kw, int C);

#ifdef __cplusplus
}
#endif
