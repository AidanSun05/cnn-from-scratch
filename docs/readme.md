# CNN from Scratch

This project demonstrates creating an optimized Convolutional Neural Network (CNN) with computational kernels written in C and training facilities written in C++.

[Benchmarks](benchmarks.md) | [Examples](examples.md)

## Features

- Dependency-free implementation of the core neural network
- General Matrix Multiply (GEMM) and 2D convolution kernels with AVX2 SIMD and OpenMP parallelism
- Fused transpose-multiply operations to efficiently compute matrix products such as $\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}$
- Allocation-free training and inference loops
- Polymorphic layer and optimizer classes for fast model construction in code
- Backpropagation with stochastic gradient descent (SGD) and pluggable optimizers: "plain" SGD and SGD with the Adam optimizer are implemented
- Independent inference path in C, separate from training loop and using optimized computational routines
  - Convolution with weight packing for improved cache locality
  - O(1) buffer allocations in the number of model layers
  - Flatten layers become no-ops
  - Batched inference allows prediction on multiple inputs in one call to `nnPredict`, exploiting parallelism

## Repository structure

- `core/`: Computation functions (GEMM, conv, max pool, etc.) in C.
  - `core/include/core/internal/gemmconf.h`: Compile-time configuration parameters for the GEMM implementation based on hardware specifications.
- `examples/`: Training setups on various datasets, and applications that demonstrate the model functioning in concrete use cases.
- `lib/`: Higher-level APIs for interfacing with models.
  - `format/`: Specifies the file format that models are saved to.
  - `inference/`: C facilities for running forward passes on models that have already been trained. Consumed by applications that only run inference.
  - `training/`: C++ facilities (layers and optimizers) for training and saving models.
- `test/`: Unit tests and benchmarks.
