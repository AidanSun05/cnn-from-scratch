# Benchmarks

This document contains benchmark results for the GEMM and convolutional kernels.

## Setup used

### Hardware

- CPU: AMD Ryzen 7 5700U
- Cores / threads: 8 cores, 16 threads
- Max clock: 4.37 GHz
- Cache sizes:
  - L1d: 32 KiB/core (256 KiB total)
  - L2: 512 KiB/core (4 MiB total)
  - L3: 8 MiB
- Memory: 64 GB DDR4, 2667 MT/s

### Software

- OS: Ubuntu 24.04
- Kernel: 6.8.0-100-generic
- Compiler: GCC 13.3.0
- Compiler flags: `-O3 -mavx2 -mfma -ffast-math -fno-exceptions -fno-plt -funroll-loops -fno-math-errno -fno-trapping-math -march=native -flto`
  - These flags will be set if using the `xmake` setup in the repository and configuring release builds, e.g., `xmake f -p linux -a x86_64 -m release -c`.

### Benchmark details

- Matrix data type: `float` (FP32)
- Threads: 8
  - These benchmarks don't target SMT (which would use up to 16 threads on this CPU instead) to prevent contention when hyperthreads on the same core try to share the same SIMD registers.
- System setup:
  ```shell
  # Run at highest possible clock speeds
  sudo cpupower frequency-set -g performance

  # OpenMP thread affinity: pin threads to physical cores and keep them close to the primary thread
  export OMP_PLACES=cores
  export OMP_PROC_BIND=close

  # Necessary for viewing detailed perf outputs
  sudo sysctl -w kernel.perf_event_paranoid=1
  sudo sysctl -w kernel.kptr_restrict=0
  ```
- `gemmconf.h`:
  ```c
  #define MR 8
  #define NR 8

  #define MC 256
  #define KC 256
  #define NC 512

  #define CACHE_LINE_SIZE 64
  #define NUM_THREADS 8
  ```

## GEMM panel size tuning

The `MC`, `KC`, and `NC` constants in `gemmconf.h` determine the panel sizes that the input matrices are packed into during multiplication. `MR=8` and `NR=8` determine the microkernel size, which is set by the AVX2 register layout. The tuning process is shown below, using the setup above as an example.

In each microkernel call, the working set is `MR` values of A and `NR` values of B, repeated up to `KC` times. With one thread per core, this will need to fit in the per-core L1 cache. Half of L1 will be reserved for other processes.

```math
KC\cdot (MR+NR)\cdot 4\:\text{bytes}\le \frac{1}{2}\:\text{L1d} \implies KC\cdot 16\cdot 4\le 16384 \implies KC\approx 256
```

The packed `ap` block, of size $MC\times KC$, is reread after every `jr` loop, so it will need to stay in per-core L2.

```math
MC\cdot KC\cdot 4\:\text{bytes}\le \frac{1}{2}\:\text{L2} \implies MC\cdot 256\cdot 4\le 262144 \implies MC\approx 256
```

The packed `bp` block, of size $KC\times NC$, is reread after every `ic` loop, so it will need to stay in L3. The 8 MiB L3 is shared across all 8 physical cores.

```math
KC\cdot NC\cdot 4\:\text{bytes}\le \frac{1}{2}\frac{8\:\text{MiB L3}}{8\:\text{threads}} \implies NC\cdot 256\cdot 4\le 524288\implies NC\approx 512
```

This procedure provides a general estimate for the panel sizes. Further tuning and benchmarks could provide the optimal panel sizes for a given problem size.

## Matrix multiplication: performance

`xmake` targets:

- `benchmark_multiply`
- `benchmark_multiply_openblas`

Source files:

- `/test/src/bench_multiply.cpp`
- `/test/src/bench_multiply_openblas.cpp`

This measures how matrix multiplication performance scales with increasing dimensions. 5 benchmarks are run, and each benchmark multiplies 2 square matrices of one of the following sizes:

- $64\times 64$
- $128\times 128$
- $256\times 256$
- $512\times 512$
- $1024\times 1024$

`benchmark_multiply` evaluates the GEMM implementation in this repository, while `benchmark_multiply_openblas` evaluates OpenBLAS under the same conditions.

`benchmark_multiply` output:

|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | multiply
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:---------
|           13,391.16 |           74,676.12 |    0.2% |      161,506.63 |       53,227.73 |  3.034 |      24,776.44 |    0.2% |      2.43 | `64x64`
|           79,814.23 |           12,529.09 |    1.2% |    1,089,674.84 |      281,371.25 |  3.873 |     168,016.55 |    0.0% |      2.47 | `128x128`
|          531,200.54 |            1,882.53 |    0.1% |    7,954,522.56 |    1,807,295.17 |  4.401 |   1,228,256.78 |    0.0% |      2.28 | `256x256`
|        4,092,102.50 |              244.37 |    0.2% |   61,005,463.25 |   13,856,759.16 |  4.403 |   9,442,889.36 |    0.1% |      2.41 | `512x512`
|       17,699,047.92 |               56.50 |    0.3% |  241,699,261.08 |   56,458,481.55 |  4.281 |  37,508,559.85 |    0.0% |      2.43 | `1024x1024`

`benchmark_multiply_openblas` output:

|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | multiply
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:---------
|            7,350.68 |          136,041.79 |    0.5% |       92,274.01 |       30,722.19 |  3.003 |       2,016.26 |    0.1% |      2.43 | `64x64`
|           44,837.05 |           22,302.99 |    0.1% |      619,436.05 |      188,034.65 |  3.294 |       8,408.30 |    0.2% |      2.32 | `128x128`
|          314,679.28 |            3,177.84 |    0.1% |    4,486,446.33 |    1,307,853.57 |  3.430 |      41,712.57 |    0.1% |      2.42 | `256x256`
|        2,585,916.20 |              386.71 |    0.2% |   34,716,182.68 |   10,142,998.30 |  3.423 |     297,262.84 |    0.1% |      2.45 | `512x512`
|       20,529,629.82 |               48.71 |    0.3% |  272,624,175.82 |   78,473,802.17 |  3.474 |   2,066,283.27 |    0.1% |      2.42 | `1024x1024`

Performance in GFLOPS is shown below for each implementation, using $\text{FLOPs}=2N^3$.

| Size | OpenBLAS | Custom GEMM |
| --- | ---: | ---: |
| `64x64` | 71.3 | 39.15 |
| `128x128` | 93.54 | 52.55 |
| `256x256` | 106.63 | 63.16 |
| `512x512` | 103.8 | 65.59 |
| `1024x1024` | 104.6 | 121.33 |

## Matrix multiplication: cache

`xmake` target: `cache_test_multiply`

Source file: `/test/src/cache_test_multiply.cpp`

This test, intended to be evaluated with `perf`, multiplies 2 $512\times 512$ matrices once.

```shell
$ perf stat -e L1-dcache-loads,L1-dcache-misses,branches,branch-misses,cycles ./cache_test_multiply

 Performance counter stats for './cache_test_multiply':

        30,831,679      L1-dcache-loads
         1,641,964      L1-dcache-misses                 #    5.33% of all L1-dcache accesses
         9,660,809      branches
           258,434      branch-misses                    #    2.68% of all branches
       157,555,487      cycles

       0.011026158 seconds time elapsed

       0.038778000 seconds user
       0.005816000 seconds sys
```

This `perf stat` output shows a low L1d miss rate and a low branch misprediction rate.

## Convolution: cache

`xmake` targets:

- `cache_test_conv`
- `cache_test_conv_packed`

Source file: `/test/src/cache_test_conv.cpp`

This test, intended to be evaluated with `perf`, applies convolution to a single $1024\times 1024\times 3$ image (i.e., $1024\times 1024$ image dimensions with 3 input channels) using 128 $3\times 3$ kernels. `cache_test_conv` uses the `conv2d` function that is used during model training. `cache_test_conv_packed` uses `conv2dPacked`, an optimized convolution routine that uses a different weight buffer memory layout and is used in the C model inference API.

These tests do not set the OpenMP thread count, so `export OMP_NUM_THREADS=N` must be used prior to running the tests.

```shell
$ export OMP_NUM_THREADS=8
$ perf stat -e L1-dcache-loads,L1-dcache-misses,branches,branch-misses,cycles ./cache_test_conv

 Performance counter stats for './cache_test_conv':

    11,443,573,399      L1-dcache-loads
       703,833,942      L1-dcache-misses                 #    6.15% of all L1-dcache accesses
     5,257,708,656      branches
        26,324,364      branch-misses                    #    0.50% of all branches
    10,015,308,605      cycles

       0.752785243 seconds time elapsed

       3.064912000 seconds user
       0.302089000 seconds sys


$ perf stat -e L1-dcache-loads,L1-dcache-misses,branches,branch-misses,cycles ./cache_test_conv_packed

 Performance counter stats for './cache_test_conv_packed':

     7,720,573,468      L1-dcache-loads
       274,394,690      L1-dcache-misses                 #    3.55% of all L1-dcache accesses
     1,708,289,612      branches
        24,343,247      branch-misses                    #    1.43% of all branches
     5,730,978,261      cycles

       0.571883981 seconds time elapsed

       1.615463000 seconds user
       0.296452000 seconds sys
```

Under these test conditions, `conv2dPacked` exhibits a lower L1d miss rate and also performs fewer loads to the L1d cache overall.
