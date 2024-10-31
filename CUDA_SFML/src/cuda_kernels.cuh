// cuda_kernels.cuh
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "matrix.hpp"
#include <cuda_runtime.h>

void matMulKernel(const Matrix* A, const Matrix* B, int width, int height);

#endif
