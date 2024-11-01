// cuda_kernels.cuh
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "matrix.hpp"
#include <cuda_runtime.h>

__global__ void matMulKernel(Matrix* A, Matrix* B, int width, int height);

// For normal memory
__global__ void matMulKernelNormal(bool* A, bool* B, int width, int height);

__device__ void setElement(Matrix *A, int row, int col, bool value);

__device__ int countAliveMembers(Matrix *A, int row, int col);

void launchMatMulKernel(Matrix* A, Matrix* B, int width, int height, std::string processingType);

#endif
