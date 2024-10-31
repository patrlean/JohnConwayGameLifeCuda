// cuda_kernels.cuh
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

void matMulKernel(const Matrix* A, const Matrix* B, int width, int height);
#endif
