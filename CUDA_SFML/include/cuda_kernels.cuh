/*
Author: Tianyou Zhao 
Class: ECE6122 
Last Date Modified: 03-11-2024
Description:
This is CUDA_KERNELS.cuh, which is used to implement the CUDA kernels.
*/
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "matrix.hpp"
#include <cuda_runtime.h>
#include <string>
#include "Globals.hpp"

__global__ void matMulKernel(Matrix* A, Matrix* B, int width, int height);

// For normal memory
__global__ void matMulKernelNormal(bool* A, bool* B, int width, int height);

__global__ void testKernel(bool *data, int size);

__device__ void setElement(Matrix *A, int row, int col, bool value);

__device__ int countAliveMembers(Matrix *A, int row, int col);

void launchMatMulKernel(Matrix* A, Matrix* B, bool* d_A, bool* d_B, int width, int height, std::string processingType);

cudaStream_t stream1, stream2; 

#endif
