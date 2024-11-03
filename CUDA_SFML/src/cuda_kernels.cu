// cuda_kernels.cu
#include "cuda_kernels.cuh"
#include <iostream>
#include <stdio.h>


// Kernel for normal memory mode
__global__ void matMulKernelNormal(bool* A, bool* B, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * width + col;
    
    if (row < height && col < width) {
        int count = 0;
        // Count alive neighbors
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                if(i == 0 && j == 0) continue;
                int newRow = row + i;
                int newCol = col + j;
                if(newRow >= 0 && newRow < height && newCol >= 0 && newCol < width) {
                    if(A[newRow * width + newCol]) count++;
                }
            }
        }
        
        // Apply rules
        if(count == 3) {
            B[index] = true;
        } else if(count == 2 && A[index]) {
            B[index] = true;
        } else {
            B[index] = false;
        }
    }
}

__global__ void matMulKernel(Matrix* A, Matrix* B, int width, int height) {
    // get position of current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * width + col;
   
    // update matrix
    if (row < height && col < width) {
        int aliveNeighbors = countAliveMembers(A, row, col);
        // check rules and generate matrix after update
        if( aliveNeighbors == 3){
            B->elements[index] = true;
        }else if( aliveNeighbors == 2 && A->elements[index]){
            B->elements[index] = true;
        }else{
            B->elements[index] = false;
        }
    }
    // now matrix B is the next frame
}

__global__ void testKernel(bool *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = !data[idx];  // flip the value
    }
}

void launchMatMulKernel(Matrix* A, Matrix* B, bool* d_A, bool* d_B, int width, int height, std::string processingType) {
    // int blockDim = (int)sqrt(numThreads);
    int blockDim = 32;
    std::cout << "blockDim: " << blockDim << std::endl;
    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);

    // testKernel<<<gridSize, blockSize>>>(A->elements, width * height);
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cout << "Test kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    if( processingType == "NORMAL" ){
        // copy data to device  

        // A is the current frame
        cudaMemcpy(d_A, A->elements, width * height * sizeof(bool), cudaMemcpyHostToDevice);

        matMulKernelNormal<<<gridSize, blockSize>>>(d_A, d_B, width, height);

        cudaDeviceSynchronize();

        // copy data to host
        cudaMemcpy(B->elements, d_B, width * height * sizeof(bool), cudaMemcpyDeviceToHost);
        
    }else{
        // check the number of white cells in A
        int countA = 0;
        for(int i = 0; i < width * height; i++) {
            if(A->elements[i]) countA++;
        }
        std::cout << "Before kernel - white cells in A: " << countA << std::endl;
        
        // check the matrix dimensions
        std::cout << "Matrix dimensions - Width: " << A->width << ", Height: " << A->height << std::endl;
        std::cout << "Grid dimensions - x: " << gridSize.x << ", y: " << gridSize.y << std::endl;
        std::cout << "Block dimensions - x: " << blockSize.x << ", y: " << blockSize.y << std::endl;

        // check the CUDA error
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cout << "CUDA error before kernel: " << cudaGetErrorString(err) << std::endl;
        }

        matMulKernel<<<gridSize, blockSize>>>(A, B, width, height);

        // check the kernel execution error
        err = cudaGetLastError();
        if(err != cudaSuccess) {
            std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        }

        // check the number of white cells in B
        cudaDeviceSynchronize();
        int count = 0;
        for( int i = 0; i < width * height; i++){
            if( B->elements[i]){
                count++;
            }
        }
        std::cout << "current white count: " << count << std::endl;
    }
}

__device__ int countAliveMembers(Matrix *A, int row, int col) {
    int count = 0;
    // iterate all neighbors
    for( int i = -1; i <= 1; i++){
        for( int j = -1; j <= 1; j++){
            // skip itself    
            if( i == 0 && j == 0){
                continue;
            }
            // count the number of alive neighbors
            if(row + i >= 0 && row + i < A->height && col + j >= 0 && col + j < A->width){
                if(A->elements[(row + i) * A->width + col + j]){
                    count++;
                }
            }
        }
    }
    return count;
}
