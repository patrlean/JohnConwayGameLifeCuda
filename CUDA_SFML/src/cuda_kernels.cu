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
            setElement(B, row, col, true);
        }else if( aliveNeighbors == 2 && A->elements[index]){
            setElement(B, row, col, true);
        }else{
            setElement(B, row, col, false);
        }
    }
    // now matrix B is the next frame
}

void launchMatMulKernel(Matrix* A, Matrix* B, bool* d_A, bool* d_B, int width, int height, std::string processingType) {
    int blockDim = (int)sqrt(numThreads);
    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);

    if( processingType == "NORMAL" ){
        // copy data to device  

        // A is the current frame
        cudaMemcpy(d_A, A->elements, width * height * sizeof(bool), cudaMemcpyHostToDevice);

        matMulKernelNormal<<<gridSize, blockSize>>>(d_A, d_B, width, height);

        // copy data to host
        cudaMemcpy(B->elements, d_B, width * height * sizeof(bool), cudaMemcpyDeviceToHost);

        // free device memory
        cudaFree(d_A);
        cudaFree(d_B);
    }else{
        matMulKernel<<<gridSize, blockSize>>>(A, B, width, height);
    }
}


__device__ void setElement(Matrix *A, int row, int col, bool value) {
    A->elements[row * A->width + col] = value;
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
