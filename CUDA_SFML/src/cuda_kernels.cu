/*
Author: Tianyou Zhao 
Class: ECE6122 
Last Date Modified: 03-11-2024
Description:
This is the implementation of cuda_kernels.cuh, which is used to implement the CUDA kernels.
*/
#include "cuda_kernels.cuh"
#include <iostream>
#include <stdio.h>
static cudaStream_t stream1, stream2; 
static bool streamsCreated = false;
static bool* device_A1 = nullptr;
static bool* device_B1 = nullptr;
static bool* device_A2 = nullptr;
static bool* device_B2 = nullptr;
static cudaEvent_t event1, event2;
// Kernel for normal memory mode
// input: A is the current frame, B is the next frame
// output: B is the next frame
// Normal mode for memory management
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

// Kernel for pinned memory mode and managed memory mode
// input: A is the current frame, B is the next frame
// output: B is the next frame
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

// Kernel for testing
__global__ void testKernel(bool *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = !data[idx];  // flip the value
    }
}

// launch the kernel
// input: A is the current frame, B is the next frame, d_A and d_B are the device memory for A and B, 
//        width and height are the dimensions of the matrix, processingType is the type of memory management
// output: B is the next frame
void launchMatMulKernel(Matrix* A, Matrix* B, bool* d_A, bool* d_B, int width, int height, std::string processingType) {
    // define the block size and grid size
    int blockDim = (int)sqrt(numThreads) + 1;
    dim3 blockSize(blockDim, blockDim);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);
    
    // launch the kernel
    if( processingType == "NORMAL" ){
        // copy data to device  
        cudaMemcpy(d_A, A->elements, width * height * sizeof(bool), cudaMemcpyHostToDevice);
        // launch the kernel
        matMulKernelNormal<<<gridSize, blockSize>>>(d_A, d_B, width, height);
        cudaDeviceSynchronize();
        // copy data to host
        cudaMemcpy(B->elements, d_B, width * height * sizeof(bool), cudaMemcpyDeviceToHost);
    }else if( processingType == "PINNED"){
        // calculate the actual size of each half
        int halfHeight = height / 2;
        int remainingHeight = height - halfHeight; // handle odd height
        
        // calculate the actual size of each part (in bytes)
        int size1 = halfHeight * width * sizeof(bool);
        int size2 = remainingHeight * width * sizeof(bool);

        if (!streamsCreated) {
            // create CUDA stream and event
            cudaStreamCreate(&stream1);
            cudaStreamCreate(&stream2);
            cudaEventCreate(&event1);
            cudaEventCreate(&event2);
            
            // allocate device memory for two streams
            cudaMalloc((void**)&device_A1, size1);
            cudaMalloc((void**)&device_B1, size1);
            cudaMalloc((void**)&device_A2, size2);
            cudaMalloc((void**)&device_B2, size2);
            
            // copy data to device
            cudaMemcpyAsync(device_A1, A->elements, size1, 
                          cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(device_A2, A->elements + (halfHeight * width), 
                          size2, cudaMemcpyHostToDevice, stream2);
            
            // wait for initial data transfer to complete
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
            
            streamsCreated = true;
        }
        // calculate the grid size
        dim3 gridSize1((width + blockSize.x - 1) / blockSize.x, 
                      (halfHeight + blockSize.y - 1) / blockSize.y);
        dim3 gridSize2((width + blockSize.x - 1) / blockSize.x, 
                      (remainingHeight + blockSize.y - 1) / blockSize.y);
        // launch the kernel in two streams
        matMulKernelNormal<<<gridSize1, blockSize, 0, stream1>>>(
            device_A1, device_B1, width, halfHeight);
        matMulKernelNormal<<<gridSize2, blockSize, 0, stream2>>>(
            device_A2, device_B2, width, remainingHeight);
        // record event
        cudaEventRecord(event1, stream1);
        cudaEventRecord(event2, stream2);
        // wait for kernel to complete
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        // swap pointers
        bool* temp1 = device_A1;
        device_A1 = device_B1;
        device_B1 = temp1;
        bool* temp2 = device_A2;
        device_A2 = device_B2;
        device_B2 = temp2;
        // copy data to host
        cudaMemcpyAsync(B->elements, device_B1, size1, 
                       cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(B->elements + (halfHeight * width), device_B2, 
                       size2, cudaMemcpyDeviceToHost, stream2);
        // use event to synchronize
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
    }else if( processingType == "MANAGED"){
        matMulKernel<<<gridSize, blockSize>>>(A, B, width, height);
        cudaDeviceSynchronize();
    }
}

// count the number of alive neighbors
// input: A is the current frame, row and col are the position of the cell
// output: the number of alive neighbors
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
