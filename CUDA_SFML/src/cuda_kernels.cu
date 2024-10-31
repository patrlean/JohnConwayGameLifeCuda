// cuda_kernels.cu
#include "cuda_kernels.cuh"
#include <iostream>
#include <stdio.h>

__global__ void matMulKernel(Matrix* A, Matrix* B, int width, int height) {
    // get position of current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * width + col;
    // printf(" thread x index is %d\n", threadIdx.x);
    // print element
    if (row == 400 && col == 200){
       printf("element is %d\n", A -> elements[index]);
       }
    
    // update matrix
    if (row < height && col < width) {
        int aliveNeighbors = countAliveMembers(A, row, col);
        // check rules and generate update matrix
        if( aliveNeighbors == 3 && !A->elements[index]){
            setElement(B, row, col, true);
        }else if( (aliveNeighbors != 2 && aliveNeighbors != 3) && A->elements[index]){
            setElement(B, row, col, false);
        }else{
            setElement(B, row, col, false);
        }
    }
}

void launchMatMulKernel(Matrix* A, Matrix* B, int width, int height) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);
    matMulKernel<<<gridSize, blockSize>>>(A, B, width, height)
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
