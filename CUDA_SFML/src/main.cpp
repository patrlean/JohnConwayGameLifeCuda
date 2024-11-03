// main.cpp
#include <SFML/Graphics.hpp>
#include "cuda_kernels.cuh"
#include <iostream>
#include <vector>
#include <random>

// time
#include <chrono>
#include <numeric>

#include "Globals.hpp"
#include "matrix.hpp"
#include "Grid.hpp"
#include "parameterParse.hpp"

int main(int argc, char *argv[]) {

    // parse arguments
    parseArgs(argc, argv);

    // check the number of CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using GPU: " << deviceProp.name << std::endl;


    // calculate the number of blocks in the window
    width = windowWidth / cellSize;
    height = windowHeight / cellSize;
    Matrix *A, *B;
    bool *d_A = nullptr, *d_B = nullptr;
    std::cout << "Processing type: " << processingType << std::endl;
    
    if( processingType == "MANAGED" ){
        // apply for memory
        cudaMallocManaged((void**)&A, sizeof(Matrix));
        cudaMallocManaged((void**)&B, sizeof(Matrix));
        int nBytes = width * height * sizeof(bool);
        cudaMallocManaged((void**)&A->elements, nBytes);
        cudaMallocManaged((void**)&B->elements, nBytes);
        // initialize matrix data
        A->width = width;
        A->height = height;
        B->width = width;
        B->height = height; 
        for (int i = 0; i < width * height; ++i) {
            A->elements[i] = rand() % 2;
            B->elements[i] = A->elements[i];
        }

    }else if( processingType == "PINNED" ){
        // apply for memory
        cudaMallocHost((void**)&A, sizeof(Matrix));
        cudaMallocHost((void**)&B, sizeof(Matrix));
        int nBytes = width * height * sizeof(bool);
        cudaMallocHost((void**)&A->elements, nBytes);
        cudaMallocHost((void**)&B->elements, nBytes);
        // initialize matrix data
        A->width = width;
        A->height = height;
        B->width = width;
        B->height = height; 
        for (int i = 0; i < width * height; ++i) {
            A->elements[i] = rand() % 2;
            B->elements[i] = A->elements[i];
        }
    }else if( processingType == "NORMAL" ){
        // normal mode
        A = new Matrix;
        B = new Matrix;
        A->elements = new bool[width * height];
        B->elements = new bool[width * height];

        // allocate device memory
        cudaMalloc((void**)&d_A, width * height * sizeof(bool));
        cudaMalloc((void**)&d_B, width * height * sizeof(bool));

        // initialize matrix data
        A->width = width;
        A->height = height;
        B->width = width;
        B->height = height; 
        for (int i = 0; i < width * height; ++i) {
            A->elements[i] = rand() % 2;
            B->elements[i] = A->elements[i];
        }

        cudaMemcpy(d_A, A->elements, width * height * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B->elements, width * height * sizeof(bool), cudaMemcpyHostToDevice);
    }else{
        std::cout << "Invalid processing type" << std::endl;
        return -1;
    }
    
    // Perform CUDA vector operation
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);

    // Set up SFML window
    sf::VideoMode vm(windowWidth, windowHeight);
    sf::RenderWindow window(vm, "CUDA + SFML");

    // Visualization of result 
    Grid grid(height, width, A);

    // timer
    std::vector<double> processingTimes;
    int generationCount = 0;
    
    while (window.isOpen()) {
        auto event = sf::Event{};
        while( window.pollEvent(event)){
            //close the window
            // if (event.type == sf::Event::KeyReleased){
                // close the window
                if ( sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                {
                    window.close();
                }
            // }
            if (event.type == sf::Event::Closed){
                window.close();
            }
        }
        auto startTime = std::chrono::high_resolution_clock::now();
        // swap A and B every loop
        launchMatMulKernel(A, B, d_A, d_B, width, height, processingType);
        // now matrix B is the next frame
        // synchronize
        
        auto midTime = std::chrono::high_resolution_clock::now();

        // update the living status of the blocks
        grid.updateLivingStatus(B);
        // show the grid
        grid.showGrid(window);
  
        auto startTime2 = std::chrono::high_resolution_clock::now();
        // swap A and B every loop
        launchMatMulKernel(B, A, d_A, d_B, width, height, processingType);
        // synchronize
        
        auto endTime = std::chrono::high_resolution_clock::now();

        grid.updateLivingStatus(A);
        // show the grid
        grid.showGrid(window);

        // calculate the total processing time
        auto processingTime = 
            std::chrono::duration_cast<std::chrono::microseconds>(midTime - startTime).count() +
            std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime2).count();
        
        processingTimes.push_back(processingTime);
        generationCount += 2;

        // output the statistics every 100 generations
        if (generationCount >= 100) {
            double averageTime = std::accumulate(processingTimes.begin(), 
                                               processingTimes.end(), 0.0) / processingTimes.size();
            
            int threadsPerBlock = numThreads;
            
            std::cout << "100 generations took " << averageTime 
                      << " microsecs with " << threadsPerBlock 
                      << " threads per block using " << processingType 
                      << " memory allocation." << std::endl;
            
            // reset the counter and the time record
            generationCount = 0;
            processingTimes.clear();
        }
    }
    // Cleanup
    if(processingType == "MANAGED") {
        // free device memory
        cudaFree(A->elements);
        cudaFree(B->elements);
        cudaFree(A);
        cudaFree(B);
    } else if(processingType == "PINNED") {
        cudaFreeHost(A->elements);
        cudaFreeHost(B->elements);
        cudaFreeHost(A);
        cudaFreeHost(B);
    } else {
        cudaFree(d_A);
        cudaFree(d_B);
        delete[] A->elements;
        delete[] B->elements;
        delete A;
        delete B;
    }
    return 0;
}
