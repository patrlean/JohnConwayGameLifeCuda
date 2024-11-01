// main.cpp
#include <SFML/Graphics.hpp>
#include "cuda_kernels.cuh"
#include <iostream>
#include <vector>
#include <random>

#include "Globals.hpp"
#include "matrix.hpp"
#include "Grid.hpp"
int main() {
    // calculate the number of blocks in the window
    width = windowWidth / cellSize;
    height = windowHeight / cellSize;
    Matrix *A, *B;

    // apply for memory
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    int nBytes = width * height * sizeof(bool);
    cudaMallocManaged((void**)&A->elements, nBytes);
    cudaMallocManaged((void**)&B->elements, nBytes);

    // initialize matrix
    A->width = width;
    A->height = height;
    B->width = width;
    B->height = height;
    for (int i = 0; i < width * height; ++i) {
        A->elements[i] = rand() % 2;
        B->elements[i] = A->elements[i];
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
    
    while (window.isOpen()) {
        auto event = sf::Event{};
        while( window.pollEvent(event)){
            //close the window
            if (event.type == sf::Event::KeyReleased){
                // close the window
                if ( sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                {
                    window.close();
                }
            }
            if (event.type == sf::Event::Closed){
                window.close();
            }
        }
        // swap A and B every loop
        launchMatMulKernel(A, B, width, height);
        // now matrix B is the next frame

        // synchronize
        cudaDeviceSynchronize();
        
        // update the living status of the blocks
        grid.updateLivingStatus(B);

        // show the grid
        grid.showGrid(window);

        // swap A and B every loop
        launchMatMulKernel(B, A, width, height);

	// synchronize
        cudaDeviceSynchronize();

        grid.updateLivingStatus(A);

        // show the grid
        grid.showGrid(window);
    }

    return 0;
}
