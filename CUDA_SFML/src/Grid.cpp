/*
Author: Tianyou Zhao 
Class: ECE6122 
Last Date Modified: 03-10-2024
Description:
This is the implementation of Grid class, which is used to store the status of the grid and some functions to generate the living status of the blocks.
*/
#include "Grid.hpp"
#include <random>

// constructor
Grid::Grid(int rows, int cols, Matrix *matrix){
    this->rows = rows;
    this->cols = cols;
    this->blocks = std::vector<std::vector<Block>> (rows, std::vector<Block>(cols)); // initialize the grid with blocks

    int row_idx = 0;
    int col_idx = 0;
    // set block position
    for (auto &row : this->blocks){
        col_idx = 0;
        for (auto &block : row){
            block.setPosition(cellSize * col_idx, cellSize * row_idx); // set the position of the block
            block.setAlive(matrix->elements[row_idx * cols + col_idx]); // set the living status of the block
            col_idx++;
        }
        row_idx++;
    }
};

// update the block status and set the color
// @input: grid - the grid of the game
void Grid::updateLivingStatus(Matrix *matrix){
    int row_idx = 0;    
    int col_idx = 0;
    for( auto &row : this -> blocks){
        col_idx = 0;
        for( auto &block : row){
            // update all blocks
            // false ^ false == false
            // true ^ true == false
            // false ^ true == true
            // true ^ false == true
            block.isAlive = matrix->elements[row_idx * cols + col_idx];
	    
            // set color
            if( block.isAlive){
                block.setFillColor(sf::Color::White);
            }else{
                block.setFillColor(sf::Color::Black);
            }
            col_idx++;
        }
        row_idx++;
    }
}

// show the grid
// @input: window - the window of the game
void Grid::showGrid(sf::RenderWindow &window){
    window.clear();
    for (auto &row : blocks){
        for (auto &block : row){
            window.draw(block);
        }
    }
   
    window.display();
}
