/*
Author: Tianyou Zhao 
Class: ECE6122 
Last Date Modified: 03-10-2024
Description:
This is the implementation of Grid class, which is used to store the status of the grid and some functions to generate the living status of the blocks.
*/
#include "Grid.hpp"

// constructor
Grid::Grid(int rows, int cols){
    this->rows = rows;
    this->cols = cols;
    this->blocks = std::vector<std::vector<Block>> (rows, std::vector<Block>(cols)); // initialize the grid with blocks
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    int row_idx = 0;
    int col_idx = 0;
    // set block position
    for (auto &row : this->blocks){
        col_idx = 0;
        for (auto &block : row){
            block.setPosition(cellSize * col_idx, cellSize * row_idx); // set the position of the block
            block.setAlive(dis(gen)); // set the living status of the block
            col_idx++;
        }
        row_idx++;
    }
};

    

// update the living status of the block by the rule
// @input: aliveNeighbors - the number of alive neighbors
// @input: isAlive - the living status of the block
// @return the living status of the block after update
bool Grid::updateByRule(int aliveNeighbors, bool isAlive){
    if( aliveNeighbors == 3 && !isAlive){
        return true;
    }else if( (aliveNeighbors != 2 && aliveNeighbors != 3) && isAlive){
        return true;
    }else{
        return false;
    }
}

// generate the living status of the blocks for a range 
// use for parallel
// @input: startRow - the start row of the range
// @input: endRow - the end row of the range
void Grid::generateLivingStatusForRange(int startRow, int endRow) {
    int aliveNeighbors = 0;
    for (int row_idx = startRow; row_idx < endRow; row_idx++) {
        for (int col_idx = 0; col_idx < cols; col_idx++) {
            auto &block = blocks[row_idx][col_idx];
            // count the number of alive neighbors
            aliveNeighbors = countAliveNeighbors(row_idx, col_idx);
            // update the living status of the block
            block.isUpdated = updateByRule(aliveNeighbors, block.isAlive);
        }
    }
}


// update the block status and set the color
// @input: grid - the grid of the game
void Grid::updateLivingStatus(){
    for( auto &row : this -> blocks){
        for( auto &block : row){
            // update all blocks
            // false ^ false == false
            // true ^ true == false
            // false ^ true == true
            // true ^ false == true
            block.isAlive = block.isAlive ^ block.isUpdated;

            // set color
            if( block.isAlive){
                block.setFillColor(sf::Color::White);
            }else{
                block.setFillColor(sf::Color::Black);
            }

            // reset isUpdated
            block.isUpdated = false;
        }
    }
}

// count the number of alive neighbors
// @input: grid - the grid of the game
// @input: x - the x coordinate of the block
// @input: y - the y coordinate of the block
// @return the number of alive neighbors
int Grid::countAliveNeighbors(int x, int y){
    int count = 0;
    // count eight neighbors
    for( int i = -1; i <= 1; i++){
        for( int j = -1; j <= 1; j++){
            // skip itself    
            if( i == 0 && j == 0){
                continue;
            }

            // count the number of alive neighbors
            if(x + i >= 0 && x+i < rows && y+j >= 0 && y+j < cols){
                if(blocks[x+i][y+j].isAlive){
                    count++;
                }
            }
        }
    }
    return count;
}