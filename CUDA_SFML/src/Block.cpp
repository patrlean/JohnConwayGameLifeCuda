/*
Author: Tianyou Zhao 
Class: ECE6122 
Last Date Modified: 03-10-2024
Description:
This is the implementation of Block class, which is used to store the status of a block in the grid.
*/

#include "Block.hpp"
#include <vector>
#include <random>


// constructor
Block::Block(){
    // set the block color
    isAlive = false;
    isUpdated = false;
    this->setSize(sf::Vector2f(cellSize, cellSize));
}

void Block::setAlive(bool alive){
    this->isAlive = alive;
    // set the block color
    if( alive){
        this -> setFillColor(sf::Color::White);
    }else{
        this -> setFillColor(sf::Color::Black);
    }
}