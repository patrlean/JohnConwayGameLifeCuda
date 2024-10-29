/*
Author: Tianyou Zhao 
Class: ECE6122 
Last Date Modified: 03-10-2024
Description:
This is the header file of Block class, which is used to store the status of a block in the grid.
Some header files are included.
*/
#ifndef BLOCK_HPP
#define BLOCK_HPP

#include <SFML/Graphics.hpp>
#include "Globals.hpp"

class Block : public sf::RectangleShape {
    public:
        bool isAlive;
        bool isUpdated = false;
        // constructor  
        Block();   

        void setAlive(bool alive);
};
#endif // BLOCK_HPP