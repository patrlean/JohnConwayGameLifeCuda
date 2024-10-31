/*
Author: Tianyou Zhao 
Class: ECE6122 
Last Date Modified: 03-10-2024
Description:
This is the implementation of Globals.hpp, which is used to store the global variables.
*/

// src/Constants.cpp
#include "Globals.hpp"

// set scene
int width;
int height;

// set window
int windowWidth = 800;
int windowHeight = 600;

// set block
int cellSize = 5;

// set thread
int numThreads = 32; // must be a multiple of 32
std::string processingType = "NORMAL"; // "NORMAL" or "PINNED" or "MANAGED"

 