#include "parameterParse.hpp"

using namespace std;

void parseArgs(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "-n" && i + 1 < argc) {
            numThreads = atoi(argv[++i]); // no less than 2
            if (numThreads < 2){
                cerr << "Error: Invalid number of threads.\n Number of threads must be greater than 1." << endl;
                exit(EXIT_FAILURE);  // exit
            }
        } else if (arg == "-c" && i + 1 < argc) {
            cellSize = atoi(argv[++i]);   // no less than 1
            if (cellSize < 1){
                cerr << "Error: Invalid cell size.\n Cell size must be greater than 0." << endl;
                exit(EXIT_FAILURE);  // exit
            }
        } else if (arg == "-x" && i + 1 < argc) {
            windowWidth = atoi(argv[++i]);
            if (windowWidth < 0){
                cerr << "Error: Invalid window width." << endl;
                exit(EXIT_FAILURE);  // exit
            }
        } else if (arg == "-y" && i + 1 < argc) {
            windowHeight = atoi(argv[++i]);
            if (windowHeight < 0){
                cerr << "Error: Invalid window height." << endl;
                exit(EXIT_FAILURE);  // exit
            }
        } else if (arg == "-t" && i + 1 < argc) {
            processingType = argv[++i];
            // check processingType valid?
            if (processingType != "NORMAL" && processingType != "PINNED" && processingType != "MANAGED") {
                cerr << "Error: Invalid processing type. Must be one of NORMAL, PINNED, MANAGED." << endl;
                exit(EXIT_FAILURE);  // exit
            }
        } else {
            cerr << "Usage: " << argv[0] << " [-n numThreads] [-c cellSize] [-x windowWidth] [-y windowHeight] [-t typeOfMemory]" << endl;
            exit(EXIT_FAILURE);
        }
    }
}