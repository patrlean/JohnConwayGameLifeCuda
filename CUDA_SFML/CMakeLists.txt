# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the required CUDA version (optional)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 80)  # or specify your target architecture here

# Project name
project(game_cuda LANGUAGES CXX CUDA)

# Include SFML
include_directories(${PROJECT_SOURCE_DIR}/../SFML/include)
link_directories(${PROJECT_SOURCE_DIR}/../SFML/lib)

# Include CUDA
# find_package(CUDA REQUIRED)

# Include CUDA headers and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})


# Include hpp
include_directories(include)

# Set CUDA architecture
# set(CUDA_ARCHITECTURES 80)

# Specify the target executable
add_executable(game_cuda 
    src/main.cpp 
    src/cuda_kernels.cu
    # src/cuda_kernels.cuh
    src/parameterParse.cpp 
    src/Globals.cpp
    src/Grid.cpp
    src/Block.cpp

    # include/parameterParse.hpp
    # include/Globals.hpp
    # include/Grid.hpp
    # include/Block.hpp
    # include/matrix.hpp
    )

# Link SFML libraries
target_link_libraries(game_cuda sfml-graphics sfml-window sfml-system ${CUDA_LIBRARIES})

# Specify C++11 standard
#set_property(TARGET cuda_sfml_app PROPERTY CXX_STANDARD 17)

set_target_properties(
    game_cuda PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${COMMON_OUTPUT_DIR}/bin"
)
# Compile Info
set_target_properties(game_cuda PROPERTIES CUDA_ARCHITECTURES "80")
target_compile_features(game_cuda PUBLIC cxx_std_11)
target_compile_features(game_cuda PUBLIC cuda_std_11)
