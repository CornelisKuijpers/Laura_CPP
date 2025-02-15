cmake_minimum_required(VERSION 3.30)
project(Laura)

set(CMAKE_CXX_STANDARD 26)

add_executable(Laura main.cpp
        src/math/Matrix.cpp
        headers/math/Matrix.h
        src/NeuralLayer.cpp
        headers/NeuralLayer.h
        headers/neural_utils.cpp
        headers/neural_utils.h
        src/NumberPredictor.cpp
        headers/NumberPredictor.h)
