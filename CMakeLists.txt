cmake_minimum_required(VERSION 3.8)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp NeuralNetwork.cpp NeuralNetwork.h)
add_executable(NeuralNetwork ${SOURCE_FILES})