cmake_minimum_required(VERSION 3.16)
project(OBRMMain LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(obrm_export src/export_main.cpp src/vectors.cpp src/learned_ann.cpp src/mlp.cpp)
target_include_directories(obrm_export PRIVATE src ../OBRMSGX/Common)
add_executable(obrm_groundtruth src/exact_groundtruth.cpp src/vectors.cpp)
target_include_directories(obrm_groundtruth PRIVATE src ../OBRMSGX/Common)
