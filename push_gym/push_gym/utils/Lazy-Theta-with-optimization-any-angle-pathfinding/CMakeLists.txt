cmake_minimum_required(VERSION 3.10)
project(LazyThetaStar)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# set the g++ compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For python binding
find_package(pybind11 REQUIRED)

# include header directories
include_directories(
   ${CMAKE_SOURCE_DIR}/include)

# find all the header files
file(GLOB HEADER_FILES_HPP include/*.hpp)
file(GLOB HEADER_FILES_H include/*.h)

# add the main file as executable
# test file

# main file
add_executable(main_single_path src/main_single_path.cpp ${HEADER_FILES_HPP})
add_executable(main_many_path src/main_many_path.cpp ${HEADER_FILES_HPP})

# Build c++ library such that python can call it
pybind11_add_module(LazyThetaStarPython src/LazyThetaStarPython.cpp)