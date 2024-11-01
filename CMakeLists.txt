
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(tiny_bvh_test LANGUAGES CXX)

enable_testing()

## Build test app ##

add_executable(${PROJECT_NAME} tiny_bvh_test.cpp)
if (NOT MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE -march=native)
endif()

add_test(NAME "${PROJECT_NAME}" COMMAND ${PROJECT_NAME})
