cmake_minimum_required(VERSION 3.10.2)

project(BallLarusFuzzing
  LANGUAGES CXX C
  DESCRIPTION "Ball-Larus path coverage fuzzing metric"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#
# LLVM
#

find_package(LLVM REQUIRED)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)
set(LLVM_LINK_LLVM_DYLIB OFF)

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif(NOT LLVM_ENABLE_RTTI)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(BEFORE
  include
  ext/aflplusplus/include
)
link_directories(${LLVM_LIBRARY_DIRS})

add_subdirectory(lib)
add_subdirectory(tools)
