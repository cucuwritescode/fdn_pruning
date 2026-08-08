cmake_minimum_required(VERSION 3.16)
project(staged_fdn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#release by default, real-time code must be optimised
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#aggressive optimisation for the target architecture
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")

# --- test runner ---
add_executable(test_staged test_staged.cpp)
target_include_directories(test_staged PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# --- benchmark ---
add_executable(benchmark benchmark.cpp)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

#link Apple Accelerate (BLAS) if available
if(APPLE)
    find_library(ACCELERATE_LIB Accelerate)
    if(ACCELERATE_LIB)
        target_link_libraries(benchmark ${ACCELERATE_LIB})
    endif()
endif()
