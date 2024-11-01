# ----------------------------------------------------------------------
# CMake Configuration
# ----------------------------------------------------------------------

cmake_minimum_required(VERSION 3.21)
project(eml
        VERSION 0.0.1
        DESCRIPTION "Embeddable machine learning"
        HOMEPAGE_URL "https://github.com/gk646/EML"
        LANGUAGES CXX
)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17) # most recent C++ standard supported by IAR

# ----------------------------------------------------------------------
# Configuration Options
# ----------------------------------------------------------------------
# NOTE: WHEN USING CMAKE CACHE, CHANGES NEED A CACHE RESET TO APPLY!

# Choose which platform file to compile with
# Currently supported:
# - windows (default)
# - unix
# - raspberry_pico
# - unix
# - custom (supply your own)
set(EML_PLATFORM "windows" CACHE STRING "The underlying platform that handles stdout, memory, etc...")

# Choose which architecture to compile for
# This conditionally enables intrinsics and other specific code for this architecture
# Currently supported:
# - amd64 (default)
# - risc-v
# - arm
set(EML_ARCH "amd64" CACHE STRING "The processing architecture")

# Choose the SIMD level to compile with
# Currently supported:
# - neon (for arm)
# - rvv (for risc-v)
# - auto (default)
# - none
set(EML_SIMD_LEVEL "auto" CACHE STRING "The SIMD level to optimize with")


# Choose to specify the number of cores (multicore)
# Currently supported:
# - 1-4
set(EML_CORES 1 CACHE STRING "The amount of cores to optimize for")

# Choose to build the test executable
# Executable can be run on the target platform or on the host
# If building with GCC or clang builds a coverage report in /testbench/artifacts
option(EML_TEST "Enable building the tests" OFF)

# ----------------------------------------------------------------------
# Build Library
# ----------------------------------------------------------------------

# Builds the library
include(cmake/BuildLibrary.cmake)

# Adds tests directory if opened in development mode
if (${PROJECT_IS_TOP_LEVEL} OR EML_BUILD_TESTS)
    add_subdirectory(testbench)
endif ()