# ----------------------------------------------------------------------
# Print Build Options
# ----------------------------------------------------------------------

message(STATUS "------------------ EML ------------------------")
message(STATUS "Building with platform:     ${EML_PLATFORM}")
if (${EML_PLATFORM} STREQUAL "custom")
    message(WARNING "   Building with a custom platform. Make sure to compile with your platform_xxx.cpp ")
endif ()
message(STATUS "Compiling for architecture: ${EML_ARCH}")
message(STATUS "Using SIMD level:           ${EML_SIMD_LEVEL}")

# ----------------------------------------------------------------------
# Add Library
# ----------------------------------------------------------------------

# Get all source files
file(GLOB_RECURSE SRC_FILES "src/*.cpp")

# Add platform header
list(APPEND SRC_FILES "${CMAKE_SOURCE_DIR}/platforms/platform_${EML_PLATFORM}.cpp")

# Create the library
add_library(eml STATIC ${SRC_FILES})

# Include directories from root for clean #includes
target_include_directories(eml PRIVATE include)
target_include_directories(eml PRIVATE src)

# Adds correct definitions for the runtime
include(cmake/AddDefines.cmake)

message(STATUS "------------------------------------------------")