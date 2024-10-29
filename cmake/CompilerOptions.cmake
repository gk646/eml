# ----------------------------------------------------------------------
# Compiler Options
# ----------------------------------------------------------------------

# Ensure the project name is defined
if (NOT TARGET ${TARGET_NAME})
    message(FATAL_ERROR "Target ${TARGET_NAME} is not defined")
endif ()

# Check for the supported compilers and architectures
if (MSVC)
    # Verify the architecture
    if (NOT "${EML_ARCH}" STREQUAL "amd64")
        message(FATAL_ERROR "Chosen platform is not supported by the current compiler")
    endif ()

    # Set compile options using generator expressions for different configurations
    target_compile_options(${TARGET_NAME} PRIVATE
            $<$<CONFIG:Debug>:/Od /Zi /RTC1>
            $<$<CONFIG:Release>:/O3 /DNDEBUG /Ob3 /W4 /EHsc /GA /fp:fast /GS- /Gy /Oi /Gw /GF /GL /GR- /Oi>
    )

    # Set link options
    target_link_options(${TARGET_NAME} PRIVATE
            /LTCG /OPT:REF /OPT:ICF
    )

elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Verify the architecture
    if ("${EML_ARCH}" STREQUAL "amd64")
        # Set compile options using generator expressions for different configurations
        target_compile_options(${TARGET_NAME} PRIVATE
                $<$<CONFIG:Debug>:-Wall -Wextra -Werror -O0 -g>
                $<$<CONFIG:Release>:-Wall -Wextra -DNDEBUG -Ofast -march=native -flto -fno-exceptions -fno-rtti -ffast-math>
        )

        # Set link options for Release configuration
        target_link_options(${TARGET_NAME} PRIVATE
                $<$<CONFIG:Release>:-flto>
        )

        # Insert test coverage flags if EML_TEST is set
        if (EML_TEST)
            target_compile_options(${TARGET_NAME} PRIVATE
                    -fprofile-arcs -ftest-coverage -fno-inline -fno-optimize-sibling-calls -exclude-unreachable-branches

            )
            target_link_options(${TARGET_NAME} PRIVATE
                    -fprofile-arcs -ftest-coverage -fno-inline
            )
        endif ()

    elseif ("${EML_ARCH}" STREQUAL "arm")
        # Add ARM-specific options here if needed
        message(STATUS "Configuring for ARM architecture")
        # Example:
        # target_compile_options(${TARGET_NAME} PRIVATE -march=armv8-a)
    elseif ("${EML_ARCH}" STREQUAL "risc-v")
        # Add RISC-V-specific options here if needed
        message(STATUS "Configuring for RISC-V architecture")
        # Example:
        # target_compile_options(${TARGET_NAME} PRIVATE -march=rv64gc)
    else ()
        message(FATAL_ERROR "Chosen platform is not supported by the current compiler")
    endif ()

else ()
    message(FATAL_ERROR "Current compiler is not supported")
endif ()