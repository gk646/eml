add_executable(eml-testbench main.cpp)

target_include_directories(eml-testbench
        PRIVATE ${CMAKE_SOURCE_DIR}/include
        PRIVATE ${CMAKE_SOURCE_DIR}/tests
)

target_link_libraries(eml-testbench PRIVATE eml)

set(TARGET_NAME eml-testbench)
include(CompilerOptions)

# Custom target to run the test executable and generate a coverage report
add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}testbench/CMakeFiles/eml-testbench.dir/main.cpp.gcda
        # Step 2: Run the test executable to generate fresh .gcda files
        COMMAND ${TARGET_NAME}

        # Step 3: Generate an HTML coverage report with gcovr
        COMMAND gcovr
        --root="${CMAKE_SOURCE_DIR}/include/eml"
        --object-directory="${CMAKE_BINARY_DIR}/testbench/CMakeFiles/eml-testbench.dir"
        --html
        --html-details
        --output="${CMAKE_BINARY_DIR}/coverage_report.html"
        --sort=uncovered-percent
        --txt-metric branch

        COMMENT "Running ${TARGET_NAME} and generating coverage report with gcovr"

        DEPENDS ${TARGET_NAME}
)