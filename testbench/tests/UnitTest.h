#ifndef EML_UNIT_TEST_H
#define EML_UNIT_TEST_H

#include <eml/platform.h>

//-----------------------------------------------
// Test
//-----------------------------------------------
// ................................................................................
// A small testing framework that handles different results and prints out stats with a clean run macro
// eml does not use frameworks that handle compiling and running the test because:
//      - Prevents running tests on the target
//      - Does not account for diverse compilers or toolchains
// ................................................................................

struct TestContext final
{
    int passedTest = 0;
    int failedTest = 0;
    int skippedTest = 0;
    int unknownTest = 0;
    int currentTest = 0;
};

enum TestResult : int
{
    TEST_FAILED,
    TEST_PASSED,
    TEST_SKIPPED,
};

inline TestContext TEST_CONTEXT{};

#define EML_RUN_TEST( func, ... )                                                                                      \
    {                                                                                                                  \
        const auto res = func( __VA_ARGS__ );                                                                          \
        const char* result_str = nullptr;                                                                              \
        switch( res )                                                                                                  \
        {                                                                                                              \
        case TestResult::TEST_PASSED:                                                                                  \
            TEST_CONTEXT.passedTest++;                                                                                 \
            result_str = "PASSED";                                                                                     \
            break;                                                                                                     \
        case TestResult::TEST_FAILED:                                                                                  \
            TEST_CONTEXT.failedTest++;                                                                                 \
            result_str = "FAILED";                                                                                     \
            break;                                                                                                     \
        case TestResult::TEST_SKIPPED:                                                                                 \
            TEST_CONTEXT.skippedTest++;                                                                                \
            result_str = "SKIPPED";                                                                                    \
            break;                                                                                                     \
        default:                                                                                                       \
            TEST_CONTEXT.unknownTest++;                                                                                \
            result_str = "UNKNOWN";                                                                                    \
            break;                                                                                                     \
        }                                                                                                              \
        PlatformPrint( "Test [%d]:%s\n     -> %s\n", TEST_CONTEXT.currentTest, #func, result_str );                    \
        ++TEST_CONTEXT.currentTest;                                                                                    \
    }

inline void PrintTestStats()
{
    eml::PlatformPrint( "----------------------------------------\n"
                        "Test stats:\n"
                        "  Total Tests   :%3d\n"
                        "  Passed Tests  :%3d (%3d%%)\n"
                        "  Failed Tests  :%3d (%3d%%)\n"
                        "  Skipped Tests :%3d\n"
                        "  Unknown Tests :%3d\n"
                        "----------------------------------------\n",
                        TEST_CONTEXT.currentTest, TEST_CONTEXT.passedTest,
                        (int)( (float)TEST_CONTEXT.passedTest / (float)TEST_CONTEXT.currentTest * 100.0F ),
                        (int)( (float)TEST_CONTEXT.failedTest / (float)TEST_CONTEXT.currentTest * 100.0F ),
                        TEST_CONTEXT.failedTest, TEST_CONTEXT.skippedTest, TEST_CONTEXT.unknownTest );
}

#endif // EML_UNIT_TEST_H