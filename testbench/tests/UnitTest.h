#ifndef EML_UNIT_TEST_H
#define EML_UNIT_TEST_H

#include <eml/platform.h>

//==============================================-
// Test
//==============================================-
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
    const char* currentFile = nullptr;
};

enum TestResult : int
{
    TEST_FAILED,
    TEST_PASSED,
    TEST_SKIPPED,
};

inline TestContext TEST_CONTEXT{};

inline void PrintTestStats()
{
    PlatformPrint( " =========================================\n"
                   "|............... S T A T S ...............|\n"
                   " =========================================\n"
                   "| %-7s : %3d                           |\n"
                   "| %-7s : %3d (%3d%%)                    |\n"
                   "| %-7s : %3d (%3d%%)                    |\n"
                   "| %-7s : %3d                           |\n"
                   "| %-7s : %3d                           |\n"
                   "==========================================-\n\n",
                   "Total", TEST_CONTEXT.currentTest, "Passed", TEST_CONTEXT.passedTest,
                   (int)( (float)TEST_CONTEXT.passedTest / (float)TEST_CONTEXT.currentTest * 100.0F ), "Failed",
                   TEST_CONTEXT.failedTest,
                   (int)( (float)TEST_CONTEXT.failedTest / (float)TEST_CONTEXT.currentTest * 100.0F ), "Skipped",
                   TEST_CONTEXT.skippedTest, "Unknown", TEST_CONTEXT.unknownTest );
}

#define EML_RUN_TEST( func, ... )                                                                                      \
    {                                                                                                                  \
        const auto res = func( __VA_ARGS__ );                                                                          \
        switch( res )                                                                                                  \
        {                                                                                                              \
        case TestResult::TEST_PASSED:                                                                                  \
            TEST_CONTEXT.passedTest++;                                                                                 \
            break;                                                                                                     \
        case TestResult::TEST_FAILED:                                                                                  \
            TEST_CONTEXT.failedTest++;                                                                                 \
            PlatformPrint( "[Test] (%d) %s :%s:%d\n     -> %s\n", TEST_CONTEXT.currentTest, #func, __FILE__, __LINE__, \
                           "FAILED" );                                                                                 \
                                                                                                                       \
            break;                                                                                                     \
        case TestResult::TEST_SKIPPED:                                                                                 \
            TEST_CONTEXT.skippedTest++;                                                                                \
            PlatformPrint( "[Test] (%d) %s :%s:%d\n     -> %s\n", TEST_CONTEXT.currentTest, #func, __FILE__, __LINE__, \
                           "SKIPPED" );                                                                                \
                                                                                                                       \
            break;                                                                                                     \
        default:                                                                                                       \
            TEST_CONTEXT.unknownTest++;                                                                                \
            PlatformPrint( "[Test] (%d) %s :%s:%d\n     -> %s\n", TEST_CONTEXT.currentTest, #func, __FILE__, __LINE__, \
                           "UNKONW" );                                                                                 \
                                                                                                                       \
            break;                                                                                                     \
        }                                                                                                              \
        ++TEST_CONTEXT.currentTest;                                                                                    \
    }

#define EML_ASSERT_TENSOR_EQUALS( expected, actual )                                                                   \
    const auto res = ops::Equals( expected, actual );                                                                  \
    if( !res )                                                                                                         \
    {                                                                                                                  \
        PlatformPrint( "Expected:\n" );                                                                                \
        expected.print();                                                                                              \
        PlatformPrint( "Actual:\n" );                                                                                  \
        actual.print();                                                                                                \
    }                                                                                                                  \
    return res;

#endif // EML_UNIT_TEST_H