#ifndef EML_UNIT_TEST_H
#define EML_UNIT_TEST_H

#include <eml/platform.h>

//===============================================
// Test
//===============================================
// ................................................................................
// A small testing framework that handles different results and prints out stats with a clean run macro
// eml does not use frameworks that handle compiling and running the test because:
//      - Prevents running tests on the target
//      - Does not account for diverse compilers or toolchains
// ................................................................................

struct TestContext final
{
    int passedTests = 0;
    int failedTest = 0;
    int currentTest = 0;
    const char* currentFile = nullptr;
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
                   "===========================================\n\n",
                   "Total", TEST_CONTEXT.currentTest, "Passed", TEST_CONTEXT.passedTests,
                   (int)( (float)TEST_CONTEXT.passedTests / (float)TEST_CONTEXT.currentTest * 100.0F ), "Failed",
                   TEST_CONTEXT.failedTest,
                   (int)( (float)TEST_CONTEXT.failedTest / (float)TEST_CONTEXT.currentTest * 100.0F ));
}

#define EML_RUN_TEST( func, ... ) func( __VA_ARGS__ );

#define EML_ASSERT_TENSOR_EQUALS( expected, actual )                                                                   \
    const auto res = ops::Equals( expected, actual );                                                                  \
    if( !res )                                                                                                         \
    {                                                                                                                  \
        PlatformPrint( "[Test] (%d) %s :%s:%d\n     -> %s\n", TEST_CONTEXT.currentTest, __FUNCTION__, __FILE__,        \
                       __LINE__, "FAILED" );                                                                           \
        PlatformPrint( "Expected:\n" );                                                                                \
        expected.print();                                                                                              \
        PlatformPrint( "Actual:\n" );                                                                                  \
        actual.print();                                                                                                \
        TEST_CONTEXT.failedTest++;                                                                                     \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        ++TEST_CONTEXT.passedTests;                                                                                    \
    }                                                                                                                  \
    ++TEST_CONTEXT.currentTest;                                                                                        \
    return res;

#endif // EML_UNIT_TEST_H