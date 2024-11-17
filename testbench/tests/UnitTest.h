#ifndef EML_UNIT_TEST_H
#define EML_UNIT_TEST_H

#include <eml/platform.h>

//===============================================
// Test
//===============================================
// ................................................................................
// A small testing framework that handles results and prints out stats with a clean run macro
// eml does not use frameworks that handle compiling and running the test because:
//      - Prevents running tests on the target
//      - Does not account for diverse compilers or toolchains
// Loosely inspired by Unity
// ................................................................................

struct TestContext final
{
    int32_t passedTests = 0;
    int32_t failedTest = 0;
    int32_t currentTest = 0;
    const char* currentFile = nullptr;

    bool isNewFile( const char* newFile ) const
    {
        const char* current = currentFile;
        while( *newFile )
        {
            if( *newFile != *current )
            {
                return false;
            }
            ++current;
            ++newFile;
        }
        return *newFile == '\0' && *current == '\0';
    }
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
                   (int32_t)( (float)TEST_CONTEXT.passedTests / (float)TEST_CONTEXT.currentTest * 100.0F ), "Failed",
                   TEST_CONTEXT.failedTest,
                   (int32_t)( (float)TEST_CONTEXT.failedTest / (float)TEST_CONTEXT.currentTest * 100.0F ) );
}

#define EML_RUN_TEST( func, ... ) func( __VA_ARGS__ );

#define EML_ASSERT_TENSOR_EQUALS( expected, actual )                                                                   \
    if( TEST_CONTEXT.currentFile != nullptr && !TEST_CONTEXT.isNewFile( __FILE_NAME__ ) )                              \
    {                                                                                                                  \
        PlatformPrint( "[Testing]: %s\n", __FILE_NAME__ );                                                             \
    }                                                                                                                  \
    TEST_CONTEXT.currentFile = __FILE_NAME__;                                                                          \
    const auto res = Equals( expected, actual );                                                                       \
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