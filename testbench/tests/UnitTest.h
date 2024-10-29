#ifndef EML_UNIT_TEST_H
#define EML_UNIT_TEST_H

#include <eml/platform.h>

//-----------------------------------------------
// Test
//-----------------------------------------------
// ................................................................................
// A small testing framework that handles different results and prints out stats with a clean run macro
// ................................................................................


struct test_context final
{
    int passed_tests = 0;
    int failed_tests = 0;
    int skipped_tests = 0;
    int unknown_tests = 0;
    int test_num = 0;
};

enum test_result : int
{
    TEST_FAILED,
    TEST_PASSED,
    TEST_SKIPPED,
};

inline test_context test_context{};


#define EML_RUN_TEST(func, ...)                                                                                        \
    {                                                                                                                  \
        const auto res = func(__VA_ARGS__);                                                                            \
        const char* result_str = nullptr;                                                                              \
        switch (res)                                                                                                   \
        {                                                                                                              \
        case test_result::TEST_PASSED:                                                                                 \
            test_context.passed_tests++;                                                                               \
            result_str = "PASSED";                                                                                     \
            break;                                                                                                     \
        case test_result::TEST_FAILED:                                                                                 \
            test_context.failed_tests++;                                                                               \
            result_str = "FAILED";                                                                                     \
            break;                                                                                                     \
        case test_result::TEST_SKIPPED:                                                                                \
            test_context.skipped_tests++;                                                                              \
            result_str = "SKIPPED";                                                                                    \
            break;                                                                                                     \
        default:                                                                                                       \
            test_context.unknown_tests++;                                                                              \
            result_str = "UNKNOWN";                                                                                    \
            break;                                                                                                     \
        }                                                                                                              \
        PlatformPrint("Test [%d]:%s\n     -> %s\n", test_context.test_num, #func, result_str);                         \
        ++test_context.test_num;                                                                                       \
    }


inline void ecml_print_test_stats()
{
    eml::PlatformPrint("----------------------------------------\n"
                       "Test stats:\n"
                       "  Total Tests   :%3d\n"
                       "  Passed Tests  :%3d (%3d%%)\n"
                       "  Failed Tests  :%3d (%3d%%)\n"
                       "  Skipped Tests :%3d\n"
                       "  Unknown Tests :%3d\n"
                       "----------------------------------------\n",
                       test_context.test_num, test_context.passed_tests,
                       (int)((float)test_context.passed_tests / (float)test_context.test_num * 100.0F),
                       (int)((float)test_context.failed_tests / (float)test_context.test_num * 100.0F),
                       test_context.failed_tests, test_context.skipped_tests, test_context.unknown_tests);
}

#endif // EML_UNIT_TEST_H