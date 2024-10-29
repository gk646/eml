#ifndef EML_TEST_TESTS_H
#define EML_TEST_TESTS_H

#include "UnitTest.h"

// Math
#include "math/TensorOps.h"

// Models
#include "nn/models/Misc.h"
#include "nn/models/LogicGates.h"

inline void RunTests()
{
    // Math
    test_math_tensor_ops();

    // Models
    test_models_handmade();

    TestLogicGates();

    // Stats
    ecml_print_test_stats();
}

#endif //EML_TEST_TESTS_H