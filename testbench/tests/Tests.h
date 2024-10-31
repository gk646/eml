#ifndef EML_TEST_TESTS_H
#define EML_TEST_TESTS_H

#include "UnitTest.h"

// Math
#include "math/TensorOps.h"

// Layers
#include "nn/layers/Conv2D.h"
#include "nn/layers/Linear.h"
#include "nn/models/LogicGates.h"

// Models
#include "nn/models/Misc.h"

inline void RunTests()
{
    // Math
    TestTensorOps();

    // Layers
    TestNNLayersLinear();
    TestNNLayersConv2D();

    // Models
    TestNNCustom();
    TestLogicGates();

    PrintTestStats();
}

#endif // EML_TEST_TESTS_H