#ifndef EML_TEST_TESTS_H
#define EML_TEST_TESTS_H

#include <eml/math/TensorOps.h>
#include <eml/nn/layers/Conv2D.h>
#include <eml/nn/layers/Linear.h>

using namespace eml::nn;
using namespace eml::ops;

#include "UnitTest.h"

// Math
#include "math/MatMul.h"
#include "math/TensorOps.h"

// Layers
#include "nn/layers/Conv2D.h"
#include "nn/layers/Linear.h"
#include "nn/models/LogicGates.h"

// Models
#include "nn/models/Misc.h"

inline void RunTests()
{
    PlatformPrint( " =========================================\n"
                   "|............... T E S T S ...............|\n"
                   " =========================================\n" );

    Tensor<float> A{ 3, 3 };
    A.allocate();
    Arange( A, 1.0F );
    A.print();
    A.transpose();
    A.print();
    return;

    // Math
    TestTensorOps();
    TestMATHMatMul();

    // Layers
    TestNNLinear();
    TestNNConv2D();

    // Models
    TestNNCustom();
    TestLogicGates();

    PrintTestStats();
}

#endif // EML_TEST_TESTS_H