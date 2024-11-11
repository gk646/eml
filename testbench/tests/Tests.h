#ifndef EML_TEST_TESTS_H
#define EML_TEST_TESTS_H

#include <eml/math/TensorOps.h>
#include <eml/nn/layers/Conv1D.h>
#include <eml/nn/layers/Conv2D.h>
#include <eml/nn/layers/LeakyReLU.h>
#include <eml/nn/layers/Linear.h>
#include <eml/nn/layers/ReLU.h>
#include <eml/nn/layers/ReflectionPad2D.h>

using namespace eml::nn;
using namespace eml::ops;

#include "UnitTest.h"

// Math
#include "math/MatMul.h"
#include "math/TensorOps.h"

// Layers
#include "nn/layers/Conv1D.h"
#include "nn/layers/Conv2D.h"
#include "nn/layers/LeakyReLU.h"
#include "nn/layers/Linear.h"
#include "nn/layers/ReLU.h"
#include "nn/layers/ReflectionPad2D.h"
#include "nn/layers/ZeroPad2D.h"
#include "nn/models/LogicGates.h"

// Models
#include "nn/models/Misc.h"

inline void RunTests()
{
    PlatformPrint( " =========================================\n"
                   "|............... T E S T S ...............|\n"
                   " =========================================\n" );
    // Math
    TestTensorOps();
    TestMATHMatMul();

    // Layers
    TestNNLinear();
    TestNNConv2D();
    TestNNReflectionPad2D();
    TestNNZeroPad2D();
    TestNNConv1D();
    TestNNReLU();
    TestNNLeakyReLU();

    // Models
    TestNNCustom();
    TestLogicGates();

    PrintTestStats();
}

#endif // EML_TEST_TESTS_H