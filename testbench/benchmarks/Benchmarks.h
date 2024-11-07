#ifndef EML_BENCHMARKS_BENCHMARKS_H
#define EML_BENCHMARKS_BENCHMARKS_H

#include "MicroBench.h"

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>
#include <eml/platform.h>

#include "math/TensorOps.h"
#include "nn/layers/Conv2D.h"

using namespace eml;

inline void RunBenchmarks()
{
    PlatformPrint( " =========================================\n" );
    PlatformPrint( "|............... B E N C H ...............|\n" );
    PlatformPrint( " =========================================\n" );

    BenchTensorOps();

    // Layers
    BenchLayersConv2D();
}

#endif // EML_BENCHMARKS_BENCHMARKS_H