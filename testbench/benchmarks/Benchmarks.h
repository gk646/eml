#ifndef EML_BENCHMARKS_BENCHMARKS_H
#define EML_BENCHMARKS_BENCHMARKS_H

#include "MicroBench.h"

#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>
#include <eml/platform.h>

#include "math/TensorOps.h"
#include "nn/layers/Conv2D.h"
#include "nn/layers/Conv2DIm2Col.h"

using namespace eml;

inline void RunBenchmarks()
{
    PlatformPrint( " =========================================\n" );
    PlatformPrint( "|............... B E N C H ...............|\n" );
    PlatformPrint( " =========================================\n" );

    BenchTensorOps();

    // Layers
    BenchLayersConv2D();
    BenchLayersConv2DIm2Col();
}

#endif // EML_BENCHMARKS_BENCHMARKS_H