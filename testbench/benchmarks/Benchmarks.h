#ifndef EML_BENCHMARKS_BENCHMARKS_H
#define EML_BENCHMARKS_BENCHMARKS_H

#include "MicroBench.h"

#include <eml/platform.h>
#include <eml/math/Tensor.h>
#include <eml/math/TensorOps.h>

#include "math/TensorOps.h"

using namespace eml;

inline void RunBenchmarks()
{
    PlatformPrint( "-------------------------------------------\n" );
    PlatformPrint( "|............... B E N C H ...............|\n" );
    PlatformPrint( "-------------------------------------------\n" );

    BenchTensorOps();
}

#endif // EML_BENCHMARKS_BENCHMARKS_H