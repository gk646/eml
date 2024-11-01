#ifndef EML_BENCHMARKS_BENCHMARKS_H
#define EML_BENCHMARKS_BENCHMARKS_H

#include "MicroBench.h"

#include "math/TensorOps.h"

inline void RunBenchmarks()
{
    BenchTensorOps();
}

#endif // EML_BENCHMARKS_BENCHMARKS_H