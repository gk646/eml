#ifndef EML_BENCHMARKS_Conv2DIm2Col_H
#define EML_BENCHMARKS_Conv2DIm2Col_H

#include <eml/nn/layers/Conv2DIm2Col.h>

using namespace eml::nn;

template <typename T>
void benchConv2DIm2ColImpl()
{
    // Benchmark 3x255x255 RGB picture
    // With 16 out channels

    constexpr int size = 500;
    constexpr int count = 20;

    Conv2DIm2Col<T> layer{ 3, 16, { 3, 3 } };

    Tensor<T> A{ 3, size, size };
    A.allocate();

    Tensor<T> C{ layer.getOutShape( A.shape() ) };
    C.allocate();

    EML_BENCHMARK_FLOPS( T, layer.forwardI, count, GetLayerOps<T>( layer, A.shape() ), A, C );
}

inline void BenchLayersConv2DIm2Col()
{
    benchConv2DIm2ColImpl<float>();
}

#endif // EML_BENCHMARKS_Conv2DIm2Col_H