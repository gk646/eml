#ifndef EML_BENCHMARKS_CONV2D_H
#define EML_BENCHMARKS_CONV2D_H

#include <eml/nn/layers/Conv2D.h>

using namespace eml::nn;

template <typename T>
void benchConv2DImpl()
{
    // Benchmark 3x255x255 RGB picture
    // With 64 out channels

    constexpr int size = 255;
    constexpr int count = 1;

    Conv2D<T> layer{ 3, 16, { 3, 3 } };

    Tensor<T> A{ 3, size, size };
    A.allocate();

    Tensor<T> C{ layer.getOutShape( A.shape() ) };
    C.allocate();

    EML_BENCHMARK_FLOPS( T, layer.forwardI, count, GetLayerOps<T>( layer, A.shape() ), A, C );
}

inline void BenchLayersConv2D()
{
    benchConv2DImpl<float>();
    benchConv2DImpl<qint8_t>();
    benchConv2DImpl<qint16_t>();
    benchConv2DImpl<qint32_t>();
}

#endif // EML_BENCHMARKS_CONV2D_H