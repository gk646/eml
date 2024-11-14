#ifndef EML_BENCHMARK_TENSOR_OPS_H
#define EML_BENCHMARK_TENSOR_OPS_H

#include <eml/nn/LayerUtil.h>

template <typename T>
void benchMatmulImpl()
{
    constexpr int size = 128;
    constexpr int count = 10;

    Tensor<float> A{ size, size };
    A.allocate();

    Tensor<float> B{ size, size };
    B.allocate();

    Tensor<float> C{ size, size };
    C.allocate();

    EML_BENCHMARK_FLOPS( T,matmul, count, size * size * size, A, B, C );
}

inline void BenchTensorOps()
{
    benchMatmulImpl<float>();
    benchMatmulImpl<qint8_t>();
    benchMatmulImpl<qint16_t>();
    benchMatmulImpl<qint32_t>();
}

#endif // EML_BENCHMARK_TENSOR_OPS_H