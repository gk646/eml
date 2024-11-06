#ifndef EML_BENCHMARK_TENSOR_OPS_H
#define EML_BENCHMARK_TENSOR_OPS_H

template <typename T>
void benchMatmulImpl()
{
    constexpr int size = 128;
    constexpr int count = 100;

    Tensor<float> A{ size, size };
    A.allocate();

    Tensor<float> B{ size, size };
    B.allocate();

    Tensor<float> C{ size, size };
    C.allocate();

    EML_BENCHMARK_MATMUL( T, ops::Matmul, count, size, A, B, C );
}

inline void benchMatmul()
{
    benchMatmulImpl<float>();
    benchMatmulImpl<qint8_t>();
    benchMatmulImpl<qint16_t>();
    benchMatmulImpl<qint32_t>();
}

inline void BenchTensorOps()
{
    benchMatmul();

    constexpr int size = 1024;

    Tensor<float> A{ size, size };
    A.allocate();

    Tensor<float> B{ size, size };
    B.allocate();

    Tensor<float> C{ size, size };
    C.allocate();

    auto start = PlatformClock();
    ops::Matmul( A, B, C );
    volatile auto end = PlatformClock();

    PlatformPrint( "Matmul time: %f\n", (end - start) / 1'000'000.0F );
}

#endif // EML_BENCHMARK_TENSOR_OPS_H