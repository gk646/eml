#ifndef EML_BENCHMARK_TENSOR_OPS_H
#define EML_BENCHMARK_TENSOR_OPS_H

#include <eml/nn/LayerUtil.h>

template <typename T>
void benchMatmulImpl()
{
    constexpr int size = 64;
    constexpr int count = 30;

    Tensor<T> A{ size, size };
    A.allocate();

    Tensor<T> B{ size, size };
    B.allocate();

    Tensor<T> C{ size, size };
    C.allocate();

    EML_BENCHMARK_FLOPS( T, matmul, count, size * size * size, A, B, C );
}

void benchMatmulBig()
{
    constexpr int size = 1000;
    constexpr int count = 1;

    Tensor<float> A{ size, size };
    A.allocate();

    Tensor<float> B{ size, size };
    B.allocate();

    Tensor<float> C{ size, size };
    C.allocate();

    EML_BENCHMARK_FLOPS( float, matmul, count, size * size * size, A, B, C );
}

template <typename T>
void matmulPrimitive(const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R)
{
    for (int i = 0; i < A.h; i++) {
        for (int j = 0; j < B.w; j++) {
            for (int k = 0; k < A.w; k++) {
                R[i * B.w + j] = A[i * A.w + k] * B[k * B.w + j];
            }
        }
    }
}

void benchMatmulBigPrimitive()
{
    constexpr int size = 1000;
    constexpr int count = 1;

    Tensor<float> A{ size, size };
    A.allocate();

    Tensor<float> B{ size, size };
    B.allocate();

    Tensor<float> C{ size, size };
    C.allocate();

    EML_BENCHMARK_FLOPS( float, matmulPrimitive, count, size * size * size, A, B, C );
}

inline void BenchTensorOps()
{
    benchMatmulImpl<qint8_t>();
    benchMatmulImpl<qint16_t>();
    benchMatmulImpl<qint32_t>();
    benchMatmulImpl<float>();
    benchMatmulBig();
    benchMatmulBigPrimitive();
}

#endif // EML_BENCHMARK_TENSOR_OPS_H