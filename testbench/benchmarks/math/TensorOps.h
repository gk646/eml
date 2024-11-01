#ifndef EML_BENCHMARK_TENSOR_OPS_H
#define EML_BENCHMARK_TENSOR_OPS_H

using namespace ops;

template <typename T>
void matmulStack( volatile int count, volatile int size )
{
    float aData[ size * size ]{};
    Tensor<T> A{ size, size };
    A.allocateCustom( aData, size * size );

    float bData[ size * size ]{};
    Tensor<T> B{ size, size };
    B.allocateCustom( bData, size * size );

    float cData[ size * size ]{};
    Tensor<T> C{ size, size };
    C.allocateCustom( cData, size * size * size );

    EML_BENCHMARK( T, Matmul, count, size, A, B, C );
}

template <typename T>
void matmulAlloc( volatile int count, volatile int size )
{
    Tensor<T> A{ size, size };
    A.allocate();

    Tensor<T> B{ size, size };
    B.allocate();

    Tensor<T> C{ size, size };
    C.allocate();

    EML_BENCHMARK( T, Matmul, count, size, A, B, C );
}



inline void BenchTensorOps()
{
    constexpr int size = 128;
    constexpr int count = 100;

    matmulStack<float>( count, size );
    matmulStack<qint8_t>( count, size );
    matmulStack<qint16_t>( count, size );
    matmulStack<qint32_t>( count, size );

    matmulAlloc<float>( count, size );
    matmulAlloc<qint8_t>( count, size );
    matmulAlloc<qint16_t>( count, size );
    matmulAlloc<qint32_t>( count, size );
}

#endif // EML_BENCHMARK_TENSOR_OPS_H