#ifndef EML_BENCHMARK_TENSOR_OPS_H
#define EML_BENCHMARK_TENSOR_OPS_H

template <typename T>
void matmulStack( const int count, const int size )
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

    EML_BENCH_OPS_IT( T, ops::Matmul, count, size, A, B, C );
}

inline void BenchTensorOps()
{
    matmulStack<float>( 1'000'000, 8 );
    matmulStack<float>( 1'000, 32 );
    matmulStack<float>( 1'000, 128 );

    matmulStack<int>( 1'000'000, 8 );
    matmulStack<int>( 1'000, 32 );
    matmulStack<int>( 1'000, 128 );
}

#endif // EML_BENCHMARK_TENSOR_OPS_H