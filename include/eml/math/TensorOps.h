#ifndef EML_TENSOR_OPS_H
#define EML_TENSOR_OPS_H

#include <eml/math/MathUtil.h>
#include <eml/math/Tensor.h>
#include <xsimd/xsimd.hpp>

// ================================================================
// TensorOps
// ================================================================
// ................................................................
// Various operations performed on tensors
// ................................................................

namespace eml::ops
{

// Matrix multiplication of A and B into R
template <typename T>
void Matmul( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R );

// Matrix multiplication of A and B into R with A being handled like its transposed
template <typename T>
void MatmulATrans( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R );

// Matrix multiplication of A and B into R with B being handled like its transposed
template <typename T>
void MatmulBTrans( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R );

template <typename AT, typename BT>
bool Equals( const Tensor<AT>& A, const Tensor<BT>& B );

// Sets all values to zero
template <typename T>
void Zero( Tensor<T>& A );

// Sets all values to random values withing the given range (inclusive)
template <typename T>
void Random( Tensor<T>& A, T min, T max );

// Sets all values in the tensor to the given val
template <typename T>
void Fill( Tensor<T>& A, T val );

// Fills the given dimension(s) with the given value - 0-based indexing
// Fills all unspecified dimensions with the given offset - if no offset given is equal to Fill()
template <typename T>
void FillDim( Tensor<T>& A, T val, int32_t n = -1, int32_t c = -1, int32_t h = -1 );

// Returns the value of the greatest element
template <typename T>
T Max( const Tensor<T>& A );

// Returns the value of the smallest element
template <typename T>
T Min( const Tensor<T>& A );

// Calls the given operation with every tensor element
template <typename T>
void ElemOp( Tensor<T>& A, void ( *op )( T& ) );

// Fills the Tensor
template <typename T>
void Arange( Tensor<T>& A, T start, T step = T( 1 ) );

} // namespace eml::ops

// IMPLEMENTATION
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//

namespace eml::ops
{

namespace impl
{

// Heavily inspired by:
// https://en.algorithmica.org/hpc/algorithms/matmul/

template <typename T, int size, bool transposeA, bool transposeB>
void Kernel( const Tensor<T>& __restrict__ A, const Tensor<T>& __restrict__ B, Tensor<T>& __restrict__ R, const int x,
             const int y, const int l, const int r )
{
    if constexpr( size == 2 )
    {
        T c00 = 0, c01 = 0, c10 = 0, c11 = 0;

        for( int k = l; k < r; k++ )
        {

            T a0 = transposeA ? A[ k * A.w + x ] : A[ x * A.w + k ];
            T a1 = transposeA ? A[ k * A.w + x + 1 ] : A[ ( x + 1 ) * A.w + k ];

            // read elements from B
            T b0 = transposeB ? B[ y * B.w + k ] : B[ k * B.w + y ];
            T b1 = transposeB ? B[ ( y + 1 ) * B.w + k ] : B[ k * B.w + y + 1 ];

            // update all combinations
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        // write the results to C
        R[ x * R.w + y ] = c00;
        R[ x * R.w + y + 1 ] = c01;
        R[ ( x + 1 ) * R.w + y ] = c10;
        R[ ( x + 1 ) * R.w + y + 1 ] = c11;
        return;
    }

    xsimd::batch<T> t[ size ]{};

    for( int k = l; k < r; k++ )
    {
        xsimd::batch<T> bBatch;
        if( !transposeB )
        {
            bBatch = xsimd::load_unaligned( &B[ k * B.w + y ] );
        }
        else
        {
            T bVals[ size ];
            for( int idx = 0; idx < size; ++idx )
            {
                bVals[ idx ] = B[ ( y + idx ) * B.w + k ];
            }
            bBatch = xsimd::load_unaligned( bVals );
        }

        for( int i = 0; i < size; ++i )
        {
            T aVal;
            if( !transposeA )
            {
                aVal = A[ ( x + i ) * A.w + k ];
            }
            else
            {
                aVal = A[ k * A.w + x + i ];
            }
            xsimd::batch<T> alpha = xsimd::batch<T>::broadcast( aVal );

            t[ i ] += alpha * bBatch;
        }
    }

    for( int i = 0; i < size; i += 4 )
    {
        const auto res0 = xsimd::load_unaligned( &R[ ( x + i ) * R.w + y ] ) + t[ i ];
        const auto res1 = xsimd::load_unaligned( &R[ ( x + i + 1 ) * R.w + y ] ) + t[ i + 1 ];
        const auto res2 = xsimd::load_unaligned( &R[ ( x + i + 2 ) * R.w + y ] ) + t[ i + 2 ];
        const auto res3 = xsimd::load_unaligned( &R[ ( x + i + 3 ) * R.w + y ] ) + t[ i + 3 ];

        xsimd::store_unaligned( &R[ ( x + i ) * R.w + y ], res0 );
        xsimd::store_unaligned( &R[ ( x + i + 1 ) * R.w + y ], res1 );
        xsimd::store_unaligned( &R[ ( x + i + 2 ) * R.w + y ], res2 );
        xsimd::store_unaligned( &R[ ( x + i + 3 ) * R.w + y ], res3 );
    }
}

template <typename T, bool transposeA, bool transposeB>
void MatmulImpl( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R )
{
    const int32_t Ah = transposeA ? A.w : A.h;
    const int32_t Aw = transposeA ? A.h : A.w;
#ifdef EML_DEBUG
    const int32_t Bh = transposeB ? B.w : B.h;
#endif
    const int32_t Bw = transposeB ? B.h : B.w;
    EML_ASSERT( Aw == Bh && R.h == Ah && R.w == Bw, "Invalid dimensions" );

    constexpr auto simdValues = xsimd::simd_type<T>::size == 0 ? 2 : xsimd::batch<T>::size;

    const int32_t stepsA = Ah - ( Ah % simdValues );
    const int32_t stepsB = Bw - ( Bw % simdValues );

    for( int x = 0; x < stepsA; x += simdValues )
        for( int y = 0; y < stepsB; y += simdValues )
            impl::Kernel<T, simdValues, transposeA, transposeB>( A, B, R, x, y, 0, Aw );

    for( int32_t h = 0; h < Ah; ++h )
    {
        if( h >= stepsA )
        {
            for( int32_t w = 0; w < Bw; ++w )
            {
                T sum = T( 0 );
                for( int32_t k = 0; k < Aw; ++k )
                {
                    const auto valA = transposeA ? A[ k * A.w + h ] : A[ h * A.w + k ];
                    const auto valB = transposeB ? B[ w * B.w + k ] : B[ k * B.w + w ];
                    sum += valA * valB;
                }
                R[ h * R.w + w ] = sum;
            }
        }
        else
        {
            for( int32_t w = stepsB; w < Bw; ++w )
            {
                T sum = T( 0 );
                for( int32_t k = 0; k < Aw; ++k )
                {
                    const auto valA = transposeA ? A[ k * A.w + h ] : A[ h * A.w + k ];
                    const auto valB = transposeB ? B[ w * B.w + k ] : B[ k * B.w + w ];
                    sum += valA * valB;
                }
                R[ h * R.w + w ] = sum;
            }
        }
    }
}

} // namespace impl

template <typename T>
void Matmul( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R )
{
    impl::MatmulImpl<T, false, false>( A, B, R );
}

template <typename T>
void MatmulATrans( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R )
{
    impl::MatmulImpl<T, true, false>( A, B, R );
}

template <typename T>
void MatmulBTrans( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R )
{
    impl::MatmulImpl<T, false, true>( A, B, R );
}

template <typename AT, typename BT>
bool Equals( const Tensor<AT>& A, const Tensor<BT>& B )
{
    if constexpr( !std::is_same_v<AT, BT> )
        return false;

    if( A.n != B.n || A.c != B.c || A.h != B.h || A.w != B.w )
        return false;

    for( int32_t i = 0; i < A.size; ++i )
    {
        if constexpr( std::is_same_v<float, AT> )
        {
            if( abst( A[ i ] - B[ i ] ) > 0.0001F )
            {
                return false;
            }
        }
        else
        {
            if( A[ i ] != B[ i ] )
                return false;
        }
    }
    return true;
}

template <typename T>
void Fill( Tensor<T>& A, T val )
{
    for( int32_t i = 0; i < A.size; ++i )
    {
        A[ i ] = val;
    }
}

template <typename T>
void FillDim( Tensor<T>& A, T val, int32_t n, int32_t c, int32_t h )
{
    if( n != -1 && c != -1 && h != -1 ) // Fill a single row (width)
    {
        const int32_t offset = n * A.chw + c * A.hw + h * A.w;
        for( int32_t i = 0; i < A.w; ++i )
        {
            A[ offset + i ] = val;
        }
    }
    else if( n != -1 && c != -1 ) // Fill an entire matrix (H x W)
    {
        int32_t offset = n * A.chw + c * A.hw;
        for( int32_t i = 0; i < A.h; ++i )
        {
            for( int32_t j = 0; j < A.w; ++j )
            {
                A[ offset + j ] = val;
            }
            offset += A.w;
        }
    }
    else if( n != -1 ) // Fill an entire batch (C x H x W)
    {
        int32_t offset = n * A.chw;
        for( int32_t i = 0; i < A.c; ++i )
        {
            int32_t matrixOff = offset;
            for( int32_t j = 0; j < A.h; ++j )
            {
                for( int32_t k = 0; k < A.w; ++k )
                {
                    A[ matrixOff + k ] = val;
                }
                matrixOff += A.w;
            }
            offset += A.hw;
        }
    }
    else
    {
        Fill( A, val );
    }
}

template <typename T>
T Max( const Tensor<T>& A )
{
    EML_ASSERT( A.size > 0, "Cannot determine max of empty tensor!" );
    T max = A[ 0 ];
    for( int32_t i = 1; i < A.size; ++i )
    {
        if( A[ i ] > max )
            max = A[ i ];
    }
    return max;
}

template <typename T>
T Min( const Tensor<T>& A )
{
    EML_ASSERT( A.size > 0, "Cannot determine min of empty tensor!" );
    T min = A[ 0 ];
    for( int32_t i = 1; i < A.size; ++i )
    {
        if( A[ i ] < min )
            min = A[ i ];
    }
    return min;
}

template <typename T>
void Zero( Tensor<T>& A )
{
    for( int32_t i = 0; i < A.size; ++i )
    {
        A[ i ] = T( 0 );
    }
}

template <typename T>
void Random( Tensor<T>& A, T min, T max )
{
    if constexpr( std::is_same_v<T, float> )
    {
        for( int32_t i = 0; i < A.size; ++i )
        {
            A[ i ] = GetRandomFloat( min, max );
        }
    }
}

template <typename T>
void ElemOp( Tensor<T>& A, void ( *op )( T& ) )
{
    for( int32_t i = 0; i < A.size; ++i )
    {
        op( A[ i ] );
    }
}

template <typename T>
void Arange( Tensor<T>& A, T start, T step )
{
    for( int32_t i = 0; i < A.size; ++i )
    {
        A[ i ] = start + T( i ) * step;
    }
}

} // namespace eml::ops

#endif // EML_TENSOR_OPS_H