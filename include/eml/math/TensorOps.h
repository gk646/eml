#ifndef EML_TENSOR_OPS_H
#define EML_TENSOR_OPS_H

#include "eml/nn/Model.h"
#include <eml/math/MathUtil.h>
#include <eml/math/Tensor.h>
#include <eml/util/Macros.h>
#include <xsimd/xsimd.hpp>

// ================================================================
// TensorOps
// ================================================================
// ................................................................
// Various operations performed on tensors
// ................................................................

namespace eml
{

template <typename T>
void matmul( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R, nn::Model<T>* model = nullptr );

// Matrix multiplication of A and B into R with A being handled like its transposed
template <typename T>
void matmulATrans( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R, nn::Model<T>* model = nullptr );

// Matrix multiplication of A and B into R with B being handled like its transposed
template <typename T>
void matmulBTrans( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R, nn::Model<T>* model = nullptr );

// Sets all values to zero
template <typename T>
void zero( Tensor<T>& A );

// Sets all values to random values withing the given range (inclusive)
template <typename T>
void rand( Tensor<T>& A, T min, T max );

// Sets all values in the tensor to the given val
template <typename T>
void fill( Tensor<T>& A, T val );

// Fills the given dimension(s) with the given value - 0-based indexing
// Fills all unspecified dimensions with the given offset - if no offset given is equal to fill()
template <typename T>
void fillDim( Tensor<T>& A, T val, int32_t n = -1, int32_t c = -1, int32_t h = -1 );

// Returns the value of the greatest element
template <typename T>
T max( const Tensor<T>& A );

// Returns the value of the smallest element
template <typename T>
T min( const Tensor<T>& A );

// Calls the given operation with every tensor element
template <typename T>
void ElemOp( Tensor<T>& A, void ( *op )( T& ) );

// Fills the Tensor
template <typename T>
void arange( Tensor<T>& A, T start, T step = T( 1 ) );

} // namespace eml

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

namespace eml
{

namespace impl
{

// Heavily inspired by:
// https://en.algorithmica.org/hpc/algorithms/matmul/

template <typename T, int32_t size, bool transposeA, bool transposeB>
void matmulKernel( const Tensor<T>& AT, const Tensor<T>& BT, Tensor<T>& RT, const int32_t x, const int32_t y,
                   const int32_t l, const int32_t r )
{
    const T* restrict A = AT.data();
    const T* restrict B = BT.data();
    T* restrict R = RT.data();
    if constexpr( size == 2 )
    {
        T c00 = 0, c01 = 0, c10 = 0, c11 = 0;
        for( int32_t k = l; k < r; k++ )
        {
            T a0 = transposeA ? A[ k * AT.w + x ] : A[ x * AT.w + k ];
            T a1 = transposeA ? A[ k * AT.w + x + 1 ] : A[ ( x + 1 ) * AT.w + k ];

            // read elements from B
            T b0 = transposeB ? B[ y * BT.w + k ] : B[ k * BT.w + y ];
            T b1 = transposeB ? B[ ( y + 1 ) * BT.w + k ] : B[ k * BT.w + y + 1 ];

            // update all combinations
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        // write the results to C
        R[ x * RT.w + y ] = c00;
        R[ x * RT.w + y + 1 ] = c01;
        R[ ( x + 1 ) * RT.w + y ] = c10;
        R[ ( x + 1 ) * RT.w + y + 1 ] = c11;
    }
    else
    {
        xsimd::batch<T> t[ size ]{};
        for( int32_t k = l; k < r; k++ )
        {
            xsimd::batch<T> bBatch;
            if constexpr( !transposeB )
            {
                bBatch = xsimd::load_unaligned( &B[ k * BT.w + y ] );
            }
            else
            {
                T bVals[ size ];
                for( int32_t idx = 0; idx < size; ++idx )
                {
                    bVals[ idx ] = B[ ( y + idx ) * BT.w + k ];
                }
                bBatch = xsimd::load_unaligned( bVals );
            }

            if constexpr( transposeA )
            {
                for( int32_t i = 0; i < size; ++i )
                {
                    xsimd::batch<T> alpha = xsimd::batch<T>::broadcast( A[ k * AT.w + x + i ] );
                    t[ i ] = xsimd::fma( alpha, bBatch, t[ i ] );
                }
            }
            else
            {
                for( int32_t i = 0; i < size; i += 4 )
                {
                    xsimd::batch<T> alpha0{ A[ ( x + i ) * AT.w + k ] };
                    xsimd::batch<T> alpha1{ A[ ( x + i + 1 ) * AT.w + k ] };
                    xsimd::batch<T> alpha2{ A[ ( x + i + 2 ) * AT.w + k ] };
                    xsimd::batch<T> alpha3{ A[ ( x + i + 3 ) * AT.w + k ] };

                    t[ i ] = xsimd::fma( alpha0, bBatch, t[ i ] );
                    t[ i + 1 ] = xsimd::fma( alpha1, bBatch, t[ i + 1 ] );
                    t[ i + 2 ] = xsimd::fma( alpha2, bBatch, t[ i + 2 ] );
                    t[ i + 3 ] = xsimd::fma( alpha3, bBatch, t[ i + 3 ] );
                }
            }
        }

        for( int32_t i = 0; i < size; i += 4 )
        {
            const auto res0 = xsimd::load_unaligned( &R[ ( x + i ) * RT.w + y ] ) + t[ i ];
            const auto res1 = xsimd::load_unaligned( &R[ ( x + i + 1 ) * RT.w + y ] ) + t[ i + 1 ];
            const auto res2 = xsimd::load_unaligned( &R[ ( x + i + 2 ) * RT.w + y ] ) + t[ i + 2 ];
            const auto res3 = xsimd::load_unaligned( &R[ ( x + i + 3 ) * RT.w + y ] ) + t[ i + 3 ];

            xsimd::store_unaligned( &R[ ( x + i ) * RT.w + y ], res0 );
            xsimd::store_unaligned( &R[ ( x + i + 1 ) * RT.w + y ], res1 );
            xsimd::store_unaligned( &R[ ( x + i + 2 ) * RT.w + y ], res2 );
            xsimd::store_unaligned( &R[ ( x + i + 3 ) * RT.w + y ], res3 );
        }
    }
}

template <typename T, bool transposeA, bool transposeB>
void matmulImpl( const Tensor<T>& restrict A, const Tensor<T>& restrict B, Tensor<T>& restrict R )
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

    for( int32_t x = 0; x < stepsA; x += simdValues )
        for( int32_t y = 0; y < stepsB; y += simdValues )
            impl::matmulKernel<T, simdValues, transposeA, transposeB>( A, B, R, x, y, 0, Aw );

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
void matmul( const Tensor<T>& restrict A, const Tensor<T>& restrict B, Tensor<T>& restrict R, nn::Model<T>* model )
{
    impl::matmulImpl<T, false, false>( A, B, R );
}

template <typename T>
void matmulATrans( const Tensor<T>& restrict A, const Tensor<T>& restrict B, Tensor<T>& restrict R,
                   nn::Model<T>* model )
{
    impl::matmulImpl<T, true, false>( A, B, R );
}

template <typename T>
void matmulBTrans( const Tensor<T>& restrict A, const Tensor<T>& restrict B, Tensor<T>& restrict R,
                   nn::Model<T>* model )
{
    impl::matmulImpl<T, false, true>( A, B, R );
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
            if( std::abs( A[ i ] - B[ i ] ) > 0.0001F )
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
void fill( Tensor<T>& A, T val )
{
    for( int32_t i = 0; i < A.size; ++i )
    {
        A[ i ] = val;
    }
}

template <typename T>
void fillDim( Tensor<T>& A, T val, int32_t n, int32_t c, int32_t h )
{
    if( n != -1 && c != -1 && h != -1 ) // fill a single row (width)
    {
        const int32_t offset = n * A.chw + c * A.hw + h * A.w;
        for( int32_t i = 0; i < A.w; ++i )
        {
            A[ offset + i ] = val;
        }
    }
    else if( n != -1 && c != -1 )       // fill an entire matrix (H x W)
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
    else if( n != -1 ) // fill an entire batch (C x H x W)
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
        fill( A, val );
    }
}

template <typename T>
T max( const Tensor<T>& A )
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
T min( const Tensor<T>& A )
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
void zero( Tensor<T>& tensor )
{
    for( int32_t i = 0; i < tensor.size; ++i )
    {
        tensor[ i ] = T( 0 );
    }
}

template <typename T>
void rand( Tensor<T>& restrict A, T min, T max )
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
void ElemOp( Tensor<T>& restrict A, void ( *op )( T& ) )
{
    for( int32_t i = 0; i < A.size; ++i )
    {
        op( A[ i ] );
    }
}

template <typename T>
void arange( Tensor<T>& restrict A, T start, T step )
{
    for( int32_t i = 0; i < A.size; ++i )
    {
        A[ i ] = start + T( i ) * step;
    }
}

} // namespace eml

#endif // EML_TENSOR_OPS_H