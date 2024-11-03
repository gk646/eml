#ifndef EML_TENSOR_OPS_H
#define EML_TENSOR_OPS_H

#include <eml/math/Tensor.h>

namespace eml::ops
{

// Matrix multiplication of A and B into R
template <typename AT, typename BT, typename RT>
void Matmul( const Tensor<AT>& A, const Tensor<BT>& B, Tensor<RT>& R );

// Returns true if both tensors have the same shape and data
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

template <typename AT, typename BT, typename RT>
void Matmul( const Tensor<AT>& A, const Tensor<BT>& B, Tensor<RT>& R )
{
    // Type
    static_assert( std::is_same_v<AT, BT> && std::is_same_v<BT, RT> && "Tensor types must match" );

    // [1][1]  A       [1][1][1]  B         [1][1][1]  R
    // [1][1]  3x2     [1][1][1]  2x3       [1][1][1]  3x3
    // [1][1]                               [1][1][1]
    EML_ASSERT( A.w == B.h && R.h == A.h && R.w == B.w, "Invalid dimensions" );

    for( int32_t height = 0; height < A.h; ++height )
    {
        const int32_t idxa = height * A.w;
        const int32_t idxr = height * R.w;
        for( int32_t width = 0; width < B.w; ++width )
        {
            AT sum = 0;
            for( int k = 0; k < A.w; ++k )
            {
                sum += A[ idxa + k ] * B[ k * B.w + width ];
            }
            R[ idxr + width ] = sum;
        }
    }
}

template <typename AT, typename BT>
bool Equals( const Tensor<AT>& A, const Tensor<BT>& B )
{
    if constexpr( !std::is_same_v<AT, BT> )
        return false;

    if( A.n != B.n || A.c != B.c || A.h != B.h || A.w != B.w )
        return false;

    for( int32_t batch = 0; batch < A.n; ++batch )
    {
        for( int32_t channel = 0; channel < A.c; ++channel )
        {
            for( int32_t height = 0; height < A.h; ++height )
            {
                for( int32_t width = 0; width < A.w; ++width )
                {
                    const auto a = A[ batch * A.chw + channel * A.hw + height * A.w + width ];
                    const auto b = B[ batch * A.chw + channel * A.hw + height * A.w + width ];
                    if( a != b )
                        return false;
                }
            }
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

} // namespace eml::ops

#endif // EML_TENSOR_OPS_H