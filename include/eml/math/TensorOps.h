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

// Applies the given operation to every tensor element
template <typename T>
void EOp( Tensor<T>& A, void ( *op )( T& ) );

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
void Zero( Tensor<T>& A )
{
    EOp( A, []( T& t ) { t = 0; } );
}

template <typename T>
void Random( Tensor<T>& A, T min, T max )
{
}

template <typename T>
void EOp( Tensor<T>& A, void ( *op )( T& ) )
{
    for( int32_t i = 0; i < A.size; ++i )
    {
        op( A[ i ] );
    }
}

} // namespace eml::ops

#endif // EML_TENSOR_OPS_H