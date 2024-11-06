#ifndef EML_TENSOR_OPS_H
#define EML_TENSOR_OPS_H

#include <eml/math/Tensor.h>
#include <xsimd/xsimd.hpp>

namespace eml::ops
{

// Matrix multiplication of A and B into R
template <typename T>
void Matmul( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R );

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

namespace impl
{

template <typename T, int size>
void Kernel( const Tensor<T>& __restrict__ A, const Tensor<T>& __restrict__ B, Tensor<T>& __restrict__ R, const int x,
             const int y, const int l, const int r )
{
    xsimd::batch<T> t[ size ]{};

    for( int k = l; k < r; k++ )
    {
        const auto bBatch = xsimd::load_unaligned( &B[ k * B.w + y ] );
        if constexpr( std::is_same_v<T, float> )
        {
            for( int i = 0; i < size; i += 4 )
            {
                xsimd::batch<T> alpha0{ A[ ( x + i ) * A.w + k ] };
                xsimd::batch<T> alpha1{ A[ ( x + i + 1 ) * A.w + k ] };
                xsimd::batch<T> alpha2{ A[ ( x + i + 2 ) * A.w + k ] };
                xsimd::batch<T> alpha3{ A[ ( x + i + 3 ) * A.w + k ] };

                t[ i ] = xsimd::fma( alpha0, bBatch, t[ i ] );
                t[ i + 1 ] = xsimd::fma( alpha1, bBatch, t[ i + 1 ] );
                t[ i + 2 ] = xsimd::fma( alpha2, bBatch, t[ i + 2 ] );
                t[ i + 3 ] = xsimd::fma( alpha3, bBatch, t[ i + 3 ] );
            }
        }
        else
        {
            for( int i = 0; i < size; i += 4 )
            {
                xsimd::batch<T> alpha0{ A[ ( x + i ) * A.w + k ] };
                xsimd::batch<T> alpha1{ A[ ( x + i + 1 ) * A.w + k ] };
                xsimd::batch<T> alpha2{ A[ ( x + i + 2 ) * A.w + k ] };
                xsimd::batch<T> alpha3{ A[ ( x + i + 3 ) * A.w + k ] };

                t[ i ] += alpha0 * bBatch;
                t[ i + 1 ] += alpha1 * bBatch;
                t[ i + 2 ] += alpha2 * bBatch;
                t[ i + 3 ] += alpha3 * bBatch;
            }
        }
    }

    for( int i = 0; i < size; i++ )
    {
        const auto res = xsimd::load_unaligned( &R[ ( x + i ) * R.w + y ] ) + t[ i ];
        xsimd::store_unaligned( &R[ ( ( x + i ) * R.w + y ) ], res );
    }
}

} // namespace impl

template <typename T>
void Matmul( const Tensor<T>& A, const Tensor<T>& B, Tensor<T>& R )
{
    // [1][1]  A       [1][1][1]  B         [1][1][1]  R
    // [1][1]  3x2     [1][1][1]  2x3       [1][1][1]  3x3
    // [1][1]                               [1][1][1]
    EML_ASSERT( A.w == B.h && R.h == A.h && R.w == B.w, "Invalid dimensions" );

    constexpr auto simdValues = static_cast<int32_t>( xsimd::simd_type<T>::size );
    const int32_t stepsA = A.w - ( A.w % simdValues );
    const int32_t stepsB = B.w - ( B.w % simdValues );

    if constexpr( simdValues != 0 )
    {
        constexpr int s3 = 512;
        constexpr int s2 = 512;
        constexpr int s1 = 512;

        /* With cache blocking
        for( int i3 = 0; i3 < stepsB; i3 += s3 )
            // now we are working with b[:][i3:i3+s3]
                for( int i2 = 0; i2 < stepsA; i2 += s2 )
                    // now we are working with a[i2:i2+s2][:]
                        for( int i1 = 0; i1 < stepsB; i1 += s1 )
                            // now we are working with b[i1:i1+s1][i3:i3+s3]
                                // and we need to update c[i2:i2+s2][i3:i3+s3] with [l:r] = [i1:i1+s1]
                                    for( int x = i2; x < std::min( i2 + s2, stepsA ); x += simdValues )
                                        for( int y = i3; y < std::min( i3 + s3, stepsB ); y += simdValues )
                                            impl::Kernel<T, simdValues>( A, B, R, x, y, i1, std::min( i1 + s1, stepsA ) );
*/

        for( int x = 0; x < stepsA; x += simdValues )
            for( int y = 0; y < stepsB; y += simdValues )
                impl::Kernel<T, simdValues>( A, B, R, x, y, 0, R.w );
    }
    else
    {
        int32_t AchannelOff = 0;
        int32_t BchannelOff = 0;
        int32_t RchannelOff = 0;
        for( int32_t c = 0; c < A.c; ++c )
        {
            int32_t idxa = AchannelOff;
            int32_t idxr = RchannelOff;
            for( int32_t h = 0; h < A.h; ++h )
            {
                for( int32_t w = 0; w < B.w; ++w )
                {
                    T sum = T( 0 );
                    int32_t idxb = BchannelOff + w;
                    for( int32_t k = 0; k < A.w; ++k )
                    {
                        sum += A[ idxa + k ] * B[ idxb ];
                        idxb += B.w;
                    }
                    R[ idxr + w ] = sum;
                }
                idxa += A.w;
                idxr += R.w;
            }
            AchannelOff += A.hw;
            BchannelOff += B.hw;
            RchannelOff += R.hw;
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