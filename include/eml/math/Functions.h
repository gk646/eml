#ifndef EML_MATH_FUNCTIONS_H
#define EML_MATH_FUNCTIONS_H

#include <eml/math/TensorOps.h>

namespace eml
{

// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
// Applies the ReLu activation function to all elements of a tensor
template <typename T>
void ReLu( Tensor<T>& A );

// Applies the ReLu activation function to the given value
template <typename T>
void Relu( T& t );

// Applies the LeakyReLu activation function to all elements of a tensor
template <typename T>
void LeakyReLu( Tensor<T>& A );

// Applies the ReLu activation function to the given value
template <typename T>
void LeakyReLu( T& t );

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

// Inplace operation
template <typename T>
void ReLu( Tensor<T>& A )
{
    ops::ElemOp( A, Relu );
}

template <typename T>
void Relu( T& t )
{

    if( t > 0 )
        return;
    t = 0;
}

template <typename T>
void LeakyReLu( Tensor<T>& A )
{
    if constexpr( std::is_same_v<T, float> )
    {
        ops::ElemOp( A, LeakyReLu );
    }
    else
    {
        static_assert( std::is_same_v<T, double> && "Not implemented" );
    }
}

template <typename T>
void LeakyReLu( T& t )
{
    if( t > 0.0F )
        return;
    t = 0.01F;
}

} // namespace eml

#endif // EML_MATH_FUNCTIONS_H